"""Static fold-aligned MRI embedding incremental-value experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..data.common import ROOT
from .phase6a_latent_state_dynamics import build_samples, fit_reference, normalize_mri, reconstruct_state_records
from .phase6b_mri_incremental_value import (
    IncrementalValueDynamics, fit_clinical_decoder, loss_function, paired_bootstrap,
    prediction_rows, subject_shuffle_structural, summarize, training_batch,
)
from .phase6c_longitudinal_mri_identifiability import cognitive_cutoff_dates


CONFIG = ROOT / "configs/phase6d1_static_embedding_incremental_dev.yaml"
PROTOCOL = ROOT / "reports/experiments/phase6d1_static_embedding_incremental_protocol.md"


ARM_GROUPS = {
    "cognition_only": "none", "metadata_only": "metadata",
    "probability_static": "probability", "embedding_static": "embedding",
    "probability_plus_embedding": "all", "subject_shuffled_embedding": "embedding",
}


def feature_mask(arm: str, config: dict[str, Any]) -> torch.Tensor:
    count = len(config["mri"]["features"]); mask = torch.zeros(count)
    start, stop = map(int, config["mri"]["embedding_indices"])
    metadata = list(map(int, config["mri"]["metadata_indices"]))
    group = ARM_GROUPS[arm]
    if group in {"embedding", "all"}: mask[start:stop] = 1
    if group in {"probability", "all"}: mask[int(config["mri"]["probability_index"])] = 1
    if group != "none": mask[metadata] = 1
    return mask


def fold_partitions(records: pd.DataFrame, fold: int, fraction: float) -> tuple[set[str], set[str], set[str]]:
    evaluation = set(records.loc[records.fold.eq(fold), "subject_id"].astype(str))
    outer = set(records.loc[records.fold.ne(fold), "subject_id"].astype(str))
    development = {subject for subject in outer
                   if int.from_bytes(hashlib.sha256(f"inner:{fold}:{subject}".encode()).digest()[:8], "big") / 2**64 < fraction}
    return outer - development, development, evaluation


def fold_scan_representation(config: dict[str, Any], fold: int, fit_subjects: set[str]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    root = ROOT / config["embedding_run"]
    scans = pd.read_csv(root / "scan_index.csv", low_memory=False)
    scans["subject_id"] = scans.subject_id.astype(str)
    scans["scan_date"] = pd.to_datetime(scans.scan_date, errors="raise")
    raw = np.asarray(np.load(root / f"embeddings_fold{fold}_float16.npy", mmap_mode="r"), np.float32)
    probability = np.asarray(np.load(root / f"probability_fold{fold}_float32.npy", mmap_mode="r"), np.float32)
    fit_rows = scans.subject_id.isin(fit_subjects).to_numpy()
    if fit_rows.sum() < int(config["mri"]["pca_dimension"]):
        raise RuntimeError("Insufficient outer-fit scans for embedding PCA")
    scaler = StandardScaler().fit(raw[fit_rows])
    standardized_fit = scaler.transform(raw[fit_rows])
    pca = PCA(n_components=int(config["mri"]["pca_dimension"]), svd_solver="full").fit(standardized_fit)
    transformed = pca.transform(scaler.transform(raw)).astype(np.float32)
    return scans, {"embedding": transformed, "probability": probability,
                   "scaler_mean": scaler.mean_.astype(np.float32), "scaler_scale": scaler.scale_.astype(np.float32),
                   "pca_components": pca.components_.astype(np.float32),
                   "pca_explained_variance_ratio": pca.explained_variance_ratio_.astype(np.float32)}


def aggregate_static_features(records: pd.DataFrame, cutoffs: dict[str, pd.Timestamp], scans: pd.DataFrame,
                              representation: dict[str, np.ndarray], config: dict[str, Any]) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    by_subject = {subject: group.sort_values(["scan_date", "scan_id"])
                  for subject, group in scans.groupby("subject_id", sort=False)}
    result: dict[str, np.ndarray] = {}; rows = []; dimension = int(config["mri"]["pca_dimension"])
    for row in records.itertuples(index=False):
        cutoff = cutoffs.get(str(row.sample_id), pd.NaT); history = by_subject.get(str(row.subject_id))
        if history is None or pd.isna(cutoff):
            result[str(row.sample_id)] = np.zeros(dimension + 3, np.float32); continue
        eligible = history[history.scan_date <= cutoff]
        if eligible.empty:
            result[str(row.sample_id)] = np.zeros(dimension + 3, np.float32); continue
        latest = int(eligible.index[-1]); recency = max((cutoff - scans.loc[latest, "scan_date"]).days / 365.25, 0.0)
        vector = np.concatenate([representation["embedding"][latest],
                                 [representation["probability"][latest], recency, np.log1p(len(eligible))]]).astype(np.float32)
        result[str(row.sample_id)] = vector
        rows.append({"state_sample_id": str(row.sample_id), "subject_id": str(row.subject_id),
                     "latest_scan_id": scans.loc[latest, "scan_id"], "scan_count": len(eligible),
                     "recency_years": recency})
    return result, pd.DataFrame(rows)


def train_arm(arm: str, fit, development, reference, decoder, config, device, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = IncrementalValueDynamics(True, arm != "cognition_only", feature_mask(arm, config),
                                     float(config["training"]["minimum_variance"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]),
                                  weight_decay=float(config["training"]["weight_decay"]))
    fit_batch, dev_batch = training_batch(fit, device), training_batch(development, device)
    best, best_loss, stale = copy.deepcopy(model.state_dict()), float("inf"), 0
    for _ in range(int(config["training"]["max_epochs"])):
        model.train(); optimizer.zero_grad(); mean, variance = model(fit_batch, reference)
        loss = loss_function(mean, variance, fit_batch, decoder, config); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip"])); optimizer.step()
        model.eval()
        with torch.no_grad():
            mean, variance = model(dev_batch, reference); score = float(loss_function(mean, variance, dev_batch, decoder, config))
        if score < best_loss - 1e-6: best, best_loss, stale = copy.deepcopy(model.state_dict()), score, 0
        else:
            stale += 1
            if stale >= int(config["training"]["patience"]): break
    model.load_state_dict(best); model.best_dev_loss = best_loss; return model


def run(config_path: Path = CONFIG) -> Path:
    config = yaml.safe_load(config_path.read_text())
    if set(config["arms"]) != set(ARM_GROUPS): raise ValueError("Phase 6D1 arm contract mismatch")
    items = yaml.safe_load((ROOT / config["item_contract"]).read_text())["primary_state_items"]
    records = reconstruct_state_records(config, items); cutoffs = cognitive_cutoff_dates(config, items)
    output = ROOT / "outputs" / config["run_id"]; output.mkdir(parents=True, exist_ok=True)
    checkpoint_root = output / "checkpoints"; checkpoint_root.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_rows = []; manifests = []; metadata_rows = []
    embedding_indices = list(range(*map(int, config["mri"]["embedding_indices"])))
    for fold in range(int(config["folds"])):
        fit_subjects, dev_subjects, evaluation_subjects = fold_partitions(records, fold, float(config["inner_dev_fraction"]))
        scans, representation = fold_scan_representation(config, fold, fit_subjects)
        features, metadata = aggregate_static_features(records, cutoffs, scans, representation, config)
        metadata["fold_encoder"] = fold; metadata_rows.append(metadata)
        samples = build_samples(records, features, config)
        fit = copy.deepcopy([sample for sample in samples if sample.subject_id in fit_subjects])
        development = copy.deepcopy([sample for sample in samples if sample.subject_id in dev_subjects])
        evaluation = copy.deepcopy([sample for sample in samples if sample.subject_id in evaluation_subjects])
        reference = fit_reference(records, fit_subjects, config); decoder = fit_clinical_decoder(records, fit_subjects)
        normalize_mri(fit, [fit, development, evaluation])
        np.savez(output / f"embedding_transform_fold{fold}.npz", scaler_mean=representation["scaler_mean"],
                 scaler_scale=representation["scaler_scale"], pca_components=representation["pca_components"],
                 explained_variance_ratio=representation["pca_explained_variance_ratio"])
        for arm in config["arms"]:
            local_fit, local_dev, local_eval = copy.deepcopy(fit), copy.deepcopy(development), copy.deepcopy(evaluation)
            changed = 0.0
            if arm == "subject_shuffled_embedding":
                local_fit, _ = subject_shuffle_structural(local_fit, embedding_indices, 10000 + fold)
                local_dev, _ = subject_shuffle_structural(local_dev, embedding_indices, 20000 + fold)
                local_eval, changed = subject_shuffle_structural(local_eval, embedding_indices, 30000 + fold)
            model = train_arm(arm, local_fit, local_dev, reference, decoder, config, device, 20261030 + fold * 101)
            model.eval()
            with torch.no_grad():
                batch = training_batch(local_eval, device); mean, variance = model(batch, reference)
            all_rows.extend(prediction_rows(arm, local_eval, mean.cpu().numpy(), variance.cpu().numpy(), decoder))
            destination = checkpoint_root / arm; destination.mkdir(exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "fold": fold, "arm": arm, "best_dev_loss": model.best_dev_loss,
                        "reference": reference, "clinical_decoder": decoder}, destination / f"fold_{fold}.pt")
            if arm == "subject_shuffled_embedding" and changed <= 0: raise RuntimeError("Embedding shuffle did not change evaluation features")
        manifests.append({"fold": fold, "fit_subjects": len(fit_subjects), "dev_subjects": len(dev_subjects),
                          "evaluation_subjects": len(evaluation_subjects), "fit_samples": len(fit),
                          "dev_samples": len(development), "evaluation_samples": len(evaluation),
                          "mri_evaluation_samples": sum(sample.mri_available for sample in evaluation),
                          "pca_explained_variance": float(representation["pca_explained_variance_ratio"].sum())})
    predictions = pd.DataFrame(all_rows); predictions.to_csv(output / "crossfit_predictions.csv", index=False)
    pd.DataFrame(manifests).to_csv(output / "fold_manifest.csv", index=False)
    pd.concat(metadata_rows, ignore_index=True).to_csv(output / "state_mri_metadata.csv", index=False)
    aggregate = summarize(predictions, ["arm"]); horizon = summarize(predictions, ["arm", "horizon_months"]); folds = summarize(predictions, ["arm", "fold"])
    aggregate.to_csv(output / "aggregate_metrics.csv", index=False); horizon.to_csv(output / "metrics_by_horizon.csv", index=False); folds.to_csv(output / "metrics_by_fold.csv", index=False)
    bootstrap = paired_bootstrap(predictions, config); bootstrap.to_csv(output / "paired_subject_bootstrap.csv", index=False)
    candidate = config["selection"]["candidate"]; comparators = config["selection"]["comparators"]
    aggregate_table = aggregate.set_index("arm"); fold_table = folds.pivot(index="fold", columns="arm", values="composite_mae"); horizon_table = horizon.pivot(index="horizon_months", columns="arm", values="composite_mae")
    primary = bootstrap[bootstrap.metric.eq("composite_absolute_error")].set_index("comparator"); comparisons = {}
    for comparator in comparators:
        comparisons[comparator] = {"overall_difference": float(aggregate_table.loc[candidate, "composite_mae"] - aggregate_table.loc[comparator, "composite_mae"]),
                                   "participant_difference": float(primary.loc[comparator, "difference"]),
                                   "ci_low": float(primary.loc[comparator, "ci_low"]), "ci_high": float(primary.loc[comparator, "ci_high"]),
                                   "folds_better": int((fold_table[candidate] < fold_table[comparator]).sum()),
                                   "horizons_noninferior": int((horizon_table[candidate] - horizon_table[comparator] <= float(config["selection"]["maximum_horizon_degradation"])).sum())}
    low, high = map(float, config["selection"]["coverage_95_bounds"]); coverage = float(aggregate_table.loc[candidate, "coverage_95"])
    gate = all(value["overall_difference"] < 0 and value["ci_high"] < 0
               and value["folds_better"] >= int(config["selection"]["minimum_folds_better"])
               and value["horizons_noninferior"] >= int(config["selection"]["minimum_horizons_noninferior"])
               for value in comparisons.values()) and low <= coverage <= high
    decision = {"run_id": config["run_id"], "original_train_only": True, "validation_or_test_loaded": False,
                "candidate": candidate, "comparisons": comparisons, "candidate_coverage_95": coverage,
                "static_embedding_gate_passed": bool(gate), "phase6d2_open": bool(gate),
                "measurement_gate_reopened": False, "confirmatory_multimodal_claim_open": False}
    (output / "development_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    lines = ["# Phase 6D1: static native-T1 embedding incremental value", "", "Original-train-only five-fold subject cross-fitting; validation and test were not loaded.", "",
             "| Arm | Memory composite MAE | Clinical NLL | 95% coverage | Latent MAE |", "|---|---:|---:|---:|---:|"]
    for row in aggregate.itertuples(index=False): lines.append(f"| {row.arm} | {row.composite_mae:.5f} | {row.composite_nll:.4f} | {row.coverage_95:.4f} | {row.latent_mae:.5f} |")
    lines.extend(["", "## Prespecified embedding comparisons", ""])
    for comparator, value in comparisons.items(): lines.append(f"- embedding minus `{comparator}` participant MAE: **{value['participant_difference']:+.6f}** (95% CI **[{value['ci_low']:+.6f}, {value['ci_high']:+.6f}]**); folds **{value['folds_better']}/5**, horizons noninferior **{value['horizons_noninferior']}/4**.")
    lines.extend(["", f"Static embedding gate passed: **`{str(gate).lower()}`**. Phase 6D2 is **{'open' if gate else 'closed'}**."])
    (output / "report.md").write_text("\n".join(lines) + "\n")
    manifest = {"config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(), "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
                "device": str(device), "state_records": len(records), "predictions": len(predictions),
                "checkpoints": len(list(checkpoint_root.glob("*/fold_*.pt"))), "pca_dimension": int(config["mri"]["pca_dimension"])}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args(); print(run(args.config))


if __name__ == "__main__": main()
