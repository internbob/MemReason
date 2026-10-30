"""Phase 6C longitudinal-MRI temporal-signal identifiability audit."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from ..data.common import ROOT
from .phase5c_memory_measurement import clean_matrix
from .phase6a_latent_state_dynamics import (
    LatentSample,
    build_samples,
    fit_reference,
)
from .phase6b_mri_incremental_value import (
    IncrementalValueDynamics,
    fit_clinical_decoder,
    loss_function,
    prediction_rows,
    reconstruct_state_records,
    summarize,
    training_batch,
)


CONFIG = ROOT / "configs/phase6c_longitudinal_mri_identifiability_dev.yaml"
PROTOCOL = ROOT / "reports/experiments/phase6c_longitudinal_mri_identifiability_protocol.md"


def _stable_seed(text: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{text}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def cognitive_cutoff_dates(config: dict[str, Any], items: list[dict[str, Any]]) -> dict[str, pd.Timestamp]:
    """Recreate Phase 5G sample IDs and map them to conservative actual dates."""
    frame = pd.read_csv(ROOT / "splits/train.csv", low_memory=False)
    raw = clean_matrix(frame, items)
    keep = np.isfinite(raw).sum(1) >= 2
    frame = frame.loc[keep].reset_index(drop=True)
    sample_ids = [f"{row.subject_id}:{row.visit_code}:{index}"
                  for index, row in enumerate(frame.itertuples())]
    column = str(config["mri"]["cognitive_cutoff_date"])
    dates = pd.to_datetime(frame[column], errors="coerce")
    return dict(zip(sample_ids, dates))


def dynamic_features(
    times_years: np.ndarray,
    burden: np.ndarray,
    disagreement: np.ndarray,
) -> np.ndarray:
    """Five prespecified actual-time longitudinal features."""
    slope = float(np.polyfit(times_years, burden, 1)[0])
    last_gap = max(float(times_years[-1] - times_years[-2]), 1e-6)
    return np.asarray([
        burden[-1] - burden[0],
        slope,
        burden[-1] - burden[-2],
        (burden[-1] - burden[-2]) / last_gap,
        disagreement[-1] - disagreement[0],
    ], dtype=np.float32)


def aggregate_longitudinal_mri(
    records: pd.DataFrame,
    cutoff_dates: dict[str, pd.Timestamp],
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], pd.DataFrame]:
    """Create true and date-permuted features without changing static MRI state."""
    source = pd.read_csv(ROOT / config["mri_burden"], low_memory=False)
    source = source[source.split.eq("train")].copy()
    source["scan_date"] = pd.to_datetime(source.scan_date, errors="coerce")
    source = source.dropna(subset=["scan_date", "burden_mean", "burden_disagreement"])
    minimum_scans = int(config["mri"]["minimum_scans"])
    minimum_span = float(config["mri"]["minimum_span_years"])
    true: dict[str, np.ndarray] = {}
    permuted: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for row in records.itertuples(index=False):
        cutoff = cutoff_dates.get(str(row.sample_id), pd.NaT)
        scans = source[(source.subject_id.astype(str) == str(row.subject_id))
                       & (source.scan_date <= cutoff)].sort_values(["scan_date", "scan_id"])
        eligible = False
        span = 0.0
        if len(scans) >= minimum_scans and pd.notna(cutoff):
            span = float((scans.scan_date.iloc[-1] - scans.scan_date.iloc[0]).days / 365.25)
            eligible = span >= minimum_span
        if not eligible:
            true[str(row.sample_id)] = np.zeros(10, np.float32)
            permuted[str(row.sample_id)] = np.zeros(10, np.float32)
            continue
        dates = scans.scan_date.to_numpy(dtype="datetime64[ns]")
        times = (dates - dates[0]) / np.timedelta64(1, "D") / 365.25
        times = np.asarray(times, float)
        burden = scans.burden_mean.to_numpy(float)
        disagreement = scans.burden_disagreement.to_numpy(float)
        recency = max(float((cutoff - scans.scan_date.iloc[-1]).days / 365.25), 0.0)
        static = np.asarray([
            burden[-1], disagreement[-1], recency,
            math.log1p(len(scans)), math.log1p(span),
        ], np.float32)
        correct_dynamic = dynamic_features(times, burden, disagreement)
        rng = np.random.default_rng(_stable_seed(str(row.sample_id), int(config["partition_seed"])))
        shift = int(rng.integers(1, len(scans)))
        order = np.roll(np.arange(len(scans)), shift)
        permuted_dynamic = dynamic_features(times, burden[order], disagreement[order])
        true[str(row.sample_id)] = np.concatenate([static, correct_dynamic]).astype(np.float32)
        permuted[str(row.sample_id)] = np.concatenate([static, permuted_dynamic]).astype(np.float32)
        rows.append({
            "state_sample_id": str(row.sample_id), "subject_id": str(row.subject_id),
            "visit_month": float(row.visit_month), "cognitive_cutoff_date": cutoff.date().isoformat(),
            "mri_scan_count": len(scans), "mri_span_years": span,
            "mri_recency_years": recency, "latest_scan_date": scans.scan_date.iloc[-1].date().isoformat(),
            "date_permutation_shift": shift,
            "date_permutation_changed": bool(not np.allclose(correct_dynamic, permuted_dynamic)),
        })
    return true, permuted, pd.DataFrame(rows)


def forecast_metadata(records: pd.DataFrame, state_metadata: pd.DataFrame,
                      horizons: list[int]) -> dict[str, dict[str, Any]]:
    eligible = state_metadata.set_index("state_sample_id").to_dict("index")
    rows = records.sort_values(["subject_id", "visit_month", "sample_id"], kind="stable")
    rows = rows.drop_duplicates(["subject_id", "visit_month"], keep="last")
    result: dict[str, dict[str, Any]] = {}
    for row in rows.itertuples(index=False):
        if str(row.sample_id) not in eligible:
            continue
        for horizon in horizons:
            key = f"{row.subject_id}:m{float(row.visit_month):g}:h{int(horizon)}"
            result[key] = eligible[str(row.sample_id)]
    return result


def apply_normalization(samples: list[LatentSample], mean: np.ndarray, std: np.ndarray) -> None:
    for sample in samples:
        sample.mri_features = ((sample.mri_features - mean) / std).astype(np.float32)


def normalization_statistics(samples: list[LatentSample]) -> tuple[np.ndarray, np.ndarray]:
    values = np.stack([sample.mri_features for sample in samples])
    mean = values.mean(0)
    std = np.where(values.std(0) > 1e-6, values.std(0), 1.0)
    return mean, std


def replace_with_aligned_features(samples: list[LatentSample], donor: list[LatentSample]) -> list[LatentSample]:
    lookup = {sample.sample_id: sample.mri_features.copy() for sample in donor}
    result = copy.deepcopy(samples)
    for sample in result:
        sample.mri_features = lookup[sample.sample_id].copy()
    return result


def subject_shuffle_dynamics(
    samples: list[LatentSample],
    metadata: dict[str, dict[str, Any]],
    dynamic_indices: list[int],
    seed: int,
) -> tuple[list[LatentSample], float, float]:
    """Exchange dynamics across subjects while retaining every static feature."""
    result = copy.deepcopy(samples)
    by_cutoff: dict[tuple[str, float], list[int]] = {}
    for index, sample in enumerate(result):
        by_cutoff.setdefault((sample.subject_id, float(sample.history_time[-1])), []).append(index)
    keys = sorted(by_cutoff)
    representative = {key: result[by_cutoff[key][0]] for key in keys}

    def stratum(key: tuple[str, float]) -> tuple[str, str]:
        info = metadata[representative[key].sample_id]
        count = int(info["mri_scan_count"])
        count_bin = "2" if count == 2 else ("3" if count == 3 else "4+")
        span = float(info["mri_span_years"])
        span_bin = "0.5-1" if span < 1 else ("1-2" if span < 2 else "2+")
        return count_bin, span_bin

    donor_for: dict[tuple[str, float], tuple[str, float]] = {}
    for key in keys:
        exact = [candidate for candidate in keys
                 if candidate[0] != key[0] and stratum(candidate) == stratum(key)]
        pool = exact or [candidate for candidate in keys if candidate[0] != key[0]]
        rng = np.random.default_rng(_stable_seed(f"{key[0]}:{key[1]}", seed))
        donor_for[key] = pool[int(rng.integers(0, len(pool)))]
    source_dynamic = {
        key: representative[key].mri_features[dynamic_indices].copy() for key in keys
    }
    changed = 0
    donor_changed = 0
    for key, indices in by_cutoff.items():
        source = source_dynamic[donor_for[key]]
        before = source_dynamic[key]
        donor_changed += int(donor_for[key][0] != key[0])
        changed += int(not np.allclose(before, source))
        for index in indices:
            result[index].mri_features[dynamic_indices] = source
    denominator = max(len(keys), 1)
    return result, changed / denominator, donor_changed / denominator


def flip_dynamics(samples: list[LatentSample], dynamic_indices: list[int]) -> list[LatentSample]:
    result = copy.deepcopy(samples)
    for sample in result:
        sample.mri_features[dynamic_indices] *= -1
    return result


def feature_mask(arm: str, config: dict[str, Any]) -> torch.Tensor:
    mask = torch.zeros(len(config["mri"]["features"]), dtype=torch.float32)
    if arm == "static_mri":
        mask[list(map(int, config["mri"]["static_feature_indices"]))] = 1
    elif arm != "cognition_only":
        mask[:] = 1
    return mask


def train_model(
    arm: str,
    fit: list[LatentSample],
    development: list[LatentSample],
    reference: tuple[float, float],
    decoder: tuple[float, float, float],
    config: dict[str, Any],
    device: torch.device,
    seed: int,
) -> IncrementalValueDynamics:
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = IncrementalValueDynamics(
        True, arm != "cognition_only", feature_mask(arm, config),
        float(config["training"]["minimum_variance"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    fit_batch = training_batch(fit, device)
    dev_batch = training_batch(development, device)
    best = copy.deepcopy(model.state_dict()); best_loss = float("inf"); stale = 0
    for _ in range(int(config["training"]["max_epochs"])):
        model.train(); optimizer.zero_grad()
        mean, variance = model(fit_batch, reference)
        loss = loss_function(mean, variance, fit_batch, decoder, config)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip"]))
        optimizer.step(); model.eval()
        with torch.no_grad():
            mean, variance = model(dev_batch, reference)
            score = float(loss_function(mean, variance, dev_batch, decoder, config))
        if score < best_loss - 1e-6:
            best = copy.deepcopy(model.state_dict()); best_loss = score; stale = 0
        else:
            stale += 1
            if stale >= int(config["training"]["patience"]):
                break
    model.load_state_dict(best); model.best_dev_loss = best_loss
    return model


def paired_bootstrap(predictions: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    candidate = str(config["selection"]["candidate"])
    rng = np.random.default_rng(int(config["statistics"]["bootstrap_seed"]))
    repetitions = int(config["statistics"]["bootstrap_replicates"])
    rows: list[dict[str, Any]] = []
    for comparator in config["arms"]:
        if comparator == candidate:
            continue
        for metric in ["composite_absolute_error", "latent_absolute_error"]:
            pair = predictions[predictions.arm.isin([candidate, comparator])].pivot_table(
                index=["sample_id", "subject_id"], columns="arm", values=metric
            ).dropna()
            differences = (pair[candidate] - pair[comparator]).groupby("subject_id").mean().to_numpy()
            draws = differences[rng.integers(0, len(differences),
                                               size=(repetitions, len(differences)))].mean(1)
            rows.append({
                "candidate": candidate, "comparator": comparator, "metric": metric,
                "difference": float(differences.mean()),
                "ci_low": float(np.quantile(draws, 0.025)),
                "ci_high": float(np.quantile(draws, 0.975)),
                "subjects": len(differences), "replicates": repetitions,
            })
    return pd.DataFrame(rows)


def run(config_path: Path = CONFIG) -> Path:
    config = yaml.safe_load(config_path.read_text())
    items = yaml.safe_load((ROOT / config["item_contract"]).read_text())["primary_state_items"]
    records = reconstruct_state_records(config, items)
    cutoffs = cognitive_cutoff_dates(config, items)
    true_map, permuted_map, state_metadata = aggregate_longitudinal_mri(records, cutoffs, config)
    true_samples = [sample for sample in build_samples(records, true_map, config) if sample.mri_available]
    permuted_samples = [sample for sample in build_samples(records, permuted_map, config) if sample.mri_available]
    if not true_samples or {sample.sample_id for sample in true_samples} != {sample.sample_id for sample in permuted_samples}:
        raise RuntimeError("Longitudinal MRI cohort is empty or controls are not aligned")
    metadata = forecast_metadata(records, state_metadata, list(map(int, config["forecast"]["horizons_months"])))
    output = ROOT / "outputs" / config["run_id"]
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_root = output / "checkpoints"; checkpoint_root.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dynamic = list(map(int, config["mri"]["dynamic_feature_indices"]))
    all_rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    for fold in range(int(config["folds"])):
        evaluation = copy.deepcopy([sample for sample in true_samples if sample.fold == fold])
        outer_fit = copy.deepcopy([sample for sample in true_samples if sample.fold != fold])
        perm_eval = [sample for sample in permuted_samples if sample.fold == fold]
        perm_outer = [sample for sample in permuted_samples if sample.fold != fold]
        dev_subjects = {
            sample.subject_id for sample in outer_fit
            if int.from_bytes(hashlib.sha256(f"inner:{fold}:{sample.subject_id}".encode()).digest()[:8], "big") / 2**64
            < float(config["inner_dev_fraction"])
        }
        fit = [sample for sample in outer_fit if sample.subject_id not in dev_subjects]
        development = [sample for sample in outer_fit if sample.subject_id in dev_subjects]
        perm_fit = [sample for sample in perm_outer if sample.subject_id not in dev_subjects]
        perm_dev = [sample for sample in perm_outer if sample.subject_id in dev_subjects]
        mean, std = normalization_statistics(fit)

        raw_groups: dict[str, tuple[list[LatentSample], list[LatentSample], list[LatentSample]]] = {
            "cognition_only": (copy.deepcopy(fit), copy.deepcopy(development), copy.deepcopy(evaluation)),
            "static_mri": (copy.deepcopy(fit), copy.deepcopy(development), copy.deepcopy(evaluation)),
            "true_longitudinal_mri": (copy.deepcopy(fit), copy.deepcopy(development), copy.deepcopy(evaluation)),
            "date_permuted_dynamics": (
                replace_with_aligned_features(fit, perm_fit),
                replace_with_aligned_features(development, perm_dev),
                replace_with_aligned_features(evaluation, perm_eval),
            ),
        }
        shuffled_fit, sf_changed, sf_donor = subject_shuffle_dynamics(fit, metadata, dynamic, 10000 + fold)
        shuffled_dev, sd_changed, sd_donor = subject_shuffle_dynamics(development, metadata, dynamic, 20000 + fold)
        shuffled_eval, se_changed, se_donor = subject_shuffle_dynamics(evaluation, metadata, dynamic, 30000 + fold)
        raw_groups["subject_shuffled_dynamics"] = (shuffled_fit, shuffled_dev, shuffled_eval)
        raw_groups["sign_flipped_dynamics"] = (
            copy.deepcopy(fit), copy.deepcopy(development), copy.deepcopy(evaluation)
        )
        for groups in raw_groups.values():
            for group in groups:
                apply_normalization(group, mean, std)
        raw_groups["sign_flipped_dynamics"] = (
            raw_groups["sign_flipped_dynamics"][0], raw_groups["sign_flipped_dynamics"][1],
            flip_dynamics(raw_groups["sign_flipped_dynamics"][2], dynamic),
        )

        for partition, true_group, perm_group in [
            ("fit", fit, perm_fit), ("development", development, perm_dev),
            ("evaluation", evaluation, perm_eval),
        ]:
            perm_lookup = {sample.sample_id: sample for sample in perm_group}
            changed = np.mean([
                not np.allclose(sample.mri_features[dynamic], perm_lookup[sample.sample_id].mri_features[dynamic])
                for sample in true_group
            ])
            static_error = max(float(np.max(np.abs(
                sample.mri_features[:5] - perm_lookup[sample.sample_id].mri_features[:5]
            ))) for sample in true_group)
            control_rows.append({
                "fold": fold, "arm": "date_permuted_dynamics", "partition": partition,
                "dynamic_changed_fraction": changed, "different_subject_donor_fraction": np.nan,
                "maximum_static_feature_difference": static_error,
            })
        for partition, changed, donor in [
            ("fit", sf_changed, sf_donor), ("development", sd_changed, sd_donor),
            ("evaluation", se_changed, se_donor),
        ]:
            control_rows.append({
                "fold": fold, "arm": "subject_shuffled_dynamics", "partition": partition,
                "dynamic_changed_fraction": changed, "different_subject_donor_fraction": donor,
                "maximum_static_feature_difference": 0.0,
            })
        control_rows.append({
            "fold": fold, "arm": "sign_flipped_dynamics", "partition": "evaluation",
            "dynamic_changed_fraction": 1.0, "different_subject_donor_fraction": np.nan,
            "maximum_static_feature_difference": 0.0,
        })

        fit_subjects = {sample.subject_id for sample in fit}
        reference = fit_reference(records, fit_subjects, config)
        decoder = fit_clinical_decoder(records, fit_subjects)
        candidate_model: IncrementalValueDynamics | None = None
        for arm in config["arms"]:
            local_fit, local_dev, local_eval = raw_groups[arm]
            if arm == "sign_flipped_dynamics":
                if candidate_model is None:
                    raise RuntimeError("true_longitudinal_mri must precede sign intervention")
                model = copy.deepcopy(candidate_model)
            else:
                model = train_model(
                    arm, local_fit, local_dev, reference, decoder, config, device,
                    20261030 + fold * 101,
                )
                if arm == "true_longitudinal_mri":
                    candidate_model = copy.deepcopy(model)
            model.eval()
            with torch.no_grad():
                batch = training_batch(local_eval, device)
                predicted, variance = model(batch, reference)
            all_rows.extend(prediction_rows(
                arm, local_eval, predicted.cpu().numpy(), variance.cpu().numpy(), decoder
            ))
            destination = checkpoint_root / arm; destination.mkdir(exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(), "fold": fold, "arm": arm,
                "reference": reference, "clinical_decoder": decoder,
                "best_dev_loss": model.best_dev_loss,
                "inference_intervention": arm == "sign_flipped_dynamics",
            }, destination / f"fold_{fold}.pt")
        manifests.append({
            "fold": fold, "fit_subjects": len(fit_subjects), "dev_subjects": len(dev_subjects),
            "evaluation_subjects": len({sample.subject_id for sample in evaluation}),
            "fit_samples": len(fit), "dev_samples": len(development),
            "evaluation_samples": len(evaluation), "reference_intercept": reference[0],
            "reference_slope_per_year": reference[1], "decoder_intercept": decoder[0],
            "decoder_slope": decoder[1], "decoder_residual_variance": decoder[2],
        })

    predictions = pd.DataFrame(all_rows)
    for column in ["mri_scan_count", "mri_span_years", "mri_recency_years"]:
        predictions[column] = predictions.sample_id.map(lambda value: metadata[value][column])
    predictions.to_csv(output / "crossfit_predictions.csv", index=False)
    state_metadata.to_csv(output / "eligible_state_mri_metadata.csv", index=False)
    pd.DataFrame(manifests).to_csv(output / "fold_manifest.csv", index=False)
    pd.DataFrame(control_rows).to_csv(output / "control_diagnostics.csv", index=False)
    aggregate = summarize(predictions, ["arm"])
    horizon = summarize(predictions, ["arm", "horizon_months"])
    folds = summarize(predictions, ["arm", "fold"])
    sensitivity = summarize(
        predictions[predictions.mri_span_years >= float(config["mri"]["sensitivity_span_years"])],
        ["arm"],
    )
    aggregate.to_csv(output / "aggregate_metrics.csv", index=False)
    horizon.to_csv(output / "metrics_by_horizon.csv", index=False)
    folds.to_csv(output / "metrics_by_fold.csv", index=False)
    sensitivity.to_csv(output / "sensitivity_12month_span_metrics.csv", index=False)
    bootstrap = paired_bootstrap(predictions, config)
    bootstrap.to_csv(output / "paired_subject_bootstrap.csv", index=False)

    candidate = str(config["selection"]["candidate"])
    aggregate_table = aggregate.set_index("arm")
    fold_table = folds.pivot(index="fold", columns="arm", values="composite_mae")
    horizon_table = horizon.pivot(index="horizon_months", columns="arm", values="composite_mae")
    comparisons: dict[str, Any] = {}
    for comparator in config["selection"]["comparators"]:
        comparisons[comparator] = {
            "overall_difference": float(aggregate_table.loc[candidate, "composite_mae"]
                                        - aggregate_table.loc[comparator, "composite_mae"]),
            "folds_better": int((fold_table[candidate] < fold_table[comparator]).sum()),
            "horizons_better": int((horizon_table[candidate] < horizon_table[comparator]).sum()),
        }
    lower, upper = map(float, config["selection"]["coverage_95_bounds"])
    coverage = float(aggregate_table.loc[candidate, "coverage_95"])
    comparison_gate = all(
        value["overall_difference"] < 0
        and value["folds_better"] >= int(config["selection"]["minimum_folds_better"])
        and value["horizons_better"] >= int(config["selection"]["minimum_horizons_better"])
        for value in comparisons.values()
    )
    coverage_gate = lower <= coverage <= upper
    supported = bool(comparison_gate and coverage_gate)
    decision = {
        "run_id": config["run_id"], "exploratory_only": True, "original_train_only": True,
        "validation_or_test_loaded": False, "candidate": candidate,
        "primary_endpoint": "future_observed_memory_composite_mae",
        "comparisons": comparisons, "candidate_coverage_95": coverage,
        "coverage_gate": coverage_gate, "longitudinal_mri_signal_supported": supported,
        "disagreement_filter_open": supported, "freeze_mri_as_static_auxiliary": not supported,
        "measurement_gate_reopened": False, "confirmatory_multimodal_claim_open": False,
    }
    (output / "development_decision.json").write_text(json.dumps(decision, indent=2) + "\n")

    primary_boot = bootstrap[bootstrap.metric.eq("composite_absolute_error")].set_index("comparator")
    lines = [
        "# Phase 6C: longitudinal MRI temporal-signal identifiability", "",
        "Original-train-only five-fold subject cross-fitting; validation and test were not loaded.", "",
        "## Primary observable endpoint", "",
        "| Arm | Memory composite MAE | Clinical NLL | 95% coverage | Interval width | Latent MAE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate.itertuples(index=False):
        lines.append(f"| {row.arm} | {row.composite_mae:.5f} | {row.composite_nll:.4f} | "
                     f"{row.coverage_95:.4f} | {row.interval_width:.4f} | {row.latent_mae:.5f} |")
    lines.extend(["", "## Prespecified temporal-signal comparisons", ""])
    for comparator, result in comparisons.items():
        boot = primary_boot.loc[comparator]
        lines.append(
            f"- true longitudinal minus `{comparator}` participant-mean MAE: "
            f"**{boot.difference:+.6f}** (95% participant-bootstrap CI "
            f"**[{boot.ci_low:+.6f}, {boot.ci_high:+.6f}]**); folds better "
            f"**{result['folds_better']}/5**; horizons better **{result['horizons_better']}/4**."
        )
    lines.extend([
        "", "## Decision", "",
        f"Candidate clinical coverage was **{coverage:.4f}**. Longitudinal MRI temporal signal "
        f"supported: **`{str(supported).lower()}`**. The next disagreement-aware filtering "
        f"experiment is **{'open' if supported else 'closed'}**; MRI is "
        f"**{'not ' if supported else ''}frozen as a static auxiliary covariate**.", "",
        "This exploratory audit does not reopen the Phase 5F measurement gate and cannot establish "
        "a confirmatory multimodal or continuous-time claim.",
    ])
    (output / "report.md").write_text("\n".join(lines) + "\n")
    manifest = {
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "phase5g_states_sha256": hashlib.sha256((ROOT / config["phase5g_states"]).read_bytes()).hexdigest(),
        "mri_source_sha256": hashlib.sha256((ROOT / config["mri_burden"]).read_bytes()).hexdigest(),
        "device": str(device), "state_records": len(records),
        "eligible_state_cutoffs": len(state_metadata), "forecast_samples": len(true_samples),
        "subjects": len({sample.subject_id for sample in true_samples}),
        "arms": len(config["arms"]), "checkpoint_files": len(config["arms"]) * int(config["folds"]),
        "independently_trained_models": (len(config["arms"]) - 1) * int(config["folds"]),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    print(run(args.config))


if __name__ == "__main__":
    main()
