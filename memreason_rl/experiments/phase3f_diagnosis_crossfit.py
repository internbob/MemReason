"""Train-only cross-fit replication of diagnosis-dependent spatial-grounding effects."""

from __future__ import annotations

import argparse
import copy
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader

from ..data.common import ROOT, save_csv, save_json
from .phase1_numeric import markdown_table, seed_everything
from .phase3b_mri_representation import (
    bootstrap_difference,
    ensemble,
    macro_huber,
    make_shuffle_indices,
    metrics,
)
from .phase3c_spatial_grounding import (
    SpatialDataset,
    SpatialGrounder,
    coordinate_grid,
    evaluate,
    prepare_arrays,
    stability_table,
)


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def build_train_frame(config: dict[str, Any]) -> pd.DataFrame:
    """Recover the exact Phase 3C train row order without loading val/test outcomes."""
    manifest = pd.read_csv(
        ROOT / "outputs" / config["sources"]["phase3c_run"] / "sample_manifest.csv",
        low_memory=False,
    )
    frame = manifest[manifest.split.eq("train")].copy()
    source = pd.read_csv(ROOT / "splits/train.csv", low_memory=False)
    source_columns = ["subject_id", "visit_month", "dxsum_diagnosis_label"]
    frame = frame.merge(
        source[source_columns],
        left_on=["subject_id", "cutoff_month"],
        right_on=["subject_id", "visit_month"],
        how="left",
        validate="one_to_one",
    )
    metadata = pd.read_csv(ROOT / "processed/mri_metadata.csv", low_memory=False)
    metadata["image_id"] = pd.to_numeric(metadata.image_id, errors="coerce")
    ida = pd.read_csv(ROOT / "idaSearch_10_12_2026 (2).csv", low_memory=False).rename(
        columns={"Image ID": "image_id", "Age": "age", "Sex": "sex"}
    )
    ida["image_id"] = pd.to_numeric(ida.image_id, errors="coerce")
    ida = ida[["image_id", "age", "sex"]].dropna(subset=["image_id"]).drop_duplicates("image_id")
    frame = frame.merge(
        metadata[["scan_id", "image_id", "cohort"]].drop_duplicates("scan_id"),
        on="scan_id", how="left", validate="one_to_one",
    ).merge(ida, on="image_id", how="left", validate="many_to_one")
    frame["age"] = pd.to_numeric(frame.age, errors="coerce")
    frame["sex_numeric"] = frame.sex.astype(str).str.upper().map({"M": 0.0, "F": 1.0})
    frame["diagnosis_group"] = frame.dxsum_diagnosis_label.fillna("Unknown").astype(str)
    if frame.age.isna().any() or frame.sex_numeric.isna().any():
        raise RuntimeError("Phase 3C train rows lost demographic covariates")
    if frame.scan_id.duplicated().any() or set(frame.split) != {"train"}:
        raise RuntimeError("Phase 3F must contain unique original-train scans only")
    expected = manifest[manifest.split.eq("train")].scan_id.astype(str).tolist()
    if frame.scan_id.astype(str).tolist() != expected:
        raise RuntimeError("Phase 3F row order no longer matches cached spatial features")
    return frame.reset_index(drop=True)


def assign_subject_folds(frame: pd.DataFrame, folds: int, seed: int) -> pd.DataFrame:
    severity = {"Unknown": 0, "CN": 1, "MCI": 2, "AD": 3}
    rows = []
    for subject, group in frame.groupby("subject_id", sort=True):
        labels = group.diagnosis_group.map(severity).fillna(0)
        diagnosis = max(severity, key=lambda label: labels.max() == severity[label])
        cohort = group.cohort.dropna().astype(str).mode()
        rows.append({"subject_id": subject, "diagnosis": diagnosis,
                     "cohort": cohort.iloc[0] if len(cohort) else "Unknown"})
    subjects = pd.DataFrame(rows)
    subjects["stratum"] = subjects.diagnosis + "|" + subjects.cohort
    counts = subjects.stratum.value_counts()
    rare = subjects.stratum.map(counts).lt(folds)
    subjects.loc[rare, "stratum"] = subjects.loc[rare, "diagnosis"]
    counts = subjects.stratum.value_counts()
    final_rare = subjects.stratum.map(counts).lt(folds)
    if final_rare.any():
        common = subjects.loc[~final_rare, "stratum"].mode().iloc[0]
        subjects.loc[final_rare, "stratum"] = common
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    subjects["fold"] = -1
    for fold, (_, held) in enumerate(splitter.split(subjects, subjects.stratum)):
        subjects.loc[held, "fold"] = fold
    if subjects.fold.lt(0).any() or subjects.subject_id.duplicated().any():
        raise RuntimeError("Invalid subject-fold registry")
    return subjects


def train_fold(
    arm: str, seed: int, fold: int, epochs: int, feature_path: Path,
    frame: pd.DataFrame, demos: np.ndarray, targets: np.ndarray, masks: np.ndarray,
    target_mean: np.ndarray, target_std: np.ndarray, coordinates: np.ndarray,
    config: dict[str, Any], output: Path, device: torch.device,
) -> tuple[pd.DataFrame, np.ndarray | None, np.ndarray, int]:
    settings = config["training"]
    optimization_seed = int(seed + 100 * fold)
    seed_everything(optimization_seed)
    train_indices = np.flatnonzero(frame.fold.ne(fold).to_numpy())
    held_indices = np.flatnonzero(frame.fold.eq(fold).to_numpy())
    train = SpatialDataset(train_indices, frame, demos, targets, masks, feature_path)
    held = SpatialDataset(held_indices, frame, demos, targets, masks, feature_path)
    local_shuffle = make_shuffle_indices(
        frame.iloc[held_indices].reset_index(drop=True),
        int(config["statistics"]["bootstrap_seed"]) + fold,
    )
    source_indices = np.arange(len(frame))
    source_indices[held_indices] = held_indices[local_shuffle]
    shuffled = SpatialDataset(
        held_indices, frame, demos, targets, masks, feature_path, source_indices=source_indices
    )
    model = SpatialGrounder(
        arm, int(config["backbone"]["feature_channels"]), int(settings["hidden_dim"]),
        len(config["targets"]), int(settings["attention_heads"]), float(settings["dropout"]),
        coordinates,
    ).to(device)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    directory = output / "checkpoints" / arm / str(seed) / f"fold{fold}"
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / "fixed_epoch.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        identity = (checkpoint["arm"], int(checkpoint["seed"]), int(checkpoint["fold"]), int(checkpoint["epochs"]))
        if identity != (arm, seed, fold, epochs):
            raise RuntimeError(f"Checkpoint identity mismatch: {checkpoint_path}")
        model.load_state_dict(checkpoint["model"])
    else:
        generator = torch.Generator().manual_seed(optimization_seed)
        loader = DataLoader(train, batch_size=int(settings["batch_size"]), shuffle=True,
                            generator=generator, num_workers=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["learning_rate"]),
                                      weight_decay=float(settings["weight_decay"]))
        curves = []
        for epoch in range(epochs):
            model.train()
            losses = []
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                prediction, _ = model(batch["tokens"].to(device), batch["demo"].to(device))
                loss = macro_huber(prediction, batch["target"].to(device), batch["mask"].to(device))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss for {arm}/{seed}/fold{fold}")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), float(settings["gradient_clip"]))
                optimizer.step()
                losses.append(float(loss.detach()))
            curves.append({"epoch": epoch, "train_loss": float(np.mean(losses))})
            print(json.dumps({"arm": arm, "seed": seed, "fold": fold,
                              "epoch": epoch, "train_loss": float(np.mean(losses))}), flush=True)
        torch.save({"model": model.state_dict(), "arm": arm, "seed": seed, "fold": fold,
                    "epochs": epochs, "optimization_seed": optimization_seed,
                    "trainable_parameters": parameter_count}, checkpoint_path)
        save_csv(pd.DataFrame(curves), directory / "curves.csv")
    target_names = list(config["targets"])
    batch_size = int(settings["batch_size"])
    observed, weights, order = evaluate(
        model, held, batch_size, device, target_names, target_mean, target_std,
        seed, "observed", return_attention=arm == "target_spatial_attention",
    )
    outputs = [observed]
    shuffled_prediction, _, _ = evaluate(
        model, shuffled, batch_size, device, target_names, target_mean, target_std,
        seed, "shuffled_subject",
    )
    outputs.append(shuffled_prediction)
    if arm == "target_spatial_attention":
        spatial = np.roll(np.arange(len(coordinates)), 1).astype(np.int64)
        query = np.roll(np.arange(len(target_names)), 1).astype(np.int64)
        for pairing, spatial_perm, query_perm in [
            ("spatial_permutation", spatial, None), ("query_permutation", None, query)
        ]:
            prediction, _, _ = evaluate(
                model, held, batch_size, device, target_names, target_mean, target_std,
                seed, pairing, spatial_permutation=spatial_perm, query_permutation=query_perm,
            )
            outputs.append(prediction)
    result = pd.concat(outputs, ignore_index=True)
    result["fold"] = fold
    return result, weights, order, parameter_count


def paired_effect(candidate: pd.DataFrame, baseline: pd.DataFrame,
                  scales: dict[str, float]) -> pd.DataFrame:
    keys = ["scan_id", "subject_id", "target", "truth"]
    paired = candidate[keys + ["prediction"]].merge(
        baseline[keys + ["prediction"]], on=keys,
        suffixes=("_candidate", "_baseline"), validate="one_to_one",
    )
    paired["difference"] = (
        (paired.prediction_candidate - paired.truth).abs()
        - (paired.prediction_baseline - paired.truth).abs()
    ) / paired.target.map(scales)
    return paired


def interaction_bootstrap(paired: pd.DataFrame, attributes: pd.DataFrame,
                          replicates: int, seed: int) -> dict[str, Any]:
    enriched = paired.join(attributes.set_index("scan_id")[["diagnosis_group"]], on="scan_id")
    groups = ["AD", "MCI"]
    subjects = sorted(enriched.subject_id.unique())
    table = []
    for subject in subjects:
        group = enriched[enriched.subject_id.eq(subject)]
        row = []
        for diagnosis in groups:
            values = group[group.diagnosis_group.eq(diagnosis)].difference
            row.extend([float(values.sum()), int(len(values))])
        table.append(row)
    table = np.asarray(table, dtype=float)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(replicates):
        sampled = table[rng.integers(0, len(table), size=len(table))].sum(0)
        if sampled[1] > 0 and sampled[3] > 0:
            draws.append(sampled[0] / sampled[1] - sampled[2] / sampled[3])
    point = (
        enriched[enriched.diagnosis_group.eq("AD")].difference.mean()
        - enriched[enriched.diagnosis_group.eq("MCI")].difference.mean()
    )
    return {"contrast": "AD_minus_MCI", "interaction": float(point),
            "ci_lower": float(np.quantile(draws, 0.025)),
            "ci_upper": float(np.quantile(draws, 0.975)),
            "replicates": len(draws)}


def fold_signature_stability(weights: dict[int, np.ndarray], frame: pd.DataFrame,
                             target_names: list[str], top_fraction: float) -> tuple[pd.DataFrame, dict[str, float]]:
    mean_weights = np.mean(np.stack([weights[s] for s in sorted(weights)]), axis=0)
    top_k = max(1, int(np.ceil(mean_weights.shape[-1] * top_fraction)))
    rows = []
    for diagnosis in ["CN", "MCI", "AD"]:
        for target_index, target in enumerate(target_names):
            signatures = {}
            for fold in sorted(frame.fold.unique()):
                keep = frame.fold.eq(fold).to_numpy() & frame.diagnosis_group.eq(diagnosis).to_numpy()
                if keep.sum() >= 5:
                    signatures[int(fold)] = mean_weights[keep, target_index].mean(0)
            for left, right in combinations(sorted(signatures), 2):
                a, b = signatures[left], signatures[right]
                rho = float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])
                top_a = set(np.argpartition(a, -top_k)[-top_k:])
                top_b = set(np.argpartition(b, -top_k)[-top_k:])
                rows.append({"diagnosis_group": diagnosis, "target": target,
                             "fold_left": left, "fold_right": right, "spearman": rho,
                             "top_token_overlap": len(top_a & top_b) / top_k})
    table = pd.DataFrame(rows)
    return table, {"median_spearman": float(table.spearman.median()),
                   "median_top_token_overlap": float(table.top_token_overlap.median())}


def main(config_path: Path, smoke: bool = False) -> None:
    config = load_config(config_path)
    if smoke:
        config = copy.deepcopy(config)
        config["run_id"] += "_smoke"
        config["seeds"] = config["seeds"][:1]
        config["statistics"]["bootstrap_replicates"] = 100
        for arm in config["arms"]:
            config["training"]["frozen_epochs"][arm] = {2026: 1}
    output = ROOT / "outputs" / config["run_id"]
    output.mkdir(parents=True, exist_ok=True)
    frame = build_train_frame(config)
    registry = assign_subject_folds(frame, int(config["folds"]), int(config["statistics"]["bootstrap_seed"]))
    frame = frame.merge(registry[["subject_id", "fold"]], on="subject_id", how="left", validate="many_to_one")
    if smoke:
        selected = registry.groupby("fold", group_keys=False).head(30).subject_id
        frame = frame[frame.subject_id.isin(selected)].reset_index(drop=True)
    feature_index = frame.index.to_numpy() if not smoke else None
    phase3b = ROOT / "outputs" / config["sources"]["phase3b_run"]
    demos, targets, masks, target_mean, target_std = prepare_arrays(
        frame, list(config["targets"]), phase3b
    )
    save_csv(registry, output / "subject_folds.csv")
    save_csv(frame[["scan_id", "subject_id", "fold", "diagnosis_group", "cohort", "cutoff_month"]],
             output / "sample_manifest.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coordinates = coordinate_grid(tuple(config["backbone"]["feature_shape"]))
    predictions, weights, counts = [], {}, {}
    orders = {}
    for arm in config["arms"]:
        for raw_seed in config["seeds"]:
            seed = int(raw_seed)
            source_features = ROOT / "outputs" / config["sources"]["phase3c_run"] / f"spatial_features_seed{seed}.npy"
            feature_path = source_features
            if smoke:
                cached = np.load(source_features, mmap_mode="r")
                original_manifest = pd.read_csv(ROOT / "outputs" / config["sources"]["phase3c_run"] / "sample_manifest.csv")
                lookup = {scan: i for i, scan in enumerate(original_manifest.scan_id.astype(str))}
                subset = np.stack([cached[lookup[str(scan)]] for scan in frame.scan_id])
                feature_path = output / f"spatial_features_seed{seed}.npy"
                np.save(feature_path, subset)
            seed_weights, seed_orders = [], []
            epochs = int(config["training"]["frozen_epochs"][arm][seed])
            for fold in range(int(config["folds"])):
                result, attention, order, count = train_fold(
                    arm, seed, fold, epochs, feature_path, frame, demos, targets, masks,
                    target_mean, target_std, coordinates, config, output, device,
                )
                predictions.append(result)
                counts[arm] = count
                if attention is not None:
                    seed_weights.append(attention)
                    seed_orders.append(order)
            if seed_weights:
                order = np.concatenate(seed_orders)
                values = np.concatenate(seed_weights)
                permutation = np.argsort(order)
                if not np.array_equal(order[permutation], np.arange(len(frame))):
                    raise RuntimeError("Out-of-fold attention does not cover each row exactly once")
                weights[seed] = values[permutation]
                orders[seed] = order[permutation]
    seed_predictions = pd.concat(predictions, ignore_index=True)
    ensembles = ensemble(seed_predictions)
    fold_map = frame.set_index("scan_id").fold
    ensembles["fold"] = ensembles.scan_id.map(fold_map)
    all_predictions = pd.concat([seed_predictions, ensembles], ignore_index=True)
    save_csv(all_predictions, output / "crossfit_predictions.csv")
    scales = dict(zip(config["targets"], target_std))
    metric = metrics(ensembles, scales)
    save_csv(metric, output / "crossfit_metrics.csv")
    candidate = ensembles[ensembles.model.eq("target_spatial_attention") & ensembles.pairing.eq("observed")]
    global_pool = ensembles[ensembles.model.eq("global_pool") & ensembles.pairing.eq("observed")]
    attributes = frame[["scan_id", "diagnosis_group", "fold"]]
    comparisons = []
    for diagnosis in ["overall", "CN", "MCI", "AD", "Unknown"]:
        scans = set(frame.scan_id) if diagnosis == "overall" else set(frame.loc[frame.diagnosis_group.eq(diagnosis), "scan_id"])
        if len(scans) == 0:
            continue
        row = bootstrap_difference(
            candidate[candidate.scan_id.isin(scans)], global_pool[global_pool.scan_id.isin(scans)],
            scales, int(config["statistics"]["bootstrap_replicates"]),
            int(config["statistics"]["bootstrap_seed"]) + len(comparisons),
        )
        row["diagnosis_group"] = diagnosis
        comparisons.append(row)
    for pairing in ["shuffled_subject", "spatial_permutation", "query_permutation"]:
        baseline = ensembles[ensembles.model.eq("target_spatial_attention") & ensembles.pairing.eq(pairing)]
        row = bootstrap_difference(
            candidate, baseline, scales, int(config["statistics"]["bootstrap_replicates"]),
            int(config["statistics"]["bootstrap_seed"]) + len(comparisons),
        )
        row["diagnosis_group"] = "overall"
        comparisons.append(row)
    bootstrap = pd.DataFrame(comparisons)
    save_csv(bootstrap, output / "bootstrap_comparisons.csv")
    paired = paired_effect(candidate, global_pool, scales).join(
        attributes.set_index("scan_id"), on="scan_id"
    )
    fold_effects = paired.groupby(["fold", "diagnosis_group"], as_index=False).agg(
        normalized_mae_difference=("difference", "mean"), n=("difference", "size"),
        subjects=("subject_id", "nunique"),
    )
    save_csv(fold_effects, output / "fold_diagnosis_effects.csv")
    interaction = interaction_bootstrap(
        paired.drop(columns=["diagnosis_group", "fold"]), attributes,
        int(config["statistics"]["bootstrap_replicates"]),
        int(config["statistics"]["bootstrap_seed"]) + 100,
    )
    save_json(interaction, output / "diagnosis_interaction.json")
    if len(weights) >= 2:
        stability, stability_summary = stability_table(
            weights, list(config["targets"]), float(config["statistics"]["attention_top_fraction"])
        )
        fold_stability, fold_summary = fold_signature_stability(
            weights, frame, list(config["targets"]), float(config["statistics"]["attention_top_fraction"])
        )
    else:
        stability, fold_stability = pd.DataFrame(), pd.DataFrame()
        stability_summary = {"median_spearman": None, "median_top_token_overlap": None}
        fold_summary = {"median_spearman": None, "median_top_token_overlap": None}
    save_csv(stability, output / "attention_stability.csv")
    save_csv(fold_stability, output / "fold_signature_stability.csv")
    if weights:
        np.savez_compressed(output / "attention_weights.npz", seeds=np.asarray(sorted(weights)),
                            scan_ids=frame.scan_id.astype(str).to_numpy(dtype=str),
                            weights=np.stack([weights[s] for s in sorted(weights)]))
    diagnosis_rows = bootstrap.set_index("diagnosis_group")
    controls = bootstrap[bootstrap.baseline_pairing.isin(
        ["shuffled_subject", "spatial_permutation", "query_permutation"]
    )]
    ad_directions = int((fold_effects[fold_effects.diagnosis_group.eq("AD")].normalized_mae_difference < 0).sum())
    mci_directions = int((fold_effects[fold_effects.diagnosis_group.eq("MCI")].normalized_mae_difference > 0).sum())
    gate_config = config["gate"]
    criteria = {
        "ad_benefit": bool(diagnosis_rows.loc["AD", "ci_upper"] < 0),
        "mci_harm": bool(diagnosis_rows.loc["MCI", "ci_lower"] > 0),
        "ad_vs_mci_interaction": bool(interaction["ci_upper"] < 0),
        "ad_direction_in_minimum_folds": ad_directions >= int(gate_config["minimum_direction_folds"]),
        "mci_direction_in_minimum_folds": mci_directions >= int(gate_config["minimum_direction_folds"]),
        "all_mechanism_controls": bool((controls.ci_upper < 0).all()),
        "attention_spearman_stable": bool(stability_summary["median_spearman"] is not None and
            stability_summary["median_spearman"] >= float(gate_config["minimum_attention_spearman"])),
        "top_token_overlap_stable": bool(stability_summary["median_top_token_overlap"] is not None and
            stability_summary["median_top_token_overlap"] >= float(gate_config["minimum_top_token_overlap"])),
        "fold_signature_spearman_stable": bool(fold_summary["median_spearman"] is not None and
            fold_summary["median_spearman"] >= float(gate_config["minimum_fold_signature_spearman"])),
        "fold_signature_overlap_stable": bool(fold_summary["median_top_token_overlap"] is not None and
            fold_summary["median_top_token_overlap"] >= float(gate_config["minimum_fold_signature_overlap"])),
    }
    passed = bool(all(criteria.values())) and not smoke
    decision = {
        "run_id": config["run_id"], "engineering_smoke_only": smoke,
        "development_only": True, "original_train_only": True,
        "validation_evaluated": False, "test_evaluated": False,
        "criteria": criteria, "ad_direction_folds": ad_directions,
        "mci_direction_folds": mci_directions,
        "attention_stability": stability_summary, "fold_signature_stability": fold_summary,
        "diagnosis_heterogeneity_replicated": passed,
        "next_action": "seek_external_diagnosis_stratified_confirmation" if passed else "do_not_build_diagnosis_conditioned_grounder",
    }
    save_json(decision, output / "development_decision.json")
    summary = metric.groupby(["model", "pairing"], as_index=False).agg(
        normalized_mae=("normalized_mae", "mean"), mae=("mae", "mean")
    ).sort_values("normalized_mae")
    report = f"""# Phase 3F diagnosis-heterogeneity cross-fit

Status: completed train-only five-fold out-of-fold replication. Validation and test outcomes were not used.

## Contract

- Original-train subjects/scans/targets: {frame.subject_id.nunique()} / {len(frame)} / {int(masks.sum())}
- Subject-level folds: {int(config['folds'])}
- Arms × seeds × folds: {len(config['arms'])} × {len(config['seeds'])} × {int(config['folds'])} = {len(config['arms']) * len(config['seeds']) * int(config['folds'])} checkpoints
- Frozen epoch schedule: `{config['training']['frozen_epochs']}`
- Trainable parameters: `{counts}`
- Diagnosis was used only for analysis, never as a model input.

## Out-of-fold summary

{markdown_table(summary)}

## Diagnosis and mechanism comparisons

Negative differences favor target-conditioned spatial attention.

{markdown_table(bootstrap)}

## AD-versus-MCI interaction

```json
{json.dumps(interaction, indent=2)}
```

## Fold-wise diagnosis effects

{markdown_table(fold_effects)}

## Attention stability

Cross-seed: `{json.dumps(stability_summary)}`

Cross-fold mean signatures: `{json.dumps(fold_summary)}`

## Frozen decision

```json
{json.dumps(decision, indent=2)}
```

This post-test hypothesis replication uses only the original training cohort and remains developmental. It cannot convert the failed Phase 3E test gate into a confirmatory result. Attention weights are predictive maps, not causal anatomical evidence.
"""
    (output / "report.md").write_text(report)
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/phase3f_diagnosis_heterogeneity_crossfit.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    main(args.config, args.smoke)
