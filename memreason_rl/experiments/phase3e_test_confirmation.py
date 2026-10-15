"""One-time frozen test confirmation of Phase 3C concurrent spatial grounding."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from ..data.common import ROOT, save_csv, save_json
from .phase1_numeric import markdown_table
from .phase3b_mri_representation import (
    ConcurrentRegressor,
    bootstrap_difference,
    ensemble,
    make_shuffle_indices,
    metrics,
    prepare_volume_cache,
)
from .phase3c_spatial_grounding import (
    SpatialDataset,
    SpatialGrounder,
    coordinate_grid,
    evaluate,
    stability_table,
)


TARGET_DATES = {
    "adas_total13": "adas_visit_date",
    "cdr_cdrsb": "cdr_visit_date",
    "faq_faqtotal": "faq_visit_date",
    "mmse_mmscore": "mmse_visit_date",
    "moca_moca": "moca_visit_date",
}


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def build_test_frame(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(ROOT / "splits/test.csv", low_memory=False)
    eligible = source[
        source.date_window_verified.fillna(False)
        & source.scan_id.notna()
        & source.image_path.notna()
        & source.mask_path.notna()
        & source.scan_date.notna()
    ].copy()
    eligible["absolute_match_days"] = pd.to_numeric(
        eligible.date_difference_days, errors="coerce"
    ).abs()
    duplicates = eligible[eligible.duplicated("scan_id", keep=False)][
        ["scan_id", "subject_id", "visit_month", "absolute_match_days"]
    ].sort_values(["scan_id", "absolute_match_days", "visit_month"])
    eligible = eligible.sort_values(
        ["scan_id", "absolute_match_days", "visit_month"]
    ).drop_duplicates("scan_id", keep="first")
    metadata = pd.read_csv(ROOT / "processed/mri_metadata.csv", low_memory=False)
    metadata["image_id"] = pd.to_numeric(metadata.image_id, errors="coerce")
    ida = pd.read_csv(ROOT / "idaSearch_10_12_2026 (2).csv", low_memory=False).rename(
        columns={"Image ID": "image_id", "Age": "age", "Sex": "sex"}
    )
    ida["image_id"] = pd.to_numeric(ida.image_id, errors="coerce")
    ida = ida[["image_id", "age", "sex"]].dropna(subset=["image_id"]).drop_duplicates("image_id")
    eligible = eligible.merge(
        metadata[["scan_id", "image_id", "site", "cohort"]].drop_duplicates("scan_id"),
        on="scan_id", how="left", validate="one_to_one"
    ).merge(ida, on="image_id", how="left", validate="many_to_one")
    eligible["age"] = pd.to_numeric(eligible.age, errors="coerce")
    eligible["sex_numeric"] = eligible.sex.astype(str).str.upper().map({"M": 0.0, "F": 1.0})
    eligible = eligible[eligible.age.notna() & eligible.sex_numeric.notna()].copy()
    scan_date = pd.to_datetime(eligible.scan_date, errors="coerce")
    targets = list(config["targets"])
    for target, date_column in TARGET_DATES.items():
        target_date = pd.to_datetime(eligible[date_column], errors="coerce")
        valid = (
            eligible[target].notna() & target_date.notna() & scan_date.notna()
            & ((target_date - scan_date).dt.days.abs() <= int(config["date_window_days"]))
        )
        eligible.loc[~valid, target] = np.nan
    eligible = eligible[eligible[targets].notna().any(axis=1)].copy()
    eligible["split"] = "test"
    eligible["anchor_key"] = eligible.apply(
        lambda row: f"{row.subject_id}:m{float(row.visit_month):g}", axis=1
    )
    gap = eligible.absolute_match_days
    eligible["gap_group"] = pd.cut(
        gap, [-0.1, 30, 60, 90], labels=["0-30d", "31-60d", "61-90d"]
    ).astype(str)
    eligible["diagnosis_group"] = eligible.dxsum_diagnosis_label.fillna("Unknown").astype(str)
    eligible["sex_group"] = eligible.sex.astype(str).str.upper()
    eligible["age_group"] = pd.cut(
        eligible.age, [-np.inf, 69.999, 79.999, np.inf], labels=["<70", "70-79", ">=80"]
    ).astype(str)
    count = eligible[targets].notna().sum(axis=1)
    eligible["availability_group"] = pd.cut(
        count, [0, 2, 4, 5], labels=["1-2", "3-4", "5"]
    ).astype(str)
    if eligible.scan_id.duplicated().any() or set(eligible.split) != {"test"}:
        raise RuntimeError("Frozen test frame must be unique by scan and contain test only")
    train_subjects = set(pd.read_csv(ROOT / "splits/train.csv", usecols=["subject_id"]).subject_id.astype(str))
    val_subjects = set(pd.read_csv(ROOT / "splits/val.csv", usecols=["subject_id"]).subject_id.astype(str))
    test_subjects = set(eligible.subject_id.astype(str))
    if test_subjects & (train_subjects | val_subjects):
        raise RuntimeError("Test confirmation contains a train/validation subject")
    return eligible.sort_values(["subject_id", "visit_month"]).reset_index(drop=True), duplicates


def prepare_arrays(frame: pd.DataFrame, targets: list[str], phase3b: Path):
    normalization = json.loads((phase3b / "normalization.json").read_text())
    mean = np.asarray([normalization["target_mean"][name] for name in targets], dtype=np.float32)
    std = np.asarray([normalization["target_std"][name] for name in targets], dtype=np.float32)
    demos = np.stack(
        [(frame.age.to_numpy() - float(normalization["age_mean"])) / float(normalization["age_std"]),
         frame.sex_numeric.to_numpy()], axis=1
    ).astype(np.float32)
    raw = frame[targets].to_numpy(dtype=np.float32)
    masks = np.isfinite(raw)
    values = np.where(masks, (raw - mean) / std, 0).astype(np.float32)
    return demos, values, masks, mean, std


def extract_test_spatial_features(
    frame: pd.DataFrame,
    volume_path: Path,
    seed: int,
    config: dict[str, Any],
    phase3b: Path,
    output: Path,
    device: torch.device,
) -> Path:
    destination = output / f"spatial_features_seed{seed}.npy"
    expected = (len(frame), int(np.prod(config["backbone"]["feature_shape"])), int(config["backbone"]["feature_channels"]))
    if destination.exists() and tuple(np.load(destination, mmap_mode="r").shape) == expected:
        return destination
    model = ConcurrentRegressor("sfcn_3d", 518, len(config["targets"]), 0.15).to(device)
    checkpoint = torch.load(
        phase3b / "checkpoints/sfcn_3d" / str(seed) / "best.pt",
        map_location=device, weights_only=False,
    )
    model.load_state_dict(checkpoint["model"])
    backbone = model.encoder.network[: int(config["backbone"]["frozen_blocks"]) * 3].eval()
    backbone.requires_grad_(False)
    volumes = np.load(volume_path, mmap_mode="r")
    temporary = output / f"spatial_features_seed{seed}.tmp.npy"
    features = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16, shape=expected)
    batch_size = 32
    with torch.no_grad():
        for start in range(0, len(frame), batch_size):
            stop = min(start + batch_size, len(frame))
            image = torch.from_numpy(np.asarray(volumes[start:stop], dtype=np.float32))[:, None].to(device)
            spatial = backbone(image)
            if tuple(spatial.shape[2:]) != tuple(config["backbone"]["feature_shape"]):
                raise RuntimeError(f"Unexpected frozen grid {tuple(spatial.shape)}")
            features[start:stop] = spatial.flatten(2).transpose(1, 2).cpu().numpy().astype(np.float16)
    features.flush()
    del features
    temporary.replace(destination)
    return destination


def load_head(arm: str, seed: int, config: dict[str, Any], phase3c: Path, device: torch.device):
    model = SpatialGrounder(
        arm, int(config["backbone"]["feature_channels"]), int(config["head"]["hidden_dim"]),
        len(config["targets"]), int(config["head"]["attention_heads"]),
        float(config["head"]["dropout"]), coordinate_grid(tuple(config["backbone"]["feature_shape"])),
    ).to(device)
    checkpoint = torch.load(
        phase3c / "checkpoints" / arm / str(seed) / "best.pt",
        map_location=device, weights_only=False,
    )
    if checkpoint["arm"] != arm or int(checkpoint["seed"]) != seed:
        raise RuntimeError("Frozen Phase 3C head checkpoint identity mismatch")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, int(checkpoint["trainable_parameters"])


def stratum_tables(
    ensemble_predictions: pd.DataFrame,
    frame: pd.DataFrame,
    scales: dict[str, float],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    attributes = frame.set_index("scan_id")[[
        "diagnosis_group", "gap_group", "sex_group", "age_group", "availability_group"
    ]]
    enriched = ensemble_predictions.join(attributes, on="scan_id")
    metric_rows, bootstrap_rows = [], []
    candidate_all = enriched[
        enriched.model.eq("target_spatial_attention") & enriched.pairing.eq("observed")
    ]
    baseline_all = enriched[enriched.model.eq("global_pool") & enriched.pairing.eq("observed")]
    for dimension in attributes.columns:
        for group in sorted(enriched[dimension].dropna().unique()):
            subset = enriched[enriched[dimension].eq(group)]
            for (model, pairing), values in subset.groupby(["model", "pairing"]):
                per_target = metrics(values, scales)
                metric_rows.append({
                    "dimension": dimension, "group": group, "model": model, "pairing": pairing,
                    "subjects": int(values.subject_id.nunique()), "targets": int(len(values)),
                    "macro_normalized_mae": float(per_target.normalized_mae.mean()),
                })
            candidate = candidate_all[candidate_all[dimension].eq(group)]
            baseline = baseline_all[baseline_all[dimension].eq(group)]
            subjects = candidate.subject_id.nunique()
            if subjects >= int(config["statistics"]["minimum_stratum_subjects"]):
                row = bootstrap_difference(
                    candidate, baseline, scales, int(config["statistics"]["bootstrap_replicates"]),
                    int(config["statistics"]["bootstrap_seed"]) + len(bootstrap_rows),
                )
                row.update({"dimension": dimension, "group": group})
                bootstrap_rows.append(row)
    return pd.DataFrame(metric_rows), pd.DataFrame(bootstrap_rows)


def main(config_path: Path) -> None:
    config = load_config(config_path)
    output = ROOT / "outputs" / config["run_id"]
    output.mkdir(parents=True, exist_ok=True)
    phase3b = ROOT / "outputs" / config["sources"]["phase3b_run"]
    phase3c = ROOT / "outputs" / config["sources"]["phase3c_run"]
    frame, duplicates = build_test_frame(config)
    targets = list(config["targets"])
    demos, values, masks, target_mean, target_std = prepare_arrays(frame, targets, phase3b)
    save_csv(duplicates, output / "duplicate_scan_resolution.csv")
    save_csv(frame[[
        "anchor_key", "scan_id", "subject_id", "split", "visit_month", "scan_date",
        "absolute_match_days", "diagnosis_group", "sex_group", "age_group", "gap_group",
        "availability_group", "image_path", "mask_path"
    ] + targets], output / "test_manifest.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    volume_path, _ = prepare_volume_cache(frame, config, output)
    feature_paths = {
        int(seed): extract_test_spatial_features(
            frame, volume_path, int(seed), config, phase3b, output, device
        ) for seed in config["seeds"]
    }
    rows = np.arange(len(frame))
    shuffle = make_shuffle_indices(frame, int(config["statistics"]["bootstrap_seed"]))
    predictions, weights, attention_order, counts = [], {}, None, {}
    for arm in config["arms"]:
        for seed in config["seeds"]:
            seed = int(seed)
            model, count = load_head(arm, seed, config, phase3c, device)
            counts[arm] = count
            observed_data = SpatialDataset(rows, frame, demos, values, masks, feature_paths[seed])
            observed, attention, order = evaluate(
                model, observed_data, int(config["evaluation"]["batch_size"]), device,
                targets, target_mean, target_std, seed, "observed",
                return_attention=arm == "target_spatial_attention",
            )
            predictions.append(observed)
            shuffled_data = SpatialDataset(
                rows, frame, demos, values, masks, feature_paths[seed], source_indices=shuffle
            )
            shuffled, _, _ = evaluate(
                model, shuffled_data, int(config["evaluation"]["batch_size"]), device,
                targets, target_mean, target_std, seed, "shuffled_subject",
            )
            predictions.append(shuffled)
            if arm == "target_spatial_attention":
                weights[seed] = attention
                if attention_order is None:
                    attention_order = order
                elif not np.array_equal(attention_order, order):
                    raise RuntimeError("Attention order differs across seeds")
                spatial = np.roll(np.arange(int(np.prod(config["backbone"]["feature_shape"]))), 1).astype(np.int64)
                query = np.roll(np.arange(len(targets)), 1).astype(np.int64)
                spatial_prediction, _, _ = evaluate(
                    model, observed_data, int(config["evaluation"]["batch_size"]), device,
                    targets, target_mean, target_std, seed, "spatial_permutation",
                    spatial_permutation=spatial,
                )
                query_prediction, _, _ = evaluate(
                    model, observed_data, int(config["evaluation"]["batch_size"]), device,
                    targets, target_mean, target_std, seed, "query_permutation",
                    query_permutation=query,
                )
                predictions.extend([spatial_prediction, query_prediction])
    seeded = pd.concat(predictions, ignore_index=True)
    ensembles = ensemble(seeded)
    save_csv(pd.concat([seeded, ensembles], ignore_index=True), output / "test_predictions.csv")
    scales = dict(zip(targets, target_std))
    metric = metrics(ensembles, scales)
    save_csv(metric, output / "test_metrics.csv")
    candidate = ensembles[
        ensembles.model.eq("target_spatial_attention") & ensembles.pairing.eq("observed")
    ]
    baselines = [
        ensembles[ensembles.model.eq("global_pool") & ensembles.pairing.eq("observed")],
        ensembles[ensembles.model.eq("target_spatial_attention") & ensembles.pairing.eq("shuffled_subject")],
        ensembles[ensembles.model.eq("target_spatial_attention") & ensembles.pairing.eq("spatial_permutation")],
        ensembles[ensembles.model.eq("target_spatial_attention") & ensembles.pairing.eq("query_permutation")],
    ]
    comparisons = []
    for baseline in baselines:
        for target in [None] + targets:
            comparisons.append(bootstrap_difference(
                candidate, baseline, scales, int(config["statistics"]["bootstrap_replicates"]),
                int(config["statistics"]["bootstrap_seed"]) + len(comparisons), target=target,
            ))
    bootstrap = pd.DataFrame(comparisons)
    save_csv(bootstrap, output / "bootstrap_comparisons.csv")
    stability, stability_summary = stability_table(
        weights, targets, float(config["statistics"]["attention_top_fraction"])
    )
    save_csv(stability, output / "attention_stability.csv")
    np.savez_compressed(
        output / "attention_weights.npz", seeds=np.asarray(sorted(weights)),
        scan_ids=frame.iloc[attention_order].scan_id.astype(str).to_numpy(),
        weights=np.stack([weights[seed] for seed in sorted(weights)]),
    )
    stratum_metric, stratum_bootstrap = stratum_tables(ensembles, frame, scales, config)
    save_csv(stratum_metric, output / "stratum_metrics.csv")
    save_csv(stratum_bootstrap, output / "stratum_bootstrap_vs_global.csv")
    overall = bootstrap[bootstrap.target.eq("overall")].set_index(["baseline", "baseline_pairing"])
    versus_global = bootstrap[
        bootstrap.baseline.eq("global_pool") & bootstrap.baseline_pairing.eq("observed")
    ]
    significant = int(((versus_global.target != "overall") & (versus_global.ci_upper < 0)).sum())
    criteria = {
        "attention_better_than_global_pool": bool(overall.loc[("global_pool", "observed"), "ci_upper"] < 0),
        "attention_better_than_subject_shuffle": bool(overall.loc[("target_spatial_attention", "shuffled_subject"), "ci_upper"] < 0),
        "attention_better_than_spatial_permutation": bool(overall.loc[("target_spatial_attention", "spatial_permutation"), "ci_upper"] < 0),
        "attention_better_than_query_permutation": bool(overall.loc[("target_spatial_attention", "query_permutation"), "ci_upper"] < 0),
        "minimum_significant_endpoints": significant >= int(config["gate"]["minimum_significant_endpoints"]),
        "attention_spearman_stable": bool(stability_summary["median_spearman"] >= float(config["gate"]["minimum_attention_spearman"])),
        "top_token_overlap_stable": bool(stability_summary["median_top_token_overlap"] >= float(config["gate"]["minimum_top_token_overlap"])),
    }
    passed = bool(all(criteria.values()))
    decision = {
        "run_id": config["run_id"], "confirmation_only": True, "training_performed": False,
        "test_evaluated": True, "criteria": criteria, "significant_endpoints_vs_global": significant,
        "attention_stability": stability_summary, "phase3c_test_confirmed": passed,
        "next_action": "open_frozen_grounding_to_structured_sft_data_construction" if passed else "retain_validation_only_phase3c_without_confirmatory_claim",
    }
    save_json(decision, output / "test_decision.json")
    summary = metric.groupby(["model", "pairing"], as_index=False).agg(
        normalized_mae=("normalized_mae", "mean"), mae=("mae", "mean")
    ).sort_values("normalized_mae")
    report = f"""# Phase 3E frozen spatial-grounding test confirmation\n\nStatus: completed one-time evaluation on the sealed subject-disjoint test split. No training or checkpoint selection was performed.\n\n## Frozen contract\n\n- Test MRI anchors: {len(frame)}\n- Test subjects: {frame.subject_id.nunique()}\n- Test targets: {int(masks.sum())}\n- Frozen SFCN/head seeds: {config['seeds']}\n- Head trainable parameters in Phase 3C: {counts}\n- Duplicate-scan resolutions: {int(len(duplicates))} source rows\n\n## Test summary\n\n{markdown_table(summary)}\n\n## Paired subject-cluster bootstrap\n\nNegative differences favor observed target-conditioned spatial attention.\n\n{markdown_table(bootstrap)}\n\n## Cross-seed attention stability\n\n```json\n{json.dumps(stability_summary, indent=2)}\n```\n\n## Frozen decision\n\n```json\n{json.dumps(decision, indent=2)}\n```\n\nSecondary diagnosis, timing-gap, sex, age, and endpoint-availability results are stored in `stratum_metrics.csv` and `stratum_bootstrap_vs_global.csv`. They do not alter the primary gate. Attention is a predictive weighting map and is not causal anatomical evidence.\n"""
    (output / "report.md").write_text(report)
    save_json({
        "run_id": config["run_id"], "device": str(device), "training_performed": False,
        "test_evaluated": True, "test_anchors": len(frame), "test_subjects": frame.subject_id.nunique(),
        "test_targets": int(masks.sum()), "parameter_counts": counts,
        "phase3b_backbones": [str(phase3b / "checkpoints/sfcn_3d" / str(seed) / "best.pt") for seed in config["seeds"]],
        "phase3c_heads": [str(phase3c / "checkpoints" / arm / str(seed) / "best.pt") for arm in config["arms"] for seed in config["seeds"]],
    }, output / "run_manifest.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/phase3e_frozen_spatial_grounding_test.yaml")
    args = parser.parse_args()
    main(args.config)
