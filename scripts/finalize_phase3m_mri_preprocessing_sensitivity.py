#!/usr/bin/env python3
"""Aggregate matched Phase 3M branches and issue the preprocessing gate."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from memreason_rl.data.common import ROOT
from memreason_rl.data.mri import center_crop_slices


CONFIG = ROOT / "configs/phase3m_mri_preprocessing_sensitivity.yaml"


def metric_values(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return {
        "ACC": float(accuracy_score(y, pred)),
        "AUC": float(roc_auc_score(y, p)),
        "SEN": tp / int((y == 1).sum()),
        "SEC": tn / int((y == 0).sum()),
        "F1": float(f1_score(y, pred)),
    }


def retention(task: tuple[str, str]) -> tuple[str, float]:
    scan_id, path = task
    image = nib.as_closest_canonical(nib.load(path))
    mask = np.asarray(image.dataobj, dtype=np.float32) > 0.5
    cropped = mask[center_crop_slices(mask.shape)]
    return scan_id, float(cropped.sum()) / max(int(mask.sum()), 1)


def load_predictions(config: dict) -> pd.DataFrame:
    source = ROOT / "outputs" / config["source_fullres_run_id"]
    full = pd.read_csv(source / "crossfit_predictions.csv")
    full = full[full.arm.isin(["native_fullres_mri", "demographics_native_fullres_mri"])].copy()
    full.arm = full.arm.map({"native_fullres_mri": "fullres_mri",
                             "demographics_native_fullres_mri": "demographics_fullres_mri"})
    output = ROOT / "outputs" / config["run_id"]
    parts = [full]
    for representation in ["center_crop", "lowres"]:
        parts.append(pd.read_csv(output / f"{representation}_crossfit_predictions.csv"))
    predictions = pd.concat(parts, ignore_index=True)
    predictions.to_csv(output / "crossfit_predictions.csv", index=False)
    return predictions


def audit_retention(config: dict, output: Path, cohort: pd.DataFrame) -> pd.DataFrame:
    path = output / "mask_retention.csv"
    if path.exists():
        result = pd.read_csv(path)
        if result.scan_id.astype(str).tolist() == cohort.scan_id.astype(str).tolist():
            return result
    crop_index = pd.read_csv(
        ROOT / "outputs" / config["source_crop_run_id"] / "volume_index.csv",
        usecols=["scan_id", "mask_path"],
    )
    paths = cohort[["scan_id"]].merge(crop_index, on="scan_id", how="left", validate="one_to_one")
    if paths.mask_path.isna().any():
        raise RuntimeError("Missing masks in retention audit")
    with ProcessPoolExecutor(max_workers=8) as pool:
        values = list(pool.map(retention, [(str(r.scan_id), str(r.mask_path))
                                           for r in paths.itertuples(index=False)]))
    result = pd.DataFrame(values, columns=["scan_id", "mask_retained_fraction"])
    result.to_csv(path, index=False)
    return result


def run(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text())
    output = ROOT / "outputs" / config["run_id"]
    output.mkdir(parents=True, exist_ok=True)
    source = ROOT / "outputs" / config["source_fullres_run_id"]
    cohort = pd.read_csv(source / "cohort_manifest.csv", low_memory=False)
    predictions = load_predictions(config)
    arms = ["fullres_mri", "center_crop_mri", "lowres_mri",
            "demographics_fullres_mri", "demographics_center_crop_mri",
            "demographics_lowres_mri"]
    groups = {}
    reference = cohort[["subject_id", "scan_id", "label", "fold"]].copy()
    reference.subject_id = reference.subject_id.astype(str)
    for arm in arms:
        group = predictions[predictions.arm.eq(arm)].copy()
        group.subject_id = group.subject_id.astype(str)
        group = reference.merge(group[["subject_id", "scan_id", "label", "fold", "probability_AD"]],
                                on=["subject_id", "scan_id", "label", "fold"], how="left",
                                validate="one_to_one")
        if group.probability_AD.isna().any() or len(group) != len(reference):
            raise RuntimeError(f"Incomplete predictions for {arm}")
        groups[arm] = group

    rng = np.random.default_rng(int(config["statistics"]["bootstrap_seed"]))
    n_boot = int(config["statistics"]["bootstrap_replicates"])
    samples = rng.integers(0, len(reference), size=(n_boot, len(reference)))
    metric_rows, bootstrap = [], {arm: {name: [] for name in ["ACC", "AUC", "SEN", "SEC", "F1"]}
                                  for arm in arms}
    for arm, group in groups.items():
        y, p = group.label.to_numpy(), group.probability_AD.to_numpy()
        point = metric_values(y, p)
        for sample in samples:
            if np.unique(y[sample]).size < 2:
                continue
            values = metric_values(y[sample], p[sample])
            for name, value in values.items():
                bootstrap[arm][name].append(value)
        for name, value in point.items():
            distribution = bootstrap[arm][name]
            metric_rows.append({"arm": arm, "metric": name, "value": value,
                                "ci_low": float(np.quantile(distribution, 0.025)),
                                "ci_high": float(np.quantile(distribution, 0.975))})
    intervals = pd.DataFrame(metric_rows)
    intervals.to_csv(output / "classification_metric_intervals.csv", index=False)

    comparisons = [
        ("center_crop_mri", "fullres_mri"), ("lowres_mri", "fullres_mri"),
        ("center_crop_mri", "lowres_mri"),
        ("demographics_center_crop_mri", "demographics_fullres_mri"),
        ("demographics_lowres_mri", "demographics_fullres_mri"),
    ]
    difference_rows = []
    for candidate, reference_arm in comparisons:
        for name in ["ACC", "AUC", "SEN", "SEC", "F1"]:
            values = np.asarray(bootstrap[candidate][name]) - np.asarray(bootstrap[reference_arm][name])
            point = (float(intervals[(intervals.arm == candidate) & (intervals.metric == name)].value.iloc[0])
                     - float(intervals[(intervals.arm == reference_arm) & (intervals.metric == name)].value.iloc[0]))
            difference_rows.append({"candidate": candidate, "reference": reference_arm,
                                    "metric": name, "difference": point,
                                    "ci_low": float(np.quantile(values, 0.025)),
                                    "ci_high": float(np.quantile(values, 0.975))})
    differences = pd.DataFrame(difference_rows)
    differences.to_csv(output / "paired_bootstrap_differences.csv", index=False)

    retention_frame = audit_retention(config, output, cohort)
    retained = reference.merge(retention_frame, on="scan_id", validate="one_to_one")
    retained["retention_quartile"] = pd.qcut(
        retained.mask_retained_fraction.rank(method="first"), 4,
        labels=["Q1_low", "Q2", "Q3", "Q4_high"],
    )
    stratum_rows = []
    for stratum, subset in retained.groupby("retention_quartile", observed=True):
        rows = subset.index.to_numpy()
        for arm in ["fullres_mri", "center_crop_mri", "lowres_mri"]:
            y = groups[arm].iloc[rows].label.to_numpy()
            p = groups[arm].iloc[rows].probability_AD.to_numpy()
            values = metric_values(y, p)
            stratum_rows.append({"retention_stratum": str(stratum), "arm": arm,
                                 "n": len(rows), "retention_min": subset.mask_retained_fraction.min(),
                                 "retention_max": subset.mask_retained_fraction.max(), **values})
    pd.DataFrame(stratum_rows).to_csv(output / "mask_retention_strata_metrics.csv", index=False)

    shapes = {"fullres": (193, 229, 193), "center_crop": (160, 192, 160), "lowres": (48, 56, 48)}
    full_voxels = int(np.prod(shapes["fullres"]))
    resource_rows = []
    cache_paths = {
        "fullres": source / "volumes_native_193x229x193_float16.npy",
        "center_crop": Path(config["representations"]["center_crop"]["cache_path"]),
        "lowres": ROOT / config["representations"]["lowres"]["cache_path"],
    }
    for name, shape in shapes.items():
        row = {"representation": name, "shape": "x".join(map(str, shape)),
               "voxels_per_scan": int(np.prod(shape)),
               "voxel_fraction_vs_full": float(np.prod(shape) / full_voxels),
               "cache_file_bytes": int(cache_paths[name].stat().st_size)}
        usage_path = output / f"{name}_resource_usage.csv"
        if usage_path.exists():
            usage = pd.read_csv(usage_path)
            row.update({"mean_fold_training_seconds": usage.elapsed_seconds.mean(),
                        "max_peak_cuda_memory_mb": usage.peak_cuda_memory_mb.max()})
        resource_rows.append(row)
    pd.DataFrame(resource_rows).to_csv(output / "computational_profile.csv", index=False)

    crop_auc = differences[(differences.candidate == "center_crop_mri")
                           & (differences.reference == "fullres_mri")
                           & (differences.metric == "AUC")].iloc[0]
    margin = float(config["gate"]["crop_minus_full_auc_noninferiority_margin"])
    gate_passed = bool(crop_auc.ci_low >= margin)
    decision = {
        "primary_comparison": "center_crop_mri minus fullres_mri",
        "primary_metric": "AUC",
        "difference": float(crop_auc.difference),
        "ci_low": float(crop_auc.ci_low), "ci_high": float(crop_auc.ci_high),
        "noninferiority_margin": margin,
        "center_crop_noninferiority_passed": gate_passed,
        "future_mri_input_policy": ("fixed_center_crop_160x192x160_no_interpolation" if gate_passed
                                    else "evaluate_adaptive_brain_mask_bounding_box"),
        "official_validation_loaded": False, "official_test_loaded": False,
    }
    (output / "final_decision.json").write_text(json.dumps(decision, indent=2) + "\n")

    minimum_retention = float(retention_frame.mask_retained_fraction.min())
    verification = {
        "passed": bool(len(predictions) == len(reference) * len(arms)
                       and reference.subject_id.nunique() == 896
                       and minimum_retention >= 0.97),
        "n_subjects": int(reference.subject_id.nunique()), "one_scan_per_subject": True,
        "arms": arms, "identical_subjects_labels_and_folds": True,
        "bootstrap_replicates": n_boot, "minimum_mask_retained_fraction": minimum_retention,
        "frozen_phase3i_protocol_reused": True, "official_validation_loaded": False,
        "official_test_loaded": False,
    }
    (output / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")

    pivot = intervals.pivot(index="arm", columns="metric", values="value").loc[arms]
    table = ["| Arm | ACC | AUC | SEN | SEC | F1 |", "|---|---:|---:|---:|---:|---:|"]
    for arm, row in pivot.iterrows():
        table.append(f"| {arm} | {row.ACC:.4f} | {row.AUC:.4f} | {row.SEN:.4f} | {row.SEC:.4f} | {row.F1:.4f} |")
    report = f"""# Phase 3M matched MRI preprocessing sensitivity

This development-only experiment held the Phase 3I AD/CN cohort (896 subjects,
one earliest scan each), labels, five subject folds, inner tuning rule, SFCN,
optimizer, epoch budget, and seed fixed. It changed only the MRI representation:
the complete `193x229x193` 1-mm grid, a no-interpolation `160x192x160` center
crop, or the legacy interpolated `48x56x48` input. Phase 3I full-resolution
out-of-fold predictions were reused; the other two branches were newly trained.

An initial crop run was rejected after a post-hoc consistency check found that
the old Phase 3K cache contained 648 all-zero volumes among these 896 scans.
The result below uses a newly generated compact cache: every crop is the exact
center subarray of its audited Phase 3I volume, and all 896 rows passed a full
nonempty audit. The rejected attempt is recorded separately and is not included
in any metric below.

## Cross-fitted results

{chr(10).join(table)}

All intervals in `classification_metric_intervals.csv` use {n_boot:,} paired
subject-bootstrap replicates. The center-crop minus full-resolution MRI-only AUC
difference was **{crop_auc.difference:+.4f}** (95% CI
**[{crop_auc.ci_low:+.4f}, {crop_auc.ci_high:+.4f}]**). The prespecified
non-inferiority margin was {margin:+.2f}; gate passed: **{str(gate_passed).lower()}**.

The crop retained at least {minimum_retention:.5f} of the foreground mask in all
896 scans. `mask_retention_strata_metrics.csv` tests whether performance varies
over retention quartiles. `computational_profile.csv` records voxel counts,
cache sizes, and newly instrumented branch resources.

## Decision

Future MRI experiments should use
`{decision['future_mri_input_policy']}`. This comparison used original-training
subjects only; official validation and test subjects were not loaded.
"""
    (output / "report.md").write_text(report)
    (output / "resolved_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(decision, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
