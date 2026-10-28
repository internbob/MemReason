#!/usr/bin/env python3
"""Independent source-level checks for the Phase 3M result bundle."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from memreason_rl.data.common import ROOT
from memreason_rl.data.mri import center_crop_slices


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    pred = (p >= 0.5).astype(int)
    return {
        "ACC": float(accuracy_score(y, pred)),
        "AUC": float(roc_auc_score(y, p)),
        "SEN": float(((pred == 1) & (y == 1)).sum() / (y == 1).sum()),
        "SEC": float(((pred == 0) & (y == 0)).sum() / (y == 0).sum()),
        "F1": float(f1_score(y, pred)),
    }


def main() -> None:
    config = yaml.safe_load((ROOT / "configs/phase3m_mri_preprocessing_sensitivity.yaml").read_text())
    output = ROOT / "outputs" / config["run_id"]
    source = ROOT / "outputs" / config["source_fullres_run_id"]
    cohort = pd.read_csv(source / "cohort_manifest.csv", low_memory=False)
    folds = pd.read_csv(source / "subject_folds.csv")
    predictions = pd.read_csv(output / "crossfit_predictions.csv")
    intervals = pd.read_csv(output / "classification_metric_intervals.csv")
    differences = pd.read_csv(output / "paired_bootstrap_differences.csv")
    decision = json.loads((output / "final_decision.json").read_text())
    cache_audit = json.loads((output / "center_crop_cache_audit.json").read_text())
    arms = sorted(predictions.arm.unique())
    expected_arms = sorted([
        "fullres_mri", "center_crop_mri", "lowres_mri",
        "demographics_fullres_mri", "demographics_center_crop_mri",
        "demographics_lowres_mri",
    ])
    checks = {
        "cohort_has_896_unique_subjects": len(cohort) == cohort.subject_id.nunique() == 896,
        "source_folds_cover_cohort": len(folds) == folds.subject_id.nunique() == 896,
        "six_expected_arms": arms == expected_arms,
        "one_prediction_per_subject_per_arm": bool(
            predictions.groupby("arm").subject_id.nunique().eq(896).all()
            and len(predictions) == 896 * 6
        ),
        "labels_and_folds_identical_across_arms": True,
        "reported_point_metrics_recomputed": True,
        "crop_cache_all_rows_nonempty": bool(cache_audit["passed"] and cache_audit["n_nonempty"] == 896),
        "crop_cache_is_exact_center_subarray": True,
        "gate_recomputed": True,
        "official_validation_not_loaded": not bool(decision["official_validation_loaded"]),
        "official_test_not_loaded": not bool(decision["official_test_loaded"]),
        "corrupt_prior_attempt_retained": (output / "invalid_attempt_corrupt_phase3k_cache.json").exists(),
    }
    reference = None
    for arm, group in predictions.groupby("arm"):
        group = group.sort_values("subject_id").reset_index(drop=True)
        identity = group[["subject_id", "scan_id", "label", "fold"]].astype(str)
        if reference is None:
            reference = identity
        elif not identity.equals(reference):
            checks["labels_and_folds_identical_across_arms"] = False
        values = metrics(group.label.to_numpy(), group.probability_AD.to_numpy())
        reported = intervals[intervals.arm.eq(arm)].set_index("metric").value.to_dict()
        if any(not np.isclose(value, reported[name], atol=1e-12) for name, value in values.items()):
            checks["reported_point_metrics_recomputed"] = False

    crop_index = pd.read_csv(output / "center_crop_volume_index.csv")
    full = np.load(source / "volumes_native_193x229x193_float16.npy", mmap_mode="r")
    crop = np.load(Path(config["representations"]["center_crop"]["cache_path"]), mmap_mode="r")
    crop_lookup = crop_index.set_index("scan_id").volume_index
    sample_rows = np.linspace(0, len(cohort) - 1, 24, dtype=int)
    slices = center_crop_slices(full.shape[1:])
    for row in sample_rows:
        full_index = int(cohort.iloc[row].native_volume_index)
        crop_index_value = int(crop_lookup.loc[cohort.iloc[row].scan_id])
        if not np.array_equal(full[full_index][slices], crop[crop_index_value]):
            checks["crop_cache_is_exact_center_subarray"] = False
            break

    contrast = differences[(differences.candidate == "center_crop_mri")
                           & (differences.reference == "fullres_mri")
                           & (differences.metric == "AUC")].iloc[0]
    margin = float(config["gate"]["crop_minus_full_auc_noninferiority_margin"])
    expected_gate = bool(float(contrast.ci_low) >= margin)
    checks["gate_recomputed"] = bool(
        np.isclose(float(contrast.difference), float(decision["difference"]), atol=1e-12)
        and np.isclose(float(contrast.ci_low), float(decision["ci_low"]), atol=1e-12)
        and expected_gate == bool(decision["center_crop_noninferiority_passed"])
    )
    payload = {"passed": bool(all(checks.values())), "checks": checks,
               "recomputed_primary_gate": expected_gate, "primary_ci_low": float(contrast.ci_low),
               "noninferiority_margin": margin}
    (output / "independent_verification.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
