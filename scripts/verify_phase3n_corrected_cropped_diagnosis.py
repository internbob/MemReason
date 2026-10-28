#!/usr/bin/env python3
"""Independent verification and requested four-arm summary for Phase 3N."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from memreason_rl.data.common import ROOT
from memreason_rl.data.mri import load_mri
from memreason_rl.experiments.phase3g_mri_diagnosis import arm_metrics


CONFIG = ROOT / "configs/phase3n_corrected_cropped_mri_diagnosis_crossfit.yaml"
OUTPUT = ROOT / "outputs/phase3n_corrected_cropped_mri_diagnosis_crossfit_20261018"
CACHE_OUTPUT = ROOT / "outputs/phase3n_corrected_cropped_t1_cache_20261018"
SOURCE = ROOT / "outputs/phase3g_mri_diagnosis_crossfit_20261016"
PRIMARY_ARMS = ["sfcn_mri", "cognition", "cognition_sfcn_mri", "cognition_shuffled_sfcn"]


def canonical(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame[columns].astype(str).sort_values(columns).reset_index(drop=True)


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    audit = json.loads((CACHE_OUTPUT / "cache_audit.json").read_text())
    index = pd.read_csv(CACHE_OUTPUT / "volume_index.csv")
    cache_path = Path(config["volume_cache"]["path"])
    cache = np.load(cache_path, mmap_mode="r")
    old_manifest = pd.read_csv(SOURCE / "diagnostic_manifest.csv", low_memory=False)
    manifest = pd.read_csv(OUTPUT / "diagnostic_manifest.csv", low_memory=False)
    old_folds = pd.read_csv(SOURCE / "subject_folds.csv")
    folds = pd.read_csv(OUTPUT / "subject_folds.csv")
    predictions = pd.read_csv(OUTPUT / "crossfit_predictions.csv")
    seed_predictions = pd.read_csv(OUTPUT / "crossfit_seed_predictions.csv")
    metrics = pd.read_csv(OUTPUT / "diagnostic_metrics.csv")
    intervals = pd.read_csv(OUTPUT / "diagnostic_metric_intervals.csv")
    decision = json.loads((OUTPUT / "development_decision.json").read_text())

    checks = {
        "cache_shape": tuple(cache.shape) == (5370, 160, 192, 160),
        "cache_all_rows_checked": audit.get("rows_checked") == 5370,
        "cache_all_rows_finite": audit.get("finite_rows") == 5370 and audit.get("all_rows_finite") is True,
        "cache_all_rows_nonempty": audit.get("nonempty_rows") == 5370 and audit.get("all_rows_nonempty") is True,
        "cache_has_no_reported_failures": audit.get("failures") == [],
        "index_unique_and_complete": len(index) == index.scan_id.nunique() == 5370,
        "raw_path_sample_matches_cache": True,
        "cohort_identical_to_phase3g": True,
        "folds_identical_to_phase3g": True,
        "five_folds_three_seeds": True,
        "fifteen_checkpoints": len(list((OUTPUT / "checkpoints").glob("fold*/seed*.pt"))) == 15,
        "four_primary_arms_present": set(PRIMARY_ARMS) <= set(predictions.arm.unique()),
        "all_probabilities_finite_and_normalized": True,
        "cognition_negative_control_unchanged": True,
        "point_metrics_recomputed": True,
        "bootstrap_uses_2000_replicates": bool(intervals.replicates.eq(2000).all()),
        "official_validation_not_loaded": decision.get("official_validation_loaded") is False,
        "official_test_not_loaded": decision.get("official_test_loaded") is False,
    }

    sample = np.linspace(0, len(index) - 1, 24, dtype=int)
    for row in index.iloc[sample].itertuples(index=False):
        rebuilt = load_mri(str(row.image_path), str(row.mask_path)).numpy()[0].astype(np.float16)
        if not np.array_equal(rebuilt, cache[int(row.volume_index)]):
            checks["raw_path_sample_matches_cache"] = False
            break

    cohort_columns = ["scan_id", "subject_id", "dxsum_diagnosis_label", "label"]
    try:
        pd.testing.assert_frame_equal(canonical(old_manifest, cohort_columns),
                                      canonical(manifest, cohort_columns))
    except AssertionError:
        checks["cohort_identical_to_phase3g"] = False
    try:
        pd.testing.assert_frame_equal(canonical(old_folds, ["subject_id", "fold"]),
                                      canonical(folds, ["subject_id", "fold"]))
    except AssertionError:
        checks["folds_identical_to_phase3g"] = False

    sfcn_seed_rows = seed_predictions[seed_predictions.arm.eq("sfcn_mri")]
    checks["five_folds_three_seeds"] = bool(
        set(sfcn_seed_rows.fold.unique()) == {0, 1, 2, 3, 4}
        and set(sfcn_seed_rows.seed.unique()) == {2026, 2027, 2028}
    )
    probability_columns = ["prob_CN", "prob_MCI", "prob_AD"]
    checks["all_probabilities_finite_and_normalized"] = bool(
        np.isfinite(predictions[probability_columns]).all().all()
        and np.allclose(predictions[probability_columns].sum(1), 1.0, atol=1e-5)
    )
    old_predictions = pd.read_csv(SOURCE / "crossfit_predictions.csv")
    old_cognition = old_predictions[old_predictions.arm.eq("cognition")].sort_values("row_index")
    new_cognition = predictions[predictions.arm.eq("cognition")].sort_values("row_index")
    checks["cognition_negative_control_unchanged"] = bool(
        old_cognition.row_index.tolist() == new_cognition.row_index.tolist()
        and np.allclose(old_cognition[probability_columns], new_cognition[probability_columns], atol=1e-8)
    )
    recomputed = arm_metrics(predictions).set_index("arm")
    reported = metrics.set_index("arm")
    metric_columns = ["accuracy", "macro_auc", "macro_sensitivity", "macro_specificity", "macro_f1"]
    checks["point_metrics_recomputed"] = all(
        np.isclose(recomputed.loc[arm, name], reported.loc[arm, name], atol=1e-12)
        for arm in reported.index for name in metric_columns
    )

    requested = intervals[
        intervals.arm.isin(PRIMARY_ARMS) & intervals.metric.isin(metric_columns)
    ].copy()
    requested.to_csv(OUTPUT / "requested_four_arm_metrics_with_ci.csv", index=False)
    requested_index = requested.set_index(["arm", "metric"])
    labels = {
        "sfcn_mri": "MRI only", "cognition": "Cognition only",
        "cognition_sfcn_mri": "Cognition + MRI",
        "cognition_shuffled_sfcn": "Cognition + shuffled MRI",
    }
    table = ["| Input | ACC | macro-AUC | macro-SEN | macro-SEC | macro-F1 |",
             "|---|---:|---:|---:|---:|---:|"]
    for arm in PRIMARY_ARMS:
        formatted = []
        for metric in metric_columns:
            row = requested_index.loc[(arm, metric)]
            formatted.append(f"{row.value:.4f} [{row.ci_low:.4f}, {row.ci_high:.4f}]")
        table.append(
            f"| {labels[arm]} | " + " | ".join(formatted) + " |"
        )
    report = f"""# Phase 3N corrected cropped-T1 three-class results

The corrected experiment contains 1,328 original-training subjects and 5,370
MRI-linked CN/MCI/AD visits. All 5,370 MRI volumes were rebuilt from the original
T1 and mask paths, center-cropped from `193x229x193` to `160x192x160` without
interpolation, and passed complete row-level cache validation.

Values are point estimates with 95% subject-cluster bootstrap confidence
intervals in brackets.

{chr(10).join(table)}

`requested_four_arm_metrics_with_ci.csv` contains all point estimates and 95%
confidence intervals from 2,000 subject-cluster bootstrap replicates. Official
validation and test subjects were not loaded.
"""
    (OUTPUT / "requested_four_arm_report.md").write_text(report)
    payload = {
        "passed": bool(all(checks.values())), "checks": checks,
        "subjects": int(manifest.subject_id.nunique()), "scans": int(len(manifest)),
        "checkpoints": len(list((OUTPUT / "checkpoints").glob("fold*/seed*.pt"))),
        "bootstrap_replicates": 2000,
    }
    (OUTPUT / "independent_verification.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
