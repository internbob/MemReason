#!/usr/bin/env python3
"""Verify the cropped-T1 diagnosis and MCI-progression reruns."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / "outputs/phase3k_cropped_mri_diagnosis_crossfit_20261017"
CONV = ROOT / "outputs/phase3l_cropped_mci_conversion_crossfit_20261017"
CACHE = ROOT / "outputs/phase3k_cropped_t1_cache_20261017"
OLD_DIAG = ROOT / "outputs/phase3g_mri_diagnosis_crossfit_20261016"
OLD_CONV = ROOT / "outputs/phase3h_mci_conversion_crossfit_20261017"
PRIMARY_ARMS = ["sfcn_mri", "cognition", "cognition_sfcn_mri", "cognition_shuffled_sfcn"]


def canonical(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame[columns].astype(str).sort_values(columns).reset_index(drop=True)


def main() -> None:
    cache_audit = json.loads((CACHE / "cache_audit.json").read_text())
    assert cache_audit["target_shape"] == [160, 192, 160]
    assert cache_audit["voxel_size_mm"] == [1.0, 1.0, 1.0]
    assert cache_audit["spatial_interpolation"] is False
    assert cache_audit["padding"] is False
    assert cache_audit["first_and_last_finite"] is True
    assert cache_audit["failures"] == []

    old_diag = pd.read_csv(OLD_DIAG / "diagnostic_manifest.csv", low_memory=False)
    new_diag = pd.read_csv(DIAG / "diagnostic_manifest.csv", low_memory=False)
    diag_columns = ["scan_id", "subject_id", "dxsum_diagnosis_label", "label"]
    pd.testing.assert_frame_equal(canonical(old_diag, diag_columns), canonical(new_diag, diag_columns))
    pd.testing.assert_frame_equal(
        canonical(pd.read_csv(OLD_DIAG / "subject_folds.csv"), ["subject_id", "fold"]),
        canonical(pd.read_csv(DIAG / "subject_folds.csv"), ["subject_id", "fold"]),
    )

    old_conv = pd.read_csv(OLD_CONV / "conversion_manifest.csv", low_memory=False)
    new_conv = pd.read_csv(CONV / "conversion_manifest.csv", low_memory=False)
    conv_columns = ["subject_id", "horizon_months", "scan_id", "label"]
    pd.testing.assert_frame_equal(canonical(old_conv, conv_columns), canonical(new_conv, conv_columns))
    pd.testing.assert_frame_equal(
        canonical(
            pd.read_csv(OLD_CONV / "subject_folds.csv"),
            ["subject_id", "horizon_months", "fold"],
        ),
        canonical(
            pd.read_csv(CONV / "subject_folds.csv"),
            ["subject_id", "horizon_months", "fold"],
        ),
    )

    diag_predictions = pd.read_csv(DIAG / "crossfit_predictions.csv")
    conv_predictions = pd.read_csv(CONV / "crossfit_predictions.csv")
    assert set(PRIMARY_ARMS) <= set(diag_predictions.arm)
    assert set(PRIMARY_ARMS) <= set(conv_predictions.arm)
    probability_columns = ["prob_CN", "prob_MCI", "prob_AD"]
    assert np.isfinite(diag_predictions[probability_columns]).all().all()
    assert np.allclose(diag_predictions[probability_columns].sum(1), 1.0, atol=1e-5)
    assert np.isfinite(conv_predictions.prob_conversion).all()

    # Cognition-only is the negative implementation control: with identical
    # cohorts and folds it must remain unchanged when only MRI preprocessing is
    # replaced.
    old_diag_predictions = pd.read_csv(OLD_DIAG / "crossfit_predictions.csv")
    old_diag_cognition = old_diag_predictions[old_diag_predictions.arm.eq("cognition")].sort_values("row_index")
    new_diag_cognition = diag_predictions[diag_predictions.arm.eq("cognition")].sort_values("row_index")
    assert old_diag_cognition.row_index.tolist() == new_diag_cognition.row_index.tolist()
    assert np.allclose(
        old_diag_cognition[probability_columns], new_diag_cognition[probability_columns], atol=1e-8
    )
    old_conv_predictions = pd.read_csv(OLD_CONV / "crossfit_predictions.csv")
    conversion_keys = ["horizon_months", "subject_id"]
    old_conv_cognition = old_conv_predictions[old_conv_predictions.arm.eq("cognition")].sort_values(conversion_keys)
    new_conv_cognition = conv_predictions[conv_predictions.arm.eq("cognition")].sort_values(conversion_keys)
    assert canonical(old_conv_cognition, conversion_keys).equals(canonical(new_conv_cognition, conversion_keys))
    assert np.allclose(
        old_conv_cognition.prob_conversion.to_numpy(),
        new_conv_cognition.prob_conversion.to_numpy(),
        atol=1e-8,
    )

    diag_intervals = pd.read_csv(DIAG / "diagnostic_metric_intervals.csv")
    metric_map = {
        "accuracy": "ACC",
        "macro_auc": "AUC",
        "macro_sensitivity": "SEN",
        "macro_specificity": "SEC",
        "macro_f1": "F1",
    }
    rows = []
    for arm in PRIMARY_ARMS:
        current = diag_intervals[diag_intervals.arm.eq(arm)].set_index("metric")
        row = {"task": "CN/MCI/AD", "horizon_months": np.nan, "arm": arm}
        for source, target in metric_map.items():
            row[target] = float(current.loc[source, "value"])
            row[f"{target}_ci_low"] = float(current.loc[source, "ci_low"])
            row[f"{target}_ci_high"] = float(current.loc[source, "ci_high"])
        row["subjects"] = int(current.subjects.iloc[0])
        rows.append(row)

    conversion = pd.read_csv(CONV / "conversion_metrics.csv")
    conv_map = {
        "accuracy": "ACC",
        "auc": "AUC",
        "sensitivity": "SEN",
        "specificity": "SEC",
        "f1": "F1",
    }
    for record in conversion[conversion.arm.isin(PRIMARY_ARMS)].itertuples(index=False):
        row = {
            "task": "MCI-to-AD",
            "horizon_months": int(record.horizon_months),
            "arm": record.arm,
            "subjects": int(record.subjects),
        }
        for source, target in conv_map.items():
            row[target] = float(getattr(record, source))
            row[f"{target}_ci_low"] = float(getattr(record, f"{source}_ci_low"))
            row[f"{target}_ci_high"] = float(getattr(record, f"{source}_ci_high"))
        rows.append(row)
    requested = pd.DataFrame(rows)
    requested.to_csv(CONV / "requested_classification_metrics_with_ci.csv", index=False)

    labels = {
        "sfcn_mri": "MRI only",
        "cognition": "Cognition only",
        "cognition_sfcn_mri": "Cognition + MRI",
        "cognition_shuffled_sfcn": "Cognition + shuffled MRI",
    }
    report_rows = []
    for record in requested.itertuples(index=False):
        task = record.task if pd.isna(record.horizon_months) else f"{int(record.horizon_months)}-month MCI-to-AD"
        cells = [task, labels[record.arm], str(int(record.subjects))]
        for metric in ["ACC", "AUC", "SEN", "SEC", "F1"]:
            value = float(getattr(record, metric))
            low = float(getattr(record, f"{metric}_ci_low"))
            high = float(getattr(record, f"{metric}_ci_high"))
            cells.append(f"{value:.3f} [{low:.3f}, {high:.3f}]")
        report_rows.append("| " + " | ".join(cells) + " |")

    report = "\n".join([
        "# Center-cropped T1 classification results",
        "",
        "All MRI inputs are deterministic, interpolation-free center crops from "
        "`193×229×193` to `160×192×160` at 1 mm. Results use the same subjects, "
        "labels, five subject-level folds, and evaluation definitions as the original runs.",
        "",
        "| Task | Input | N | ACC (95% CI) | AUC (95% CI) | SEN (95% CI) | SEC (95% CI) | F1 (95% CI) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
        *report_rows,
        "",
        "For CN/MCI/AD, AUC, sensitivity, specificity, and F1 are macro averages. "
        "For MCI-to-AD, they refer to conversion as the positive class. Confidence "
        "intervals use 2,000 subject-cluster bootstrap replicates.",
        "",
        "The experiment uses original-training subjects only. Official validation "
        "and test subjects were not loaded.",
        "",
    ])
    (CONV / "cropped_classification_summary.md").write_text(report)

    payload = {
        "passed": True,
        "input_shape": [160, 192, 160],
        "voxel_size_mm": [1.0, 1.0, 1.0],
        "spatial_interpolation": False,
        "diagnostic_cohort_and_folds_reused": True,
        "conversion_cohorts_and_folds_reused": True,
        "cognition_only_negative_control_unchanged": True,
        "diagnostic_subjects": int(new_diag.subject_id.nunique()),
        "diagnostic_scans": int(len(new_diag)),
        "conversion_subjects_by_horizon": {
            str(int(horizon)): int(group.subject_id.nunique())
            for horizon, group in new_conv.groupby("horizon_months")
        },
        "bootstrap_replicates": 2000,
        "official_validation_loaded": False,
        "official_test_loaded": False,
    }
    (CONV / "suite_verification.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
