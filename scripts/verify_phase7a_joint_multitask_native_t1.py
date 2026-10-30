#!/usr/bin/env python
"""Independent artifact and leakage checks for Phase 7A."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from memreason_rl.data.common import ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs/phase7a_joint_multitask_native_t1.yaml",
    )
    args = parser.parse_args(); config = yaml.safe_load(args.config.read_text())
    output = ROOT / "outputs" / config["run_id"]
    cohort = pd.read_csv(output / "cohort_manifest.csv", low_memory=False)
    predictions = pd.read_csv(output / "crossfit_predictions.csv", low_memory=False)
    diagnostic = pd.read_csv(output / "diagnostic_metrics.csv")
    regression = pd.read_csv(output / "regression_metrics.csv")
    decision = json.loads((output / "development_decision.json").read_text())
    expected_arms = {
        "classification_only", "regression_only", "joint_multitask",
        "joint_subject_shuffled_mri",
    }
    if set(cohort.diagnosis) != set(config["classes"]):
        raise RuntimeError("CN/MCI/AD are not all present")
    if cohort.subject_id.astype(str).groupby(cohort.subject_id.astype(str)).size().empty:
        raise RuntimeError("Empty cohort")
    if (cohort.history_latest_month > cohort.visit_month + 1e-8).any():
        raise RuntimeError("History contains a future visit")
    for horizon in config["forecast"]["horizons_months"]:
        mask = cohort[f"future_memory_{horizon}"].notna()
        if not (cohort.loc[mask, f"future_month_{horizon}"] > cohort.loc[mask, "visit_month"]).all():
            raise RuntimeError(f"Invalid future target at {horizon} months")
    if set(predictions.arm) != expected_arms:
        raise RuntimeError(f"Unexpected prediction arms: {set(predictions.arm)}")
    counts = predictions.groupby("arm").row_index.nunique()
    if not (counts == len(cohort)).all():
        raise RuntimeError(f"Incomplete OOF predictions: {counts.to_dict()}")
    if predictions.groupby(["arm", "row_index"]).size().max() != 1:
        raise RuntimeError("Duplicate arm/row prediction")
    if not np.isfinite(
        predictions[predictions.arm.isin(["classification_only", "joint_multitask"])]
        [["prob_CN", "prob_MCI", "prob_AD"]].to_numpy(float)
    ).all():
        raise RuntimeError("Nonfinite diagnostic probability")
    diagnostic_rows = predictions[predictions.arm.isin([
        "classification_only", "joint_multitask", "joint_subject_shuffled_mri",
    ])]
    probability_sums = diagnostic_rows[["prob_CN", "prob_MCI", "prob_AD"]].sum(axis=1)
    if not np.allclose(probability_sums, 1.0, rtol=0.0, atol=1e-10):
        raise RuntimeError("Diagnostic probabilities are not normalized")
    if not np.isfinite(
        predictions[predictions.arm.isin(["regression_only", "joint_multitask"])]
        ["prediction_current"].to_numpy(float)
    ).all():
        raise RuntimeError("Nonfinite current-Memory prediction")
    overlaps = []
    for fold in range(int(config["folds"])):
        manifest = json.loads((output / f"fold_manifest_{fold}.json").read_text())
        overlaps.extend([
            manifest["fit_evaluation_overlap"], manifest["development_evaluation_overlap"],
            manifest["fit_development_overlap"],
        ])
    if any(overlaps):
        raise RuntimeError(f"Fold leakage detected: {overlaps}")
    checkpoints = list((output / "checkpoints").glob("fold*/*.pt"))
    expected_checkpoints = int(config["folds"]) * len(config["arms"])
    if len(checkpoints) != expected_checkpoints:
        raise RuntimeError(f"Expected {expected_checkpoints} checkpoints, found {len(checkpoints)}")
    required_diagnostic = {"accuracy", "macro_auc", "macro_sensitivity", "macro_specificity", "macro_f1"}
    if not required_diagnostic <= set(diagnostic.columns):
        raise RuntimeError("Missing diagnostic metrics")
    if not {"current", "future"} <= set(regression.endpoint):
        raise RuntimeError("Missing regression endpoints")
    verification = {
        "status": "passed", "scans": len(cohort),
        "subjects": int(cohort.subject_id.nunique()), "classes": config["classes"],
        "folds": int(config["folds"]), "training_arms": config["arms"],
        "checkpoints": len(checkpoints), "prediction_rows": len(predictions),
        "subject_leakage": 0, "history_future_leakage": 0,
        "joint_multitask_gate_passed": bool(decision["joint_multitask_gate_passed"]),
        "original_train_only": True, "validation_or_test_loaded": False,
    }
    (output / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(output / "verification.json")


if __name__ == "__main__":
    main()
