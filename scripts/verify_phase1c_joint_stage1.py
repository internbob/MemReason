#!/usr/bin/env python3
"""Independent artifact checks for the frozen Phase 1C joint Stage I run."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from memreason_rl.data.common import ROOT, read_csv
from memreason_rl.data.dictionary import load_dictionary
from memreason_rl.experiments.phase1c_joint_stage1 import (
    build_response_vocab,
    clean_frame,
    load_config,
)


RUN = ROOT / "outputs/phase1c_joint_stage1_20261004"
EXPECTED_ARMS = {"numeric_only", "task_semantics", "response_aware"}
EXPECTED_TASKS = {
    "future_forecast",
    "cross_adas_cdr_to_moca",
    "cross_mmse_cdr_to_adas",
    "masked_item",
    "masked_instrument",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    manifest = json.loads((RUN / "manifest.json").read_text())
    status = json.loads((RUN / "status.json").read_text())
    gate = json.loads((RUN / "gate_decision.json").read_text())
    archived_gate = json.loads(
        (ROOT / "reports/experiments/phase1c_gate_decision.json").read_text()
    )
    config = load_config(ROOT / "configs/phase1c_joint_stage1.yaml")
    features = manifest["features"]
    targets = manifest["targets"]

    require(status["completed"] and status["test_evaluated"], "formal run is incomplete")
    require(not status["engineering_smoke_only"], "formal output is marked as smoke")
    require(len(features) == 68 and manifest["dictionary_verified"], "feature contract failed")
    require(set(manifest["arms"]) == EXPECTED_ARMS, "arm contract failed")
    require(len(set(manifest["parameter_counts"].values())) == 1, "arms are not parameter matched")
    require(not manifest["diagnosis_input"], "diagnosis leakage declared in manifest")
    require(manifest["response_vocab_source"] == "train_only", "response vocabulary source failed")

    checkpoints = sorted(RUN.glob("checkpoints/*/*/best.pt"))
    curves = sorted(RUN.glob("checkpoints/*/*/curves.csv"))
    require(len(checkpoints) == 9 and len(curves) == 9, "expected 9 checkpoints and curves")
    for path in checkpoints:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        require(checkpoint["features"] == features, f"feature mismatch: {path}")
        require(checkpoint["targets"] == targets, f"target mismatch: {path}")
        require(checkpoint["parameter_count"] == next(iter(manifest["parameter_counts"].values())), f"parameter mismatch: {path}")

    dictionary = load_dictionary()
    train = clean_frame(read_csv(ROOT / "splits/train.csv"), features, targets, dictionary)
    rebuilt_vocab = build_response_vocab(train, features)
    stored_vocab = json.loads((RUN / "response_vocab_train.json").read_text())
    require(rebuilt_vocab == stored_vocab, "stored response vocabulary is not the train-only vocabulary")

    predictions = pd.read_csv(RUN / "test_predictions.csv")
    require(set(predictions.task.unique()) == EXPECTED_TASKS, "test task set failed")
    require(np.isfinite(predictions[["truth", "prediction"]].to_numpy()).all(), "nonfinite predictions")
    raw = predictions[~predictions.model.str.endswith("ensemble")]
    ensembles = predictions[predictions.model.str.endswith("ensemble")]
    require(set(raw.model.unique()) == EXPECTED_ARMS, "raw prediction arm set failed")
    require(set(raw.seed.astype(int).unique()) == set(config["neural_seeds"]), "seed set failed")

    keys = ["sample_id", "subject_id", "task", "target", "horizon_months", "truth"]
    raw_counts = raw.groupby(keys, dropna=False).size()
    ensemble_counts = ensembles.groupby(keys, dropna=False).size()
    require(raw_counts.eq(9).all(), "each target must have 3 arms x 3 raw predictions")
    require(ensemble_counts.eq(3).all(), "each target must have one ensemble prediction per arm")
    require(len(raw_counts) == len(ensemble_counts), "raw and ensemble target keys differ")

    bootstrap = pd.read_csv(RUN / "paired_subject_bootstrap.csv")
    require(len(bootstrap) == 10, "expected two contrasts for each of five tasks")
    require(set(bootstrap.task) == EXPECTED_TASKS, "bootstrap task set failed")
    require(bootstrap.replicates.eq(config["statistics"]["bootstrap_replicates"]).all(), "bootstrap replicate count failed")
    response = bootstrap[bootstrap.candidate.eq("response_aware ensemble")].set_index("task")
    recomputed = {
        "masked_item_improved": bool(response.loc["masked_item", "ci_upper"] < 0),
        "masked_instrument_improved": bool(response.loc["masked_instrument", "ci_upper"] < 0),
        "cross_scale_two_pairs_improved": bool(
            response.loc[["cross_adas_cdr_to_moca", "cross_mmse_cdr_to_adas"], "ci_upper"].lt(0).all()
        ),
        "no_diagnosis_leakage": gate["criteria"]["no_diagnosis_leakage"],
        "future_noninferior": bool(
            response.loc["future_forecast", "ci_upper"]
            <= config["gate"]["future_noninferiority_margin_normalized_mae"]
        ),
    }
    require(recomputed == gate["criteria"], "gate criteria do not match frozen bootstrap results")
    expected_pass = all(
        recomputed[name]
        for name in [
            "masked_item_improved",
            "masked_instrument_improved",
            "cross_scale_two_pairs_improved",
            "no_diagnosis_leakage",
        ]
    )
    require(gate["stage1_gate_passed"] == expected_pass, "gate decision failed")
    for key in [
        "run_id",
        "criteria",
        "stage1_gate_passed",
        "future_noninferiority_supportive",
        "missingness_only_validation_balanced_accuracy",
        "advance_to_mri_stage2",
    ]:
        require(archived_gate[key] == gate[key], f"archived gate mismatch: {key}")

    result = {
        "verified": True,
        "run_id": manifest["run_id"],
        "features": len(features),
        "parameter_count_per_arm": next(iter(manifest["parameter_counts"].values())),
        "checkpoints": len(checkpoints),
        "test_target_records": int(len(raw_counts)),
        "bootstrap_rows": len(bootstrap),
        "bootstrap_replicates": int(bootstrap.replicates.iloc[0]),
        "stage1_gate_passed": gate["stage1_gate_passed"],
    }
    (RUN / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
