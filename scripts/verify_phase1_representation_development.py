#!/usr/bin/env python3
"""Verify Phase 1D/1E/1F development artifacts and test isolation."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from memreason_rl.data.common import ROOT
from memreason_rl.experiments.phase1c_joint_stage1 import load_config


RUNS = {
    "phase1d_typed_representation_dev_20261004": {
        "config": "phase1d_typed_representation_dev.yaml",
        "arms": 4,
    },
    "phase1e_gated_aligned_dev_20261009": {
        "config": "phase1e_gated_aligned_dev.yaml",
        "arms": 3,
    },
    "phase1f_task_gated_dev_20261009": {
        "config": "phase1f_task_gated_dev.yaml",
        "arms": 3,
    },
}
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
    summary = {}
    for run_id, contract in RUNS.items():
        output = ROOT / "outputs" / run_id
        status = json.loads((output / "status.json").read_text())
        manifest = json.loads((output / "manifest.json").read_text())
        decision = json.loads((output / "development_decision.json").read_text())
        config = load_config(ROOT / "configs" / contract["config"])
        resolved = json.loads((output / "resolved_config.json").read_text())
        normalized_config = json.loads(json.dumps(config))
        require(resolved == normalized_config, f"resolved config mismatch: {run_id}")
        require(status["completed"] and not status["engineering_smoke_only"], f"incomplete: {run_id}")
        require(not status["test_evaluated"] and not manifest["test_evaluated"], f"test evaluated: {run_id}")
        require(not decision["test_evaluated"], f"decision reports test evaluation: {run_id}")
        require(not decision["advance_to_mri_stage2"], f"MRI advanced without gate: {run_id}")
        require(len(set(manifest["parameter_counts"].values())) == 1, f"unmatched parameters: {run_id}")

        checkpoints = sorted(output.glob("checkpoints/*/*/best.pt"))
        require(len(checkpoints) == contract["arms"], f"checkpoint count failed: {run_id}")
        for checkpoint_path in checkpoints:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            require(checkpoint["features"] == manifest["features"], f"feature mismatch: {checkpoint_path}")
            require(checkpoint["targets"] == manifest["targets"], f"target mismatch: {checkpoint_path}")

        predictions = pd.read_csv(output / "validation_predictions.csv")
        require(set(predictions.split) == {"val"}, f"non-validation prediction: {run_id}")
        require(set(predictions.task) == EXPECTED_TASKS, f"task set mismatch: {run_id}")
        require(np.isfinite(predictions[["truth", "prediction"]].to_numpy()).all(), f"nonfinite: {run_id}")
        arms = set(config["arms"])
        require(set(predictions.model) == arms, f"prediction arms mismatch: {run_id}")
        keys = ["sample_id", "subject_id", "task", "target", "horizon_months", "truth"]
        require(predictions.groupby(keys, dropna=False).size().eq(len(arms)).all(), f"unpaired predictions: {run_id}")

        bootstrap = pd.read_csv(output / "paired_subject_bootstrap.csv")
        require(len(bootstrap) == (len(arms) - 1) * 5, f"bootstrap rows failed: {run_id}")
        require(set(bootstrap.task) == EXPECTED_TASKS, f"bootstrap tasks failed: {run_id}")
        require(bootstrap.replicates.eq(config["statistics"]["bootstrap_replicates"]).all(), f"replicates failed: {run_id}")

        record = {
            "verified": True,
            "test_evaluated": False,
            "arms": len(arms),
            "checkpoints": len(checkpoints),
            "validation_target_records": int(predictions.groupby(keys, dropna=False).ngroups),
            "bootstrap_rows": len(bootstrap),
            "provisional_validation_passed": decision["provisional_validation_passed"],
        }
        (output / "verification.json").write_text(json.dumps(record, indent=2) + "\n")
        summary[run_id] = record

    source_text = "\n".join(
        (ROOT / "memreason_rl/experiments" / filename).read_text()
        for filename in [
            "phase1d_typed_representation.py",
            "phase1e_gated_aligned.py",
            "phase1f_task_gated.py",
        ]
    )
    require('splits/test.csv' not in source_text, "development source references test split")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
