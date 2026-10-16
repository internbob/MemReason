#!/usr/bin/env python3
"""Independent artifact verification for Phase 5D."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    config_path = ROOT / "configs/phase5d_anchored_mirt_dev.yaml"
    config = yaml.safe_load(config_path.read_text())
    contract_path = ROOT / config["item_contract"]
    output = ROOT / "outputs" / config["run_id"]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["config_sha256"] == hashlib.sha256(config_path.read_bytes()).hexdigest()
    assert manifest["contract_sha256"] == hashlib.sha256(contract_path.read_bytes()).hexdigest()
    phase5c_predictions = ROOT / config["phase5c_output"] / "crossfit_predictions.csv"
    assert manifest["phase5c_predictions_sha256"] == hashlib.sha256(phase5c_predictions.read_bytes()).hexdigest()
    assert not manifest["validation_or_test_loaded"]
    predictions = pd.read_csv(output / "crossfit_predictions.csv", low_memory=False)
    states = pd.read_csv(output / "cross_view_memory_states.csv")
    folds = pd.read_csv(output / "fold_manifest.csv")
    anchors = pd.read_csv(output / "anchor_stability.csv")
    dif = pd.read_csv(output / "dif_audit.csv")
    expected_arms = set(config["arms"] + config["baseline_arms"])
    assert set(predictions.arm) == expected_arms
    assert set(predictions.task) == {"heldout_item", "heldout_instrument"}
    assert predictions.fold.nunique() == len(folds) == config["folds"]
    assert predictions.groupby(["sample_id", "task", "item"]).arm.nunique().eq(len(expected_arms)).all()
    train = set(pd.read_csv(ROOT / "splits/train.csv", usecols=["subject_id"]).subject_id.astype(str))
    val = set(pd.read_csv(ROOT / "splits/val.csv", usecols=["subject_id"]).subject_id.astype(str))
    test = set(pd.read_csv(ROOT / "splits/test.csv", usecols=["subject_id"]).subject_id.astype(str))
    used = set(predictions.subject_id.astype(str))
    assert used <= train and not used & (val | test)
    assert set(states.arm) == set(config["arms"])
    assert states.fold.nunique() == config["folds"]
    assert set(dif.group_type) == set(config["dif"]["group_types"])
    if folds.time_dif_items.sum() == 0:
        invariant = predictions[predictions.arm.eq("invariant_mirt")].sort_values(
            ["sample_id", "task", "item"]).prediction_z.to_numpy()
        partial = predictions[predictions.arm.eq("partial_invariant_mirt")].sort_values(
            ["sample_id", "task", "item"]).prediction_z.to_numpy()
        assert np.array_equal(invariant, partial)
    fresh_item = predictions.groupby(["arm", "task", "item"], as_index=False).absolute_error.mean()
    fresh = fresh_item.groupby(["arm", "task"]).absolute_error.mean().sort_index()
    stored = pd.read_csv(output / "aggregate_metrics.csv").set_index(["arm", "task"]).macro_item_mae.sort_index()
    assert np.allclose(fresh, stored)
    decision = json.loads((output / "development_decision.json").read_text())
    comparisons = pd.read_csv(output / "subject_bootstrap.csv")
    instrument = comparisons[comparisons.task.eq("heldout_instrument")].set_index("baseline")
    fixed_pass = instrument.loc[config["gate"]["fixed_baseline"], "ci_high"] < 0
    ae_pass = instrument.loc[config["gate"]["predictive_baseline"], "ci_high"] <= config["gate"]["ae_noninferiority_margin"]
    state_ci = json.loads((output / "cross_view_ccc_bootstrap.json").read_text())
    ccc_pass = state_ci["ci_low"] > config["gate"]["memory_cross_view_ccc_lower_bound"]
    stable_memory = anchors[anchors.domain.eq("Memory") & anchors.stable_for_gate]
    anchor_pass = (len(stable_memory) >= config["gate"]["minimum_memory_anchors"]
                   and stable_memory.instrument.nunique() >= config["gate"]["minimum_anchor_instruments"])
    assert decision["fixed_score_improvement"] == bool(fixed_pass)
    assert decision["autoencoder_noninferiority"] == bool(ae_pass)
    assert decision["cross_view_gate"] == bool(ccc_pass)
    assert decision["anchor_gate"] == bool(anchor_pass)
    expected_gate = all([decision["fixed_score_improvement"], decision["autoencoder_noninferiority"],
                         decision["cross_view_gate"], decision["anchor_gate"],
                         decision["fold_reproducibility_gate"]])
    assert decision["phase5d_measurement_gate_passed"] == expected_gate
    assert decision["continuous_time_dynamics_open"] == expected_gate
    result = {"status": "passed", "folds": len(folds), "arms": len(expected_arms),
              "predictions": len(predictions), "cross_view_pairs": len(states),
              "dif_decisions": len(dif), "stable_memory_anchors": len(stable_memory),
              "train_only": True, "validation_or_test_loaded": False,
              "phase5d_measurement_gate_passed": expected_gate,
              "continuous_time_dynamics_open": expected_gate}
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
