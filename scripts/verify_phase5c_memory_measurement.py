#!/usr/bin/env python
"""Independent checks for Phase 5C Stage 0/1 artifacts."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))


def main() -> None:
    config = yaml.safe_load((ROOT / "configs/phase5c_memory_measurement_dev.yaml").read_text())
    contract_path = ROOT / config["item_contract"]
    contract = yaml.safe_load(contract_path.read_text())
    output = ROOT / "outputs" / config["run_id"]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["config_sha256"] == hashlib.sha256((ROOT / "configs/phase5c_memory_measurement_dev.yaml").read_bytes()).hexdigest()
    assert manifest["contract_sha256"] == hashlib.sha256(contract_path.read_bytes()).hexdigest()
    assert not manifest["validation_or_test_loaded"]
    items = contract["primary_state_items"]
    assert len(items) == 70 and sum(x["domain"] == "Memory" for x in items) == 14
    assert {x["field"] for x in items if x["instrument"] == "LOGMEM"} == {
        "LogMemIA_ImmediateScore", "LogMemIIA_DelayedScore"}
    assert not any(x["field"].startswith("AVLT_") for x in items)
    audit = pd.read_csv(ROOT / "reports/data_audit/phase5c_memory_field_audit.csv")
    assert len(audit) == 12 and audit.semantics_verified.all() and audit.coding_rule_frozen.all()
    assert audit[audit.field.str.match(r"AVLT_Trial\dInt")].official_description.str.contains("intrusions").all()
    predictions = pd.read_csv(output / "crossfit_predictions.csv")
    states = pd.read_csv(output / "cross_view_states.csv")
    folds = pd.read_csv(output / "fold_manifest.csv")
    assert set(predictions.arm) == set(config["arms"])
    assert set(predictions.task) == {"heldout_item", "heldout_instrument"}
    assert predictions.fold.nunique() == len(folds) == config["folds"]
    assert predictions.groupby(["sample_id", "task", "item"]).arm.nunique().eq(len(config["arms"])).all()
    train_subjects = set(pd.read_csv(ROOT / "splits/train.csv", usecols=["subject_id"]).subject_id.astype(str))
    val_subjects = set(pd.read_csv(ROOT / "splits/val.csv", usecols=["subject_id"]).subject_id.astype(str))
    test_subjects = set(pd.read_csv(ROOT / "splits/test.csv", usecols=["subject_id"]).subject_id.astype(str))
    used = set(predictions.subject_id.astype(str)); assert used <= train_subjects and not used & (val_subjects | test_subjects)
    item = predictions.groupby(["arm", "task", "item"], as_index=False).absolute_error.mean()
    fresh = item.groupby(["arm", "task"]).absolute_error.mean()
    stored = pd.read_csv(output / "aggregate_metrics.csv").set_index(["arm", "task"]).macro_item_mae
    assert np.allclose(fresh.sort_index(), stored.sort_index())
    summary = pd.read_csv(output / "cross_view_summary.csv").set_index("arm")
    for arm, group in states.groupby("arm"):
        assert np.isclose(summary.loc[arm, "mean_absolute_difference"], group.absolute_difference.mean())
    decision = json.loads((output / "development_decision.json").read_text())
    assert decision["development_only"] and decision["original_train_only"]
    assert not decision["validation_loaded"] and not decision["test_loaded"]
    assert not decision["phase5c_complete"]
    result = {"status": "passed", "primary_items": 70, "memory_items": 14,
              "folds": 5, "predictions": len(predictions), "cross_view_states": len(states),
              "train_only": True, "validation_or_test_loaded": False,
              "stage1_measurement_gate_passed": decision["stage1_measurement_gate_passed"],
              "phase5c_complete": False}
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
