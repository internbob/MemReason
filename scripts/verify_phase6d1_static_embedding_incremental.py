#!/usr/bin/env python3
import json
import numpy as np
import pandas as pd
import yaml
from memreason_rl.data.common import ROOT


def main() -> None:
    config = yaml.safe_load((ROOT / "configs/phase6d1_static_embedding_incremental_dev.yaml").read_text())
    output = ROOT / "outputs" / config["run_id"]
    predictions = pd.read_csv(output / "crossfit_predictions.csv"); aggregate = pd.read_csv(output / "aggregate_metrics.csv")
    decision = json.loads((output / "development_decision.json").read_text())
    assert set(predictions.arm) == set(config["arms"]) == set(aggregate.arm)
    assert not predictions.duplicated(["sample_id", "arm"]).any()
    assert set(predictions.fold) == set(range(int(config["folds"])))
    assert np.isfinite(aggregate.select_dtypes(include=[np.number]).to_numpy()).all()
    assert len(list((output / "checkpoints").glob("*/fold_*.pt"))) == len(config["arms"]) * int(config["folds"])
    assert not decision["validation_or_test_loaded"] and not decision["measurement_gate_reopened"]
    result = {"status": "passed", "arms": len(config["arms"]), "folds": int(config["folds"]),
              "predictions": len(predictions), "checkpoints": len(config["arms"]) * int(config["folds"]),
              "static_embedding_gate_passed": decision["static_embedding_gate_passed"], "validation_or_test_loaded": False}
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(output / "verification.json")


if __name__ == "__main__": main()
