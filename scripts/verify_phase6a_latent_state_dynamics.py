#!/usr/bin/env python3
"""Independent structural checks for Phase 6A latent dynamics."""

import json

import numpy as np
import pandas as pd
import yaml

from memreason_rl.data.common import ROOT


def main() -> None:
    output = ROOT / "outputs/phase6a_latent_state_dynamics_dev_20261029"
    config = yaml.safe_load((ROOT / "configs/phase6a_latent_state_dynamics_dev.yaml").read_text())
    predictions = pd.read_csv(output / "crossfit_predictions.csv")
    aggregate = pd.read_csv(output / "aggregate_metrics.csv")
    manifest = pd.read_csv(output / "fold_manifest.csv")
    decision = json.loads((output / "development_decision.json").read_text())
    expected_arms = set(config["arms"])
    assert set(predictions.arm) == expected_arms == set(aggregate.arm)
    assert set(predictions.fold) == set(range(int(config["folds"])))
    assert not predictions.duplicated(["sample_id", "arm"]).any()
    numeric = aggregate[["latent_mae", "latent_nll", "wasserstein2", "coverage_95", "composite_mae"]]
    assert np.isfinite(numeric.to_numpy()).all()
    assert aggregate.coverage_95.between(0, 1).all()
    assert len(manifest) == int(config["folds"])
    checkpoints = list((output / "checkpoints").glob("*/fold_*.pt"))
    assert len(checkpoints) == (len(expected_arms) - 1) * int(config["folds"])
    assert decision["original_train_only"] and not decision["validation_or_test_loaded"]
    assert not decision["measurement_gate_reopened"]
    assert not decision["confirmatory_continuous_time_claim_open"]
    verification = {"status": "passed", "arms": len(expected_arms), "folds": int(config["folds"]),
                    "predictions": len(predictions), "checkpoints": len(checkpoints),
                    "measurement_gate_reopened": False, "validation_or_test_loaded": False}
    (output / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(output / "verification.json")


if __name__ == "__main__":
    main()
