#!/usr/bin/env python3
"""Independent structural and decision checks for Phase 6B."""

import json

import numpy as np
import pandas as pd
import yaml

from memreason_rl.data.common import ROOT


def main() -> None:
    output = ROOT / "outputs/phase6b_mri_incremental_value_dev_20261029"
    config = yaml.safe_load((ROOT / "configs/phase6b_mri_incremental_value_dev.yaml").read_text())
    predictions = pd.read_csv(output / "crossfit_predictions.csv")
    aggregate = pd.read_csv(output / "aggregate_metrics.csv")
    horizon = pd.read_csv(output / "metrics_by_horizon.csv")
    folds = pd.read_csv(output / "metrics_by_fold.csv")
    controls = pd.read_csv(output / "control_diagnostics.csv")
    manifest = json.loads((output / "manifest.json").read_text())
    decision = json.loads((output / "development_decision.json").read_text())
    expected = set(config["arms"])

    assert set(predictions.arm) == expected == set(aggregate.arm)
    assert set(predictions.fold) == set(range(int(config["folds"])))
    assert not predictions.duplicated(["sample_id", "arm"]).any()
    counts = predictions.groupby("arm").sample_id.nunique()
    assert counts.nunique() == 1
    assert set(horizon.horizon_months) == set(config["forecast"]["horizons_months"])
    assert len(folds) == len(expected) * int(config["folds"])
    numeric = aggregate[["composite_mae", "composite_nll", "coverage_95", "interval_width",
                         "latent_mae", "latent_nll", "wasserstein2"]]
    assert np.isfinite(numeric.to_numpy()).all()
    assert aggregate.coverage_95.between(0, 1).all()
    assert (predictions.loc[predictions.mri_available, "mri_recency_years"] >= 0).all()
    assert (predictions.loc[predictions.mri_available, "mri_scan_count"] >= 1).all()

    metadata = predictions.pivot_table(
        index="sample_id", columns="arm", values=["mri_available", "mri_recency_years", "mri_scan_count"]
    )
    for field in ["mri_available", "mri_recency_years", "mri_scan_count"]:
        reference = metadata[field]["real_mri"]
        for arm in expected:
            assert np.allclose(reference, metadata[field][arm], equal_nan=True)

    subject_control = controls[controls.arm.eq("subject_shuffled_mri")]
    time_control = controls[controls.arm.eq("time_shuffled_mri")]
    assert np.allclose(subject_control.evaluation_changed_fraction, 1.0)
    assert (time_control.evaluation_changed_fraction > 0.90).all()
    checkpoints = list((output / "checkpoints").glob("*/fold_*.pt"))
    assert len(checkpoints) == len(expected) * int(config["folds"])
    assert manifest["neural_checkpoints"] == len(checkpoints)

    candidate = config["selection"]["candidate"]
    aggregate_table = aggregate.set_index("arm")
    fold_table = folds.pivot(index="fold", columns="arm", values="composite_mae")
    horizon_table = horizon.pivot(index="horizon_months", columns="arm", values="composite_mae")
    recomputed = {}
    for comparator in config["selection"]["comparators"]:
        recomputed[comparator] = {
            "overall_difference": float(aggregate_table.loc[candidate, "composite_mae"]
                                        - aggregate_table.loc[comparator, "composite_mae"]),
            "folds_better": int((fold_table[candidate] < fold_table[comparator]).sum()),
            "horizons_noninferior": int((horizon_table[candidate] - horizon_table[comparator]
                                         <= float(config["selection"]["maximum_horizon_degradation"])).sum()),
        }
    for comparator, values in recomputed.items():
        stored = decision["comparisons"][comparator]
        assert np.isclose(values["overall_difference"], stored["overall_difference"])
        assert values["folds_better"] == stored["folds_better"]
        assert values["horizons_noninferior"] == stored["horizons_noninferior"]
    assert decision["original_train_only"] and not decision["validation_or_test_loaded"]
    assert decision["primary_endpoint"] == "future_observed_memory_composite_mae"
    assert not decision["measurement_gate_reopened"]
    assert not decision["confirmatory_multimodal_claim_open"]
    supported = all(
        value["overall_difference"] < 0
        and value["folds_better"] >= int(config["selection"]["minimum_folds_better"])
        and value["horizons_noninferior"] >= int(config["selection"]["minimum_horizons_noninferior"])
        for value in recomputed.values()
    ) and bool(decision["coverage_gate"])
    assert supported == bool(decision["mri_content_signal_supported"])

    verification = {
        "status": "passed", "arms": len(expected), "folds": int(config["folds"]),
        "predictions": len(predictions), "checkpoints": len(checkpoints),
        "primary_endpoint": decision["primary_endpoint"],
        "mri_content_signal_supported": supported,
        "validation_or_test_loaded": False, "measurement_gate_reopened": False,
    }
    (output / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(output / "verification.json")


if __name__ == "__main__":
    main()
