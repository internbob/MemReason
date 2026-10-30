#!/usr/bin/env python3
"""Independent structural and decision checks for Phase 6C."""

import json

import numpy as np
import pandas as pd
import yaml

from memreason_rl.data.common import ROOT


def main() -> None:
    output = ROOT / "outputs/phase6c_longitudinal_mri_identifiability_dev_20261030"
    config = yaml.safe_load((ROOT / "configs/phase6c_longitudinal_mri_identifiability_dev.yaml").read_text())
    predictions = pd.read_csv(output / "crossfit_predictions.csv")
    aggregate = pd.read_csv(output / "aggregate_metrics.csv")
    horizon = pd.read_csv(output / "metrics_by_horizon.csv")
    folds = pd.read_csv(output / "metrics_by_fold.csv")
    controls = pd.read_csv(output / "control_diagnostics.csv")
    eligibility = pd.read_csv(output / "eligible_state_mri_metadata.csv")
    manifest = json.loads((output / "manifest.json").read_text())
    decision = json.loads((output / "development_decision.json").read_text())
    expected = set(config["arms"])

    assert set(predictions.arm) == expected == set(aggregate.arm)
    assert set(predictions.fold) == set(range(int(config["folds"])))
    assert not predictions.duplicated(["sample_id", "arm"]).any()
    counts = predictions.groupby("arm").sample_id.nunique()
    assert counts.nunique() == 1 and counts.iloc[0] == manifest["forecast_samples"]
    assert predictions.groupby("arm").subject_id.nunique().nunique() == 1
    assert set(horizon.horizon_months) == set(config["forecast"]["horizons_months"])
    assert len(folds) == len(expected) * int(config["folds"])
    numeric = aggregate[["composite_mae", "composite_nll", "coverage_95", "interval_width",
                         "latent_mae", "latent_nll", "wasserstein2"]]
    assert np.isfinite(numeric.to_numpy()).all()
    assert aggregate.coverage_95.between(0, 1).all()

    assert (eligibility.mri_scan_count >= int(config["mri"]["minimum_scans"])).all()
    assert (eligibility.mri_span_years >= float(config["mri"]["minimum_span_years"])).all()
    assert (pd.to_datetime(eligibility.latest_scan_date)
            <= pd.to_datetime(eligibility.cognitive_cutoff_date)).all()
    shared = predictions.pivot_table(
        index="sample_id", columns="arm",
        values=["mri_available", "mri_scan_count", "mri_span_years", "mri_recency_years"],
    )
    for field in ["mri_available", "mri_scan_count", "mri_span_years", "mri_recency_years"]:
        reference = shared[field]["true_longitudinal_mri"]
        for arm in expected:
            assert np.allclose(reference, shared[field][arm], equal_nan=True)

    assert np.allclose(controls.maximum_static_feature_difference, 0.0)
    date_control = controls[controls.arm.eq("date_permuted_dynamics")]
    subject_control = controls[controls.arm.eq("subject_shuffled_dynamics")]
    assert (date_control.dynamic_changed_fraction > .90).all()
    assert (subject_control.dynamic_changed_fraction > .90).all()
    assert np.allclose(subject_control.different_subject_donor_fraction, 1.0)
    checkpoints = list((output / "checkpoints").glob("*/fold_*.pt"))
    assert len(checkpoints) == len(expected) * int(config["folds"])
    assert manifest["checkpoint_files"] == len(checkpoints)

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
            "horizons_better": int((horizon_table[candidate] < horizon_table[comparator]).sum()),
        }
    for comparator, values in recomputed.items():
        stored = decision["comparisons"][comparator]
        assert np.isclose(values["overall_difference"], stored["overall_difference"])
        assert values["folds_better"] == stored["folds_better"]
        assert values["horizons_better"] == stored["horizons_better"]
    supported = all(
        value["overall_difference"] < 0
        and value["folds_better"] >= int(config["selection"]["minimum_folds_better"])
        and value["horizons_better"] >= int(config["selection"]["minimum_horizons_better"])
        for value in recomputed.values()
    ) and bool(decision["coverage_gate"])
    assert supported == bool(decision["longitudinal_mri_signal_supported"])
    assert decision["original_train_only"] and not decision["validation_or_test_loaded"]
    assert not decision["measurement_gate_reopened"]
    assert not decision["confirmatory_multimodal_claim_open"]

    figure_files = list((output / "figures").glob("figure_*.pdf"))
    png_files = list((output / "figures").glob("figure_*.png"))
    assert len(figure_files) == 6 == len(png_files)
    verification = {
        "status": "passed", "arms": len(expected), "folds": int(config["folds"]),
        "predictions": len(predictions), "forecast_samples_per_arm": int(counts.iloc[0]),
        "subjects": manifest["subjects"], "checkpoints": len(checkpoints),
        "figures_pdf": len(figure_files), "figures_png": len(png_files),
        "primary_endpoint": decision["primary_endpoint"],
        "longitudinal_mri_signal_supported": supported,
        "validation_or_test_loaded": False, "measurement_gate_reopened": False,
    }
    (output / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(output / "verification.json")


if __name__ == "__main__":
    main()
