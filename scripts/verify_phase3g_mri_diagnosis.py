"""Independent integrity checks for Phase 3G MRI diagnosis cross-fitting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/phase3g_mri_diagnosis_crossfit.yaml"


def subject_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("subject_id")["subject_id"].transform("size").to_numpy(dtype=float)
    return 1.0 / counts


def macro_auc(frame: pd.DataFrame) -> float:
    truth = frame.label.to_numpy(dtype=int)
    probability = frame[["prob_CN", "prob_MCI", "prob_AD"]].to_numpy(dtype=float)
    weights = subject_weights(frame)
    values = []
    for label in range(3):
        values.append(roc_auc_score(truth == label, probability[:, label], sample_weight=weights))
    return float(np.mean(values))


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    output = ROOT / "outputs" / config["run_id"]
    manifest = pd.read_csv(output / "diagnostic_manifest.csv")
    folds = pd.read_csv(output / "subject_folds.csv")
    seed_predictions = pd.read_csv(output / "crossfit_seed_predictions.csv")
    predictions = pd.read_csv(output / "crossfit_predictions.csv")
    metrics = pd.read_csv(output / "diagnostic_metrics.csv").set_index("arm")
    requested = pd.read_csv(output / "diagnostic_requested_metrics.csv").set_index("arm")
    intervals = pd.read_csv(output / "diagnostic_metric_intervals.csv")
    per_class = pd.read_csv(output / "diagnostic_per_class_metrics.csv")
    bootstrap = pd.read_csv(output / "diagnostic_bootstrap.csv")
    curves = pd.read_csv(output / "decision_curves.csv")
    decision = json.loads((output / "development_decision.json").read_text())

    train_subjects = set(pd.read_csv(ROOT / "splits/train.csv", usecols=["subject_id"]).subject_id.astype(str))
    validation_subjects = set(pd.read_csv(ROOT / "splits/val.csv", usecols=["subject_id"]).subject_id.astype(str))
    test_subjects = set(pd.read_csv(ROOT / "splits/test.csv", usecols=["subject_id"]).subject_id.astype(str))
    used = set(manifest.subject_id.astype(str))
    assert used <= train_subjects
    assert not used & validation_subjects
    assert not used & test_subjects
    assert manifest.split.eq("train").all()
    assert manifest.scan_id.is_unique
    assert len(manifest) == 5370 and len(used) == 1328
    assert set(manifest.dxsum_diagnosis_label) == {"CN", "MCI", "AD"}
    assert manifest.diagnosis_date_difference_days.abs().le(config["date_window_days"]).all()

    assert folds.subject_id.is_unique and set(folds.subject_id.astype(str)) == used
    assert set(folds.fold) == set(range(int(config["outer_folds"])))
    assert manifest.groupby("subject_id").fold.nunique().max() == 1
    fold_map = folds.set_index("subject_id").fold
    assert (
        manifest.fold.to_numpy()
        == manifest.subject_id.map(fold_map).to_numpy()
    ).all()

    checkpoints = list((output / "checkpoints").glob("fold*/seed*.pt"))
    expected_checkpoints = int(config["outer_folds"]) * len(config["neural_seeds"])
    assert len(checkpoints) == expected_checkpoints == 15
    expected_arms = {
        "demographics", "global_mri", "sfcn_mri", "cognition",
        "cognition_global_mri", "cognition_sfcn_mri", "cognition_shuffled_sfcn",
    }
    assert set(predictions.arm) == expected_arms
    assert len(predictions) == len(manifest) * len(expected_arms)
    assert not predictions.duplicated(["row_index", "arm"]).any()
    assert predictions.groupby("row_index").arm.nunique().eq(len(expected_arms)).all()
    assert predictions.groupby("subject_id").fold.nunique().max() == 1
    assert set(predictions.scan_id) == set(manifest.scan_id)

    probability_columns = ["prob_CN", "prob_MCI", "prob_AD"]
    probability = predictions[probability_columns].to_numpy(dtype=float)
    assert np.isfinite(probability).all()
    assert ((probability >= 0.0) & (probability <= 1.0)).all()
    np.testing.assert_allclose(probability.sum(1), 1.0, atol=2e-6)

    seed_counts = seed_predictions.groupby(["row_index", "arm"]).seed.nunique()
    assert seed_counts.loc[seed_counts.index.get_level_values("arm").isin(
        ["sfcn_mri", "cognition_sfcn_mri", "cognition_shuffled_sfcn"]
    )].eq(3).all()
    assert set(seed_predictions.loc[seed_predictions.seed.ge(0), "seed"].astype(int)) == set(config["neural_seeds"])

    for arm, group in predictions.groupby("arm"):
        actual_auc = macro_auc(group)
        truth = group.label.to_numpy(dtype=int)
        predicted = group[probability_columns].to_numpy().argmax(1)
        weights = subject_weights(group)
        actual_balanced = balanced_accuracy_score(
            truth, predicted, sample_weight=weights
        )
        actual_accuracy = accuracy_score(truth, predicted, sample_weight=weights)
        actual_f1 = f1_score(truth, predicted, average="macro", sample_weight=weights)
        specificities = []
        for label in range(3):
            negative = truth != label
            specificities.append(np.average(predicted[negative] != label, weights=weights[negative]))
        assert np.isclose(actual_auc, metrics.loc[arm, "macro_auc"], atol=1e-12)
        assert np.isclose(actual_balanced, metrics.loc[arm, "balanced_accuracy"], atol=1e-12)
        assert np.isclose(actual_accuracy, requested.loc[arm, "accuracy"], atol=1e-12)
        assert np.isclose(actual_balanced, requested.loc[arm, "macro_sensitivity"], atol=1e-12)
        assert np.isclose(np.mean(specificities), requested.loc[arm, "macro_specificity"], atol=1e-12)
        assert np.isclose(actual_f1, requested.loc[arm, "macro_f1"], atol=1e-12)
    assert len(intervals) == len(expected_arms) * 5
    assert intervals.replicates.eq(config["statistics"]["bootstrap_replicates"]).all()
    assert len(per_class) == len(expected_arms) * 3

    macro = bootstrap[bootstrap.metric.eq("macro_auc")].set_index("comparison")
    gate = config["gate"]
    mri = macro.loc["sfcn_mri_minus_demographics"]
    increment = macro.loc["cognition_sfcn_mri_minus_cognition"]
    mechanism = macro.loc["cognition_sfcn_mri_minus_cognition_shuffled_sfcn"]
    curve_table = curves.pivot(index="threshold", columns="arm", values="net_benefit")
    gain_thresholds = int((curve_table.cognition_sfcn_mri > curve_table.cognition).sum())
    reproduced = {
        "mri_signal_effect": bool(mri.difference >= gate["minimum_mri_vs_demographics_macro_auc_difference"]),
        "mri_signal_ci": bool(mri.ci_low > gate["minimum_ci_low"]),
        "fusion_increment_effect": bool(increment.difference >= gate["minimum_fusion_vs_cognition_macro_auc_difference"]),
        "fusion_increment_ci": bool(increment.ci_low > gate["minimum_ci_low"]),
        "mechanism_effect": bool(mechanism.difference >= gate["minimum_fusion_vs_shuffled_macro_auc_difference"]),
        "mechanism_ci": bool(mechanism.ci_low > gate["minimum_ci_low"]),
        "balanced_accuracy_noninferiority": bool(
            metrics.loc["cognition_sfcn_mri", "balanced_accuracy"]
            - metrics.loc["cognition", "balanced_accuracy"]
            >= gate["minimum_balanced_accuracy_difference"]
        ),
        "brier_noninferiority": bool(
            metrics.loc["cognition_sfcn_mri", "brier"] - metrics.loc["cognition", "brier"]
            <= gate["maximum_brier_increase"]
        ),
        "decision_curve_gain": bool(
            gain_thresholds >= gate["require_positive_ad_net_benefit_gain_at_thresholds"]
        ),
    }
    assert reproduced == decision["criteria"]
    assert decision["mri_concurrent_signal_supported"] == (
        reproduced["mri_signal_effect"] and reproduced["mri_signal_ci"]
    )
    assert decision["mri_increment_beyond_cognition_supported"] == all(reproduced.values())
    assert decision["mci_conversion_open"] == all(reproduced.values())
    assert not decision["official_validation_loaded"] and not decision["official_test_loaded"]
    assert bootstrap.replicates.eq(config["statistics"]["bootstrap_replicates"]).all()

    result = {
        "status": "passed",
        "subjects": len(used),
        "scans": len(manifest),
        "folds": int(config["outer_folds"]),
        "checkpoints": len(checkpoints),
        "bootstrap_replicates": int(config["statistics"]["bootstrap_replicates"]),
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "mri_concurrent_signal_supported": decision["mri_concurrent_signal_supported"],
        "mri_increment_beyond_cognition_supported": decision["mri_increment_beyond_cognition_supported"],
        "mci_conversion_open": decision["mci_conversion_open"],
    }
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
