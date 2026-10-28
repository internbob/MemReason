"""Independent artifact and metric checks for Phase 3H MCI conversion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/phase3h_mci_conversion_crossfit.yaml"
MONTH_DAYS = 365.25 / 12.0


def metrics(truth: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(truth, predicted),
        "auc": roc_auc_score(truth, probability),
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "f1": f1_score(truth, predicted, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(truth, predicted),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    output = ROOT / "outputs" / config["run_id"]
    manifest = pd.read_csv(
        output / "conversion_manifest.csv",
        parse_dates=["index_diagnosis_date", "conversion_date", "last_followup_date"],
    )
    folds = pd.read_csv(output / "subject_folds.csv")
    seed_predictions = pd.read_csv(output / "crossfit_seed_predictions.csv")
    predictions = pd.read_csv(output / "crossfit_predictions.csv")
    reported = pd.read_csv(output / "conversion_metrics.csv")
    contrasts = pd.read_csv(output / "conversion_auc_contrasts.csv")
    decision = json.loads((output / "development_decision.json").read_text())
    contract = json.loads((output / "feature_contract.json").read_text())

    train_subjects = set(pd.read_csv(ROOT / "splits/train.csv", usecols=["subject_id"]).subject_id.astype(str))
    validation_subjects = set(pd.read_csv(ROOT / "splits/val.csv", usecols=["subject_id"]).subject_id.astype(str))
    test_subjects = set(pd.read_csv(ROOT / "splits/test.csv", usecols=["subject_id"]).subject_id.astype(str))
    used = set(manifest.subject_id.astype(str))
    assert used <= train_subjects
    assert not used & validation_subjects
    assert not used & test_subjects
    assert contract["post_baseline_input"] is False and contract["diagnosis_input"] is False

    expected = {24: (609, 128), 36: (554, 183)}
    for horizon, (subjects, converters) in expected.items():
        group = manifest[manifest.horizon_months.eq(horizon)]
        assert len(group) == subjects and group.subject_id.nunique() == subjects
        assert int(group.label.sum()) == converters
        assert group.subject_id.is_unique
        positive = group.label.eq(1)
        elapsed = (group.conversion_date - group.index_diagnosis_date).dt.days / MONTH_DAYS
        assert elapsed[positive].gt(0).all() and elapsed[positive].le(horizon).all()
        assert group.loc[~positive, "conversion_date"].isna().all()
        negative_followup = (
            group.loc[~positive, "last_followup_date"]
            - group.loc[~positive, "index_diagnosis_date"]
        ).dt.days / MONTH_DAYS
        assert negative_followup.ge(horizon).all()
        horizon_folds = folds[folds.horizon_months.eq(horizon)]
        assert horizon_folds.subject_id.is_unique
        assert set(horizon_folds.fold) == set(range(int(config["outer_folds"])))
        assert set(horizon_folds.subject_id.astype(str)) == set(group.subject_id.astype(str))

    checkpoints = list((output / "checkpoints").glob("horizon*/fold*/seed*.pt"))
    assert len(checkpoints) == 30
    expected_arms = set(config["arms"])
    assert set(predictions.arm) == expected_arms
    for horizon, (subjects, _) in expected.items():
        group = predictions[predictions.horizon_months.eq(horizon)]
        assert len(group) == subjects * len(expected_arms)
        assert not group.duplicated(["subject_id", "arm"]).any()
        assert group.groupby("subject_id").arm.nunique().eq(len(expected_arms)).all()
        assert group.groupby("subject_id").fold.nunique().eq(1).all()
    probability = predictions.prob_conversion.to_numpy(dtype=float)
    assert np.isfinite(probability).all()
    assert ((probability >= 0) & (probability <= 1)).all()

    seeded = seed_predictions[seed_predictions.seed.ge(0)]
    assert set(seeded.seed.astype(int)) == set(config["neural_seeds"])
    assert seeded.groupby(["horizon_months", "subject_id", "arm"]).seed.nunique().eq(3).all()

    threshold = float(config["classification_threshold"])
    for row in reported.itertuples(index=False):
        group = predictions[
            predictions.horizon_months.eq(row.horizon_months) & predictions.arm.eq(row.arm)
        ].sort_values("subject_id")
        actual = metrics(
            group.label.to_numpy(dtype=int), group.prob_conversion.to_numpy(dtype=float), threshold
        )
        for name in [
            "accuracy", "auc", "sensitivity", "specificity", "f1", "balanced_accuracy",
            "tn", "fp", "fn", "tp",
        ]:
            assert np.isclose(actual[name], getattr(row, name))

    for row in contrasts.itertuples(index=False):
        candidate, baseline = row.comparison.split("_minus_")
        candidate_group = predictions[
            predictions.horizon_months.eq(row.horizon_months) & predictions.arm.eq(candidate)
        ].sort_values("subject_id")
        baseline_group = predictions[
            predictions.horizon_months.eq(row.horizon_months) & predictions.arm.eq(baseline)
        ].sort_values("subject_id")
        assert candidate_group.subject_id.tolist() == baseline_group.subject_id.tolist()
        truth = candidate_group.label.to_numpy(dtype=int)
        difference = (
            roc_auc_score(truth, candidate_group.prob_conversion)
            - roc_auc_score(truth, baseline_group.prob_conversion)
        )
        assert np.isclose(difference, row.auc_difference)

    assert not decision["official_validation_loaded"] and not decision["official_test_loaded"]
    assert decision["external_confirmation_required"]
    result = {
        "status": "passed",
        "horizons": expected,
        "arms": len(expected_arms),
        "checkpoints": len(checkpoints),
        "bootstrap_replicates": int(config["statistics"]["bootstrap_replicates"]),
        "official_validation_loaded": False,
        "official_test_loaded": False,
    }
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
