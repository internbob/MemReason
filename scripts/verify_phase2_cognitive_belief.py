"""Independent artifact checks for the cognitive belief dynamics experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memreason_rl.data.common import ROOT, read_csv
from memreason_rl.data.dictionary import load_dictionary
from memreason_rl.experiments.phase1_numeric import clean_frame, fit_normalization
from memreason_rl.experiments.phase1b_semantic import select_verified_scalar_features


RUN = ROOT / "outputs/phase2_cognitive_belief_dev_20261010"
CONFIG_PATH = ROOT / "configs/phase2_cognitive_belief_dev.yaml"


def in_fraction(subject: str, seed: int, fraction: float) -> bool:
    digest = hashlib.sha256(f"{seed}:{subject}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < fraction


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    resolved = json.loads((RUN / "resolved_config.json").read_text())
    require(resolved == json.loads(json.dumps(config)), "Resolved configuration differs from frozen YAML")
    status = json.loads((RUN / "status.json").read_text())
    manifest = json.loads((RUN / "manifest.json").read_text())
    decision = json.loads((RUN / "development_decision.json").read_text())
    require(status["completed"] and not status["engineering_smoke_only"], "Run is not formal/completed")
    require(not status["test_evaluated"] and not manifest["test_evaluated"], "Test was marked evaluated")
    require(not decision["semantic_superiority_claim"], "Universal semantic claim was reintroduced")

    arms, seeds = list(config["arms"]), list(config["neural_seeds"])
    checkpoints = sorted((RUN / "checkpoints").glob("*/*/best.pt"))
    require(len(checkpoints) == len(arms) * len(seeds) == 9, "Expected nine checkpoints")
    dictionary = load_dictionary()
    features = select_verified_scalar_features(config, dictionary)
    targets = list(config["targets"])
    require(manifest["features"] == features and manifest["targets"] == targets, "Feature/target contract changed")
    require(len(features) == 68 and len(targets) == 5, "Unexpected feature or target count")

    # Recompute fit-only normalization independently from the source split.
    train = clean_frame(read_csv(ROOT / "splits/train.csv"), features, targets, dictionary)
    validation_source = clean_frame(read_csv(ROOT / "splits/val.csv"), features, targets, dictionary)
    require(not train.duplicated(["subject_id", "visit_month"]).any(), "Train has duplicate subject/month visits")
    require(not validation_source.duplicated(["subject_id", "visit_month"]).any(), "Validation has duplicate subject/month visits")
    seed = int(config["development_partition"]["subject_hash_seed"])
    fraction = float(config["development_partition"]["train_tune_fraction"])
    fit_subjects = {sid for sid in train.subject_id.unique() if not in_fraction(str(sid), seed, fraction)}
    fit = train[train.subject_id.isin(fit_subjects)]
    normalization = fit_normalization(fit, features, targets)
    expected = {
        "feature_mean": [normalization[name]["mean"] for name in features],
        "feature_std": [normalization[name]["std"] for name in features],
        "target_mean": [normalization[name]["mean"] for name in targets],
        "target_std": [normalization[name]["std"] for name in targets],
    }
    for path in checkpoints:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        require(checkpoint["features"] == features and checkpoint["targets"] == targets, f"Contract mismatch: {path}")
        for name, values in expected.items():
            require(np.allclose(checkpoint[name], values), f"Fit-only normalization mismatch: {path}/{name}")

    calibration_seed = read_csv(RUN / "calibration_seed_predictions.csv")
    evaluation_seed = read_csv(RUN / "evaluation_seed_predictions.csv")
    evaluation = read_csv(RUN / "evaluation_predictions.csv")
    require(set(calibration_seed.split) == {"calibration"}, "Calibration file contains another split")
    require(set(evaluation_seed.split) == {"evaluation"}, "Evaluation seed file contains another split")
    require(set(calibration_seed.subject_id).isdisjoint(set(evaluation_seed.subject_id)), "Calibration/evaluation subject leakage")
    require(set(evaluation.task) == {"forecast", "interpolation"}, "Both dynamics tasks were not evaluated")
    require(set(evaluation.model) == set(arms), "Not all arms reached evaluation")
    per_model = evaluation.groupby("model").size()
    require(per_model.nunique() == 1 and int(per_model.iloc[0]) == 5936, "Arms do not share 5,936 targets")
    require(set(evaluation.horizon_months) == {0, 6, 12, 24, 36}, "Unexpected horizon coverage")

    # Recompute the ensemble total variance from the seed-level rows.
    keys = [
        "sample_id", "subject_id", "split", "task", "horizon_months", "target",
        "truth", "truth_norm", "history_length", "observed_fraction", "query_month",
        "target_month", "target_mean", "target_std", "model",
    ]
    recomputed = evaluation_seed.groupby(keys, as_index=False, dropna=False).agg(
        mean_check=("mean_norm", "mean"),
        within=("variance_norm", "mean"),
        between=("mean_norm", "var"),
    )
    recomputed["variance_check"] = recomputed.within + recomputed.between.fillna(0.0)
    paired = evaluation.merge(recomputed, on=keys, validate="one_to_one")
    require(np.allclose(paired.mean_norm, paired.mean_check), "Ensemble mean mismatch")
    require(np.allclose(paired.variance_norm, paired.variance_check), "Total variance mismatch")
    require(np.isfinite(evaluation[["mean_norm", "variance_norm", "calibrated_variance_norm"]]).all().all(), "Non-finite prediction")
    require((evaluation[["variance_norm", "calibrated_variance_norm"]] > 0).all().all(), "Non-positive variance")

    # Recompute every stored exact variance temperature from calibration subjects.
    calibration_ensemble = read_csv(RUN / "calibration_ensemble_predictions.csv")
    temperatures = read_csv(RUN / "calibration_temperatures.csv")
    exact = temperatures[temperatures.level.eq("exact")]
    low, high = map(float, config["calibration"]["temperature_clip"])
    for row in exact.itertuples():
        group = calibration_ensemble[
            calibration_ensemble.model.eq(row.model)
            & calibration_ensemble.task.eq(row.task)
            & calibration_ensemble.target.eq(row.target)
            & calibration_ensemble.horizon_months.eq(row.horizon_months)
        ]
        ratio = (group.truth_norm - group.mean_norm) / np.sqrt(group.variance_norm)
        expected_temperature = float(np.clip(np.sqrt(np.mean(np.square(ratio))), low, high))
        require(np.isclose(row.temperature, expected_temperature), f"Temperature mismatch: {row}")

    metrics = read_csv(RUN / "evaluation_metrics.csv")
    overall = metrics[(metrics.scope == "overall") & metrics.calibrated].set_index("model")
    require(set(overall.index) == set(arms), "Missing calibrated overall metrics")
    bootstrap = read_csv(RUN / "paired_subject_bootstrap.csv")
    require(len(bootstrap) == 6 and set(bootstrap.task) == {"overall", "forecast", "interpolation"}, "Bootstrap contract mismatch")
    gate_row = bootstrap[
        bootstrap.candidate.eq(config["gate"]["candidate"]) & bootstrap.task.eq("overall")
    ].iloc[0]
    candidate = overall.loc[config["gate"]["candidate"]]
    criteria = {
        "normalized_mae_noninferior": bool(gate_row.mae_difference_ci_upper <= config["gate"]["normalized_mae_noninferiority_margin"]),
        "calibrated_nll_improved": bool(gate_row.nll_difference_ci_upper < 0),
        "multilevel_calibration_error_within_bound": bool(candidate.multilevel_calibration_error <= config["gate"]["maximum_multilevel_calibration_error"]),
        "coverage_95_within_bounds": bool(config["gate"]["coverage_95_bounds"][0] <= candidate.coverage_95 <= config["gate"]["coverage_95_bounds"][1]),
    }
    require(criteria == decision["criteria"], "Stored gate criteria do not reproduce")
    require(bool(all(criteria.values())) == decision["cognitive_dynamics_gate_passed"], "Gate result does not reproduce")

    result = {
        "verified": True,
        "run_id": config["run_id"],
        "test_evaluated": False,
        "checkpoint_count": len(checkpoints),
        "features": len(features),
        "targets_per_model": int(per_model.iloc[0]),
        "evaluation_subjects": int(evaluation.subject_id.nunique()),
        "calibration_subjects": int(calibration_seed.subject_id.nunique()),
        "tasks": sorted(evaluation.task.unique()),
        "horizons_months": sorted(int(value) for value in evaluation.horizon_months.unique()),
        "fit_only_normalization_verified": True,
        "calibration_evaluation_subject_disjoint": True,
        "ensemble_total_variance_verified": True,
        "temperature_recomputation_verified": True,
        "unique_subject_month_visits_verified": True,
        "gate_recomputed": decision["cognitive_dynamics_gate_passed"],
    }
    (RUN / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
