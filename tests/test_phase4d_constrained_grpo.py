import json

import pandas as pd

from memreason_rl.experiments.phase4d_constrained_grpo import outcome_records, reward_components, stable_cap


def test_stable_cap_is_deterministic_and_outcome_blind():
    rows = [{"sample_id": f"s{i}", "subject_id": f"p{i}"} for i in range(10)]
    assert stable_cap(rows, 4, 7) == stable_cap(list(reversed(rows)), 4, 7)
    assert len(stable_cap(rows, 4, 7)) == 4


def test_reward_has_candidate_varying_and_frozen_components():
    domains = ["Memory", "Executive", "Attention", "Language", "Orientation"]
    row = {"completion": {
        "current_state": {domain: {"status": "within_training_reference"} for domain in domains},
        "trend": {domain: {"label": "stable"} for domain in domains},
    }}
    statuses = {domain: "within_training_reference" for domain in domains}
    trends = {domain: "stable" for domain in domains}
    weights = {"current_state": .35, "temporal": .35, "outcome": .1, "evidence": .05,
               "domain": .05, "uncertainty": .05, "format": .05}
    perfect = reward_components(row, statuses, trends, .5, weights)
    trends["Memory"] = "decline"
    imperfect = reward_components(row, statuses, trends, .5, weights)
    assert perfect["current_state"] == perfect["temporal"] == 1
    assert perfect["outcome"] == .5 and perfect["evidence"] == perfect["format"] == 1
    assert imperfect["total"] < perfect["total"]
    json.dumps(perfect, allow_nan=False)


def test_outcome_reward_skips_unavailable_targets():
    frame = pd.DataFrame([
        {"sample_id": "a", "subject_id": "s", "target": "x", "prediction": 2.0, "truth": 1.0},
        {"sample_id": "a", "subject_id": "s", "target": "y", "prediction": 3.0, "truth": float("nan")},
    ])
    result = outcome_records(frame, {"a"}, {"x": {"std": 2.0}, "y": {"std": 1.0}})
    assert result.loc[0, "available_targets"] == 1
    assert abs(result.loc[0, "outcome_reward"] - 0.6065306597) < 1e-8
