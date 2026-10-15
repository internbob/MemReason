import json

import numpy as np
import pandas as pd

from memreason_rl.experiments.phase4e_trajectory_action_grpo import (
    anchor_action,
    direction,
    load_config,
    score_action,
)


def synthetic_rows(config):
    return pd.DataFrame([
        {"target": target, "prediction": 10.0, "truth": 12.0, "ensemble_std": 1.0}
        for target in config["targets"]
    ])


def synthetic_norm(config):
    return {target: {"mean": 10.0, "std": 4.0, "minimum": 0.0, "maximum": 30.0}
            for target in config["targets"]}


def synthetic_example(config):
    return {"sample_id": "s:m0:h12", "model_input": {
        "last_targets": {target: 10.0 for target in config["targets"]}
    }}


def test_anchor_action_reproduces_frozen_mean():
    config = load_config()
    score = score_action(synthetic_example(config), anchor_action(config), synthetic_rows(config),
                         synthetic_norm(config), config)
    assert np.isclose(score["normalized_mae"], 0.5)
    assert score["clinical_range_rate"] == 1.0


def test_positive_half_std_action_reaches_truth():
    config = load_config()
    action = anchor_action(config)
    for target in config["targets"]:
        action[target]["residual"] = "boost"
    score = score_action(synthetic_example(config), action, synthetic_rows(config),
                         synthetic_norm(config), config)
    assert np.isclose(score["normalized_mae"], 0.0)
    assert score["total_reward"] > 0.8


def test_direction_respects_scale_orientation():
    assert direction(12, 10, 4, True, 0.1) == "decline"
    assert direction(12, 10, 4, False, 0.1) == "improve"
    assert direction(10.1, 10, 4, True, 0.1) == "stable"


def test_prepared_phase4e_partitions_are_subject_disjoint():
    config = load_config()
    root = __import__("pathlib").Path(__file__).parents[1] / "outputs" / config["run_id"]
    sets = []
    for split, expected in [("sft", 256), ("rollout", 96), ("development", 256)]:
        rows = [json.loads(line) for line in (root / f"{split}_examples.jsonl").read_text().splitlines()]
        subjects = {row["subject_id"] for row in rows}
        assert len(rows) == len(subjects) == expected
        sets.append(subjects)
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
