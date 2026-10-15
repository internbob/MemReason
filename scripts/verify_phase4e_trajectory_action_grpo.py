#!/usr/bin/env python
"""Independent artifact and metric verification for Phase 4E."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memreason_rl.experiments.phase4e_trajectory_action_grpo import (
    load_config,
    prediction_groups,
    read_jsonl,
    score_action,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    config = load_config(); output = ROOT / "outputs" / config["run_id"]
    manifest = json.loads((output / "manifest.json").read_text())
    selection = json.loads((output / "development_selection.json").read_text())
    decision = json.loads((output / "final_decision.json").read_text())
    training = json.loads((output / "training_state.json").read_text())
    split_sets = []
    for split, expected in [("sft", 256), ("rollout", 96), ("development", 256)]:
        rows = read_jsonl(output / f"{split}_examples.jsonl")
        subjects = {row["subject_id"] for row in rows}
        assert len(rows) == len(subjects) == expected
        assert all("truth" not in json.dumps(row["model_input"]) for row in rows)
        split_sets.append(subjects)
    assert not any(split_sets[i] & split_sets[j] for i in range(3) for j in range(i + 1, 3))
    phase4d = {row["subject_id"] for row in read_jsonl(
        ROOT / "outputs" / config["phase4d_run_id"] / "rollout_examples.jsonl")}
    assert not any(subjects & phase4d for subjects in split_sets)
    assert manifest["subject_disjoint"] and not manifest["original_test_loaded"]

    curves = pd.read_csv(output / "training_curves.csv")
    rollouts = read_jsonl(output / "on_policy_rollouts.jsonl")
    assert len(curves) == training["updates_completed"] == 32
    assert len(rollouts) == training["rollout_candidates"] == 256
    assert curves.mean_kl.max() <= float(config["policy"]["maximum_mean_kl"])
    assert not training["safety_abort"]
    for item in rollouts:
        assert set(item["actions"]) == set(config["targets"])
        assert np.isfinite(item["metrics"]["total_reward"])

    rows = read_jsonl(output / "development_examples.jsonl")
    lookup = {row["sample_id"]: row for row in rows}
    predictions = prediction_groups(config)
    normalization = json.loads((output / "normalization.json").read_text())
    fresh = []
    arms = ["frozen", "action_sft", "grpo_u8", "grpo_u16", "grpo_u24", "grpo_u32"]
    distributions = {}
    for arm in arms:
        generated = read_jsonl(output / f"development_actions_{arm}.jsonl")
        assert len(generated) == 256
        counts = Counter()
        for item in generated:
            action = item["actions"]
            metrics = score_action(lookup[item["sample_id"]], action, predictions[item["sample_id"]],
                                   normalization, config)
            fresh.append({"sample_id": item["sample_id"], "subject_id": item["subject_id"],
                          "arm": arm, **metrics})
            counts.update((target, values["residual"], values["uncertainty"])
                          for target, values in action.items())
        distributions[arm] = {"|".join(key): value for key, value in sorted(counts.items())}
    fresh = pd.DataFrame(fresh).sort_values(["arm", "sample_id"]).reset_index(drop=True)
    stored = pd.read_csv(output / "development_per_report_metrics.csv").sort_values(["arm", "sample_id"]).reset_index(drop=True)
    assert fresh[["sample_id", "subject_id", "arm"]].equals(stored[["sample_id", "subject_id", "arm"]])
    for column in [name for name in fresh if name not in {"sample_id", "subject_id", "arm"}]:
        assert np.allclose(fresh[column], stored[column], atol=1e-12)
    aggregate = fresh.groupby("arm").mean(numeric_only=True)
    best = min([arm for arm in arms if arm.startswith("grpo_")],
               key=lambda arm: (aggregate.loc[arm, "normalized_mae"], int(arm.split("u")[-1])))
    assert best == selection["selected_arm"] == "grpo_u8"
    assert not selection["development_gate_passed"] and not decision["trajectory_action_rlvr_supported"]
    assert not decision["original_test_loaded"] and not decision["confirmation_opened"]
    assert not (output / "confirmation_examples.jsonl").exists()
    assert distributions["action_sft"] == distributions["grpo_u8"] == distributions["grpo_u32"]

    result = {
        "status": "passed", "sft_subjects": 256, "rollout_subjects": 96,
        "development_subjects": 256, "updates": 32, "on_policy_candidates": 256,
        "selected_update": 8, "active_group_fraction": selection["active_group_fraction"],
        "frozen_normalized_mae": selection["frozen_normalized_mae"],
        "selected_normalized_mae": selection["selected_normalized_mae"],
        "grpo_minus_frozen_normalized_mae": decision["grpo_minus_frozen_normalized_mae"],
        "development_gate_passed": False, "original_test_loaded": False,
        "trajectory_action_rlvr_supported": False,
        "selected_adapter_sha256": digest(output / "checkpoints/update_0008/adapter_model.safetensors"),
        "greedy_action_diagnosis": "all_action_sft_and_grpo_targets_selected_lower_plus_wide",
    }
    (output / "action_distribution.json").write_text(json.dumps(distributions, indent=2) + "\n")
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
