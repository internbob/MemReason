"""Independent artifact, reward, and isolation checks for Phase 4D."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memreason_rl.data.common import DOMAINS
from memreason_rl.experiments.phase4b_reasoning_sft import score_generation
from memreason_rl.experiments.phase4d_constrained_grpo import reward_components


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    config = yaml.safe_load((ROOT / "configs/phase4d_cognitive_constrained_grpo.yaml").read_text())
    output = ROOT / "outputs" / config["run_id"]
    manifest = json.loads((output / "manifest.json").read_text())
    training = json.loads((output / "training_state.json").read_text())
    selection = json.loads((output / "development_selection.json").read_text())
    decision = json.loads((output / "final_decision.json").read_text())
    rollout = read_jsonl(output / "rollout_examples.jsonl")
    development = read_jsonl(output / "development_examples.jsonl")
    assert len(rollout) == len({row["subject_id"] for row in rollout}) == 64
    assert len(development) == len({row["subject_id"] for row in development}) == 167
    assert not {row["subject_id"] for row in rollout} & {row["subject_id"] for row in development}
    assert manifest["phase4b_confirmation_reused"] is False
    assert manifest["phase4c_confirmation_reused"] is False
    assert manifest["original_test_loaded"] is False
    assert manifest["future_truth_in_prompt"] is False and manifest["future_truth_reward_only"] is True
    for row in rollout + development:
        assert set(row["model_input"]) == {"cutoff", "horizon", "evidence_trajectories", "domain_statistics",
                                           "last_targets", "frozen_forecast", "frozen_confidence"}
        assert all(float(item["time_month"]) <= float(row["model_input"]["cutoff"]) + 1e-6
                   for item in row["model_input"]["evidence_trajectories"])

    curves = pd.read_csv(output / "training_curves.csv")
    rollouts = read_jsonl(output / "on_policy_rollouts.jsonl")
    assert len(curves) == training["updates_completed"] == 32
    assert len(rollouts) == training["rollout_candidates"] == 256
    assert len({(row["sample_id"], row["candidate"]) for row in rollouts}) == 256
    assert set(curves["update"]) == set(range(1, 33))
    assert np.isfinite(curves.select_dtypes(include=[np.number])).all().all()
    assert curves.mean_kl.max() <= config["policy"]["maximum_mean_kl"]
    assert training["real_on_policy_rollouts"] and training["reference_frozen"] and training["lora_only"]
    assert not training["safety_abort"]
    outcomes = dict(zip(pd.read_csv(output / "rollout_outcome_rewards.csv").sample_id,
                        pd.read_csv(output / "rollout_outcome_rewards.csv").outcome_reward))
    source = {row["sample_id"]: row for row in rollout}
    for candidate in rollouts:
        fresh = reward_components(source[candidate["sample_id"]], candidate["statuses"], candidate["trends"],
                                  outcomes[candidate["sample_id"]], config["reward_weights"])
        for key, value in fresh.items():
            assert np.isclose(value, candidate["reward_components"][key])
        assert candidate["reward_components"]["evidence"] == 1
        assert candidate["reward_components"]["domain"] == 1
        assert candidate["reward_components"]["uncertainty"] == 1
        assert candidate["reward_components"]["format"] == 1
    for update in config["policy"]["checkpoint_updates"]:
        adapter = output / "checkpoints" / f"update_{int(update):04d}" / "adapter_model.safetensors"
        assert adapter.stat().st_size > 1_000_000

    frames = []
    lookup = {row["sample_id"]: row for row in development}
    for arm in ["sft"] + [f"grpo_u{int(update)}" for update in config["policy"]["checkpoint_updates"]]:
        generated = read_jsonl(output / f"development_generations_{arm}.jsonl")
        assert len(generated) == 167 and {row["sample_id"] for row in generated} == set(lookup)
        local = []
        for item in generated:
            source_row = lookup[item["sample_id"]]
            score = score_generation(source_row, item["generation"], arm)
            report = json.loads(item["generation"])
            state = np.mean([report["current_state"][domain]["status"] == source_row["completion"]["current_state"][domain]["status"] for domain in DOMAINS])
            trend = np.mean([report["trend"][domain]["label"] == source_row["completion"]["trend"][domain]["label"] for domain in DOMAINS])
            score.update({"state_label_accuracy": state, "trend_label_accuracy": trend,
                          "action_composite": (state + trend) / 2})
            local.append(score)
        frames.append(pd.DataFrame(local))
    fresh = pd.concat(frames, ignore_index=True)
    stored = pd.read_csv(output / "development_per_report_metrics.csv")
    keys = ["sample_id", "subject_id", "arm"]
    fresh = fresh.sort_values(keys).reset_index(drop=True); stored = stored.sort_values(keys).reset_index(drop=True)
    assert fresh[keys].equals(stored[keys])
    for column in [name for name in fresh if name not in keys]:
        assert np.allclose(fresh[column].astype(float), stored[column].astype(float))
    aggregate = fresh.groupby("arm").mean(numeric_only=True)
    best_arm = max([arm for arm in aggregate.index if arm.startswith("grpo_")],
                   key=lambda arm: (aggregate.loc[arm, "action_composite"], -int(arm.split("u")[-1])))
    assert best_arm == selection["selected_arm"] == "grpo_u32"
    assert np.isclose(aggregate.loc["sft", "action_composite"], selection["sft_action_composite"])
    assert np.isclose(aggregate.loc[best_arm, "action_composite"], selection["selected_action_composite"])
    assert not selection["development_gate_passed"]
    assert selection["criteria"]["action_composite_improved"] is False
    assert all(value for key, value in selection["criteria"].items() if key != "action_composite_improved")
    assert decision["real_on_policy_grpo_completed"] and not decision["confirmation_opened"]
    assert not decision["original_test_loaded"] and not decision["cognitive_rlvr_supported"]
    assert not (output / "confirmation_examples.jsonl").exists()

    result = {"status": "passed", "rollout_subjects": 64, "development_subjects": 167,
              "on_policy_candidates": 256, "updates": 32, "selected_update_for_audit": 32,
              "sft_action_composite": selection["sft_action_composite"],
              "best_grpo_action_composite": selection["selected_action_composite"],
              "development_gate_passed": False, "original_test_loaded": False,
              "cognitive_rlvr_supported": False,
              "selected_adapter_sha256": digest(output / "checkpoints/update_0032/adapter_model.safetensors")}
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
