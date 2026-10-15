"""Independent artifact and source-isolation checks for Phase 4B."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memreason_rl.experiments.phase4b_reasoning_sft import load_rows, stable_subset
from memreason_rl.models.reasoning_llm import parse_report


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def main() -> None:
    config = yaml.safe_load((ROOT / "configs/phase4b_cognitive_reasoning_sft.yaml").read_text())
    output = ROOT / "outputs" / config["run_id"]
    manifest = json.loads((output / "manifest.json").read_text())
    selection = json.loads((output / "selection_manifest.json").read_text())
    training = json.loads((output / "training_state.json").read_text())
    decision = json.loads((output / "development_decision.json").read_text())
    phase4a = ROOT / "outputs" / config["source_run_id"]

    assert manifest["examples"] == 16577 and manifest["sample_counts"] == {
        "build": 13238, "confirmation": 1757, "dev": 1582}
    assert not manifest["future_truth_in_prompt"] and not manifest["diagnosis_input"] and not manifest["mri_input"]
    assert not manifest["target_truncation"]
    lengths = pd.read_csv(output / "token_lengths.csv")
    assert len(lengths) == manifest["examples"]
    assert lengths.total_tokens.max() == manifest["token_lengths"]["total_tokens"]["max"] <= config["prompt"]["max_length"]

    all_rows = load_rows(output)
    build = stable_subset([x for x in all_rows if x["reasoning_split"] == "build"],
                          per_subject=config["training"]["maximum_examples_per_build_subject"],
                          seed=config["training"]["seed"])
    dev = stable_subset([x for x in all_rows if x["reasoning_split"] == "dev"],
                        per_subject_horizon=config["training"]["maximum_examples_per_dev_subject_horizon"],
                        seed=config["training"]["seed"])
    confirmation = stable_subset([x for x in all_rows if x["reasoning_split"] == "confirmation"],
                                 per_subject=config["evaluation"]["maximum_examples_per_confirmation_subject"],
                                 seed=config["training"]["seed"])
    assert [x["sample_id"] for x in build] == selection["training_sample_ids"]
    assert [x["sample_id"] for x in dev] == selection["dev_sample_ids"]
    assert [x["sample_id"] for x in confirmation] == selection["confirmation_sample_ids"]
    assert len(build) == training["build_examples"] == 2471
    assert len(dev) == training["dev_examples"] == 488
    assert len(confirmation) == selection["confirmation_subjects"] == 168
    assert len({x["subject_id"] for x in build}) == 1336
    assert len({x["subject_id"] for x in confirmation}) == 168
    assert {int(x["model_input"]["horizon"]) for x in confirmation} == {6, 12, 24, 36}

    registry = pd.read_csv(phase4a / "subject_registry.csv")
    train_subjects = set(pd.read_csv(ROOT / "splits/train.csv", usecols=["subject_id"]).subject_id.astype(str))
    val_subjects = set(pd.read_csv(ROOT / "splits/val.csv", usecols=["subject_id"]).subject_id.astype(str))
    test_subjects = set(pd.read_csv(ROOT / "splits/test.csv", usecols=["subject_id"]).subject_id.astype(str))
    used = set(registry.subject_id.astype(str))
    assert used <= train_subjects and not used & (val_subjects | test_subjects)
    for row in all_rows:
        assert set(row["model_input"]) == {"cutoff", "horizon", "evidence_trajectories", "domain_statistics",
                                           "last_targets", "frozen_forecast", "frozen_confidence"}
        assert all(float(x["time_month"]) <= float(row["model_input"]["cutoff"]) + 1e-6
                   for x in row["model_input"]["evidence_trajectories"])

    assert training["epochs"] == 1 and training["optimizer_updates"] == 155
    assert training["completion_only_labels"] and not training["confirmation_loaded_during_training"]
    adapter_config = json.loads((output / "adapter/adapter_config.json").read_text())
    assert adapter_config["r"] == config["training"]["lora_rank"]
    assert set(adapter_config["target_modules"]) == set(config["training"]["lora_targets"])
    adapter_file = output / "adapter/adapter_model.safetensors"
    assert adapter_file.stat().st_size > 1_000_000

    expected = {x["sample_id"]: x for x in confirmation}
    parsed_counts, exact_counts = {}, {}
    generation_hashes = {}
    for arm in ["base", "sft"]:
        path = output / f"confirmation_generations_{arm}.jsonl"
        generations = load_jsonl(path)
        assert len(generations) == 168 and len({x["sample_id"] for x in generations}) == 168
        assert {x["sample_id"] for x in generations} == set(expected)
        parsed = 0; exact = 0
        for generated in generations:
            try:
                report = parse_report(generated["generation"].strip())
                parsed += 1
                exact += int(report == expected[generated["sample_id"]]["completion"])
                cutoff = float(expected[generated["sample_id"]]["model_input"]["cutoff"])
                assert all(float(e["time_month"]) <= cutoff + 1e-6 for e in report["evidence"])
            except (ValueError, json.JSONDecodeError, TypeError, KeyError):
                pass
        parsed_counts[arm] = parsed; exact_counts[arm] = exact
        generation_hashes[arm] = digest(path)
    assert parsed_counts == {"base": 0, "sft": 166}
    assert exact_counts == {"base": 0, "sft": 131}

    per_report = pd.read_csv(output / "confirmation_per_report_metrics.csv")
    aggregate = pd.read_csv(output / "aggregate_metrics.csv").set_index("arm")
    assert len(per_report) == 336
    for arm in ["base", "sft"]:
        local = per_report[per_report.arm.eq(arm)]
        assert np.isclose(local.format_validity.mean(), aggregate.loc[arm, "format_validity"])
        assert np.isclose(local.exact_report_match.mean(), aggregate.loc[arm, "exact_report_match"])
    assert np.isclose(aggregate.loc["sft", "format_validity"], 166 / 168)
    assert np.isclose(aggregate.loc["sft", "exact_report_match"], 131 / 168)
    assert aggregate.loc["sft", "future_evidence_leakage"] == 0
    assert decision["criteria"]["improvement_over_base"]
    assert not decision["phase4b_sft_passed"] and not decision["cognitive_grpo_ready"]
    assert not decision["multimodal_reasoning_ready"]

    result = {
        "status": "passed", "source_examples": manifest["examples"],
        "training_examples": len(build), "dev_examples": len(dev),
        "confirmation_examples": len(confirmation), "confirmation_subjects": 168,
        "base_valid_json": parsed_counts["base"], "sft_valid_json": parsed_counts["sft"],
        "sft_exact_reports": exact_counts["sft"], "adapter_sha256": digest(adapter_file),
        "generation_sha256": generation_hashes, "original_validation_used": False,
        "original_test_used": False, "future_truth_in_prompt": False,
        "phase4b_sft_passed": False, "cognitive_grpo_ready": False,
    }
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
