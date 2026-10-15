"""Independent provenance and metric checks for Phase 4C."""

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

from memreason_rl.experiments.phase4b_reasoning_sft import score_generation
from memreason_rl.models.reasoning_llm import parse_report


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    config = yaml.safe_load((ROOT / "configs/phase4c_grammar_constrained_decoding.yaml").read_text())
    output = ROOT / "outputs" / config["run_id"]
    manifest = json.loads((output / "manifest.json").read_text())
    selection = json.loads((output / "dev_selection_decision.json").read_text())
    decision = json.loads((output / "development_decision.json").read_text())
    dev = load_jsonl(output / "dev_examples.jsonl")
    confirmation = load_jsonl(output / "new_confirmation_examples.jsonl")

    assert len(dev) == len({row["subject_id"] for row in dev}) == 167
    assert len(confirmation) == len({row["subject_id"] for row in confirmation}) == 234
    assert {int(row["model_input"]["horizon"]) for row in confirmation} == {6, 12, 24, 36}
    train_subjects = set(pd.read_csv(ROOT / "splits/train.csv", usecols=["subject_id"]).subject_id.astype(str))
    val_subjects = set(pd.read_csv(ROOT / "splits/val.csv", usecols=["subject_id"]).subject_id.astype(str))
    test_subjects = set(pd.read_csv(ROOT / "splits/test.csv", usecols=["subject_id"]).subject_id.astype(str))
    dev_subjects = {row["subject_id"] for row in dev}
    confirmation_subjects = {row["subject_id"] for row in confirmation}
    assert dev_subjects <= train_subjects
    assert confirmation_subjects <= val_subjects
    assert not dev_subjects & confirmation_subjects
    assert not confirmation_subjects & test_subjects
    assert manifest["original_test_used"] is False
    assert manifest["previous_confirmation_reused"] is False
    assert manifest["future_truth_in_prompt"] is False
    assert manifest["diagnosis_input"] is False and manifest["mri_input"] is False
    for row in dev + confirmation:
        assert set(row["model_input"]) == {
            "cutoff", "horizon", "evidence_trajectories", "domain_statistics",
            "last_targets", "frozen_forecast", "frozen_confidence",
        }
        assert all(float(item["time_month"]) <= float(row["model_input"]["cutoff"]) + 1e-6
                   for item in row["model_input"]["evidence_trajectories"])

    assert selection["selection_uses_dev_only"] and not selection["previous_confirmation_used"]
    assert selection["selected_arm"] == "grammar_model"
    assert all(selection["criteria"]["grammar_model"].values())
    assert decision["phase4c_structured_decoding_passed"]
    assert decision["stable_sft_policy_for_grpo"] and decision["cognitive_grpo_ready"]
    assert not decision["multimodal_reasoning_ready"]

    status_choices = set(config["decoding"]["status_choices"])
    trend_choices = set(config["decoding"]["trend_choices"])
    expected = {row["sample_id"]: row for row in confirmation}
    recomputed = []
    parsed_counts: dict[str, int] = {}
    exact_counts: dict[str, int] = {}
    generation_hashes: dict[str, str] = {}
    for arm in ["raw_sft", "grammar_model"]:
        path = output / f"new_confirmation_generations_{arm}.jsonl"
        generated = load_jsonl(path)
        assert len(generated) == len(expected)
        assert {row["sample_id"] for row in generated} == set(expected)
        parsed = exact = 0
        for row in generated:
            source = expected[row["sample_id"]]
            recomputed.append(score_generation(source, row["generation"], arm))
            try:
                report = parse_report(row["generation"].strip())
                parsed += 1
                exact += int(report == source["completion"])
                assert all(float(item["time_month"]) <= float(source["model_input"]["cutoff"]) + 1e-6
                           for item in report["evidence"])
                if arm == "grammar_model":
                    assert all(value["status"] in status_choices for value in report["current_state"].values())
                    assert all(value["label"] in trend_choices for value in report["trend"].values())
                    assert report["forecast"] == source["model_input"]["frozen_forecast"]
                    assert report["confidence"] == source["model_input"]["frozen_confidence"]
            except (ValueError, json.JSONDecodeError, TypeError, KeyError):
                pass
        parsed_counts[arm] = parsed
        exact_counts[arm] = exact
        generation_hashes[arm] = digest(path)

    assert parsed_counts == {"raw_sft": 226, "grammar_model": 234}
    assert exact_counts == {"raw_sft": 192, "grammar_model": 201}
    fresh = pd.DataFrame(recomputed)
    stored = pd.read_csv(output / "confirmation_per_report_metrics.csv")
    keys = ["sample_id", "subject_id", "arm"]
    fresh = fresh.sort_values(keys).reset_index(drop=True)
    stored = stored.sort_values(keys).reset_index(drop=True)
    assert fresh[keys].equals(stored[keys])
    for column in [name for name in fresh if name not in keys]:
        assert np.allclose(fresh[column].astype(float), stored[column].astype(float))

    aggregate = fresh.groupby("arm").mean(numeric_only=True)
    assert aggregate.loc["grammar_model", "format_validity"] == 1
    assert aggregate.loc["grammar_model", "evidence_precision"] == 1
    assert aggregate.loc["grammar_model", "evidence_recall"] == 1
    assert aggregate.loc["grammar_model", "forecast_preservation"] == 1
    assert aggregate.loc["grammar_model", "uncertainty_preservation"] == 1
    assert aggregate.loc["grammar_model", "hallucinated_evidence_rate"] == 0
    assert aggregate.loc["grammar_model", "future_evidence_leakage"] == 0
    assert aggregate.loc["grammar_model", "current_state_consistency"] >= .99
    assert aggregate.loc["grammar_model", "temporal_consistency"] >= .99
    bootstrap = pd.read_csv(output / "subject_cluster_bootstrap.csv")
    assert (bootstrap.ci_low > 0).all()

    result = {
        "status": "passed",
        "dev_examples": len(dev),
        "dev_subjects": len(dev_subjects),
        "new_confirmation_examples": len(confirmation),
        "new_confirmation_subjects": len(confirmation_subjects),
        "raw_valid_json": parsed_counts["raw_sft"],
        "grammar_model_valid_json": parsed_counts["grammar_model"],
        "raw_exact_reports": exact_counts["raw_sft"],
        "grammar_model_exact_reports": exact_counts["grammar_model"],
        "generation_sha256": generation_hashes,
        "original_test_used": False,
        "future_truth_in_prompt": False,
        "selected_arm": selection["selected_arm"],
        "phase4c_structured_decoding_passed": True,
        "cognitive_grpo_ready": True,
        "multimodal_reasoning_ready": False,
    }
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
