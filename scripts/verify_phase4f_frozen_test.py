#!/usr/bin/env python
"""Independent provenance and metric verification for Phase 4F."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memreason_rl.experiments.phase4a_structured_reasoning import ensemble_predictions
from memreason_rl.experiments.phase4b_reasoning_sft import score_generation


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    config = yaml.safe_load((ROOT / "configs/phase4f_frozen_grud_phase4c_test.yaml").read_text())
    output = ROOT / "outputs" / config["run_id"]
    lock = json.loads((output / "pretest_lock.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    decision = json.loads((output / "final_decision.json").read_text())
    assert lock["lock_created_before_test_construction"] and lock["grpo_adapters_excluded"]
    assert not lock["training_performed"] and not lock["checkpoint_selection_performed"]
    for item in lock["artifacts"].values():
        assert digest(Path(item["path"])) == item["sha256"]
    assert manifest["pretest_lock_written_first"] and manifest["original_test_loaded"]
    assert not manifest["future_truth_in_prompt"] and not manifest["diagnosis_input"]
    assert not manifest["mri_input"] and not manifest["grpo_adapter_used"]
    assert not manifest["training_performed"] and not manifest["checkpoint_selection_performed"]

    examples = read_jsonl(output / "test_reasoning_examples.jsonl")
    subjects = {row["subject_id"] for row in examples}
    assert len(examples) == len(subjects) == manifest["reasoning_subjects"] == 495
    train = set(pd.read_csv(ROOT / "splits/train.csv", usecols=["subject_id"]).subject_id.astype(str))
    val = set(pd.read_csv(ROOT / "splits/val.csv", usecols=["subject_id"]).subject_id.astype(str))
    test = set(pd.read_csv(ROOT / "splits/test.csv", usecols=["subject_id"]).subject_id.astype(str))
    assert subjects <= test and not subjects & (train | val)
    for row in examples:
        assert set(row["model_input"]) == {"cutoff", "horizon", "evidence_trajectories",
                                            "domain_statistics", "last_targets",
                                            "frozen_forecast", "frozen_confidence"}
        assert all(float(item["time_month"]) <= float(row["model_input"]["cutoff"]) + 1e-6
                   for item in row["model_input"]["evidence_trajectories"])
        assert "truth" not in json.dumps(row["model_input"])

    seeds = pd.read_csv(output / "test_seed_forecasts.csv")
    stored = pd.read_csv(output / "test_ensemble_forecasts.csv")
    fresh = ensemble_predictions(seeds)
    keys = ["sample_id", "subject_id", "reasoning_split", "target", "horizon_months", "cutoff_month"]
    fresh = fresh.sort_values(keys).reset_index(drop=True); stored = stored.sort_values(keys).reset_index(drop=True)
    assert len(seeds) == manifest["test_forecast_samples"] * 5 * 3
    assert len(fresh) == len(stored) == manifest["test_forecast_samples"] * 5
    for column in ["prediction", "ensemble_std", "truth", "target_month"]:
        assert np.allclose(fresh[column], stored[column], equal_nan=True)
    available = stored[stored.truth.notna()].copy()
    assert len(available) == manifest["available_forecast_targets"] == 20007
    normalization = json.loads((ROOT / "outputs" / config["phase4a_run_id"] / "normalization.json").read_text())
    available["fresh_nmae"] = (available.prediction - available.truth).abs() / available.target.map(
        lambda target: normalization[target]["std"])
    assert np.allclose(available.fresh_nmae, available.normalized_absolute_error)
    cells = available.groupby(["target", "horizon_months"]).fresh_nmae.mean()
    summary = json.loads((output / "forecast_summary.json").read_text())
    assert np.isclose(cells.mean(), summary["macro_normalized_mae"])
    paired = available[available.last_observed.notna()].copy()
    paired["difference"] = paired.normalized_absolute_error-paired.locf_normalized_absolute_error
    subject_difference = paired.groupby("subject_id").difference.mean().to_numpy()
    assert np.isclose(subject_difference.mean(), summary["paired_grud_minus_locf_subject_mean"])
    correlation = spearmanr(available.normalized_uncertainty, available.normalized_absolute_error)
    assert np.isclose(correlation.statistic, summary["uncertainty_error_spearman"])

    lookup = {row["sample_id"]: row for row in examples}; recomputed = []; hashes = {}
    parsed = {}; exact = {}
    for arm in ["raw_sft", "grammar_model", "grammar_executable", "teacher"]:
        path = output / f"test_generations_{arm}.jsonl"; generated = read_jsonl(path)
        assert len(generated) == len(examples) and {row["sample_id"] for row in generated} == set(lookup)
        local = [score_generation(lookup[row["sample_id"]], row["generation"], arm) for row in generated]
        recomputed.extend(local); parsed[arm] = int(sum(row["format_validity"] for row in local))
        exact[arm] = int(sum(row["exact_report_match"] for row in local)); hashes[arm] = digest(path)
    fresh_scores = pd.DataFrame(recomputed).sort_values(["arm", "sample_id"]).reset_index(drop=True)
    stored_scores = pd.read_csv(output / "reasoning_per_report_metrics.csv").sort_values(["arm", "sample_id"]).reset_index(drop=True)
    assert fresh_scores[["sample_id", "subject_id", "arm"]].equals(stored_scores[["sample_id", "subject_id", "arm"]])
    for column in [name for name in fresh_scores if name not in {"sample_id", "subject_id", "arm"}]:
        assert np.allclose(fresh_scores[column].astype(float), stored_scores[column].astype(float))
    aggregate = fresh_scores.groupby("arm").mean(numeric_only=True)
    grammar = aggregate.loc["grammar_model"]
    assert grammar.format_validity == grammar.evidence_precision == grammar.evidence_recall == 1
    assert grammar.forecast_preservation == grammar.uncertainty_preservation == 1
    assert grammar.hallucinated_evidence_rate == grammar.future_evidence_leakage == 0
    mismatch = pd.read_csv(output / "grammar_label_mismatch_cases.csv")
    state_mismatch = mismatch[mismatch.field.eq("current_state")]
    trend_mismatch = mismatch[mismatch.field.eq("trend")]
    state_labels = len(examples) * 5
    trend_labels = len(examples) * 5
    assert len(state_mismatch) == 36 and state_mismatch.sample_id.nunique() == 32
    assert len(trend_mismatch) == 100 and trend_mismatch.sample_id.nunique() == 67
    # nested_score averages three fields per state domain and four per trend domain.
    expected_state_score = 1 - len(state_mismatch) / (state_labels * 3)
    expected_temporal_score = 1 - len(trend_mismatch) / (trend_labels * 4)
    assert np.isclose(grammar.current_state_consistency, expected_state_score)
    assert np.isclose(grammar.temporal_consistency, expected_temporal_score)
    bootstrap = pd.read_csv(output / "reasoning_subject_bootstrap.csv").set_index("metric")
    assert bootstrap.loc["format_validity", "ci_low"] > 0
    failed = {key for key, value in decision["criteria"].items() if not value}
    assert failed == {"temporal_consistency"}
    assert not decision["phase4f_frozen_system_confirmed"]
    assert not decision["paper_primary_method_ready"] and not decision["grpo_supported"]
    assert not decision["multimodal_reasoning_supported"] and not decision["further_test_tuning_permitted"]
    assert not (output / "checkpoints").exists()

    result = {
        "status": "passed", "forecast_samples": manifest["test_forecast_samples"],
        "available_forecast_targets": manifest["available_forecast_targets"],
        "forecast_subjects": manifest["test_forecast_subjects"],
        "reasoning_reports": len(examples), "reasoning_subjects": len(subjects),
        "macro_normalized_mae": summary["macro_normalized_mae"],
        "grud_minus_locf_subject_mean": summary["paired_grud_minus_locf_subject_mean"],
        "uncertainty_error_spearman": summary["uncertainty_error_spearman"],
        "valid_json": parsed, "exact_reports": exact, "generation_sha256": hashes,
        "grammar_current_state_label_mismatches": int(len(state_mismatch)),
        "grammar_current_state_reports_affected": int(state_mismatch.sample_id.nunique()),
        "grammar_trend_label_mismatches": int(len(trend_mismatch)),
        "grammar_trend_reports_affected": int(trend_mismatch.sample_id.nunique()),
        "grammar_domain_labels_per_field": int(trend_labels),
        "current_state_metric_definition": "mean nested agreement over 3 fields x 5 domains per report",
        "temporal_metric_definition": "mean nested agreement over 4 fields x 5 domains per report",
        "grammar_temporal_consistency": float(grammar.temporal_consistency),
        "failed_criteria": sorted(failed),
        "phase4f_frozen_system_confirmed": False, "paper_primary_method_ready": False,
        "training_performed": False, "grpo_adapter_used": False,
        "further_test_tuning_permitted": False,
    }
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
