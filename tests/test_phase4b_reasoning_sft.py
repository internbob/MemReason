import json

from memreason_rl.experiments.phase4b_reasoning_sft import compact_history, phase4b_input, score_generation, stable_subset


def example():
    evidence = {"event_id": "s:v1:ADAS", "item": "adas_q1", "time_month": 12.0,
                "visit_month": 12.0, "value": 2.0, "domain": "Memory"}
    completion = {
        "current_state": {"Memory": {"status": "elevated_impairment", "severity_z": 1.0, "items": 1}},
        "evidence": [{key: evidence[key] for key in ["event_id", "item", "time_month", "value", "domain"]}],
        "trend": {"Memory": {"label": "decline", "start_severity_z": 0.0,
                              "end_severity_z": 1.0, "timepoints": 2}},
        "forecast": {"adas_total13": {"mean": 10.0, "horizon_months": 12, "change_from_last": "decline"}},
        "confidence": {"adas_total13": {"ensemble_std": 1.0, "normalized_std": .1, "level": "high"}},
    }
    source = {"sample_id": "s:m12:h12", "subject_id": "s", "reasoning_split": "confirmation",
              "prompt": {"evidence_cutoff_time_month": 12.0, "forecast_horizon_months": 12,
                         "verified_history": [{**evidence, "event_id": "s:v0:ADAS", "time_month": 0.0,
                                                "visit_month": 0.0, "value": 0.0}, evidence],
                         "last_observed_targets": {"adas_total13": 8.0}},
              "completion": completion}
    row = {"sample_id": source["sample_id"], "subject_id": "s", "reasoning_split": "confirmation",
           "model_input": phase4b_input(source), "completion": completion}
    return source, row


def test_compact_history_keeps_endpoints():
    source, _ = example()
    compact = compact_history(source["prompt"]["verified_history"])
    assert [row["time_month"] for row in compact] == [0.0, 12.0]


def test_exact_generation_scores_one():
    _, row = example()
    score = score_generation(row, json.dumps(row["completion"]), "sft")
    for key, value in score.items():
        if key in {"sample_id", "subject_id", "arm", "hallucinated_evidence_rate", "future_evidence_leakage", "forbidden_claim"}:
            continue
        assert value == 1.0
    assert score["hallucinated_evidence_rate"] == 0.0
    assert score["future_evidence_leakage"] == 0.0


def test_invalid_generation_fails_schema():
    _, row = example()
    score = score_generation(row, "not json", "base")
    assert score["format_validity"] == 0.0
    assert score["evidence_precision"] == 0.0


def test_stable_subset_is_deterministic_and_grouped():
    _, row = example()
    rows = [{**row, "sample_id": f"s:m{i}:h{h}", "model_input": {**row["model_input"], "horizon": h}}
            for i, h in enumerate([6, 6, 12, 12])]
    left = stable_subset(rows, per_subject_horizon=1, seed=7)
    right = stable_subset(list(reversed(rows)), per_subject_horizon=1, seed=7)
    assert [x["sample_id"] for x in left] == [x["sample_id"] for x in right]
    assert len(left) == 2
