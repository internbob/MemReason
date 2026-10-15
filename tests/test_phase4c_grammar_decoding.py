import json

from memreason_rl.experiments.phase4c_grammar_decoding import (
    executable_status, executable_trend, latest_evidence, render_report,
)


def row():
    return {
        "sample_id": "s:m12:h12", "subject_id": "s", "reasoning_split": "dev",
        "model_input": {
            "cutoff": 12.0, "horizon": 12,
            "evidence_trajectories": [
                {"event_id": "e0", "item": "x", "time_month": 0.0, "visit_month": 0.0,
                 "value": 0.0, "domain": "Memory"},
                {"event_id": "e1", "item": "x", "time_month": 12.0, "visit_month": 12.0,
                 "value": 1.0, "domain": "Memory"},
            ],
            "domain_statistics": {
                domain: {"current_severity_z": 1.0 if domain == "Memory" else 0.0,
                         "items": 1 if domain == "Memory" else 0,
                         "start_severity_z": 0.0, "end_severity_z": 1.0 if domain == "Memory" else 0.0,
                         "timepoints": 2 if domain == "Memory" else 0}
                for domain in ["Memory", "Executive", "Attention", "Language", "Orientation"]
            },
            "last_targets": {},
            "frozen_forecast": {"adas_total13": {"mean": 2.0, "horizon_months": 12,
                                                    "change_from_last": "stable"}},
            "frozen_confidence": {"adas_total13": {"ensemble_std": .1, "normalized_std": .01,
                                                      "level": "high"}},
        },
    }


def test_executable_enums():
    assert executable_status({"current_severity_z": .6}) == "elevated_impairment"
    assert executable_status({"current_severity_z": -.6}) == "lower_impairment"
    assert executable_status({"current_severity_z": 0.0}) == "within_training_reference"
    assert executable_trend({"timepoints": 1, "start_severity_z": 0, "end_severity_z": 2}) == "insufficient_history"
    assert executable_trend({"timepoints": 2, "start_severity_z": 0, "end_severity_z": .3}) == "decline"


def test_latest_evidence_and_renderer_are_valid_json():
    sample = row()
    evidence = latest_evidence(sample["model_input"])
    assert len(evidence) == 1 and evidence[0]["event_id"] == "e1"
    status = {domain: executable_status(stats) for domain, stats in sample["model_input"]["domain_statistics"].items()}
    trend = {domain: executable_trend(stats) for domain, stats in sample["model_input"]["domain_statistics"].items()}
    report = render_report(sample, status, trend)
    assert set(report) == {"current_state", "evidence", "trend", "forecast", "confidence"}
    json.loads(json.dumps(report, allow_nan=False))
