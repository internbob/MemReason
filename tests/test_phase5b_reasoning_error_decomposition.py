import numpy as np

from memreason_rl.experiments.phase5b_reasoning_error_decomposition import (
    expected_state,
    expected_trend,
    margin_band,
    reconstruct_decisions,
)


def test_frozen_label_boundaries_and_margin_bands():
    assert expected_state(-0.51) == "lower_impairment"
    assert expected_state(-0.50) == "within_training_reference"
    assert expected_state(0.50) == "within_training_reference"
    assert expected_state(0.51) == "elevated_impairment"
    assert expected_trend(0.0, 0.26, 2) == "decline"
    assert expected_trend(0.0, -0.26, 2) == "improve"
    assert expected_trend(0.0, 0.25, 2) == "stable"
    assert expected_trend(0.0, 1.0, 1) == "insufficient_history"
    assert margin_band(0.049) == "[0,0.05)"
    assert margin_band(0.05) == "[0.05,0.10)"
    assert margin_band(0.10) == "[0.10,0.25)"
    assert margin_band(0.25) == ">=0.25"
    assert margin_band(np.nan) == "unavailable_or_insufficient"


def test_reconstructs_one_state_and_one_trend_decision_per_domain():
    stats = {"Memory": {"current_severity_z": 0.6, "items": 2,
                        "start_severity_z": 0.0, "end_severity_z": 0.3, "timepoints": 2}}
    completion = {
        "current_state": {"Memory": {"status": "elevated_impairment"}},
        "trend": {"Memory": {"label": "decline"}},
    }
    example = {"sample_id": "s:h6", "subject_id": "s", "model_input": {
        "horizon": 6, "domain_statistics": stats,
        "last_targets": {"a": None, "b": 1.0}}, "completion": completion}
    generated_report = {
        "current_state": {"Memory": {"status": "within_training_reference"}},
        "trend": {"Memory": {"label": "decline"}},
    }
    generation = {"sample_id": "s:h6", "generation": __import__("json").dumps(generated_report)}
    frame = reconstruct_decisions([example], [generation])
    assert len(frame) == 2
    assert frame.set_index("field").loc["current_state", "mismatch"] == 1
    assert frame.set_index("field").loc["trend", "mismatch"] == 0
    assert np.isclose(frame.set_index("field").loc["current_state", "threshold_margin"], 0.1)
    assert np.isclose(frame.set_index("field").loc["trend", "threshold_margin"], 0.05)
