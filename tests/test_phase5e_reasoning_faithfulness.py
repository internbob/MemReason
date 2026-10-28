from dataclasses import replace

import numpy as np

from memreason_rl.experiments.phase1_numeric import ForecastSample
from memreason_rl.experiments.phase5e_reasoning_faithfulness import (
    drop_visit,
    reflected_memory_sample,
    reversed_time_sample,
    transform_events,
)


def sample() -> ForecastSample:
    return ForecastSample(
        sample_id="s:m24:h12", subject_id="s", split="test", cutoff_month=24.0,
        horizon_months=12, months=np.asarray([0.0, 6.0, 24.0]),
        features=np.asarray([[0.0, 2.0], [1.0, 3.0], [0.0, 4.0]], dtype=np.float32),
        feature_mask=np.ones((3, 2), dtype=bool),
        history_targets=np.zeros((3, 1), dtype=np.float32),
        history_target_mask=np.ones((3, 1), dtype=bool),
        targets=np.zeros(1, dtype=np.float32), target_mask=np.ones(1, dtype=bool),
        target_months=np.asarray([36.0], dtype=np.float32),
    )


def test_drop_visit_removes_the_complete_time_step():
    result = drop_visit(sample(), 1)
    assert result.months.tolist() == [0.0, 24.0]
    assert result.features.shape == (2, 2)
    assert result.history_targets.shape == (2, 1)


def test_reverse_gaps_keeps_endpoints_and_reverses_intervals():
    result, mapping = reversed_time_sample(sample())
    assert result.months.tolist() == [0.0, 18.0, 24.0]
    assert mapping == {0.0: 0.0, 6.0: 18.0, 24.0: 24.0}


def test_memory_reflection_changes_only_memory_features_and_events():
    dictionary = {"memory": {"domain": "Memory"}, "other": {"domain": "Attention"}}
    normalization = {
        "memory": {"minimum": 0.0, "maximum": 1.0},
        "other": {"minimum": 2.0, "maximum": 4.0},
    }
    result = reflected_memory_sample(sample(), ["memory", "other"], dictionary, normalization)
    assert result.features[:, 0].tolist() == [1.0, 0.0, 1.0]
    assert result.features[:, 1].tolist() == [2.0, 3.0, 4.0]
    events = [{
        "event_id": "e", "visit_month": 6.0, "time_month": 6.2,
        "payload": {"memory": 1.0, "other": 3.0},
    }]
    reflected = transform_events(events, "memory_score_reflection", dictionary, normalization)
    assert reflected[0]["payload"] == {"memory": 0.0, "other": 3.0}
    assert events[0]["payload"]["memory"] == 1.0
