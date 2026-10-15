from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from memreason_rl.experiments.phase1_numeric import ForecastSample
from memreason_rl.experiments.phase3a_global_mri_anchor import (
    AnchorSample,
    MRIAnchorForecaster,
    construct_arm,
    shuffle_maps,
)


def _base(subject: str, cutoff: float = 0.0) -> ForecastSample:
    return ForecastSample(
        sample_id=f"{subject}:m{cutoff:g}:h12",
        subject_id=subject,
        split="train",
        cutoff_month=cutoff,
        horizon_months=12,
        months=np.asarray([cutoff], dtype=np.float32),
        features=np.asarray([[1.0]], dtype=np.float32),
        feature_mask=np.asarray([[True]]),
        history_targets=np.asarray([[1.0]], dtype=np.float32),
        history_target_mask=np.asarray([[True]]),
        targets=np.asarray([2.0], dtype=np.float32),
        target_mask=np.asarray([True]),
        target_months=np.asarray([cutoff + 12], dtype=np.float32),
    )


def _anchor(subject: str, cutoff: float, scan: str) -> AnchorSample:
    return AnchorSample(
        base=_base(subject, cutoff),
        anchor_key=f"{subject}:m{cutoff:g}",
        scan_id=scan,
        source_scan_id=scan,
        mri_features=np.empty(0, dtype=np.float32),
        mri_available=1.0,
    )


def test_shuffle_controls_corrupt_subject_and_available_time() -> None:
    samples = [
        _anchor("A", 0, "A0"),
        _anchor("A", 12, "A1"),
        _anchor("B", 0, "B0"),
    ]
    subject_map, time_map, diagnostics = shuffle_maps(samples, 7)
    assert all(subject_map[s.anchor_key][0] != s.base.subject_id for s in samples)
    assert time_map["A:m0"] == "A1"
    assert time_map["A:m12"] == "A0"
    assert time_map["B:m0"] == "B0"
    assert diagnostics["shuffled_subject_changed_fraction"] == 1.0


def test_construct_arm_uses_identical_samples_and_disables_mri() -> None:
    samples = [_anchor("A", 0, "A0"), _anchor("B", 0, "B0")]
    raw = {
        "A0": np.asarray([1.0, 2.0], dtype=np.float32),
        "B0": np.asarray([3.0, 4.0], dtype=np.float32),
    }
    mean = np.asarray([2.0, 3.0], dtype=np.float32)
    std = np.ones(2, dtype=np.float32)
    subject_map = {"A:m0": "B0", "B:m0": "A0"}
    time_map = {"A:m0": "A0", "B:m0": "B0"}
    cognition = construct_arm(samples, "cognition_only", raw, mean, std, subject_map, time_map)
    observed = construct_arm(samples, "observed_mri", raw, mean, std, subject_map, time_map)
    assert [s.base.sample_id for s in cognition] == [s.base.sample_id for s in observed]
    assert all(s.mri_available == 0 for s in cognition)
    assert all(np.array_equal(s.mri_features, np.zeros(2)) for s in cognition)
    assert np.array_equal(observed[0].mri_features, np.asarray([-1.0, -1.0]))


def test_forecaster_is_finite_with_and_without_mri() -> None:
    model = MRIAnchorForecaster(2, 4, 1, hidden=8, layers=1, dropout=0.0)
    batch = {
        "values": torch.zeros(2, 2, 2),
        "mask": torch.ones(2, 2, 2, dtype=torch.bool),
        "times": torch.tensor([[0.0, 1.0], [0.0, 2.0]]),
        "event_mask": torch.ones(2, 2, dtype=torch.bool),
        "horizon": torch.ones(2),
        "mri_features": torch.randn(2, 4),
        "mri_available": torch.tensor([[0.0], [1.0]]),
    }
    output = model(batch)
    assert output.shape == (2, 1)
    assert torch.isfinite(output).all()


def test_phase3a_source_never_loads_test_split() -> None:
    source = (
        Path(__file__).parents[1]
        / "memreason_rl"
        / "experiments"
        / "phase3a_global_mri_anchor.py"
    ).read_text()
    assert "splits/test.csv" not in source
    assert 'splits/test' not in source
