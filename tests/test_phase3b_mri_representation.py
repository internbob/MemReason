from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from memreason_rl.experiments.phase3b_mri_representation import (
    ConcurrentRegressor,
    ensemble,
    macro_huber,
    make_shuffle_indices,
)


def test_subject_shuffle_always_changes_subject() -> None:
    frame = pd.DataFrame(
        {"subject_id": ["A", "A", "B", "C"], "scan_id": ["A0", "A1", "B0", "C0"]}
    )
    shuffled = make_shuffle_indices(frame, 7)
    subjects = frame.subject_id.to_numpy()
    assert np.all(subjects[shuffled] != subjects)


def test_all_concurrent_models_have_finite_forward() -> None:
    demo = torch.randn(2, 2)
    inputs = {
        "demographics_only": torch.zeros(2, 1),
        "global_mri": torch.randn(2, 10),
        "sfcn_3d": torch.randn(2, 1, 48, 56, 48),
    }
    for representation, image in inputs.items():
        model = ConcurrentRegressor(representation, global_dim=10, n_targets=5, dropout=0.0)
        prediction, embedding = model(demo, image)
        assert prediction.shape == (2, 5)
        assert torch.isfinite(prediction).all()
        assert torch.isfinite(embedding).all()


def test_macro_huber_ignores_missing_targets() -> None:
    prediction = torch.tensor([[1.0, 999.0], [3.0, 999.0]])
    target = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    mask = torch.tensor([[True, False], [True, False]])
    assert torch.isclose(macro_huber(prediction, target, mask), torch.tensor(0.5))


def test_delta_ensemble_accepts_future_columns() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": ["x", "x"],
            "scan_id": ["s", "s"],
            "subject_id": ["p", "p"],
            "split": ["val", "val"],
            "model": ["delta_sfcn_3d", "delta_sfcn_3d"],
            "seed": [2026, 2027],
            "pairing": ["observed", "observed"],
            "target": ["adas_total13", "adas_total13"],
            "horizon_months": [12, 12],
            "current": [10.0, 10.0],
            "truth": [2.0, 2.0],
            "future_value": [12.0, 12.0],
            "prediction": [1.0, 3.0],
        }
    )
    result = ensemble(frame)
    assert len(result) == 1
    assert result.prediction.iloc[0] == 2.0


def test_phase3b_source_does_not_load_test_split() -> None:
    source = (
        Path(__file__).parents[1]
        / "memreason_rl"
        / "experiments"
        / "phase3b_mri_representation.py"
    ).read_text()
    assert "splits/test.csv" not in source
    assert 'splits/test' not in source
