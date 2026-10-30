import numpy as np
import pandas as pd
import torch

from memreason_rl.experiments.phase7a_joint_multitask_native_t1 import (
    SubjectBalancedSampler,
    _nearest_future,
    memory_history,
    multitask_loss,
    normalize_diagnostic_probabilities,
)


def test_memory_history_never_reads_after_cutoff():
    months = np.asarray([0.0, 6.0, 12.0, 24.0])
    values = np.asarray([0.1, 0.2, 9.0, 10.0])
    features, latest = memory_history(months, values, 6.0)
    assert latest == 6.0
    assert np.isclose(features[0], 0.2)
    assert np.isclose(features[1], 0.15)


def test_nearest_future_is_strict_and_tolerance_limited():
    months = np.asarray([0.0, 6.0, 12.0, 24.0])
    values = np.asarray([0.1, 0.2, 0.3, 0.4])
    value, month = _nearest_future(months, values, 6.0, 6, 3.0)
    assert value == 0.3 and month == 12.0
    missing, missing_month = _nearest_future(months, values, 6.0, 30, 3.0)
    assert np.isnan(missing) and np.isnan(missing_month)


def test_subject_balanced_sampler_draws_one_scan_per_subject():
    frame = pd.DataFrame({"subject_id": ["a", "a", "b", "b", "b", "c"]})
    sampler = SubjectBalancedSampler(frame, np.arange(len(frame)), 7)
    selected = list(iter(sampler))
    assert len(selected) == 3
    assert frame.iloc[selected].subject_id.nunique() == 3


def test_classification_only_loss_does_not_backpropagate_regression_heads():
    logits = torch.zeros((2, 3), requires_grad=True)
    current = torch.zeros(2, requires_grad=True)
    future = torch.zeros((2, 4), requires_grad=True)
    outputs = {
        "diagnosis_logits": logits,
        "current_mean": current,
        "current_logvar": torch.zeros(2, requires_grad=True),
        "future_mean": future,
        "future_logvar": torch.zeros((2, 4), requires_grad=True),
    }
    batch = {
        "label": torch.tensor([0, 2]), "current": torch.tensor([0.0, 1.0]),
        "future": torch.zeros((2, 4)), "future_mask": torch.ones((2, 4), dtype=torch.bool),
        "weight": torch.ones(2),
    }
    config = {"training": {
        "label_smoothing": 0.0, "diagnosis_weight": 1.0,
        "current_memory_weight": 0.5, "future_memory_weight": 1.0,
    }}
    loss, _ = multitask_loss(outputs, batch, "classification_only", torch.ones(3), config)
    loss.backward()
    assert logits.grad is not None
    assert current.grad is None
    assert future.grad is None


def test_diagnostic_probabilities_are_renormalized_after_serialization():
    frame = pd.DataFrame({
        "prob_CN": [0.333, np.nan],
        "prob_MCI": [0.333, np.nan],
        "prob_AD": [0.333, np.nan],
    })
    normalized = normalize_diagnostic_probabilities(frame)
    assert np.isclose(normalized.loc[0, ["prob_CN", "prob_MCI", "prob_AD"]].sum(), 1.0)
    assert normalized.loc[1].isna().all()
