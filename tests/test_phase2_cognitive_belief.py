import numpy as np
import pandas as pd
import torch

from memreason_rl.data.common import ROOT
from memreason_rl.data.dictionary import load_dictionary
from memreason_rl.experiments.phase1b_semantic import select_verified_scalar_features
from memreason_rl.experiments.phase1c_joint_stage1 import load_config
from memreason_rl.experiments.phase1d_typed_representation import feature_metadata
from memreason_rl.experiments.phase2_cognitive_belief import (
    DynamicsCollator,
    DirectGaussianDynamics,
    TypedBeliefDynamics,
    apply_temperatures,
    build_domain_assignment,
    build_interpolation_samples,
    fit_temperatures,
    partition_frame,
)


def test_subject_partitions_are_deterministic_and_disjoint():
    frame = pd.DataFrame(
        {"subject_id": [f"S{i:03d}" for i in range(100) for _ in range(2)], "value": range(200)}
    )
    retained, heldout = partition_frame(frame, 2026, 0.25)
    retained_again, heldout_again = partition_frame(frame, 2026, 0.25)
    assert set(retained.subject_id).isdisjoint(set(heldout.subject_id))
    assert set(retained.subject_id) == set(retained_again.subject_id)
    assert set(heldout.subject_id) == set(heldout_again.subject_id)


def test_interpolation_removes_complete_middle_visit():
    features = ["item_a", "item_b"]
    targets = ["score"]
    frame = pd.DataFrame(
        {
            "subject_id": ["001_S_0001"] * 3,
            "visit_month": [0.0, 12.0, 24.0],
            "item_a": [0.0, 1.0, 2.0],
            "item_b": [1.0, 2.0, 3.0],
            "score": [10.0, 11.0, 12.0],
        }
    )
    samples = build_interpolation_samples(frame, "unit", features, targets, 3)
    assert len(samples) == 1
    sample = samples[0]
    assert sample.query_month == 12.0
    assert sample.months.tolist() == [0.0, 24.0]
    assert sample.targets.tolist() == [11.0]
    assert not np.any(np.isclose(sample.months, sample.query_month))


def test_probabilistic_arms_return_finite_positive_variance():
    config = load_config(ROOT / "configs/phase2_cognitive_belief_dev.yaml")
    dictionary = load_dictionary()
    features = select_verified_scalar_features(config, dictionary)
    feature_types, _, _, _, _ = feature_metadata(features, dictionary)
    domains = build_domain_assignment(features, dictionary)
    batch_size, steps = 2, 3
    mask = torch.ones(batch_size, steps, len(features), dtype=torch.bool)
    batch = {
        "values": torch.randn(batch_size, steps, len(features)),
        "mask": mask,
        "severity": torch.rand(batch_size, steps, len(features)),
        "times": torch.tensor([[0.0, 1.0, 2.0], [0.0, 1.5, 3.0]]),
        "left_mask": torch.tensor([[True, True, False], [True, True, True]]),
        "right_mask": torch.tensor([[False, False, True], [False, False, False]]),
        "query_time": torch.tensor([1.5, 4.0]),
        "task_id": torch.tensor([1, 0]),
    }
    models = [
        DirectGaussianDynamics(len(features), 5, feature_types, domains, True, config),
        TypedBeliefDynamics(len(features), 5, feature_types, domains, config),
    ]
    for model in models:
        mean, logvar = model(batch)
        assert mean.shape == (batch_size, 5)
        assert torch.isfinite(mean).all()
        assert torch.isfinite(logvar).all()
        assert (logvar.exp() > 0).all()
        (mean.mean() + logvar.mean()).backward()


def test_calibration_temperature_is_fit_on_named_model_and_applied():
    frame = pd.DataFrame(
        {
            "model": ["belief"] * 4,
            "task": ["forecast"] * 4,
            "target": ["score"] * 4,
            "horizon_months": [12] * 4,
            "truth_norm": [-2.0, -1.0, 1.0, 2.0],
            "mean_norm": [0.0] * 4,
            "variance_norm": [1.0] * 4,
            "std": [1.0] * 4,
        }
    )
    config = {
        "calibration": {"minimum_group_targets": 2, "temperature_clip": [0.25, 4.0]}
    }
    temperatures = fit_temperatures(frame, config)
    applied = apply_temperatures(frame, temperatures)
    expected = np.sqrt(np.mean(np.square(frame.truth_norm)))
    assert np.allclose(applied.temperature, expected)
    assert np.allclose(applied.calibrated_variance_norm, expected**2)
