import numpy as np
import torch

from memreason_rl.experiments.phase6a_latent_state_dynamics import LatentSample
from memreason_rl.experiments.phase6c_longitudinal_mri_identifiability import (
    dynamic_features,
    feature_mask,
    flip_dynamics,
    subject_shuffle_dynamics,
)


def sample(subject: str, cutoff: float, dynamic: tuple[float, ...], horizon: int = 12) -> LatentSample:
    return LatentSample(
        sample_id=f"{subject}:m{cutoff:g}:h{horizon}", subject_id=subject, fold=0,
        horizon_months=horizon, history_mean=np.asarray([0.1], np.float32),
        history_var=np.asarray([0.2], np.float32),
        history_time=np.asarray([cutoff / 12], np.float32), target_mean=0.2,
        target_var=0.3, target_time=cutoff / 12 + 1, target_composite=0.4,
        mri_features=np.asarray([1, 2, 0.2, 1.1, 0.7, *dynamic], np.float32),
        mri_available=True,
    )


def config():
    return {
        "mri": {
            "features": list("abcdefghij"),
            "static_feature_indices": [0, 1, 2, 3, 4],
            "dynamic_feature_indices": [5, 6, 7, 8, 9],
        }
    }


def test_dynamic_features_use_actual_time_and_recent_interval():
    result = dynamic_features(
        np.asarray([0.0, 1.0, 3.0]), np.asarray([0.0, 1.0, 5.0]),
        np.asarray([0.2, 0.4, 0.5]),
    )
    assert np.allclose(result[[0, 2, 3, 4]], [5.0, 4.0, 2.0, 0.3])
    assert np.isclose(result[1], np.polyfit([0, 1, 3], [0, 1, 5], 1)[0])


def test_static_mask_excludes_all_longitudinal_features():
    assert torch.equal(feature_mask("static_mri", config()),
                       torch.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=torch.float32))
    assert not feature_mask("cognition_only", config()).any()


def test_sign_intervention_changes_only_normalized_dynamics():
    original = sample("A", 12, (1, -2, 3, -4, 5))
    flipped = flip_dynamics([original], [5, 6, 7, 8, 9])[0]
    assert np.array_equal(flipped.mri_features[:5], original.mri_features[:5])
    assert np.array_equal(flipped.mri_features[5:], -original.mri_features[5:])


def test_subject_shuffle_preserves_static_and_is_horizon_consistent():
    samples = [
        sample("A", 12, (1, 2, 3, 4, 5), 12),
        sample("A", 12, (1, 2, 3, 4, 5), 24),
        sample("B", 12, (6, 7, 8, 9, 10), 12),
    ]
    metadata = {
        value.sample_id: {"mri_scan_count": 2, "mri_span_years": 0.75}
        for value in samples
    }
    shuffled, changed, donors = subject_shuffle_dynamics(samples, metadata, [5, 6, 7, 8, 9], 9)
    assert changed == 1.0 and donors == 1.0
    for before, after in zip(samples, shuffled):
        assert np.array_equal(before.mri_features[:5], after.mri_features[:5])
    assert np.array_equal(shuffled[0].mri_features[5:], shuffled[1].mri_features[5:])
    assert np.array_equal(shuffled[0].mri_features[5:], samples[2].mri_features[5:])
