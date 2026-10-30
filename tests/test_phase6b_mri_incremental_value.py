import numpy as np
import torch

from memreason_rl.experiments.phase6a_latent_state_dynamics import LatentSample
from memreason_rl.experiments.phase6b_mri_incremental_value import (
    IncrementalValueDynamics,
    clinical_distribution,
    feature_mask,
    subject_shuffle_structural,
    time_shuffle_structural,
)


def sample(subject: str, cutoff: float, structural: tuple[float, float, float]) -> LatentSample:
    return LatentSample(
        sample_id=f"{subject}:m{cutoff}:h12", subject_id=subject, fold=0, horizon_months=12,
        history_mean=np.asarray([0.1], np.float32), history_var=np.asarray([0.2], np.float32),
        history_time=np.asarray([cutoff], np.float32), target_mean=0.2, target_var=0.3,
        target_time=cutoff + 1, target_composite=0.4,
        mri_features=np.asarray([*structural, 0.25, 1.1], np.float32), mri_available=True,
    )


def config():
    return {"mri": {"features": ["a", "b", "c", "recency", "count"],
                    "metadata_feature_indices": [3, 4]}}


def test_metadata_feature_mask_excludes_structural_content():
    mask = feature_mask("metadata", config())
    assert torch.equal(mask, torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0]))


def test_subject_shuffle_changes_structure_but_preserves_metadata_and_identity():
    values = [sample("A", 0.0, (1, 2, 3)), sample("B", 0.0, (4, 5, 6))]
    shuffled, changed = subject_shuffle_structural(values, [0, 1, 2], 7)
    assert changed == 1.0
    assert [value.subject_id for value in shuffled] == ["A", "B"]
    assert np.allclose(shuffled[0].mri_features[3:], values[0].mri_features[3:])
    assert np.allclose(shuffled[0].mri_features[:3], values[1].mri_features[:3])


def test_time_shuffle_is_consistent_across_horizons_and_preserves_metadata():
    first = sample("A", 0.0, (1, 2, 3))
    duplicate = sample("A", 0.0, (1, 2, 3)); duplicate.sample_id = "A:m0:h24"
    second = sample("A", 1.0, (4, 5, 6))
    shuffled, changed = time_shuffle_structural([first, duplicate, second], [0, 1, 2], 9)
    assert changed == 1.0
    assert np.allclose(shuffled[0].mri_features[:3], shuffled[1].mri_features[:3])
    assert np.allclose(shuffled[0].mri_features[3:], first.mri_features[3:])


def test_clinical_distribution_adds_decoder_residual_variance():
    mean, variance = clinical_distribution(torch.tensor([2.0]), torch.tensor([0.5]), (0.1, 0.2, 0.03))
    assert torch.allclose(mean, torch.tensor([0.5]))
    assert torch.allclose(variance, torch.tensor([0.05]))


def test_mri_only_forward_is_finite_without_using_cognitive_means():
    model = IncrementalValueDynamics(False, True, torch.ones(5), 1e-4)
    batch = {
        "history_mean": torch.tensor([[100.0]]), "history_var": torch.tensor([[0.2]]),
        "history_time": torch.tensor([[0.0]]), "history_mask": torch.tensor([[True]]),
        "target_time": torch.tensor([1.0]), "mri": torch.zeros((1, 5)),
        "mri_available": torch.tensor([True]),
    }
    mean, variance = model(batch, (0.0, 0.0))
    assert torch.isfinite(mean).all() and torch.isfinite(variance).all()
    batch["history_mean"] = torch.tensor([[-100.0]])
    mean_changed, _ = model(batch, (0.0, 0.0))
    assert torch.allclose(mean, mean_changed)
