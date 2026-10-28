import numpy as np
import pandas as pd

from memreason_rl.experiments.phase3g_mri_diagnosis import (
    expected_calibration_error,
    multiclass_brier,
    shuffled_indices,
    subject_weights,
)


def test_subject_weights_give_each_subject_equal_total_mass():
    frame = pd.DataFrame({"subject_id": ["a", "a", "b"]})
    weights = subject_weights(frame)
    assert np.isclose(weights[:2].sum(), weights[2:].sum())


def test_perfect_probabilities_have_zero_brier_and_ece():
    labels = np.asarray([0, 1, 2])
    probabilities = np.eye(3)
    weights = np.ones(3)
    assert multiclass_brier(labels, probabilities, weights) == 0.0
    assert expected_calibration_error(labels, probabilities, weights) == 0.0


def test_shuffled_indices_never_retain_subject_identity():
    frame = pd.DataFrame({"subject_id": ["a", "a", "b", "b", "c", "c"]})
    indices = np.arange(len(frame))
    shuffled = shuffled_indices(frame, indices, seed=7)
    assert np.all(frame.iloc[indices].subject_id.to_numpy() != frame.iloc[shuffled].subject_id.to_numpy())
