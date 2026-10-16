import numpy as np

from memreason_rl.experiments.phase5c_memory_measurement import (
    deterministic_targets,
    subject_fold,
    to_severity,
)


def test_subject_fold_is_stable_and_bounded():
    assert subject_fold("001_S_0001", 2026, 5) == subject_fold("001_S_0001", 2026, 5)
    assert 0 <= subject_fold("001_S_0001", 2026, 5) < 5


def test_item_and_instrument_masks_never_reveal_targets():
    mask = np.array([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=bool)
    instruments = np.array(["A", "A", "B", "B"])
    for task in ["heldout_item", "heldout_instrument"]:
        inputs, targets = deterministic_targets(mask, instruments, ["x", "y"], task, .2)
        assert not np.any(inputs & targets)
        assert np.all((inputs | targets) <= mask)
        assert inputs[targets.any(1)].any(1).all()


def test_severity_direction_is_consistent():
    raw = np.array([[0.0, 0.0], [1.0, 1.0]])
    scaling = {"mean": np.array([.5, .5]), "std": np.array([.5, .5]),
               "direction": np.array([1.0, -1.0])}
    z = to_severity(raw, scaling)
    assert z[1, 0] > z[0, 0]
    assert z[1, 1] < z[0, 1]
