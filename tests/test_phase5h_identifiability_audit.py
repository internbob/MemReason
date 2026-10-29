import math

from memreason_rl.experiments.phase5h_identifiability_audit import spearman_brown


def test_spearman_brown_known_value():
    assert math.isclose(spearman_brown(0.5), 2 / 3)


def test_spearman_brown_rejects_minus_one():
    assert math.isnan(spearman_brown(-1.0))
