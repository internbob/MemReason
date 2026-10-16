import numpy as np
import torch

from memreason_rl.experiments.phase5d_anchored_mirt import (
    AnchoredBifactorIRT,
    assign_time_bins,
    concordance_correlation,
    response_representation,
)


def _items():
    return [
        {"field": "binary", "domain": "Memory", "value_type": "binary",
         "direction": "higher=better", "valid_values": [0, 1], "range": None},
        {"field": "continuous", "domain": "Language", "value_type": "continuous",
         "direction": "higher=worse", "valid_values": None, "range": [0, 10]},
        {"field": "ordinal", "domain": "Memory", "value_type": "ordinal",
         "direction": "higher=worse", "valid_values": [0, .5, 1, 2, 3], "range": None},
    ]


def test_response_representation_orients_higher_as_worse():
    raw = np.asarray([[1, 0, 0], [0, 10, 3]], float)
    fraction, category, response_type = response_representation(raw, _items())
    assert np.allclose(fraction[0], [0, 0, 0])
    assert np.allclose(fraction[1], [1, 1, 1])
    assert category[:, 2].tolist() == [0, 4]
    assert response_type.tolist() == [0, 1, 2]


def test_identification_constraints_and_ordinal_probabilities():
    model = AnchoredBifactorIRT(_items(), np.asarray([0, 1, 2]),
                               np.asarray([False, True, False]), 3.0)
    global_loading, memory_loading = model.discriminations()
    assert torch.all(global_loading > 0)
    assert memory_loading[0] > 0 and memory_loading[1] == 0 and memory_loading[2] > 0
    assert torch.allclose(model.dif_parameters()[:, 0], torch.zeros(3))
    probabilities = model.ordinal_probabilities(torch.zeros(4, 3), 2)
    assert probabilities.shape == (4, 5)
    assert torch.allclose(probabilities.sum(1), torch.ones(4), atol=1e-6)


def test_time_bins_and_concordance_are_deterministic():
    assert assign_time_bins(np.asarray([0, 6, 12, 18, 24, 36])).tolist() == [0, 1, 1, 2, 2, 3]
    x = np.asarray([-1.0, 0.0, 1.0])
    assert np.isclose(concordance_correlation(x, x), 1.0)
    assert concordance_correlation(x, -x) < 0
