import torch

from memreason_rl.experiments.phase6a_latent_state_dynamics import analytic_transition, precision_update


def test_precision_update_reduces_variance_and_averages_equal_evidence():
    mean, variance = precision_update(torch.tensor([0.0]), torch.tensor([1.0]),
                                      torch.tensor([2.0]), torch.tensor([1.0]))
    assert torch.allclose(mean, torch.tensor([1.0]))
    assert torch.allclose(variance, torch.tensor([0.5]))


def test_analytic_transition_zero_gap_is_identity():
    mean, variance = analytic_transition(torch.tensor([1.2]), torch.tensor([0.4]), torch.tensor([0.0]),
                                         torch.tensor(0.3), torch.tensor(0.1), torch.tensor(0.2), torch.tensor(0.5))
    assert torch.allclose(mean, torch.tensor([1.2]))
    assert torch.allclose(variance, torch.tensor([0.4]))


def test_analytic_transition_accumulates_process_uncertainty():
    _, variance = analytic_transition(torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([1.0]),
                                      torch.tensor(0.2), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.5))
    assert variance.item() > 0
