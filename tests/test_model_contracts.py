import pytest
import torch
from memreason_rl.models.belief_state import BeliefState
from memreason_rl.models.temporal_dynamics import ContinuousTransition
from memreason_rl.models.decoders import ItemDecoders
from memreason_rl.losses.grounding import contrastive_grounding
from memreason_rl.data.dictionary import build_dictionary
from memreason_rl.eval.calibration import gaussian_calibration
from memreason_rl.models.trajectory_generator import TrajectoryGenerator

def test_continuous_time_zero_is_identity_and_negative_rejected():
    transition=ContinuousTransition(8)
    mu,lv=torch.randn(2,8),torch.zeros(2,8)
    m,v=transition(mu,lv,torch.zeros(2))
    torch.testing.assert_close(m,mu)
    torch.testing.assert_close(v,lv)
    with pytest.raises(ValueError):
        transition(mu,lv,torch.tensor([-1.,1.]))

def test_ordinal_thresholds_are_ordered():
    dictionary=build_dictionary(['cdr_cdmemory'])
    decoder=ItemDecoders(['cdr_cdmemory'],dictionary,8)
    logits=decoder(torch.randn(10,8),'cdr_cdmemory')
    assert (logits[:,1:]<=logits[:,:-1]).all()

def test_grounding_does_not_use_same_subject_as_negative():
    with pytest.raises(ValueError):
        contrastive_grounding(torch.randn(2,8),torch.randn(2,8),['a','a'],torch.ones(2))
    loss=contrastive_grounding(torch.randn(3,8),torch.randn(3,8),['a','a','b'],torch.ones(3))
    assert torch.isfinite(loss)

def test_unobserved_targets_do_not_enter_calibration():
    result=gaussian_calibration([0.,100.],[1.,1.],[0.,-100.],[True,False])
    assert result['n']==1 and result['coverage_95']==1.

def test_stochastic_trajectories_share_horizons_but_differ():
    generator=TrajectoryGenerator(ContinuousTransition(8))
    candidates=generator(torch.zeros(2,8),torch.zeros(2,8),[.5,1.,2.],4)
    assert len(candidates)==4 and all(len(c)==3 for c in candidates)
    assert not torch.equal(candidates[0][0][0],candidates[1][0][0])
