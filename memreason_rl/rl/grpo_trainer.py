import torch
from torch import nn

def grpo_loss(new_logp,old_logp,reference_logp,rewards,completion_mask,clip=.2,beta=.1):
    """Shapes [histories, group, tokens], rewards [histories, group].

    Mask applies only to completion tokens; prompt tokens are never optimized.
    Equal-reward groups produce zero policy advantage (KL remains active).
    """
    if rewards.shape[-1]<2:
        raise ValueError('GRPO requires at least two candidates per history')
    if new_logp.shape!=completion_mask.shape or new_logp.shape[:2]!=rewards.shape:
        raise ValueError('Mismatched GRPO tensor shapes')
    if not torch.isfinite(rewards).all():
        raise ValueError('Missing total rewards must be filtered before GRPO')
    advantages=(rewards-rewards.mean(-1,keepdim=True))/rewards.std(-1,keepdim=True,unbiased=False).clamp_min(1e-6)
    ratio=(new_logp-old_logp.detach()).clamp(-20,20).exp()
    advantage=advantages.detach().unsqueeze(-1)
    policy=torch.minimum(ratio*advantage,ratio.clamp(1-clip,1+clip)*advantage)
    log_ratio=(reference_logp.detach()-new_logp).clamp(-20,20)
    kl=log_ratio.exp()-log_ratio-1
    valid=completion_mask.to(new_logp.dtype)
    per_candidate=((beta*kl-policy)*valid).sum(-1)/valid.sum(-1).clamp_min(1)
    active=valid.sum(-1)>0
    if not active.any():
        raise ValueError('No completion tokens')
    return per_candidate[active].mean()

class GRPOTrainer:
    def __init__(self,policy,reference,optimizer,frozen_backbones=(),beta=.1):
        self.policy,self.reference,self.optimizer,self.beta=policy,reference,optimizer,beta
        reference.requires_grad_(False)
        reference.eval()
        for module in frozen_backbones:
            module.requires_grad_(False)
        frozen_ids={id(p) for module in frozen_backbones for p in module.parameters()}
        if any(id(p) in frozen_ids for g in optimizer.param_groups for p in g['params']):
            raise ValueError('GRPO optimizer contains frozen representation parameters')
    def step(self,new_logp,old_logp,reference_logp,rewards,completion_mask):
        loss=grpo_loss(new_logp,old_logp,reference_logp,rewards,completion_mask,beta=self.beta)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(),1.)
        self.optimizer.step()
        return float(loss.detach())
