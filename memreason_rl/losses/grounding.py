import torch
from torch.nn import functional as F

def contrastive_grounding(cognition,mri,subject_ids,time_weights,temperature=.07):
    """Other visits of the same subject are not accidental negative pairs."""
    if len(cognition)<2 or len(set(subject_ids))<2:
        raise ValueError('Grounding needs at least two distinct subjects')
    logits=F.normalize(cognition,dim=-1)@F.normalize(mri,dim=-1).T/temperature
    same=torch.tensor([[a==b for b in subject_ids] for a in subject_ids],device=logits.device)
    eye=torch.eye(len(logits),device=logits.device,dtype=torch.bool)
    logits=logits.masked_fill(same & ~eye,-torch.inf)
    labels=torch.arange(len(logits),device=logits.device)
    losses=(F.cross_entropy(logits,labels,reduction='none')+F.cross_entropy(logits.T,labels,reduction='none'))/2
    if not torch.isfinite(time_weights).all() or (time_weights<0).any():
        raise ValueError('Grounding time weights require known nonnegative real-date weights')
    return (losses*time_weights).sum()/time_weights.sum().clamp_min(1e-8)
