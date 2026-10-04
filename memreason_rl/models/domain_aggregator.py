import math
import torch
from torch import nn

class DomainAggregator(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.queries=nn.Parameter(torch.randn(5,dim)/math.sqrt(dim))
        self.context=nn.Linear(dim,dim)
        self.gate=nn.Linear(2*dim,dim)
    def forward(self,tokens,domains,observed):
        slots=[]
        masks=[]
        for d in range(5):
            valid=observed & (domains==d)
            scores=(tokens*self.queries[d]).sum(-1)/math.sqrt(tokens.shape[-1])
            weights=torch.softmax(scores.masked_fill(~valid,-1e4),dim=-1)*valid
            weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-8)
            slots.append((tokens*weights.unsqueeze(-1)).sum(-2))
            masks.append(valid.any(-1))
        state=torch.stack(slots,dim=-2)
        available=torch.stack(masks,dim=-1)
        mem=state[...,0,:]
        ctx=state[...,1:,:].sum(-2)/available[...,1:].sum(-1,keepdim=True).clamp_min(1)
        gate=torch.sigmoid(self.gate(torch.cat([mem,ctx],dim=-1)))
        # No synthetic domain observation is added for unavailable slots.
        output=mem+gate*self.context(ctx)*available[...,1:].any(-1,keepdim=True)
        return output,state,available
