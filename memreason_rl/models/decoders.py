import torch
from torch import nn
from torch.nn import functional as F

class ItemDecoders(nn.Module):
    """Separate item heads within each instrument; ordinal codes use ordered thresholds."""
    def __init__(self,items,dictionary,dim=32):
        super().__init__()
        self.items,self.dictionary=items,dictionary
        heads={}
        for k in items:
            e=dictionary[k]
            n=len(e['valid_values'] or [])
            width=1 if e['value_type']=='binary' else n-1 if e['value_type']=='ordinal' else n if e['value_type']=='categorical' and n else 2
            heads[k]=nn.Linear(dim,width)
        self.heads=nn.ModuleDict(heads)
    def forward(self,z,item):
        raw=self.heads[item](z)
        if self.dictionary[item]['value_type']=='ordinal':
            # Nonincreasing cumulative P(Y > threshold) logits.
            return torch.cat([raw[...,:1],raw[...,:1]-F.softplus(raw[...,1:]).cumsum(-1)],-1)
        return raw
