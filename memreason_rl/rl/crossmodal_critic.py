import torch
from torch import nn

class CrossmodalCritic(nn.Module):
    def __init__(self,cognitive_dim=32,mri_dim=256):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(cognitive_dim+mri_dim,128),nn.GELU(),nn.Linear(128,1))
    def forward(self,cognition,mri):
        return self.network(torch.cat([cognition,mri],dim=-1)).squeeze(-1)
