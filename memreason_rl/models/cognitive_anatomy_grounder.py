import torch
from torch import nn

class CognitiveAnatomyGrounder(nn.Module):
    def __init__(self,dim=256,heads=4):
        super().__init__()
        self.attention=nn.MultiheadAttention(dim,heads,batch_first=True)
        self.fuse=nn.Sequential(nn.Linear(2*dim,dim),nn.GELU(),nn.LayerNorm(dim))
    def forward(self,cognitive_tokens,mri_spatial_tokens):
        structural,attention=self.attention(cognitive_tokens,mri_spatial_tokens,mri_spatial_tokens,need_weights=True,average_attn_weights=False)
        return self.fuse(torch.cat([cognitive_tokens,structural],-1)),attention
