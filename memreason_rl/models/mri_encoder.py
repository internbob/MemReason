"""3D ViT retaining spatial tokens before question-to-anatomy grounding."""
import torch
from torch import nn

class MRIEncoder(nn.Module):
    def __init__(self,dim=256,patch_size=16,heads=4,layers=2):
        super().__init__()
        self.patch=nn.Conv3d(1,dim,patch_size,stride=patch_size)
        self.position=nn.Linear(3,dim)
        self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(dim,heads,dim*4,batch_first=True,norm_first=True),layers,enable_nested_tensor=False)
    def forward(self,volume):
        grid=self.patch(volume)
        coords=torch.stack(torch.meshgrid(*[torch.linspace(-1,1,n,device=volume.device) for n in grid.shape[2:]],indexing='ij'),-1).reshape(-1,3)
        tokens=grid.flatten(2).transpose(1,2)+self.position(coords)[None]
        tokens=self.encoder(tokens)
        return {'spatial_tokens':tokens,'global_token':tokens.mean(1),'coordinates':coords}
