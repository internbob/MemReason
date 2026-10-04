import torch
from torch import nn

class TimeAwareTransformer(nn.Module):
    def __init__(self,dim=256,heads=4,layers=2,dropout=.1):
        super().__init__()
        self.heads=heads
        self.time=nn.Sequential(nn.Linear(3,dim),nn.Tanh(),nn.Linear(dim,dim))
        layer=nn.TransformerEncoderLayer(dim,heads,dim*4,dropout,batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,layers,enable_nested_tensor=False)
    def forward(self,x,times,event_mask):
        gaps=torch.cat([torch.zeros_like(times[:,:1]),times[:,1:]-times[:,:-1]],dim=1).clamp_min(0)
        features=torch.stack([times,torch.sin(times),torch.log1p(gaps)],dim=-1)
        x=x+self.time(features)
        # Same-date instrument events may see each other; later events are masked.
        # Padded queries may attend valid keys, avoiding all-masked softmax/NaNs
        # that could contaminate subsequent layers. Their outputs are discarded.
        causal=((times[:,None,:]>times[:,:,None]) & event_mask[:,:,None]).repeat_interleave(self.heads,dim=0)
        encoded=self.encoder(x,mask=causal,src_key_padding_mask=~event_mask)
        lengths=event_mask.sum(1)-1
        return encoded[torch.arange(len(x),device=x.device),lengths],encoded

class ContinuousTransition(nn.Module):
    def __init__(self,dim=32):
        super().__init__()
        self.drift=nn.Sequential(nn.Linear(dim+1,dim*2),nn.Tanh(),nn.Linear(dim*2,dim))
        self.diffusion=nn.Sequential(nn.Linear(dim+1,dim),nn.Softplus())
    def forward(self,mu,logvar,delta_years):
        if (delta_years<0).any():
            raise ValueError('Forecast transition cannot use negative time')
        dt=delta_years.unsqueeze(-1)
        features=torch.cat([mu,torch.log1p(dt)],-1)
        next_mu=mu+dt*self.drift(features)
        var=logvar.exp()+dt*self.diffusion(features)
        return next_mu,var.clamp_min(1e-6).log()
