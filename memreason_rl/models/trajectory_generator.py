import torch
from torch import nn

class TrajectoryGenerator(nn.Module):
    def __init__(self,transition):
        super().__init__()
        self.transition=transition
    def forward(self,mu,logvar,horizons_years,num_candidates=4,generator=None):
        if list(horizons_years)!=sorted(horizons_years) or min(horizons_years)<0:
            raise ValueError('Horizons must be nonnegative and sorted')
        trajectories=[]
        for _ in range(num_candidates):
            noise=torch.randn(mu.shape,device=mu.device,dtype=mu.dtype,generator=generator)
            z=mu+(.5*logvar).exp()*noise
            lv=logvar
            previous=0.
            states=[]
            for h in horizons_years:
                z,lv=self.transition(z,lv,torch.full((len(z),),h-previous,device=z.device))
                states.append((z,lv))
                previous=h
            trajectories.append(states)
        return trajectories
