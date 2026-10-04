import torch
from torch import nn

class BeliefState(nn.Module):
    def __init__(self,input_dim=256,state_dim=32):
        super().__init__()
        self.posterior=nn.Linear(input_dim,2*state_dim)
        self.correction=nn.Linear(input_dim,2*state_dim)
        self.gate=nn.Linear(input_dim+state_dim,state_dim)
    def forward(self,context):
        mu,logvar=self.posterior(context).chunk(2,-1)
        return mu,logvar.clamp(-8,6)
    def update(self,mu,logvar,evidence):
        shift,variance_shift=self.correction(evidence).chunk(2,-1)
        g=torch.sigmoid(self.gate(torch.cat([mu,evidence],-1)))
        return mu+g*shift,(logvar+g*torch.tanh(variance_shift)).clamp(-8,6)
