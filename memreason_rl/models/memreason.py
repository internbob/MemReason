import torch
from torch import nn
from .cognitive_encoder import CognitiveEncoder
from .domain_aggregator import DomainAggregator
from .temporal_dynamics import TimeAwareTransformer,ContinuousTransition
from .belief_state import BeliefState
from .decoders import ItemDecoders

class MemReason(nn.Module):
    def __init__(self,items,dictionary,normalizer,dim=256,state_dim=32,heads=4,layers=2,text_model=None,target_items=None):
        super().__init__()
        self.cognitive=CognitiveEncoder(items,dictionary,normalizer,dim,text_model)
        self.domains=DomainAggregator(dim)
        self.temporal=TimeAwareTransformer(dim,heads,layers)
        self.belief=BeliefState(dim,state_dim)
        self.transition=ContinuousTransition(state_dim)
        self.decoders=ItemDecoders(target_items or items,dictionary,state_dim)
        self.items,self.dictionary,self.normalizer=items,dictionary,normalizer
    def encode(self,batch,mri_embeddings=None):
        tok=self.cognitive(batch['item_ids'],batch['values'],batch['observed'])
        x,domains,availability=self.domains(tok,batch['domains'],batch['observed'])
        if batch['mri_paths'] and mri_embeddings is None:
            raise ValueError('MRI events require actual encoded MRI evidence; zero anchors are prohibited')
        if mri_embeddings:
            x=x.clone()
            for (b,t),value in mri_embeddings.items():
                x[b,t]=value
        summary,states=self.temporal(x,batch['times'],batch['event_mask'])
        mu,logvar=self.belief(summary)
        last_time=batch['times'].masked_fill(~batch['event_mask'],-torch.inf).max(1).values
        mu,logvar=self.transition(mu,logvar,(batch['cutoff']-last_time).clamp_min(0))
        return mu,logvar
    def forward(self,batch):
        mu,logvar=self.encode(batch)
        z,logv=self.transition(mu,logvar,batch['horizon'])
        return {'mu':z,'logvar':logv,'current_mu':mu,'current_logvar':logvar}
