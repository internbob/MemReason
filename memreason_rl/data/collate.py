import torch
from .common import DOMAINS

class EventCollator:
    def __init__(self,store):
        self.store=store
    def __call__(self,samples):
        if not samples:
            raise ValueError('Empty batch')
        batch=len(samples)
        ne=max(len(s['history']) for s in samples)
        ni=max(max((len(e['state_payload']) for e in s['history']),default=1) for s in samples)
        ni=max(ni,1)
        ids=torch.zeros(batch,ne,ni,dtype=torch.long)
        values=torch.zeros(batch,ne,ni)
        observed=torch.zeros(batch,ne,ni,dtype=torch.bool)
        domains=torch.zeros_like(ids)
        times=torch.zeros(batch,ne)
        event_mask=torch.zeros(batch,ne,dtype=torch.bool)
        mri_paths={}
        # Padding zeros are explicitly masked; they never become cognitive observations.
        for b,s in enumerate(samples):
            for t,e in enumerate(s['history']):
                times[b,t]=e['time_month']/12
                event_mask[b,t]=True
                if e['role']=='mri':
                    mri_paths[b,t]=e['image_path']
                for j,(k,v) in enumerate(e['state_payload'].items()):
                    ids[b,t,j]=self.store.item_to_id[k]
                    n=self.store.norm[k]
                    values[b,t,j]=(v-n['mean'])/n['std']
                    domains[b,t,j]=DOMAINS.index(self.store.dictionary[k]['domain'])
                    observed[b,t,j]=True
        targets=[]
        for b,s in enumerate(samples):
            for e in s['targets']:
                for k,v in e['payload'].items():
                    if k not in self.store.target_to_id:
                        continue
                    targets.append((b,self.store.target_to_id[k],float(v),float(e['time_month']-s['cutoff'])/12,k))
        return dict(item_ids=ids,values=values,observed=observed,domains=domains,times=times,event_mask=event_mask,
            cutoff=torch.tensor([s['cutoff']/12 for s in samples]),horizon=torch.tensor([s['horizon']/12 for s in samples]),
            targets=targets,mri_paths=mri_paths,sample_ids=[s['sample_id'] for s in samples])
