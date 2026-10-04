"""Evidence is filtered before collation; targets never enter the history."""
import json
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from .common import ROOT, read_csv, DOMAINS
from .dictionary import load_dictionary, state_items

class EventStore:
    def __init__(self, root=ROOT, exploratory=False, allow_nominal_time=False):
        self.root=Path(root)
        self.dictionary=load_dictionary(self.root/'configs/cognitive_item_dictionary.yaml')
        self.items=state_items(self.dictionary,exploratory)
        if not self.items:
            raise ValueError('No dictionary-verified state items. Verify dictionary first, or explicitly use exploratory=True for engineering checks.')
        self.item_to_id={k:i for i,k in enumerate(self.items)}
        self.target_items=self.items+[k for k,e in self.dictionary.items() if e['use_for_outcome'] and not k.startswith('dxsum_') and (exploratory or e['verified_from_dictionary'])]
        self.target_to_id={k:i for i,k in enumerate(self.target_items)}
        self.norm=json.loads((self.root/'processed/train_normalization.json').read_text())['items']
        self.events={}
        self.subjects={}
        for row in read_csv(self.root/'processed/events.csv').to_dict('records'):
            row['payload']=json.loads(row.pop('payload_json'))
            state=json.loads(row.pop('state_payload_json'))
            row['state_payload']={k:v for k,v in state.items() if k in self.item_to_id}
            row['time_eligible']=allow_nominal_time or row['time_source']=='collection_date'
            self.events[row['event_id']]=row
            self.subjects.setdefault(row['subject_id'],[]).append(row)
        self.exploratory=exploratory

    def history(self, sid, cutoff, include_mri=False):
        return [e for e in self.subjects[sid] if e['time_month']<=cutoff and e['time_eligible'] and
                ((e['role']=='cognition' and e['state_payload']) or (include_mri and e['role']=='mri'))]

class ForecastDataset(Dataset):
    def __init__(self, store, split='train', max_samples=None, include_mri=False):
        self.store=store
        self.include_mri=include_mri
        rows=read_csv(store.root/f'splits/{split}_forecast.csv').to_dict('records')
        self.samples=[]
        for row in rows:
            targets=[store.events[k] for k in json.loads(row['target_event_ids_json'])]
            if not store.history(row['subject_id'],row['cutoff_month'],include_mri):
                continue
            if not any(e['time_eligible'] and any(k in store.target_to_id for k in e['payload']) for e in targets):
                continue
            self.samples.append(row)
            if max_samples and len(self.samples)>=max_samples:
                break
    def __len__(self):
        return len(self.samples)
    def __getitem__(self,index):
        row=self.samples[index]
        history=self.store.history(row['subject_id'],row['cutoff_month'],self.include_mri)
        targets=[self.store.events[k] for k in json.loads(row['target_event_ids_json'])]
        targets=[e for e in targets if e['time_eligible']]
        assert all(e['time_month']>row['cutoff_month'] for e in targets)
        return dict(history=history,targets=targets,cutoff=row['cutoff_month'],horizon=row['horizon_months'],sample_id=row['sample_id'])

class MaskedObservationDataset(Dataset):
    """Masks observed entries, including whole instruments, without hidden-score leakage."""
    def __init__(self,store,split='train',seed=2026,item_prob=.15,instrument_prob=.2,max_samples=None):
        self.store,self.seed,self.epoch=store,seed,0
        self.item_prob,self.instrument_prob=item_prob,instrument_prob
        groups={}
        for e in store.events.values():
            if e['split']==split and e['role']=='cognition' and e['time_eligible'] and e['state_payload']:
                groups.setdefault(e['visit_id'],[]).append(e)
        self.groups=[v for _,v in sorted(groups.items()) if sum(len(e['state_payload']) for e in v)>=2]
        if max_samples:
            self.groups=self.groups[:max_samples]
    def __len__(self):
        return len(self.groups)
    def __getitem__(self,index):
        rng=np.random.default_rng(self.seed+self.epoch*1000003+index)
        original=self.groups[index]
        positions=[(i,k) for i,e in enumerate(original) for k in e['state_payload']]
        instruments={e['instrument'] for e in original}
        held={s for s in sorted(instruments) if rng.random()<self.instrument_prob}
        hidden={(i,k) for i,k in positions if original[i]['instrument'] in held or rng.random()<self.item_prob}
        if not hidden:
            hidden.add(positions[int(rng.integers(len(positions)))])
        if len(hidden)==len(positions):
            hidden.remove(positions[int(rng.integers(len(positions)))])
        history,targets=[],[]
        for i,e in enumerate(original):
            observed={k:v for k,v in e['state_payload'].items() if (i,k) not in hidden}
            target={k:v for k,v in e['state_payload'].items() if (i,k) in hidden}
            if observed:
                # Remove the original payload as well, so hidden responses cannot reach text prompts.
                h={**e,'state_payload':observed,'payload':observed}
                history.append(h)
            if target:
                targets.append({**e,'payload':target})
        return dict(history=history,targets=targets,cutoff=max(e['time_month'] for e in original),horizon=0.,sample_id=original[0]['visit_id'])
