"""Real-data end-to-end checks, plus a bounded MRI tensor/gradient check on CPU."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
import torch
from memreason_rl.data.common import ROOT,save_json
from memreason_rl.data.dataset import EventStore,ForecastDataset
from memreason_rl.data.collate import EventCollator
from memreason_rl.data.mri import load_mri
from memreason_rl.models import MemReason
from memreason_rl.models.mri_encoder import MRIEncoder
from memreason_rl.models.cognitive_anatomy_grounder import CognitiveAnatomyGrounder
from memreason_rl.losses import supervised_loss
from memreason_rl.rl.rewards import availability_weighted_reward

torch.set_num_threads(1)
torch.manual_seed(2026)
store=EventStore(exploratory=True,allow_nominal_time=True)
dataset=ForecastDataset(store,'train',max_samples=2)
batch=EventCollator(store)([dataset[0],dataset[1]])
model=MemReason(store.items,store.dictionary,store.norm,dim=32,state_dim=16,layers=1,target_items=store.target_items)
output=model(batch)
loss=supervised_loss(model,batch,output)
loss.backward()
assert torch.isfinite(loss)
assert model.belief.posterior.weight.grad is not None
assert not set(store.items)&{k for k in store.target_items if k.startswith(('faq_','dxsum_'))}
scan=pd.read_csv(ROOT/'processed/mri_primary.csv').image_path.iloc[0]
volume=load_mri(scan)[None]
mri=MRIEncoder(dim=32,patch_size=16,heads=4,layers=1)
encoded=mri(volume)
grounder=CognitiveAnatomyGrounder(32,4)
query=torch.randn(1,3,32)
evidence,attention=grounder(query,encoded['spatial_tokens'])
assert encoded['spatial_tokens'].shape[1]>1
assert torch.isfinite(evidence).all()
assert torch.allclose(attention.sum(-1),torch.ones_like(attention.sum(-1)),atol=1e-5)
result=dict(engineering_smoke_only=True,forecast_forward_backward_finite=True,forecast_loss=float(loss.detach()),
    state_items=len(store.items),target_items=len(store.target_items),faq_and_diagnosis_absent_from_state=True,
    real_mri_path=scan,preprocessed_mri_shape=list(volume.shape),mri_spatial_tokens=list(encoded['spatial_tokens'].shape),
    grounding_attention_finite=True,absent_mri_reward_skipped=availability_weighted_reward({'outcome':.8,'mri':None},{'outcome':1,'mri':1})==.8,
    clinical_qc='not_evaluated',scientific_gate_passed=False)
save_json(result,ROOT/'reports/experiments/integration_smoke.json')
print(json.dumps(result,indent=2))
