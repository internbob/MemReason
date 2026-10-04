"""Gated stage-II training API for verified, date-matched spatial MRI batches."""
import torch
from ..losses.grounding import contrastive_grounding
from .gates import require_gate

def train_grounding_epoch(cognitive_encoder,mri_encoder,grounder,loader,optimizer,gate_file):
    require_gate('stage2',gate_file)
    cognitive_encoder.eval()
    cognitive_encoder.requires_grad_(False)
    mri_encoder.train(); grounder.train()
    records=[]
    for batch in loader:
        if not all(batch['qc_pass']) or not torch.isfinite(batch['date_difference_days']).all():
            raise ValueError('Stage II requires QC-passed scans and real date differences')
        if (batch['date_difference_days'].abs()>90).any():
            raise ValueError('Stage II pair lies outside the main 90-day window')
        with torch.no_grad():
            cognition=cognitive_encoder(batch['item_ids'],batch['values'],batch['observed'])
        structural=mri_encoder(batch['volume'])
        evidence,attention=grounder(cognition,structural['spatial_tokens'])
        mask=batch['observed'].unsqueeze(-1)
        c=(cognition*mask).sum(1)/mask.sum(1).clamp_min(1)
        e=(evidence*mask).sum(1)/mask.sum(1).clamp_min(1)
        weights=torch.exp(-batch['date_difference_days'].abs()/90)
        # Spatial evidence prediction plus matched correspondence; no pooled MRI before grounding.
        loss=contrastive_grounding(c,structural['global_token'],batch['subject_ids'],weights)
        loss=loss+.1*torch.nn.functional.mse_loss(e,c.detach())
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        entropy=-(attention.clamp_min(1e-8)*attention.clamp_min(1e-8).log()).sum(-1).mean()
        records.append(dict(loss=float(loss.detach()),attention_entropy=float(entropy.detach())))
    if not records:
        raise ValueError('No eligible grounding batches')
    return records
