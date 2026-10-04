"""Gated post-training APIs. Caller supplies local LLM/tokenized SFT or on-policy rollouts."""
import torch
from .gates import require_gate
from ..models.reasoning_llm import parse_report

def train_sft_epoch(reasoner,loader,optimizer):
    reasoner.train()
    losses=[]
    for batch in loader:
        for report in batch['reports']:
            parse_report(report)
        if not (batch['labels']!=-100).any():
            raise ValueError('No completion labels; prompts must be masked with -100')
        output=reasoner(batch['input_ids'],batch['attention_mask'],batch['labels'])
        optimizer.zero_grad(set_to_none=True); output.loss.backward()
        torch.nn.utils.clip_grad_norm_(reasoner.parameters(),1.)
        optimizer.step(); losses.append(float(output.loss.detach()))
    return losses

def train_grpo_epoch(trainer,rollout_batches,gate_file):
    require_gate('stage4',gate_file)
    records=[]
    for batch in rollout_batches:
        if 'reward_components' not in batch:
            raise ValueError('Every GRPO group must retain reward decomposition')
        loss=trainer.step(batch['new_logp'],batch['old_logp'],batch['reference_logp'],batch['rewards'],batch['completion_mask'])
        records.append(dict(loss=loss,reward_components=batch['reward_components']))
    return records
