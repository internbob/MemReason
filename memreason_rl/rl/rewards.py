import math
import numpy as np
from ..models.reasoning_llm import parse_report
from .evidence_verifier import verify_evidence
from .domain_verifier import domain_reward

DEFAULT_WEIGHTS={'outcome':1.,'evidence':1.,'temporal':.5,'domain':.5,'mri':.5,'crossmodal':.5,'uncertainty':.2,'format':.05}

def availability_weighted_reward(components,weights=None):
    weights=weights or DEFAULT_WEIGHTS
    active={k:float(v) for k,v in components.items() if v is not None and k in weights and weights[k]>0}
    if any(not math.isfinite(v) for v in active.values()):
        raise ValueError('Nonfinite available reward')
    if not active:
        return None
    return sum(weights[k]*v for k,v in active.items())/sum(weights[k] for k in active)

def outcome_reward(predictions,targets,normalizer):
    values=[]
    # None = absent target. Missing predictions for an available target are penalized.
    for key,target in targets.items():
        if target is None or not np.isfinite(target):
            continue
        pred=predictions.get(key)
        values.append(0. if pred is None or not np.isfinite(pred) else math.exp(-abs(pred-target)/max(normalizer[key]['std'],1e-6)))
    return float(np.mean(values)) if values else None

def score_report(report,event_store,dictionary,subject_id,cutoff,predictions,targets,normalizer,**available_components):
    components={k:None for k in DEFAULT_WEIGHTS}
    try:
        parsed=parse_report(report)
    except (ValueError,TypeError):
        components.update(format=0.,evidence=0.,outcome=outcome_reward(predictions,targets,normalizer))
    else:
        components.update(format=1.,evidence=verify_evidence(parsed['evidence'],event_store,subject_id,cutoff)[0],
            domain=domain_reward(parsed['evidence'],dictionary),outcome=outcome_reward(predictions,targets,normalizer))
    for key in ['temporal','mri','crossmodal','uncertainty']:
        components[key]=available_components.get(key)
    return {'total':availability_weighted_reward(components),'components':components}
