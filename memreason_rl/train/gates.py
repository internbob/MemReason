import json
from pathlib import Path

REQUIRED={
    'stage1':['dictionary_verified','no_diagnosis_input'],
    'stage2':['semantic_beats_numeric_reconstruction','cross_scale_two_pairs_improved','no_diagnosis_leakage'],
    'stage3':['alignment_beats_shuffled','mri_reproducible_gain','attention_not_collapsed'],
    'stage4':['forecast_beats_locf_lme_grud_12_24m','uncertainty_calibrated','variable_history_stable','sft_schema_valid'],
}

def require_gate(stage,path):
    path=Path(path)
    if not path.is_file():
        raise RuntimeError(f'{stage} gate evidence missing: {path}')
    record=json.loads(path.read_text())
    if record.get('engineering_smoke_only') or record.get('split')=='test':
        raise RuntimeError('Smoke results or test-set selection cannot satisfy an advancement gate')
    missing=[k for k in REQUIRED[stage] if record.get('criteria',{}).get(k) is not True]
    evidence=record.get('artifacts',[])
    if missing or not evidence or any(not Path(p).is_file() for p in evidence):
        raise RuntimeError(f'{stage} gate unsatisfied: {missing}; validation artifacts required')
    return record
