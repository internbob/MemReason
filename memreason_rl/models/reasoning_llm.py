"""Strict JSON evidence schema plus optional local causal LM for SFT."""
import json
import math
from torch import nn

REQUIRED={'current_state':dict,'evidence':list,'trend':dict,'forecast':dict,'confidence':dict}

def parse_report(text):
    value=json.loads(text) if isinstance(text,str) else text
    if not isinstance(value,dict) or set(value)!=set(REQUIRED):
        raise ValueError('Report must have exactly the five documented sections')
    for k,typ in REQUIRED.items():
        if not isinstance(value[k],typ):
            raise ValueError(f'Invalid section type: {k}')
    for citation in value['evidence']:
        if not isinstance(citation,dict) or not {'event_id','item','time_month','value','domain'}<=citation.keys():
            raise ValueError('Incomplete evidence citation')
        if not all(isinstance(citation[k],(int,float)) and math.isfinite(citation[k]) for k in ['time_month','value']):
            raise ValueError('Evidence time and value must be finite numbers')
    # JSON NaN/Infinity are rejected throughout nested output.
    json.dumps(value,allow_nan=False)
    return value

def evidence_report(history,dictionary,forecast):
    citations=[]
    for e in history:
        for key,value in e.get('state_payload',{}).items():
            if dictionary[key]['verified_from_dictionary']:
                citations.append(dict(event_id=e['event_id'],item=key,time_month=e['time_month'],value=value,domain=dictionary[key]['domain']))
    return parse_report(dict(current_state={},evidence=citations,trend={},forecast=forecast,confidence={}))

class ReasoningLLM(nn.Module):
    def __init__(self,model_path,lora_rank=8):
        super().__init__()
        from transformers import AutoModelForCausalLM,AutoTokenizer
        from peft import LoraConfig,get_peft_model
        self.tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
        model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True)
        self.model=get_peft_model(model,LoraConfig(r=lora_rank,lora_alpha=2*lora_rank,target_modules='all-linear',task_type='CAUSAL_LM'))
    def forward(self,input_ids,attention_mask,labels):
        return self.model(input_ids=input_ids,attention_mask=attention_mask,labels=labels)
