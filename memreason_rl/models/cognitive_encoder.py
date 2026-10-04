import torch
from torch import nn

class NumericEncoder(nn.Module):
    def __init__(self,n_items,dim):
        super().__init__()
        self.identity=nn.Embedding(n_items,dim)
        self.value=nn.Sequential(nn.Linear(1,dim),nn.GELU(),nn.Linear(dim,dim))
    def forward(self,ids,values):
        return self.identity(ids)+self.value(values.unsqueeze(-1))

class FrozenTextEncoder(nn.Module):
    """Explicit local pretrained model; never downloads an unrequested backbone."""
    def __init__(self,model_path,dim):
        super().__init__()
        from transformers import AutoModel,AutoTokenizer
        self.tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
        self.model=AutoModel.from_pretrained(model_path,local_files_only=True)
        self.model.requires_grad_(False)
        self.projection=nn.Linear(self.model.config.hidden_size,dim)
    def forward(self,texts,device):
        self.model.eval()
        encoded=self.tokenizer(texts,padding=True,truncation=True,max_length=192,return_tensors='pt').to(device)
        with torch.no_grad():
            x=self.model(**encoded).last_hidden_state
            mask=encoded['attention_mask'].unsqueeze(-1)
            x=(x*mask).sum(1)/mask.sum(1).clamp_min(1)
        return self.projection(x)

class CognitiveEncoder(nn.Module):
    def __init__(self,items,dictionary,normalizer,dim=256,text_model=None):
        super().__init__()
        self.numeric=NumericEncoder(len(items),dim)
        self.items,self.dictionary,self.normalizer=items,dictionary,normalizer
        self.text=FrozenTextEncoder(text_model,dim) if text_model else None
        self.gate=nn.Linear(2*dim,dim)
    def forward(self,ids,values,observed):
        numeric=self.numeric(ids,values)
        if self.text is None:
            return numeric*observed.unsqueeze(-1)
        positions=observed.nonzero(as_tuple=True)
        texts=[]
        for item_id,value in zip(ids[positions].tolist(),values[positions].detach().tolist()):
            key=self.items[item_id]
            e=self.dictionary[key]
            raw=value*self.normalizer[key]['std']+self.normalizer[key]['mean']
            label=e.get('response_labels',{}).get(str(raw),f'raw code {raw:g}; no unverified category inference')
            texts.append(f"Instrument: {e['instrument']}; Domain: {e['domain']}; Task: {e['task_description']}; Response: {label}; Score: {raw:g}")
        semantic=torch.zeros_like(numeric)
        if texts:
            semantic[positions]=self.text(texts,values.device)
        gate=torch.sigmoid(self.gate(torch.cat([semantic,numeric],dim=-1)))
        return (gate*semantic+(1-gate)*numeric)*observed.unsqueeze(-1)
