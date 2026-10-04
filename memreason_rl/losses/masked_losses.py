import torch
from torch.nn import functional as F

def item_loss(prediction,value,entry,normalizer):
    target=prediction.new_tensor(value)
    typ=entry['value_type']
    if typ=='binary':
        return F.binary_cross_entropy_with_logits(prediction.squeeze(-1),target)
    if typ=='ordinal':
        rank=entry['valid_values'].index(value)
        thresholds=torch.arange(len(entry['valid_values'])-1,device=prediction.device)
        return F.binary_cross_entropy_with_logits(prediction,(rank>thresholds).to(prediction.dtype))
    if typ=='categorical' and entry['valid_values']:
        rank=entry['valid_values'].index(value)
        return F.cross_entropy(prediction.unsqueeze(0),torch.tensor([rank],device=prediction.device))
    y=(target-normalizer['mean'])/normalizer['std']
    mean,logvar=prediction.unbind(-1)
    logvar=logvar.clamp(-8,6)
    return F.huber_loss(mean,y)+.05*(logvar+(y-mean).square()/logvar.exp())

def supervised_loss(model,batch,output):
    losses=[]
    for b,item_id,value,delta,key in batch['targets']:
        # Reconstruction may refer to an earlier item in the same instrument bucket.
        dt=max(delta,0.) if batch['horizon'][b]==0 else delta
        if dt<0:
            raise ValueError('Future target precedes cutoff')
        z,_=model.transition(output['current_mu'][b:b+1],output['current_logvar'][b:b+1],output['mu'].new_tensor([dt]))
        if model.training:
            _,lv=model.transition(output['current_mu'][b:b+1],output['current_logvar'][b:b+1],output['mu'].new_tensor([dt]))
            z=z+torch.randn_like(z)*(.5*lv).exp()
        pred=model.decoders(z[0],key)
        losses.append(item_loss(pred,value,model.dictionary[key],model.normalizer[key]))
    if not losses:
        return output['mu'].sum()*0
    mu,lv=output['current_mu'],output['current_logvar']
    kl=.5*(mu.square()+lv.exp()-1-lv).mean()
    return torch.stack(losses).mean()+.001*kl

def masked_mean(values,mask):
    selected=values[mask]
    return selected.mean() if selected.numel() else values.nan_to_num().sum()*0
