import numpy as np
import pandas as pd
import torch

def prediction_value(pred,entry,norm):
    typ=entry['value_type']
    if typ=='binary':
        return float(pred.sigmoid().squeeze())
    if typ=='ordinal':
        vals=pred.new_tensor(entry['valid_values'])
        return float(vals[0]+(pred.sigmoid()*(vals[1:]-vals[:-1])).sum())
    if typ=='categorical' and entry['valid_values']:
        return float(entry['valid_values'][int(pred.argmax())])
    return float(pred[0]*norm['std']+norm['mean'])

def metrics(rows):
    frame=pd.DataFrame(rows)
    result=[]
    if frame.empty:
        return pd.DataFrame(columns=['item','n','mae','rmse','pearson','spearman'])
    keys=['item']+(['horizon_months'] if 'horizon_months' in frame else [])
    for key,g in frame.groupby(keys):
        if not isinstance(key,tuple):
            key=(key,)
        err=g.prediction-g.target
        record=dict(zip(keys,key))
        record.update(n=len(g),mae=float(err.abs().mean()),rmse=float(np.sqrt((err**2).mean())),
            pearson=float(g.prediction.corr(g.target)) if len(g)>2 and g.prediction.std()>0 and g.target.std()>0 else None,
            spearman=float(g.prediction.corr(g.target,method='spearman')) if len(g)>2 and g.prediction.std()>0 and g.target.std()>0 else None)
        result.append(record)
    return pd.DataFrame(result)

@torch.no_grad()
def evaluate_batch(model,batch,output):
    result=[]
    for b,_,value,delta,key in batch['targets']:
        z,_=model.transition(output['current_mu'][b:b+1],output['current_logvar'][b:b+1],output['mu'].new_tensor([max(delta,0.)]))
        pred=model.decoders(z[0],key)
        prediction=prediction_value(pred,model.dictionary[key],model.normalizer[key])
        result.append(dict(sample_id=batch['sample_ids'][b],item=key,target=value,prediction=prediction,absolute_error=abs(prediction-value),horizon_months=round(float(batch['horizon'][b])*12)))
    return result
