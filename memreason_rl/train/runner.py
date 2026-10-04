import argparse
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from ..data.common import ROOT,save_csv,save_json
from ..data.dataset import EventStore,MaskedObservationDataset,ForecastDataset
from ..data.collate import EventCollator
from ..models import MemReason
from ..losses import supervised_loss
from ..eval.forecast import metrics,evaluate_batch
from .gates import require_gate

def run(task='mask'):
    p=argparse.ArgumentParser()
    p.add_argument('--exploratory',action='store_true')
    p.add_argument('--smoke',action='store_true',help='Bounded software check; cannot satisfy scientific gates')
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--device',default='cpu')
    p.add_argument('--text-model',default=None)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.smoke and not args.exploratory:
        raise ValueError('Smoke checks on provisional dictionary require --exploratory')
    stage='stage1' if task=='mask' else 'stage3'
    if not args.smoke:
        require_gate(stage,ROOT/f'reports/experiments/{stage}_entry_gate.json')
    if task=='forecast' and not args.checkpoint and not args.smoke:
        raise ValueError('Dynamics training requires a pretrained representation checkpoint')
    seed=2026
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1)
    store=EventStore(exploratory=args.exploratory,allow_nominal_time=args.exploratory)
    klass=MaskedObservationDataset if task=='mask' else ForecastDataset
    train=klass(store,'train',max_samples=12 if args.smoke else None)
    val=klass(store,'val',max_samples=8 if args.smoke else None)
    if not train or not val:
        raise ValueError('No usable training/validation observations')
    collate=EventCollator(store)
    trainloader=DataLoader(train,batch_size=2 if args.smoke else args.batch_size,shuffle=True,collate_fn=collate)
    valloader=DataLoader(val,batch_size=2 if args.smoke else args.batch_size,collate_fn=collate)
    kwargs=dict(items=store.items,target_items=store.target_items,dictionary=store.dictionary,normalizer=store.norm,dim=32 if args.smoke else 256,state_dim=16 if args.smoke else 32,heads=4,layers=1 if args.smoke else 2,text_model=args.text_model)
    model=MemReason(**kwargs).to(args.device)
    if args.checkpoint:
        checkpoint=torch.load(args.checkpoint,map_location=args.device,weights_only=True)
        if checkpoint['items']!=store.items:
            raise ValueError('Checkpoint item vocabulary differs from active dictionary')
        model.load_state_dict(checkpoint['model'])
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr)
    output=args.output
    output=output or ROOT/'outputs'/f'{stage}_{"smoke" if args.smoke else "training"}'
    output.mkdir(parents=True,exist_ok=True)
    curves=[]; best=float('inf'); last_predictions=[]
    epochs=1 if args.smoke else args.epochs
    for epoch in range(epochs):
        if hasattr(train,'epoch'):
            train.epoch=epoch
        model.train(); losses=[]
        for step,batch in enumerate(trainloader):
            if args.smoke and step>=3:
                break
            batch={k:v.to(args.device) if torch.is_tensor(v) else v for k,v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            out=model(batch)
            loss=supervised_loss(model,batch,out)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite train loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step(); losses.append(float(loss.detach()))
        model.eval(); val_losses=[]; last_predictions=[]
        with torch.no_grad():
            for batch in valloader:
                batch={k:v.to(args.device) if torch.is_tensor(v) else v for k,v in batch.items()}
                out=model(batch)
                val_losses.append(float(supervised_loss(model,batch,out)))
                last_predictions.extend(evaluate_batch(model,batch,out))
        record=dict(epoch=epoch,train_loss=float(np.mean(losses)),val_loss=float(np.mean(val_losses)))
        curves.append(record); print(json.dumps(record),flush=True)
        if record['val_loss']<best:
            best=record['val_loss']
            torch.save(dict(model=model.state_dict(),items=store.items,target_items=store.target_items,engineering_smoke_only=args.smoke,epoch=epoch,dim=kwargs['dim'],state_dim=kwargs['state_dim'],layers=kwargs['layers']),output/'best.pt')
    import pandas as pd
    save_csv(pd.DataFrame(curves),output/'curves.csv')
    save_csv(metrics(last_predictions),output/'metrics.csv')
    save_csv(pd.DataFrame(last_predictions).sort_values('absolute_error',ascending=False).head(50),output/'failure_cases.csv')
    save_json(dict(stage=stage,engineering_smoke_only=args.smoke,exploratory=args.exploratory,semantic_encoder=bool(args.text_model),
        train_samples=len(train),val_samples=len(val),train_steps=min(len(trainloader),3) if args.smoke else len(trainloader)*epochs,
        dictionary_verified=all(store.dictionary[k]['verified_from_dictionary'] for k in store.items),next_stage_gate_passed=False,
        interpretation='Execution/gradient check only; not a trained scientific baseline' if args.smoke else 'Validation results require separate paired ablations before gate advancement'),output/'status.json')

if __name__=='__main__':
    run()
