"""Observed-value LOCF baseline on a sealed subject split; no outcome input to state."""
import argparse
from ..data.common import ROOT,save_csv,save_json
from ..data.dataset import EventStore,ForecastDataset
from .forecast import metrics
import pandas as pd

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--split',choices=['val','test'],default='val')
    p.add_argument('--exploratory',action='store_true')
    args=p.parse_args()
    store=EventStore(exploratory=args.exploratory,allow_nominal_time=args.exploratory)
    dataset=ForecastDataset(store,args.split)
    rows=[]
    n_available=0
    for sample in dataset:
        latest={}
        for event in store.subjects[sample['history'][0]['subject_id']]:
            if event['time_month']<=sample['cutoff'] and event['time_eligible']:
                latest.update({k:v for k,v in event['payload'].items() if k in store.target_to_id})
        for event in sample['targets']:
            for key,value in event['payload'].items():
                if key not in store.target_to_id:
                    continue
                n_available+=1
                if key not in latest:
                    continue
                rows.append(dict(sample_id=sample['sample_id'],item=key,horizon_months=sample['horizon'],target=value,prediction=latest[key],absolute_error=abs(value-latest[key])))
    output=ROOT/'reports/experiments'/f'locf_{args.split}'
    save_csv(metrics(rows),output/'metrics.csv')
    save_csv(pd.DataFrame(rows).sort_values('absolute_error',ascending=False).head(100),output/'failure_cases.csv')
    save_json(dict(method='LOCF',exploratory=args.exploratory,split=args.split,predicted_targets=len(rows),available_targets=n_available,
        coverage=len(rows)/max(n_available,1),caveat='Targets without an observed same-item history are skipped; report coverage with errors.'),output/'status.json')

if __name__=='__main__':
    main()
