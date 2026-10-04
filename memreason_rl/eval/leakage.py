"""Diagnosis is used solely as the target of this protocol leakage audit."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score,accuracy_score
from ..data.common import ROOT,read_csv,save_json

def main():
    df=read_csv(ROOT/'processed/adni_visits_with_mri.csv')
    columns=[c for c in df if c.endswith('_present')]
    x=np.column_stack([df[c].astype(str).str.lower().isin(['true','1','1.0']).to_numpy() for c in columns]).astype(float)
    valid=df.dxsum_diagnosis_label.isin(['CN','MCI','AD'])
    tr=(df.split=='train')&valid
    va=(df.split=='val')&valid
    model=LogisticRegression(max_iter=500,class_weight='balanced',random_state=2026).fit(x[tr],df.loc[tr,'dxsum_diagnosis_label'])
    prediction=model.predict(x[va])
    target=df.loc[va,'dxsum_diagnosis_label']
    phase_results={}
    for phase,g in df[va].groupby('dxsum_phase',dropna=False):
        idx=va[va].index.get_indexer(g.index)
        if g.dxsum_diagnosis_label.nunique()>1:
            phase_results[str(phase)]={'n':len(g),'balanced_accuracy':float(balanced_accuracy_score(g.dxsum_diagnosis_label,prediction[idx]))}
    result=dict(only_features=columns,target='dxsum_diagnosis_label',train_subjects=int(df.loc[tr,'subject_id'].nunique()),
        validation_subjects=int(df.loc[va,'subject_id'].nunique()),validation_accuracy=float(accuracy_score(target,prediction)),
        validation_balanced_accuracy=float(balanced_accuracy_score(target,prediction)),by_phase=phase_results,
        interpretation='Screen for protocol confounding only. No causal claim; repeated visits cluster within subject. Not a representation clinical validity result.')
    save_json(result,ROOT/'reports/data_audit/protocol_leakage.json')
    print(result)

if __name__=='__main__':
    main()
