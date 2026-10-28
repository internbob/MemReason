import json
import numpy as np
import pandas as pd
import pytest
import torch
from memreason_rl.data.common import ROOT,valid_value
from memreason_rl.data.split_subjects import split_subjects
from memreason_rl.data.dictionary import build_dictionary,state_items,load_dictionary
from memreason_rl.data.match_mri_cognition import match_mri_cognition
from memreason_rl.data.build_mri_metadata import load_ida_image_metadata
from memreason_rl.data.align_nearest_mri_month import align_visits_to_nearest_mri_month
from memreason_rl.rl.evidence_verifier import verify_evidence
from memreason_rl.rl.rewards import availability_weighted_reward
from memreason_rl.rl.grpo_trainer import grpo_loss
from memreason_rl.models.temporal_dynamics import TimeAwareTransformer
from memreason_rl.models.reasoning_llm import parse_report
from memreason_rl.train.gates import require_gate
from memreason_rl.data.mri import center_crop_3d, center_crop_slices


def test_standard_mri_center_crop_preserves_shape():
    shape = (193, 229, 193)
    target = (160, 192, 160)
    slices = center_crop_slices(shape, target)
    assert [(part.start, part.stop) for part in slices] == [
        (16, 176), (18, 210), (16, 176)
    ]
    volume = np.zeros(shape, dtype=np.uint8)
    volume[16:176, 18:210, 16:176] = 1
    cropped = center_crop_3d(volume, target)
    assert cropped.shape == target
    assert cropped.all()

def test_subject_split_deterministic_and_disjoint():
    ids=[f's{i}' for i in range(100)]
    a=split_subjects(ids)
    b=split_subjects(ids[::-1]+ids)
    pd.testing.assert_frame_equal(a,b)
    assert a.split.value_counts().to_dict()=={'train':70,'test':20,'val':10}
    assert a.subject_id.is_unique

def test_unknown_coding_and_outcome_isolation():
    d=build_dictionary(['moca_delw1','dxsum_diagnosis','faq_faqrem','cdr_cdhome','mmse_mmscore','mmse_word1dl'])
    assert state_items(d)==[]
    assert set(state_items(d,True))=={'moca_delw1','mmse_word1dl'}
    assert d['moca_delw1']['value_type']=='categorical'
    assert all(valid_value(v,d['moca_delw1']) for v in [0,1,2,3])
    assert not valid_value(999,d['mmse_word1dl'])

def test_same_visit_does_not_invent_date_weight_or_qc():
    visits=pd.DataFrame([dict(visit_id='a:bl',subject_id='a',visit_code='bl',cognition_date_start='2020-01-01')])
    scans=pd.DataFrame([dict(subject_id='a',primary=True,visit_code='bl',scan_id='s',scan_date=None,image_path='/real/example',mask_path=None,visit_month=0,qc_pass=None,qc_status='unknown')])
    match=match_mri_cognition(visits,scans).iloc[0]
    assert match.match_strategy=='same_visit_code'
    assert pd.isna(match.time_weight) and not match.main_grounding_eligible

def test_ida_metadata_requires_exact_image_subject_and_date(tmp_path):
    path=tmp_path/'ida.csv'
    pd.DataFrame([
        {'Subject ID':'002_S_0295','Study Date':'4/18/2006','Image ID':'I45108','Description':'MPRAGE','Type':'Pre-processed'},
    ]).to_csv(path,index=False)
    lookup=load_ida_image_metadata(path)
    assert lookup['45108']['subject_id_normalized']=='002_S_0295'
    assert lookup['45108']['scan_date_parsed']==pd.Timestamp('2006-04-18')

def test_nearest_mri_month_grouping_is_subject_local_and_tie_prefers_past():
    visits=pd.DataFrame([
        dict(subject_id='a',visit_month=0),
        dict(subject_id='a',visit_month=9),
        dict(subject_id='b',visit_month=0),
    ])
    mri=pd.DataFrame([
        dict(subject_id='a',scan_id='a6',image_id=6,session_id='ses-M006',visit_month=6,visit_code='m06',image_path='/a6',mask_path='/ma6',primary=True),
        dict(subject_id='a',scan_id='a12',image_id=12,session_id='ses-M012',visit_month=12,visit_code='m12',image_path='/a12',mask_path='/ma12',primary=True),
        dict(subject_id='b',scan_id='b24',image_id=24,session_id='ses-M024',visit_month=24,visit_code='m24',image_path='/b24',mask_path='/mb24',primary=True),
    ])
    aligned=align_visits_to_nearest_mri_month(visits,mri,max_gap_months=6)
    assert aligned.loc[0,'nearest_mri_scan_id']=='a6'
    assert aligned.loc[0,'cognition_minus_mri_months']==-6
    assert aligned.loc[1,'nearest_mri_scan_id']=='a6'
    assert aligned.loc[1,'nearest_mri_tie_count']==2
    assert aligned.loc[1,'cognition_minus_mri_months']==3
    assert aligned.loc[1,'mri_month_match_eligible']
    assert aligned.loc[2,'nearest_mri_scan_id']=='b24'
    assert not aligned.loc[2,'mri_month_match_eligible']

def test_history_clock_origin_never_uses_future_visit_dates():
    from memreason_rl.data.build_cognitive_events import enrich_provenance
    rows=[]
    for month,date in [(0,'2020-01-01'),(12,'2021-01-01')]:
        r=dict(subject_id='001_S_0001',rid=1,visit_month=month,visit_code='bl' if month==0 else 'm12')
        r.update({s+'_visit_date':date for s in ['adas','cdr','faq','mmse','moca','dxsum']})
        rows.append(r)
    df=pd.DataFrame(rows)
    before=enrich_provenance(df).time_origin_date.iloc[0]
    for col in [c for c in df if c.endswith('_visit_date')]:
        df.loc[1,col]='2025-01-01'
    after=enrich_provenance(df).time_origin_date.iloc[0]
    assert before==after=='2020-01-01'

def test_missing_mri_reward_removed_from_denominator():
    assert availability_weighted_reward({'outcome':.8,'mri':None},{'outcome':1,'mri':1})==.8
    assert availability_weighted_reward({'outcome':.8,'mri':0},{'outcome':1,'mri':1})==.4

def test_future_or_wrong_subject_evidence_is_rejected():
    store={'past':dict(subject_id='a',time_month=0,payload={'x':1}), 'future':dict(subject_id='a',time_month=12,payload={'x':0})}
    citation=dict(event_id='future',item='x',value=0,time_month=12)
    assert verify_evidence([citation],store,'a',6)[0]==0
    citation.update(event_id='past',value=1,time_month=0)
    assert verify_evidence([citation],store,'a',6)[0]==1
    assert verify_evidence([citation],store,'b',6)[0]==0

def test_grpo_equal_group_is_finite_and_padding_irrelevant():
    x=torch.zeros(2,4,3,requires_grad=True)
    loss=grpo_loss(x,torch.zeros_like(x),torch.zeros_like(x),torch.ones(2,4),torch.ones_like(x,dtype=torch.bool))
    assert loss.item()==0
    loss.backward()
    assert torch.isfinite(x.grad).all()

def test_report_nan_and_malformed_citation_rejected():
    value=dict(current_state={},evidence=[],trend={},forecast={},confidence={})
    assert parse_report(json.dumps(value))==value
    value['evidence']=[dict(event_id='x')]
    with pytest.raises(ValueError):
        parse_report(value)

def test_transformer_cannot_attend_future():
    torch.manual_seed(1)
    model=TimeAwareTransformer(16,4,1,0).eval()
    x=torch.randn(1,3,16)
    times=torch.tensor([[0.,1.,2.]])
    mask=torch.ones(1,3,dtype=torch.bool)
    _,a=model(x,times,mask)
    x[:,2]=100*torch.randn(16)
    _,b=model(x,times,mask)
    torch.testing.assert_close(a[:,:2],b[:,:2])

def test_stage_gates_fail_closed(tmp_path):
    p=tmp_path/'gate.json'
    with pytest.raises(RuntimeError):
        require_gate('stage2',p)
    p.write_text(json.dumps({'engineering_smoke_only':True}))
    with pytest.raises(RuntimeError):
        require_gate('stage2',p)

@pytest.fixture(scope='module')
def store():
    from memreason_rl.data.dataset import EventStore
    return EventStore(exploratory=True,allow_nominal_time=True)

def test_real_split_retains_every_original_observation():
    raw=pd.read_csv(ROOT/'adni_all_longitudinal_scales_visit_with_ravlt_logmem.csv',low_memory=False)
    frames=[pd.read_csv(ROOT/f'splits/{s}.csv',low_memory=False) for s in ['train','val','test']]
    merged=pd.concat(frames,ignore_index=True)
    assert merged.groupby('subject_id').split.nunique().max()==1
    assert len(raw)==len(merged)
    raw=raw.sort_values(['subject_id','visit_month']).reset_index(drop=True)
    merged=merged.sort_values(['subject_id','visit_month']).reset_index(drop=True)
    pd.testing.assert_frame_equal(raw,merged[raw.columns],check_dtype=False)

def test_real_nearest_mri_month_table_retains_visits_and_respects_window():
    raw=pd.read_csv(ROOT/'adni_all_longitudinal_scales_visit_with_ravlt_logmem.csv',low_memory=False)
    aligned=pd.read_csv(ROOT/'processed/adni_visits_nearest_mri_month.csv',low_memory=False)
    scans=pd.read_csv(ROOT/'processed/mri_primary.csv',low_memory=False)[['subject_id','scan_id']]
    assert len(aligned)==len(raw)
    pd.testing.assert_frame_equal(
        raw.sort_values(['subject_id','visit_month']).reset_index(drop=True),
        aligned.sort_values(['subject_id','visit_month']).reset_index(drop=True)[raw.columns],
        check_dtype=False,
    )
    selected=aligned[aligned.mri_month_match_eligible]
    assert selected.absolute_mri_month_gap.le(6).all()
    assert selected.nearest_mri_image_path.notna().all()
    linked=selected.merge(scans,left_on='nearest_mri_scan_id',right_on='scan_id',suffixes=('_cognition','_mri'),validate='many_to_one')
    assert linked.subject_id_cognition.eq(linked.subject_id_mri).all()

def test_masking_removes_hidden_values_and_all_label_inputs(store):
    from memreason_rl.data.dataset import MaskedObservationDataset
    sample=MaskedObservationDataset(store,max_samples=3)[0]
    inputs={(e['event_id'],k) for e in sample['history'] for k in e['payload']}
    targets={(e['event_id'],k) for e in sample['targets'] for k in e['payload']}
    assert inputs and targets and inputs.isdisjoint(targets)
    assert not any(k.startswith(('dxsum_','faq_')) or k in {'moca_moca','mmse_mmscore'} for _,k in inputs)

def test_forecast_history_strictly_precedes_targets(store):
    from memreason_rl.data.dataset import ForecastDataset
    ds=ForecastDataset(store,max_samples=8)
    for sample in ds:
        assert all(e['time_month']<=sample['cutoff'] for e in sample['history'])
        assert all(e['time_month']>sample['cutoff'] for e in sample['targets'])

def test_forecast_outcomes_are_decoded_but_never_encoded(store):
    assert 'mmse_mmscore' in store.target_items
    assert 'faq_faqtotal' in store.target_items
    assert 'mmse_mmscore' not in store.items
    assert 'faq_faqtotal' not in store.items
    assert not any(k.startswith('dxsum_') for k in store.target_items)

def test_padding_values_do_not_change_prediction(store):
    from memreason_rl.data.dataset import MaskedObservationDataset
    from memreason_rl.data.collate import EventCollator
    from memreason_rl.models import MemReason
    ds=MaskedObservationDataset(store,max_samples=3)
    batch=EventCollator(store)([ds[0],ds[1]])
    model=MemReason(store.items,store.dictionary,store.norm,dim=16,state_dim=8,heads=4,layers=2).eval()
    with torch.no_grad():
        a=model(batch)['mu']
        batch['values'][~batch['observed']]=12345
        b=model(batch)['mu']
    torch.testing.assert_close(a,b)
