"""Instrument events with individual dates; no synthetic missing instrument tokens."""
import json
import numpy as np
import pandas as pd
from .common import DAYS_PER_MONTH, SCALES, dates, valid_value

def enrich_provenance(df, item_source=None, clinical_dir=None):
    df = df.copy()
    for scale in SCALES:
        df[f'{scale}_phase'] = None
        df[f'{scale}_source_qc_error'] = None
        if clinical_dir is None:
            continue
        paths = sorted(clinical_dir.glob(f'{scale.upper()}_*.csv'))
        if not paths:
            continue
        raw = pd.read_csv(paths[-1], low_memory=False)
        datecol = 'EXAMDATE' if scale == 'dxsum' else 'VISDATE'
        if datecol not in raw or 'RID' not in raw or 'PHASE' not in raw:
            continue
        raw['_date'] = dates(raw[datecol]).dt.strftime('%Y-%m-%d')
        # Match phase only on RID + actual instrument collection date.
        grouped = raw.dropna(subset=['_date']).groupby(['RID','_date'])
        phases = grouped.PHASE.agg(lambda x: '|'.join(sorted(set(x.dropna().astype(str)))))
        phase_map = phases.to_dict()
        df[f'{scale}_phase'] = [phase_map.get((r, d)) for r,d in zip(df.rid, dates(df[f'{scale}_visit_date']).dt.strftime('%Y-%m-%d'))]
        if 'HAS_QC_ERROR' in raw:
            q = grouped.HAS_QC_ERROR.agg(lambda x: '|'.join(sorted(set(x.dropna().astype(str))))).to_dict()
            df[f'{scale}_source_qc_error'] = [q.get((r,d)) for r,d in zip(df.rid, dates(df[f'{scale}_visit_date']).dt.strftime('%Y-%m-%d'))]
    if item_source is not None and item_source.is_file():
        header = pd.read_csv(item_source, nrows=0).columns
        cols = ['subject_id', 'visit_month'] + [c for c in header if c.endswith(('_ExamDate', '_QuestionnaireNotAttempted')) and c.startswith(('AVLT_', 'LogMem'))]
        # Reproduce the source row selection used by merge_ravlt_logmem.py, not a fresh arbitrary row.
        raw = pd.read_csv(item_source, low_memory=False)
        selected = [c for c in df if c.startswith(('AVLT_', 'LogMem'))]
        raw['_n_selected'] = raw[selected].notna().sum(axis=1)
        raw['_n_all'] = raw.drop(columns='_n_selected').notna().sum(axis=1)
        raw = raw.sort_values(['_n_selected', '_n_all'], ascending=False, kind='stable').drop_duplicates(['subject_id','visit_month'])
        df = df.merge(raw[cols], on=['subject_id','visit_month'], how='left', validate='one_to_one')
    datecols = [f'{s}_visit_date' for s in SCALES] + [c for c in df if c.endswith('_ExamDate')]
    dateframe = pd.DataFrame({c:dates(df[c]) for c in datecols})
    df['cognition_date_start'] = dateframe.min(axis=1).dt.strftime('%Y-%m-%d')
    df['cognition_date_end'] = dateframe.max(axis=1).dt.strftime('%Y-%m-%d')
    # Use only the earliest nominal visit to anchor the clock. A subject-wide
    # median would make historical time features depend on future visit dates.
    estimates = dateframe.min(axis=1) - pd.to_timedelta(df.visit_month * DAYS_PER_MONTH, unit='D')
    first_indices = df.sort_values(['subject_id','visit_month'],kind='stable').drop_duplicates('subject_id').index
    baseline = pd.Series(estimates.loc[first_indices].to_numpy(),index=df.loc[first_indices,'subject_id'])
    df['time_origin_date'] = df.subject_id.map(baseline).dt.strftime('%Y-%m-%d')
    df['visit_id'] = [f'{sid}:{code}' for sid,code in zip(df.subject_id,df.visit_code)]
    return df

def event_time(row, date_column):
    date = pd.to_datetime(row.get(date_column), errors='coerce')
    origin = pd.to_datetime(row.get('time_origin_date'), errors='coerce')
    if pd.notna(date) and pd.notna(origin):
        return float((date-origin).total_seconds()/86400/DAYS_PER_MONTH), str(date.date()), 'collection_date'
    return float(row['visit_month']), None, 'nominal_visit_month'

def build_cognitive_events(df, dictionary):
    groups = {s.upper():[c for c in dictionary if c.startswith(f'{s}_')] for s in SCALES}
    groups.update(AVLT=[c for c in dictionary if c.startswith('AVLT_Trial')],
        AVLT_DELAY=[c for c in dictionary if c.startswith('AVLT_Delay')],
        LOGMEM_IMMEDIATE=[c for c in dictionary if c.startswith('LogMemIA_')],
        LOGMEM_DELAYED=[c for c in dictionary if c.startswith('LogMemIIA_')])
    datecols = {s.upper():f'{s}_visit_date' for s in SCALES}
    datecols.update(AVLT='AVLT_ExamDate', AVLT_DELAY='AVLT_Delay_ExamDate', LOGMEM_IMMEDIATE='LogMemIA_ExamDate', LOGMEM_DELAYED='LogMemIIA_ExamDate')
    events, invalid = [], []
    for row in df.to_dict('records'):
        for inst, columns in groups.items():
            payload = {}
            for c in columns:
                value = row[c]
                if pd.isna(value):
                    continue
                if dictionary[c]['value_type']=='categorical' and c.startswith('dxsum_'):
                    payload[c] = value
                elif valid_value(value, dictionary[c]):
                    payload[c] = float(value)
                else:
                    invalid.append(dict(visit_id=row['visit_id'], subject_id=row['subject_id'], item=c, raw_value=value, reason='outside_candidate_range_or_missing_code'))
            if not payload:
                continue
            t, date, time_source = event_time(row, datecols[inst])
            role = 'outcome' if inst in ['FAQ','DXSUM'] else 'cognition'
            state_payload = {c:v for c,v in payload.items() if dictionary[c]['use_for_state'] and not dictionary[c]['default_drop']}
            attempted = {'AVLT':'AVLT_QuestionnaireNotAttempted', 'AVLT_DELAY':'AVLT_Delay_QuestionnaireNotAttempted', 'LOGMEM_IMMEDIATE':'LogMemIA_QuestionnaireNotAttempted','LOGMEM_DELAYED':'LogMemIIA_QuestionnaireNotAttempted'}.get(inst)
            if attempted and str(row.get(attempted)) in {'1', '1.0'}:
                state_payload = {}
            events.append(dict(event_id=f"{row['visit_id']}:{inst}", subject_id=row['subject_id'], rid=row['rid'], visit_id=row['visit_id'], visit_code=row['visit_code'], visit_month=row['visit_month'],
                time_month=t, event_date=date, time_source=time_source, event_type='COG_'+inst if role=='cognition' else inst+'_OUTCOME',
                instrument=inst, role=role, payload_json=json.dumps(payload,ensure_ascii=False), state_payload_json=json.dumps(state_payload),
                image_path=None, scan_id=None, split=row['split'], phase=row.get(inst.lower()+'_phase'),
                main_eligible=bool(state_payload) and all(dictionary[c]['verified_from_dictionary'] for c in state_payload) and time_source=='collection_date'))
    return pd.DataFrame(events), pd.DataFrame(invalid, columns=['visit_id','subject_id','item','raw_value','reason'])
