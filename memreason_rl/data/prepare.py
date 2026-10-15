"""Run with python -m memreason_rl.data.prepare; raw inputs are read-only."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from .common import ROOT, DAYS_PER_MONTH, SCALES, read_config, read_csv, save_csv, save_json, sha256, dates, valid_value
from .dictionary import build_dictionary
from .split_subjects import split_subjects
from .build_mri_metadata import build_mri_metadata
from .build_cognitive_events import enrich_provenance, build_cognitive_events
from .match_mri_cognition import match_mri_cognition
from .align_nearest_mri_month import (
    align_visits_to_nearest_mri_month,
    summarize_nearest_month_alignment,
)

def audit_columns(df, dictionary):
    records = []
    for c in df:
        values = df[c].dropna()
        numeric = pd.to_numeric(values, errors='coerce')
        invalid = sum(not valid_value(v,dictionary[c]) for v in values) if c in dictionary and not c.startswith('dxsum_') else 0
        records.append(dict(column=c, dtype=str(df[c].dtype), non_null=int(len(values)), missing=int(df[c].isna().sum()), n_unique=int(values.nunique()),
            minimum=float(numeric.min()) if numeric.notna().any() else None, maximum=float(numeric.max()) if numeric.notna().any() else None,
            unique_values=json.dumps(sorted(values.astype(str).unique().tolist())) if values.nunique()<=100 else None,
            invalid_or_missing_code_count=invalid, verified_from_dictionary=dictionary.get(c,{}).get('verified_from_dictionary'),
            role='state_candidate' if dictionary.get(c,{}).get('use_for_state') else 'outcome' if dictionary.get(c,{}).get('use_for_outcome') else 'metadata_or_excluded'))
    return pd.DataFrame(records)

def forecast_index(events, cfg):
    result = []
    histories = {sid:g[g.role.isin(['cognition','mri'])].time_month.to_numpy() for sid,g in events.groupby('subject_id')}
    for sid, frame in events[events.role!='mri'].groupby('subject_id', sort=True):
        cog = frame[(frame.role=='cognition') & (frame.state_payload_json!='{}')]
        if cog.empty:
            continue
        cutoffs = cog.groupby('visit_id').time_month.max().sort_values()
        # No score imputations: every target points to a genuinely observed instrument event.
        for vid, cutoff in cutoffs.items():
            future = frame[frame.time_month > cutoff]
            for horizon in cfg['horizons_months']:
                tolerance = cfg['tolerance_months'][horizon]
                candidates = future[(future.time_month-cutoff-horizon).abs()<=tolerance].copy()
                if candidates.empty:
                    continue
                candidates['_gap'] = (candidates.time_month-cutoff-horizon).abs()
                selected = candidates.sort_values(['_gap','time_month','event_id']).drop_duplicates('instrument')
                # Diagnostic labels are separate validity targets, not forecast inputs.
                selected = selected[selected.instrument!='DXSUM']
                if selected.empty:
                    continue
                result.append(dict(sample_id=f'{vid}:h{horizon}', subject_id=sid, split=frame.split.iloc[0],
                    cutoff_visit_id=vid, cutoff_month=float(cutoff), horizon_months=horizon,
                    target_event_ids_json=json.dumps(selected.event_id.tolist()),
                    n_history_events=int((histories[sid]<=cutoff).sum()),
                    target_time_min=float(selected.time_month.min()), target_time_max=float(selected.time_month.max())))
    return pd.DataFrame(result)

def main(config_path=None):
    cfg = read_config(config_path)
    source = Path(cfg['cognition_csv'])
    if not source.is_absolute():
        source = ROOT/source
    raw = read_csv(source)
    if raw.duplicated(['subject_id','visit_month']).any() or raw.subject_id.isna().any():
        raise ValueError('Non-unique or missing subject/visit identity')
    if (raw.groupby('subject_id').rid.nunique()>1).any() or (raw.groupby('rid').subject_id.nunique()>1).any():
        raise ValueError('RID and subject_id do not form a one-to-one mapping')
    print(f'Loaded {len(raw)} visits / {raw.subject_id.nunique()} subjects', flush=True)
    dictionary_path = ROOT/'configs/cognitive_item_dictionary.yaml'
    if dictionary_path.exists():
        dictionary = yaml.safe_load(dictionary_path.read_text())['items']
        if set(dictionary)!=set(build_dictionary(raw.columns)):
            raise ValueError('Input schema changed; review dictionary before rebuilding')
    else:
        dictionary = build_dictionary(raw.columns)
        dictionary_path.write_text(yaml.safe_dump(dict(schema_version=1, status='provisional_requires_dictionary_verification',items=dictionary), sort_keys=False, allow_unicode=True))
    save_csv(audit_columns(raw,dictionary), ROOT/'reports/data_audit/item_stats.csv')
    availability = pd.DataFrame(index=raw.index)
    for scale in SCALES:
        cols = [c for c in dictionary if c.startswith(f'{scale}_')]
        availability[scale] = raw[cols].notna().any(axis=1)
    availability['ravlt'] = raw[[c for c in raw if c.startswith('AVLT_')]].notna().any(axis=1)
    availability['logmem'] = raw[[c for c in raw if c.startswith('LogMem')]].notna().any(axis=1)
    save_csv((availability.astype(int).T@availability.astype(int)).rename_axis('instrument').reset_index(), ROOT/'reports/data_audit/scale_cooccurrence.csv')
    save_csv(availability.sum().rename('available_visits').rename_axis('instrument').reset_index(), ROOT/'reports/data_audit/scale_availability.csv')
    print('Inventorying MRI paths', flush=True)
    mri = build_mri_metadata(cfg)
    primary = mri[mri.primary]
    subjects = split_subjects(raw.subject_id, cfg['seed'], cfg['split_fractions'])
    split_map = subjects.set_index('subject_id').split.to_dict()
    mri['split'] = mri.subject_id.map(split_map).fillna('outside_cognitive_cohort')
    subjects['n_visits'] = subjects.subject_id.map(raw.groupby('subject_id').size()).fillna(0).astype(int)
    subjects['n_mri'] = subjects.subject_id.map(primary.groupby('subject_id').size()).fillna(0).astype(int)
    subjects['has_mri'] = subjects.n_mri>0
    subjects['longitudinal'] = subjects.n_visits>=2
    save_csv(subjects, ROOT/'splits/subjects.csv')
    save_csv(mri, ROOT/'processed/mri_metadata.csv')
    save_csv(mri[mri.primary], ROOT/'processed/mri_primary.csv')
    save_csv(mri[~mri.primary], ROOT/'processed/mri_alternates.csv')
    comparison = mri.groupby('source').agg(scans=('scan_id','size'), subjects=('subject_id','nunique'),
        scans_with_subject_session_mapping=('visit_month','count'), true_scan_dates=('scan_date','count')).reset_index()
    comparison['subjects_overlapping_cognition'] = comparison.source.map(mri[mri.subject_id.isin(raw.subject_id)].groupby('source').subject_id.nunique()).fillna(0).astype(int)
    save_csv(comparison, ROOT/'reports/data_audit/mri_root_comparison.csv')
    date_coverage = mri.groupby(['source', 'primary'], dropna=False).agg(
        scans=('scan_id', 'size'),
        scans_with_image_id=('image_id', 'count'),
        scans_with_true_date=('scan_date', 'count'),
        subjects=('subject_id', 'nunique'),
    ).reset_index()
    date_coverage['true_date_fraction'] = date_coverage.scans_with_true_date / date_coverage.scans
    save_csv(date_coverage, ROOT/'reports/data_audit/mri_date_coverage.csv')
    df = enrich_provenance(raw, Path(cfg['item_provenance_csv']), Path(cfg['clinical_source_dir']))
    df['split'] = df.subject_id.map(split_map)
    nearest_window = cfg['mri'].get('nearest_month_window', 6)
    nearest_month = align_visits_to_nearest_mri_month(df, mri, nearest_window)
    save_csv(nearest_month, ROOT/'processed/adni_visits_nearest_mri_month.csv')
    save_csv(
        nearest_month[nearest_month.mri_month_match_eligible],
        ROOT/f'processed/adni_visits_nearest_mri_month_within_{nearest_window}m.csv',
    )
    save_csv(
        summarize_nearest_month_alignment(nearest_month),
        ROOT/'reports/matching/nearest_mri_month_stats.csv',
    )
    save_csv(
        nearest_month.groupby(['split', 'mri_month_match_class'], dropna=False)
        .size().rename('visits').reset_index(),
        ROOT/'reports/matching/nearest_mri_month_class_counts.csv',
    )
    matches = match_mri_cognition(df,mri,cfg['mri']['matching_window_days'],cfg['mri']['tau_days'])
    save_csv(matches, ROOT/'reports/matching/mri_cognition_matching.csv')
    stats = []
    for window in cfg['mri']['sensitivity_days']:
        window_matches = match_mri_cognition(df,mri,window,cfg['mri']['tau_days'])
        matched_scans = window_matches.scan_id.nunique()
        verified = window_matches.date_difference_days.notna()
        stats.append(dict(window_days=window, same_visit_pairs=int((window_matches.match_strategy=='same_visit_code').sum()),
            true_date_pairs_evaluable=int(verified.sum()), true_date_pairs_within_window=int((verified & (window_matches.date_difference_days<=window)).sum()),
            matched_pairs=len(window_matches), matched_mri=matched_scans, unmatched_cognition_visits=len(df)-len(window_matches),
            unmatched_primary_mri=len(primary)-matched_scans, sensitivity_status='not_evaluable_without_scan_dates' if not verified.any() else 'evaluated',
            median_gap_days=float(window_matches.date_difference_days.median()) if verified.any() else None))
    save_csv(pd.DataFrame(stats),ROOT/'reports/mri_cognition_matching_stats.csv')
    wide = df.merge(matches.drop(columns='subject_id'),on='visit_id',how='left',validate='one_to_one')
    wide['mri_available'] = wide.mri_available.eq(True)
    wide['subject_n_mri'] = wide.subject_id.map(subjects.set_index('subject_id').n_mri)
    wide['subject_n_visits'] = wide.subject_id.map(subjects.set_index('subject_id').n_visits)
    save_csv(wide,ROOT/'processed/adni_visits_with_mri.csv')
    print('Building asynchronous cognitive events',flush=True)
    events, invalid = build_cognitive_events(df,dictionary)
    save_csv(invalid,ROOT/'reports/data_audit/invalid_observations.csv')
    mri_events=[]
    origin_by_subject = pd.to_datetime(
        df.groupby('subject_id').time_origin_date.first(), errors='coerce'
    ).to_dict()
    for scan in mri[mri.primary & mri.subject_id.isin(raw.subject_id)].to_dict('records'):
        scan_date = pd.to_datetime(scan.get('scan_date'), errors='coerce')
        origin = origin_by_subject.get(scan['subject_id'])
        has_true_time = pd.notna(scan_date) and pd.notna(origin)
        event_time = float((scan_date-origin).days/DAYS_PER_MONTH) if has_true_time else scan['visit_month']
        mri_events.append(dict(event_id=scan['scan_id'],subject_id=scan['subject_id'],rid=scan['rid'],visit_id=f"{scan['subject_id']}:{scan['visit_code']}",visit_code=scan['visit_code'],visit_month=scan['visit_month'],
            time_month=event_time,event_date=scan_date.date().isoformat() if has_true_time else None,
            time_source='ida_study_date' if has_true_time else 'nominal_session_month',event_type='MRI_T1',instrument='MRI',role='mri',
            payload_json='{}',state_payload_json='{}',image_path=scan['image_path'],scan_id=scan['scan_id'],split=scan['split'],phase=scan['cohort'],main_eligible=False))
    events=pd.concat([events,pd.DataFrame(mri_events)],ignore_index=True).sort_values(['subject_id','time_month','event_id']).reset_index(drop=True)
    if events.event_id.duplicated().any():
        raise ValueError('Duplicate event identity')
    save_csv(events,ROOT/'processed/events.csv')
    # Train-only numeric statistics, never fit on validation/test observations.
    normalizer={}
    for c,e in dictionary.items():
        if c.startswith('dxsum_'):
            continue
        values=pd.to_numeric(raw.loc[raw.subject_id.map(split_map).eq('train'),c],errors='coerce')
        values=values[values.map(lambda v: valid_value(v,e))]
        normalizer[c]=dict(mean=float(values.mean()) if len(values) else 0.,std=max(float(values.std(ddof=0)),1e-6) if len(values) else 1.,n=int(len(values)))
    save_json({'fit_split':'train','items':normalizer},ROOT/'processed/train_normalization.json')
    print('Building future-target indices',flush=True)
    forecasts=forecast_index(events,cfg['forecast'])
    save_csv(forecasts,ROOT/'processed/forecast_samples.csv')
    split_stats=[]
    for split in ['train','val','test']:
        w=wide[wide.split==split]
        s=subjects[subjects.split==split]
        save_csv(w,ROOT/f'splits/{split}.csv')
        save_csv(w[w.mri_available],ROOT/f'splits/{split}_mri_matched.csv')
        save_csv(w[(w.subject_n_mri>0)&(w.subject_n_visits>=2)],ROOT/f'splits/{split}_longitudinal_mri_subjects.csv')
        save_csv(events[events.split==split],ROOT/f'splits/{split}_events.csv')
        save_csv(forecasts[forecasts.split==split],ROOT/f'splits/{split}_forecast.csv')
        save_csv(mri[(mri.split==split)&mri.primary],ROOT/f'splits/{split}_mri.csv')
        nearest_split = nearest_month[nearest_month.split == split]
        save_csv(
            nearest_split,
            ROOT/f'splits/{split}_nearest_mri_month_all.csv',
        )
        save_csv(
            nearest_split[nearest_split.mri_month_match_eligible],
            ROOT/f'splits/{split}_nearest_mri_month.csv',
        )
        split_stats.append(dict(split=split,subjects=len(s),visits=len(w),subjects_with_mri=int(s.has_mri.sum()),longitudinal_subjects=int(s.longitudinal.sum()),
            primary_mri=int(s.n_mri.sum()),matched_visits=int(w.mri_available.sum()),nearest_mri_month_visits=int(nearest_split.mri_month_match_eligible.sum()),forecast_samples=int((forecasts.split==split).sum())))
    save_csv(pd.DataFrame(split_stats),ROOT/'reports/data_audit/split_summary.csv')
    followup=raw.groupby('subject_id').agg(n_visits=('visit_month','size'),first_month=('visit_month','min'),last_month=('visit_month','max')).reset_index()
    followup['span_months']=followup.last_month-followup.first_month
    followup['n_mri']=followup.subject_id.map(primary.groupby('subject_id').size()).fillna(0).astype(int)
    save_csv(followup,ROOT/'reports/data_audit/followup_distribution.csv')
    save_csv(events.groupby(['instrument','time_source'],dropna=False).size().rename('events').reset_index(),ROOT/'reports/data_audit/event_time_sources.csv')
    save_csv(events.groupby(['instrument','phase'],dropna=False).size().rename('events').reset_index(),ROOT/'reports/data_audit/phase_distribution.csv')
    # Header checks are a preprocessing audit, never official MRI QC.
    import nibabel as nib
    manifest=[]
    for name,g in mri.groupby('source'):
        p=g.image_path.iloc[0]
        img=nib.load(p)
        manifest.append(dict(source=name,sampled_path=p,shape=list(img.shape),spacing=[float(x) for x in img.header.get_zooms()],orientation=list(nib.aff2axcodes(img.affine)),
            scope='one_header_per_source; not full QC',skull_stripping='unverified',bias_correction='unverified',clinical_qc='unknown'))
    save_json({'sources':manifest,'online_preprocessing':{'orientation':'RAS','spacing_mm':[2.,2.,2.],'crop_pad':[96,112,96],'intensity':'finite foreground z-score','upstream_corrections':'not claimed'}},ROOT/'processed/preprocessing_manifest.json')
    dated = mri.scan_date.notna()
    expected_date_source = str(Path(cfg['mri']['image_metadata']))
    checks={'subject_split_disjoint':subjects.subject_id.is_unique,'every_visit_retained':len(wide)==len(raw),
            'no_missing_value_imputation':True,'original_sha256':sha256(source),
            'all_primary_paths_exist':bool(mri[mri.primary].path_exists.all()),
            'all_scan_dates_have_auditable_ida_source':bool((mri.loc[dated, 'date_metadata_source']==expected_date_source).all()),
            'primary_scans_with_true_dates':int(mri.loc[mri.primary, 'scan_date'].notna().sum()),
            'primary_scans_without_true_dates':int(mri.loc[mri.primary, 'scan_date'].isna().sum()),
            'events_unique':bool(events.event_id.is_unique),'forecast_targets_strictly_future':bool((forecasts.target_time_min>forecasts.cutoff_month).all()),
            'normalization_fit_split':'train','split_seed':cfg['seed'],'input_visits':len(raw),'input_subjects':raw.subject_id.nunique(),
            'matched_visits':len(matches),'primary_scans_in_cognitive_cohort':len(mri_events),'invalid_observations':len(invalid)}
    checks['nearest_month_all_visits_retained'] = len(nearest_month) == len(raw)
    checks['nearest_month_window'] = nearest_window
    checks['nearest_month_eligible_visits'] = int(nearest_month.mri_month_match_eligible.sum())
    save_json(checks,ROOT/'reports/data_audit/validation.json')
    input_paths=[source,Path(cfg['mri']['session_metadata']),Path(cfg['mri']['image_metadata']),dictionary_path,Path(config_path) if config_path else ROOT/'configs/data.yaml',Path(cfg['item_provenance_csv'])]
    input_paths.extend(sorted(Path(cfg['clinical_source_dir']).glob('*_*.csv')))
    save_json({'inputs':{str(p):sha256(p) for p in input_paths if p.is_file()},'split_algorithm':'sorted subjects; numpy.default_rng(seed).shuffle; floor 70%,floor 10%,remainder 20%'},ROOT/'reports/data_audit/provenance.json')
    primary_true_dates = int(primary.scan_date.notna().sum())
    primary_missing_dates = int(primary.scan_date.isna().sum())
    summary=f'''# MemReason Phase 0 数据审计

原始数据：{len(raw)} visits，{raw.subject_id.nunique()} subjects，{len(raw.columns)} columns。原始 CSV 未修改；缺失值未填补。

## MRI 数据源

主目录：`{cfg['mri']['primary_root']}`。全目录 {len(primary)} scans；其中 {primary_true_dates} scans 通过 Image ID 与 IDA Study Date 精确连接，{primary_missing_dates} scans 因缺少或未命中 Image ID 仍无真实日期；认知队列内 {len(mri_events)} scans；90 天主协议匹配 {len(matches)} visits。

详见 `mri_root_comparison.csv`。保留全部认知访视；MRI 不作为纳入认知队列的条件。

## Split

按 subject 随机划分，seed={cfg['seed']}，70/10/20。所有 visits、events、MRI、future targets 随 subject 归属。主文件 `splits/train.csv`、`val.csv`、`test.csv` 保留原始 161 列及真实 MRI 路径。`*_mri_matched.csv` 是同步匹配子集，不能取代全部认知队列。

{pd.DataFrame(split_stats).to_string(index=False)}

## 必须保留的限制

- IDA 导出 `{cfg['mri']['image_metadata']}` 提供 Image ID 级 `Study Date`；仅在 Image ID 和 subject 同时一致时写入 `scan_date`。真实日期覆盖和 30/60/90/180 天匹配敏感性见 `mri_date_coverage.csv` 与 `mri_cognition_matching_stats.csv`。
- 未获得日期的 MRI 继续使用明确标记的 nominal session month。scanner、field strength 和临床 QC 仍未知，因此 `main_grounding_eligible` 继续为 false；MRI header 检查不等于 QC。
- ADAS/CDR/FAQ/MMSE/MoCA 的官方编码核验由独立脚本维护；重建 Phase 0 数据不会覆盖已核验字典。当前核验状态见 `reports/data_audit/dictionary_verification.md`；`verified_from_dictionary=false` 的字段不能进入正式语义实验。
- RAVLT Trial*Int 不按学习词数解释；原值保留在 CSV 和事件证据中，默认不进入 state。999/9999 及候选范围异常单独记录在 `invalid_observations.csv`，原始表不改写。
- FAQ、诊断、总分不进入 memory state encoder；CDR function items 只作 outcome。源文件 presence flags 保留，availability 另按实际 payload 统计。
- `*_userdate` 不作为采集日期；缺失真实日期时，time_source 标为 nominal。原始 ITEM 的可用测量日期按原合并规则补回，未创造日期。
- 原始 study phase 按 RID+instrument date 回溯，匹配不到仍未知。缺失模式预测 diagnosis 的检查见后续 leakage audit。

## Gate 状态

Phase 0 数据与 split 构建完成；{len(invalid)} 个观测被标记为候选范围异常/特殊码。Item 字典核验与 Phase 1B 的状态由独立报告记录；MRI、SFT 与 RL 仍须依规范通过各自实证 gate。
'''
    (ROOT/'reports/data_audit/summary.md').write_text(summary)
    print(json.dumps(checks,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path)
    main(parser.parse_args().config)
