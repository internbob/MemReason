"""Same-session joins and true-date matching are distinct evidence classes."""
import math
import pandas as pd

def match_mri_cognition(visits, mri, window_days=90, tau_days=90):
    groups = {sid:g.to_dict('records') for sid,g in mri[mri.primary].groupby('subject_id')}
    matches = []
    for row in visits.to_dict('records'):
        cdate = pd.to_datetime(row.get('cognition_date_start'), errors='coerce')
        candidates = []
        for scan in groups.get(row['subject_id'], []):
            same_visit = row['visit_code'] == scan['visit_code']
            sdate = pd.to_datetime(scan.get('scan_date'), errors='coerce')
            delta = abs((sdate-cdate).days) if pd.notna(sdate) and pd.notna(cdate) else None
            if not same_visit and (delta is None or delta > window_days):
                continue
            if scan.get('qc_pass') is False:
                continue
            rank = (not same_visit, scan.get('qc_pass') is not True, delta if delta is not None else float('inf'), scan['scan_id'])
            candidates.append((rank, scan, delta, same_visit))
        if candidates:
            _, scan, delta, same = min(candidates, key=lambda x:x[0])
            matches.append(dict(visit_id=row['visit_id'], subject_id=row['subject_id'], scan_id=scan['scan_id'],
                image_path=scan['image_path'], mask_path=scan['mask_path'], mri_visit_month=scan['visit_month'],
                scan_date=scan['scan_date'], match_strategy='same_visit_code' if same else 'nearest_true_date',
                date_difference_days=delta, time_weight=math.exp(-delta/tau_days) if delta is not None else None,
                date_window_verified=delta is not None and delta<=window_days, qc_pass=scan['qc_pass'],
                qc_status=scan['qc_status'], n_candidates=len(candidates), mri_available=True,
                main_grounding_eligible=delta is not None and delta<=window_days and scan.get('qc_pass') is True))
    columns = ['visit_id','subject_id','scan_id','image_path','mask_path','mri_visit_month','scan_date','match_strategy','date_difference_days','time_weight','date_window_verified','qc_pass','qc_status','n_candidates','mri_available','main_grounding_eligible']
    return pd.DataFrame(matches, columns=columns)
