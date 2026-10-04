"""Inventory files without inferring dates/QC from filenames or NIfTI headers."""
import re
from pathlib import Path
import pandas as pd
from .common import read_csv, subject_id, visit_code

def build_mri_metadata(config):
    cfg = config['mri']
    source = read_csv(cfg['session_metadata'], dtype={'image_id':str})
    source['subject_id'] = source.subject_id.map(subject_id)
    source['visit_month'] = source.session_id.str.extract(r'M(\d+)')[0].astype(float)
    if source.duplicated(['subject_id','session_id']).any():
        raise ValueError('Conflicting session metadata; resolve before selecting MRI')
    lookup = source.set_index(['subject_id','session_id']).to_dict('index')
    image_lookup = {}
    for row in source.to_dict('records'):
        image_lookup.setdefault((row['subject_id'], str(row['image_id'])), []).append(row)
    records = []
    root = Path(cfg['primary_root'])
    for path in sorted(root.glob('sub-*/ses-*/t1/affine.nii.gz')):
        sid, session = subject_id(path.parts[-4]), path.parts[-3]
        month = float(re.search(r'M(\d+)', session)[1])
        info = lookup.get((sid, session), {})
        records.append(dict(subject_id=sid, rid=int(sid[-4:]), scan_id=f'linear:{sid}:{session}',
            image_id=info.get('image_id'), session_id=session, visit_month=month, visit_code=visit_code(month),
            scan_date=None, scan_year=None, time_source='session_month', image_path=str(path),
            mask_path=str(path.with_name('mask.nii.gz')) if path.with_name('mask.nii.gz').is_file() else None,
            source='adni_linear', sequence_name='T1', field_strength=None, site=sid[:3], scanner=None,
            qc_pass=None, qc_status='unknown', primary=True, path_exists=True, file_size_bytes=path.stat().st_size,
            cohort=info.get('cohort'), metadata_link='exact_subject_session' if info else 'directory_session_only',
            metadata_source=cfg['session_metadata'], preprocessing='affine filename; upstream provenance unverified'))
    for name, root in cfg.get('alternate_roots', {}).items():
        for path in sorted(Path(root).glob('*/brain.nii.gz')):
            match = re.fullmatch(r'(\d{3}_S_\d{4})-(\d{4})-I(\d+)', path.parent.name)
            if not match:
                continue
            sid, year, image = match.groups()
            matches = image_lookup.get((sid, image), [])
            info = matches[0] if len(matches)==1 else {}
            month = info.get('visit_month')
            records.append(dict(subject_id=sid, rid=int(sid[-4:]), scan_id=f'{name}:{path.parent.name}', image_id=image,
                session_id=info.get('session_id'), visit_month=month, visit_code=visit_code(month) if month is not None else None,
                scan_date=None, scan_year=int(year), time_source='session_month' if info else 'year_only',
                image_path=str(path), mask_path=None, source=name, sequence_name='T1', field_strength=None,
                site=sid[:3], scanner=None, qc_pass=None, qc_status='unknown', primary=False, path_exists=True,
                file_size_bytes=path.stat().st_size, cohort=info.get('cohort'),
                metadata_link='exact_subject_image_id' if info else 'unresolved_image_id',
                metadata_source=cfg['session_metadata'] if info else None,
                preprocessing='brain filename; upstream provenance unverified'))
    mri = pd.DataFrame(records)
    if mri.empty:
        raise ValueError('No MRI files found; cannot create a real data split')
    if mri.image_path.duplicated().any() or mri.scan_id.duplicated().any():
        raise ValueError('Duplicate MRI inventory identity')
    # Alternates are retained for auditing, never substituted across preprocessing pipelines.
    return mri
