"""Inventory MRI files and attach only auditable image-level metadata."""
import re
from pathlib import Path
import pandas as pd
from .common import read_csv, subject_id, visit_code


def normalize_image_id(value):
    """Return the numeric ADNI Image ID as a canonical string."""
    if pd.isna(value):
        return None
    text = str(value).strip().lstrip('Ii')
    try:
        numeric = float(text)
    except ValueError:
        return text or None
    return str(int(numeric)) if numeric.is_integer() else text


def load_ida_image_metadata(path):
    """Load an IDA export keyed by Image ID and validate date/subject identity."""
    if not path:
        return {}
    path = Path(path)
    frame = read_csv(path, dtype={'Subject ID': str, 'Image ID': str})
    required = {'Subject ID', 'Study Date', 'Image ID'}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f'IDA metadata is missing required columns: {sorted(missing)}')
    frame = frame.copy()
    frame['image_id_key'] = frame['Image ID'].map(normalize_image_id)
    if frame.image_id_key.isna().any():
        raise ValueError('IDA metadata contains missing Image IDs')
    frame['subject_id_normalized'] = frame['Subject ID'].map(subject_id)
    frame['scan_date_parsed'] = pd.to_datetime(frame['Study Date'], format='%m/%d/%Y', errors='coerce')
    if frame.scan_date_parsed.isna().any():
        examples = frame.loc[frame.scan_date_parsed.isna(), 'Study Date'].head().tolist()
        raise ValueError(f'IDA metadata contains unparseable Study Date values: {examples}')
    conflicts = frame.groupby('image_id_key').agg(
        subjects=('subject_id_normalized', 'nunique'),
        dates=('scan_date_parsed', 'nunique'),
    )
    if ((conflicts.subjects > 1) | (conflicts.dates > 1)).any():
        raise ValueError('IDA Image IDs map to conflicting subjects or Study Dates')
    frame = frame.drop_duplicates('image_id_key', keep='first')
    return frame.set_index('image_id_key').to_dict('index')


def image_metadata(image_id, sid, lookup):
    key = normalize_image_id(image_id)
    info = lookup.get(key, {}) if key is not None else {}
    if info and info['subject_id_normalized'] != sid:
        raise ValueError(
            f'IDA subject mismatch for Image ID {key}: '
            f"{sid} != {info['subject_id_normalized']}"
        )
    return key, info

def build_mri_metadata(config):
    cfg = config['mri']
    source = read_csv(cfg['session_metadata'], dtype={'image_id':str})
    source['subject_id'] = source.subject_id.map(subject_id)
    source['image_id'] = source.image_id.map(normalize_image_id)
    source['visit_month'] = source.session_id.str.extract(r'M(\d+)')[0].astype(float)
    if source.duplicated(['subject_id','session_id']).any():
        raise ValueError('Conflicting session metadata; resolve before selecting MRI')
    lookup = source.set_index(['subject_id','session_id']).to_dict('index')
    image_lookup = {}
    for row in source.to_dict('records'):
        image_lookup.setdefault((row['subject_id'], row['image_id']), []).append(row)
    ida_path = cfg.get('image_metadata')
    ida_lookup = load_ida_image_metadata(ida_path)
    records = []
    root = Path(cfg['primary_root'])
    for path in sorted(root.glob('sub-*/ses-*/t1/affine.nii.gz')):
        sid, session = subject_id(path.parts[-4]), path.parts[-3]
        month = float(re.search(r'M(\d+)', session)[1])
        info = lookup.get((sid, session), {})
        image_id, ida = image_metadata(info.get('image_id'), sid, ida_lookup)
        scan_date = ida.get('scan_date_parsed')
        scan_date = scan_date.date().isoformat() if scan_date is not None else None
        records.append(dict(subject_id=sid, rid=int(sid[-4:]), scan_id=f'linear:{sid}:{session}',
            image_id=image_id, session_id=session, visit_month=month, visit_code=visit_code(month),
            scan_date=scan_date, scan_year=int(scan_date[:4]) if scan_date else None,
            time_source='ida_study_date' if scan_date else 'session_month', image_path=str(path),
            mask_path=str(path.with_name('mask.nii.gz')) if path.with_name('mask.nii.gz').is_file() else None,
            source='adni_linear', sequence_name='T1', field_strength=None, site=sid[:3], scanner=None,
            qc_pass=None, qc_status='unknown', primary=True, path_exists=True, file_size_bytes=path.stat().st_size,
            cohort=info.get('cohort'), metadata_link='exact_subject_session' if info else 'directory_session_only',
            metadata_source=cfg['session_metadata'], date_metadata_source=str(ida_path) if scan_date else None,
            ida_description=ida.get('Description'), ida_type=ida.get('Type'),
            preprocessing='affine filename; upstream provenance unverified'))
    for name, root in cfg.get('alternate_roots', {}).items():
        for path in sorted(Path(root).glob('*/brain.nii.gz')):
            match = re.fullmatch(r'(\d{3}_S_\d{4})-(\d{4})-I(\d+)', path.parent.name)
            if not match:
                continue
            sid, year, image = match.groups()
            sid = subject_id(sid)
            matches = image_lookup.get((sid, image), [])
            info = matches[0] if len(matches)==1 else {}
            month = info.get('visit_month')
            image_id, ida = image_metadata(image, sid, ida_lookup)
            scan_date = ida.get('scan_date_parsed')
            scan_date = scan_date.date().isoformat() if scan_date is not None else None
            records.append(dict(subject_id=sid, rid=int(sid[-4:]), scan_id=f'{name}:{path.parent.name}', image_id=image_id,
                session_id=info.get('session_id'), visit_month=month, visit_code=visit_code(month) if month is not None else None,
                scan_date=scan_date, scan_year=int(scan_date[:4]) if scan_date else int(year),
                time_source='ida_study_date' if scan_date else ('session_month' if info else 'year_only'),
                image_path=str(path), mask_path=None, source=name, sequence_name='T1', field_strength=None,
                site=sid[:3], scanner=None, qc_pass=None, qc_status='unknown', primary=False, path_exists=True,
                file_size_bytes=path.stat().st_size, cohort=info.get('cohort'),
                metadata_link='exact_subject_image_id' if info else 'unresolved_image_id',
                metadata_source=cfg['session_metadata'] if info else None,
                date_metadata_source=str(ida_path) if scan_date else None,
                ida_description=ida.get('Description'), ida_type=ida.get('Type'),
                preprocessing='brain filename; upstream provenance unverified'))
    mri = pd.DataFrame(records)
    if mri.empty:
        raise ValueError('No MRI files found; cannot create a real data split')
    if mri.image_path.duplicated().any() or mri.scan_id.duplicated().any():
        raise ValueError('Duplicate MRI inventory identity')
    # Alternates are retained for auditing, never substituted across preprocessing pipelines.
    return mri
