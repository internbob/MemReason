from pathlib import Path
import hashlib
import json
import re
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
DAYS_PER_MONTH = 365.25 / 12
SCALES = ['adas', 'cdr', 'faq', 'mmse', 'moca', 'dxsum']
DOMAINS = ['Memory', 'Executive', 'Attention', 'Language', 'Orientation']

def read_config(path=None):
    return yaml.safe_load(Path(path or ROOT / 'configs/data.yaml').read_text())

def subject_id(value):
    match = re.search(r'(\d{3})_?S_?(\d{4})', str(value))
    if not match:
        raise ValueError(f'Unrecognized ADNI subject: {value!r}')
    return f'{match[1]}_S_{match[2]}'

def visit_code(month):
    return 'bl' if float(month) == 0 else f'm{int(float(month)):02d}'

def read_csv(path, **kwargs):
    return pd.read_csv(path, low_memory=False, na_values=['NULL', 'null', 'None'], **kwargs)

def save_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)

def save_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str) + '\n')

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def dates(series):
    # Only collection dates; USERDATE is a data-entry timestamp.
    return pd.to_datetime(series.astype(str).str[:10], errors='coerce')

def valid_value(value, entry):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(value) or value in entry.get('missing_codes', [-4, -1, 999]):
        return False
    if entry.get('valid_values') is not None:
        return value in entry['valid_values']
    bounds = entry.get('range')
    return bounds is None or bounds[0] <= value <= bounds[1]
