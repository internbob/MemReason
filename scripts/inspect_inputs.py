"""Bounded metadata inspection; run on a compute node."""
import json
from pathlib import Path
import pandas as pd

base = Path(__file__).resolve().parents[1]
df = pd.read_csv(base / 'adni_all_longitudinal_scales_visit_with_ravlt_logmem.csv')
print('cognition', df.shape, df.subject_id.nunique(), 'duplicate visits', df.duplicated(['subject_id', 'visit_month']).sum())
for c in ['moca_delw1','adas_q13score','AVLT_Trial1Int','AVLT_Delay_Rec','LogMemIA_ImmediateScore','LogMemIIA_DelayedScore','LogMemIIA_ReminderGiven']:
    print(c, df[c].value_counts(dropna=False).head(14).to_dict())
for root, pattern in [
    ('/public_bme2/bme-dgshen/WeilinZhou/AD_longitudinal/ADNI/ADNI_Linear', 'sub-*/ses-*/t1/affine.nii.gz'),
    ('/public_bme/data/lianzf2024/UniSurf/surfGT/T1/ADNI/Step01_sMRI_data', '*/brain.nii.gz'),
    ('/public_bme/data/lianzf2024/UniSurf/Longitudinal/UniSurf/ADNI/Step01_sMRI_data', '*/brain.nii.gz')]:
    files = sorted(Path(root).glob(pattern))
    print(root, 'scans', len(files), 'example', str(files[0]) if files else '')
    if files:
        import nibabel as nib
        img = nib.load(files[0])
        print('shape', img.shape, 'spacing', img.header.get_zooms(), 'orientation', nib.aff2axcodes(img.affine))

