# FreeSurfer MRI inventory audit

This inventory is an additive data source. It does not replace or modify
`processed/mri_primary.csv`.

## Coverage

| split | directories | subjects | image_eligible_scans | image_eligible_subjects | roi_eligible_scans | surface_eligible_scans | subjects_with_at_least_2_images | adjacent_pairs | adjacent_pairs_6_36_months | subjects_with_6_36_month_pairs | same_day_primary_matches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 376 | 72 | 374 | 72 | 368 | 369 | 71 | 302 | 148 | 63 | 310 |
| train | 291 | 57 | 290 | 57 | 284 | 285 | 56 | 233 | 113 | 49 | 242 |
| val | 25 | 4 | 25 | 4 | 25 | 25 | 4 | 21 | 13 | 4 | 21 |
| test | 60 | 11 | 59 | 11 | 59 | 59 | 11 | 48 | 22 | 10 | 47 |

## File coverage

| file | available_scans | total_scans | coverage_fraction |
| --- | --- | --- | --- |
| mri/brainmask.mgz | 374 | 376 | 0.9946808510638298 |
| mri/brainmask.auto.mgz | 374 | 376 | 0.9946808510638298 |
| mri/T1.mgz | 375 | 376 | 0.9973404255319149 |
| mri/brain.mgz | 369 | 376 | 0.9813829787234043 |
| mri/brain.finalsurfs.mgz | 369 | 376 | 0.9813829787234043 |
| mri/rawavg.mgz | 376 | 376 | 1.0 |
| mri/orig.mgz | 376 | 376 | 1.0 |
| mri/norm.mgz | 374 | 376 | 0.9946808510638298 |
| mri/aseg.mgz | 369 | 376 | 0.9813829787234043 |
| mri/aparc+aseg.mgz | 369 | 376 | 0.9813829787234043 |
| mri/aparc.DKTatlas+aseg.mgz | 369 | 376 | 0.9813829787234043 |
| mri/aparc.a2009s+aseg.mgz | 369 | 376 | 0.9813829787234043 |
| mri/wmparc.mgz | 369 | 376 | 0.9813829787234043 |
| stats/aseg.stats | 368 | 376 | 0.9787234042553191 |
| stats/lh.aparc.stats | 368 | 376 | 0.9787234042553191 |
| stats/rh.aparc.stats | 368 | 376 | 0.9787234042553191 |
| surf/lh.thickness | 369 | 376 | 0.9813829787234043 |
| surf/rh.thickness | 369 | 376 | 0.9813829787234043 |

`brainmask.mgz` is a FreeSurfer-conformed intensity image, not a binary label.
The mask contract for downstream loading is `brainmask.mgz > 0`.

## Missing brainmask files

| subject_id | image_id | freesurfer_directory_name |
| --- | --- | --- |
| 072_S_0315 | 366226 | 072_S_0315-2013-I366226 |
| 123_S_0113 | 362152 | 123_S_0113-2013-I362152 |

## Pair rule

`processed/mri_freesurfer_pairs.csv` contains consecutive, real-date scans
within each subject. `interval_6_36_months` marks the current longitudinal
analysis window. No registration, scanner-protocol, motion-artifact, or
clinical-QC exclusion is applied.
