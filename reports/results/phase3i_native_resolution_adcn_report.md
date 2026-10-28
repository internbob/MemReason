# Phase 3I native-resolution AD-vs-CN MRI benchmark

This train-only, subject-wise five-fold experiment used one earliest eligible
MRI per subject. Every image retained the complete `193 x 229 x 193`, 1-mm
voxel grid. No interpolation, cropping, padding, or slice selection was used.
Float16 was used only for cache storage.

## Cohort

- Subjects/scans: 896
- CN: 503
- AD: 393
- Official validation/test loaded: no/no

## Cross-fitted results

| Arm | ACC | AUC | SEN | SEC | F1 |
|---|---:|---:|---:|---:|---:|
| demographics | 0.6016 | 0.6339 | 0.6209 | 0.5865 | 0.5775 |
| demographics_native_fullres_mri | 0.7210 | 0.7997 | 0.6997 | 0.7376 | 0.6875 |
| native_fullres_mri | 0.7221 | 0.7986 | 0.7074 | 0.7336 | 0.6907 |

The prespecified native-MRI sanity threshold was AUC >=
0.80. The observed primary
MRI-plus-demographics AUC was 0.7997; gate passed: `false`.
See `classification_metric_intervals.csv` for 2,000 subject-bootstrap confidence
intervals.
