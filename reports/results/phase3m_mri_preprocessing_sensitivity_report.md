# Phase 3M matched MRI preprocessing sensitivity

This development-only experiment held the Phase 3I AD/CN cohort (896 subjects,
one earliest scan each), labels, five subject folds, inner tuning rule, SFCN,
optimizer, epoch budget, and seed fixed. It changed only the MRI representation:
the complete `193x229x193` 1-mm grid, a no-interpolation `160x192x160` center
crop, or the legacy interpolated `48x56x48` input. Phase 3I full-resolution
out-of-fold predictions were reused; the other two branches were newly trained.

An initial crop run was rejected after a post-hoc consistency check found that
the old Phase 3K cache contained 648 all-zero volumes among these 896 scans.
The result below uses a newly generated compact cache: every crop is the exact
center subarray of its audited Phase 3I volume, and all 896 rows passed a full
nonempty audit. The rejected attempt is recorded separately and is not included
in any metric below.

## Cross-fitted results

| Arm | ACC | AUC | SEN | SEC | F1 |
|---|---:|---:|---:|---:|---:|
| fullres_mri | 0.7221 | 0.7986 | 0.7074 | 0.7336 | 0.6907 |
| center_crop_mri | 0.7243 | 0.7960 | 0.7023 | 0.7416 | 0.6909 |
| lowres_mri | 0.7667 | 0.8135 | 0.7354 | 0.7913 | 0.7344 |
| demographics_fullres_mri | 0.7210 | 0.7997 | 0.6997 | 0.7376 | 0.6875 |
| demographics_center_crop_mri | 0.7277 | 0.7954 | 0.6997 | 0.7495 | 0.6927 |
| demographics_lowres_mri | 0.7623 | 0.8184 | 0.7303 | 0.7873 | 0.7294 |

All intervals in `classification_metric_intervals.csv` use 2,000 paired
subject-bootstrap replicates. The center-crop minus full-resolution MRI-only AUC
difference was **-0.0026** (95% CI
**[-0.0287, +0.0233]**). The prespecified
non-inferiority margin was -0.02; gate passed: **false**.

The crop retained at least 0.98437 of the foreground mask in all
896 scans. `mask_retention_strata_metrics.csv` tests whether performance varies
over retention quartiles. `computational_profile.csv` records voxel counts,
cache sizes, and newly instrumented branch resources.

## Decision

Future MRI experiments should use
`evaluate_adaptive_brain_mask_bounding_box`. This comparison used original-training
subjects only; official validation and test subjects were not loaded.
