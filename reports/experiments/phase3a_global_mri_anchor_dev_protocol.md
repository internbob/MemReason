# Phase 3A global MRI-anchor development protocol

Date frozen: 2026-10-13

## Scientific question

Does a technically valid, real-date-matched global structural MRI anchor improve future cognitive forecasting beyond the confirmed cognitive GRU-D baseline, and does any apparent gain disappear when MRI identity or timing is deliberately corrupted?

This is a revised, validation-only MRI development protocol. The earlier universal semantic-superiority gate was not met, so this experiment does not revive that claim and does not evaluate the test split.

## Cohort and temporal contract

- Use only official train and validation subjects; test is never loaded.
- Use the same verified 68 scalar cognitive inputs and five future outcomes as the formal baseline suite.
- Retain only forecasting samples whose cutoff visit has a primary MRI with `date_window_verified=true` under the 90-day protocol.
- Require every retained outcome collection date to occur after the MRI Study Date.
- All comparison arms use exactly the same samples and targets.

## MRI representation and project QC

The inputs are the existing preprocessed `affine.nii.gz` and `mask.nii.gz` files. Project QC requires readable finite 3D data, matching image/mask shape, a nonempty nonconstant foreground, and a mask fraction between 0.01 and 0.80. This is technical QC of the supplied processed files, not an ADNI clinical-QC claim.

The first-stage global anchor contains train-normalized whole-brain volume/shape summaries and coarse aligned-grid mask/intensity summaries. It does not support a regional neuroanatomical-attention claim. Spatial question-to-anatomy learning is conditional on this global experiment.

## Arms

1. `cognition_only`: GRU-D history with the MRI channel disabled.
2. `observed_mri`: the correctly paired global MRI anchor.
3. `shuffled_subject_mri`: MRI features replaced by a different subject within the same split.
4. `shuffled_time_mri`: MRI features replaced by another available scan from the same subject; singleton subjects remain unchanged and the changed fraction is reported.

All four arms share the same architecture and trainable-parameter count. Each arm runs seeds 2026, 2027, and 2028.

## Outcomes and statistics

Primary metric: equal target/horizon macro normalized MAE on validation. Secondary metrics: raw MAE, RMSE, Spearman correlation, per-target results, and 6/12/24/36-month strata. Candidate comparisons use 2,000-replicate subject-cluster bootstrap intervals on paired normalized absolute-error differences. Negative differences favor the candidate.

## Frozen development gate

The global MRI anchor advances only if:

1. `observed_mri` improves over `cognition_only` with an overall 95% CI upper bound below 0;
2. `observed_mri` improves over both shuffled controls under the same criterion;
3. at least one of the 12- or 24-month comparisons against cognition-only has a 95% CI upper bound below 0; and
4. at least 95% of required scans pass project technical QC.

Failure stops progression to spatial grounding on the current validation cohort. Passing permits a separately frozen question-to-anatomy experiment; it does not itself establish clinical anatomical validity.
