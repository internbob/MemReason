# Phase 3I: Native-resolution AD-vs-CN MRI sanity benchmark

## Purpose

This development experiment determines whether the modest MRI performance in
Phases 3G/3H is primarily caused by the former `48 x 56 x 48` interpolation or
by the intrinsically difficult MCI tasks. It uses the complete preprocessed
`193 x 229 x 193`, 1-mm affine T1 grid. No spatial interpolation, cropping,
padding, slice selection, or post-hoc subject exclusion is permitted.

## Frozen cohort and split

- Source: original training subjects from the Phase 3G concurrent diagnosis
  manifest; official validation and test subjects remain unopened.
- Endpoint: concurrent AD versus CN diagnosis within the already verified
  90-day scan/diagnosis window.
- Unit: one earliest eligible scan per subject, selected before fold creation.
- Split: five subject-wise stratified outer folds. The outer-training subjects
  are split again into fit and tuning subjects for early stopping and
  temperature selection.

## Inputs and arms

- `demographics`: age and sex.
- `native_fullres_mri`: a full-grid masked, foreground-z-normalized affine T1.
- `demographics_native_fullres_mri`: age, sex, and the native-resolution SFCN
  embedding.

The MRI cache is float16 storage only; tensors are converted to float32 or AMP
inside training. Float16 storage does not alter the voxel grid. Every cached
scan must match `193 x 229 x 193` and 1-mm isotropic spacing.

## Outcomes and decision

Report subject-level ACC, ROC AUC, sensitivity, specificity, and F1, with 2,000
subject-bootstrap confidence intervals. An MRI AUC of at least 0.80 opens the
three-seed native-resolution CN/MCI/AD and MCI-progression reruns. A lower AUC
triggers an MRI representation/preprocessing redesign before further fusion.

This is a development sanity benchmark, not an external-validation claim.
