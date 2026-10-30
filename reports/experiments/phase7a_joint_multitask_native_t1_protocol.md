# Phase 7A frozen protocol: native-T1 diagnosis and Memory multitask learning

## Status and question

This protocol is frozen before fitting Phase 7A. Phase 7A is a new development
branch; it does not reinterpret the negative Phase 6D1 gate and does not reopen
Phase 6D2 or Phase 6E. The question is whether end-to-end shared MRI training
allows continuous Memory supervision to help CN/MCI/AD diagnosis and allows
diagnosis supervision to help future observed Memory prediction.

Only original-train subjects are eligible. Official validation and test data are
not loaded. All scans, visits and targets from one participant remain in one
outer fold.

## Cohort and timing

The MRI registry is the native `193 x 229 x 193`, 1-mm longitudinal T1 registry
used by Phase 6D0. A scan is eligible when it maps to an exact original-train
visit with a CN, MCI or AD label and an observed Memory composite. The Memory
composite is reconstructed from the frozen Phase 5C item contract. Diagnosis is
never used as a regression input.

For each scan cutoff, the current Memory target is the composite at the same
visit. Future targets are the nearest strictly later observed composites around
6/12/24/36 months, within frozen tolerances of 3/3/6/6 months. Cognition-history
features use only observations at or before the scan cutoff: last value, mean,
per-year slope, log count and recency. Target normalization and history
standardization are fitted on the outer-fit participants only.

## Splits and leakage controls

The five Phase 5G/6 subject folds are retained. Within each outer-fit set, 12%
of participants are stratified into an inner development set using their modal
diagnosis. A fold-specific Phase 6D0 encoder may initialize the backbone only
because that encoder excluded every participant in the same Phase 7A evaluation
fold. No evaluation participant may enter encoder warm-start fitting, current
normalization, history normalization, class weighting, early stopping or model
selection.

## Model and arms

All neural arms use the same 3D SFCN encoder and 128-dimensional representation.
The diagnosis head outputs CN/MCI/AD logits. The current-Memory head outputs a
Gaussian mean and variance. The future head fuses the MRI representation with
the five cutoff-safe cognition-history features and outputs Gaussian parameters
for all four horizons.

Three independently trained arms are required in every fold:

1. `classification_only`: weighted three-class cross-entropy only;
2. `regression_only`: current and future Memory Gaussian NLL only;
3. `joint_multitask`: the sum of classification, current-Memory and future-Memory losses.

An inference-only `joint_subject_shuffled_mri` control permutes MRI volumes
between evaluation participants while retaining their labels, histories and
targets. It is not used for checkpoint selection.

Participants contribute equal total mass regardless of scan count. Training
draws one scan per fit participant per epoch. Class weights are estimated from
outer-fit participant mass. Anatomically unsafe flips, rotations, resampling,
cropping, padding, MixUp and CutMix are prohibited.

## Optimization and selection

The SFCN is initialized from the leakage-safe Phase 6D0 fold checkpoint, then
fully unfrozen. AdamW uses learning rates `1e-4` for the encoder and `5e-4` for
new heads, weight decay `1e-4`, bfloat16 autocast, gradient accumulation of two
four-volume batches and gradient clipping at 1.0. Maximum training is 16 epochs
with patience four. Each arm selects the minimum matching inner-development
objective; the evaluation fold is never consulted.

## Endpoints

Diagnosis endpoints are subject-balanced ACC, macro one-vs-rest AUC, macro SEN,
macro SPE, macro F1, and class-specific sensitivity/specificity/F1. Regression
endpoints are subject-balanced observed Memory MAE, Gaussian NLL and 95%
coverage overall and by horizon. Participant-cluster bootstrap uses 2,000
replicates.

The primary classification contrast is `joint_multitask - classification_only`
macro AUC. The primary regression contrast is `joint_multitask - regression_only`
future observed Memory MAE. Mutual benefit is supported only if the AUC point
difference is nonnegative with a 95% lower bound above -0.01, future MAE has an
upper 95% bound below zero, future MAE improves in at least four of five folds,
at least three of four horizons are noninferior, and joint 95% coverage lies in
0.90--0.98. Failure is reported without threshold changes.

## Artifacts

Required artifacts are the frozen configuration/protocol hashes, cohort and
fold leakage audits, 15 selected checkpoints, training curves, out-of-fold
predictions, classification and regression metrics, per-class/per-horizon
tables, bootstrap contrasts, decision JSON, report and independent verification.
Participant rows, native arrays and checkpoints remain git-ignored. The approximately
72.7 GB float16 native-array cache is held on project scratch storage rather than in
the repository or its Git history. Identical runtime copies may be staged on each
allocated node's local temporary disk to prevent shared-filesystem I/O from idling the GPU.
