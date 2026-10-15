# Phase 3B MRI representation validation protocol

Date frozen: 2026-10-14

## Question

Do the supplied aligned structural MRIs contain reproducible information about concurrent cognitive state, and does a learned volumetric representation add information about 12/24-month cognitive change after controlling for current score, age, and sex?

Phase 3A showed that coarse global MRI features did not improve the full cognitive-history forecaster. Phase 3B separates MRI representation quality from longitudinal fusion. It is a new validation-only development experiment and does not load the test split.

## Cohort and temporal contract

- Use the exact train/validation MRI anchors that passed the Phase 3A real-date and project technical-QC contract.
- For concurrent prediction, an outcome is eligible only when its instrument date is within 90 days of the MRI Study Date.
- For future-change prediction, use 12- and 24-month observed outcomes strictly after the scan and require an observed concurrent score for the same target.
- Normalize outcomes, demographics, and MRI features with training subjects only.
- Keep subjects disjoint between train and validation. Do not read `splits/test.csv`.

## Representations

1. `demographics_only`: age and sex.
2. `global_mri`: Phase 3A whole-brain shape, mask occupancy, and coarse intensity features plus demographics.
3. `sfcn_3d`: a compact 3D convolutional encoder trained directly from masked, foreground-normalized aligned MRI volumes downsampled to 48×56×48, plus demographics.

ROI morphometry is not included because the supplied directory contains only `affine.nii.gz` and `mask.nii.gz`, with no audited subject-level anatomical segmentation. Atlas labels will not be fabricated from filenames or coarse grids.

Each learned model runs seeds 2026, 2027, and 2028. Every MRI model is also evaluated after replacing each validation MRI with an MRI from a different validation subject. This subject-shuffle evaluation uses the same trained checkpoint and unchanged labels.

## Primary task: concurrent cognition

Predict ADAS13, CDR-SB, FAQ, MMSE, and MoCA from each MRI anchor. Optimize equal-target normalized Huber loss and select checkpoints by validation macro normalized MAE. Report three-seed ensembles, endpoint metrics, and 2,000-replicate subject-cluster bootstrap intervals.

A representation passes the concurrent signal criterion only if it:

1. improves overall normalized absolute error over `demographics_only` with the 95% CI upper bound below zero;
2. improves over its own subject-shuffled evaluation with the 95% CI upper bound below zero; and
3. significantly improves at least two of five endpoints over demographics.

## Secondary task: future cognitive change

Fit fixed-alpha ridge heads separately for each target and horizon. The control uses current score, age, and sex. MRI arms add either train-normalized global features or frozen SFCN embeddings. SFCN ridge predictions are ensembled across the three independently trained encoders. Report correctly paired and subject-shuffled evaluations.

Future-change evidence requires an overall 95% CI upper bound below zero versus the no-MRI control and a supported improvement at either 12 or 24 months. This secondary result is interpreted only together with the concurrent signal test.

## Decision

The experiment determines whether MRI representation learning is scientifically viable on the current cohort. Failure does not block engineering work, but it prevents a claim that the current MRI representation supports spatial cognitive grounding. Passing concurrent but not future-change criteria supports concurrent MRI pretraining only; passing both supports a separately frozen spatial-grounding protocol.
