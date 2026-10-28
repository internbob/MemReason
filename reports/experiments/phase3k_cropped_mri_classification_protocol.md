# Phase 3K/L: Native-spacing cropped-T1 classification rerun

## Objective

This experiment replaces the former `48 x 56 x 48` interpolated SFCN input
with a deterministic `160 x 192 x 160` center crop of the registered
`193 x 229 x 193`, 1-mm ADNI_Linear T1. It evaluates CN/MCI/AD concurrent
classification and 24-/36-month MCI-to-AD progression.

## Frozen data and comparison contract

- Reuse the Phase 3G diagnostic cohort, labels, five subject folds, three
  seeds, inner-tuning rule, model architecture, and statistical definitions.
- Reuse the Phase 3H 24-/36-month cohorts, outcome definitions, subject folds,
  three seeds, and 0.5 classification threshold.
- Use original-train subjects only. Official validation and test are not
  loaded.
- Crop both T1 and mask with fixed slices `[16:176, 18:210, 16:176]` after
  canonical orientation. Preserve 1-mm spacing; do not interpolate or pad.
- Normalize retained mask foreground per scan and store the cache as float16;
  training tensors use float32/AMP.
- Retain demographics, MRI-only, cognition-only, cognition+MRI, and
  subject-shuffled MRI controls. The existing global-MRI arms remain secondary
  compatibility controls.

## Outcomes

Report subject-balanced macro ACC, one-vs-rest macro AUC, macro sensitivity,
macro specificity, and macro F1 for CN/MCI/AD. For MCI conversion report ACC,
ROC-AUC, conversion sensitivity, nonconversion specificity, and positive-class
F1 at 24 and 36 months. Use 2,000 subject-level bootstrap replicates and paired
AUC contrasts for incremental-value claims.

This is a resolution/preprocessing sensitivity experiment. It does not alter
the previously reported Phase 3G/3H results and does not open official test
evaluation.
