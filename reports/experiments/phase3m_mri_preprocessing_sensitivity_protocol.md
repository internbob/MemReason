# Phase 3M: Matched MRI Preprocessing Sensitivity

## Question

Does deterministic `160x192x160` center cropping preserve AD-vs-CN MRI
classification performance relative to the full `193x229x193` registered T1
grid, and how does it compare with the former interpolated `48x56x48` input?

## Frozen design

- Cohort: the 896 original-training subjects in Phase 3I, one earliest eligible
  scan per subject (CN/AD only).
- Labels and partitions: the exact Phase 3I labels, outer five folds, and inner
  tuning rule.
- Model: the same SFCN encoder, embedding classifier, optimizer, epoch budget,
  early stopping, temperature selection, threshold, and neural seed.
- Inputs: full 1-mm T1, fixed center crop without interpolation, and legacy
  low-resolution interpolation.
- Endpoints: ACC, AUC, sensitivity, specificity, and F1.
- Statistics: 2,000 paired subject-bootstrap replicates for each metric and
  preprocessing contrast; performance is also reported by foreground-mask
  retention quartile.
- Data isolation: development uses original-training subjects only. Official
  validation and test subjects remain unloaded.

## Prespecified gate

The center crop is retained for future MRI experiments when the lower 95%
confidence bound of `AUC(center crop) - AUC(full resolution)` is at least
`-0.02`. If it fails, the next preprocessing experiment must test an adaptive
brain-mask bounding box before further multimodal tuning.
