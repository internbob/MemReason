# Phase 3N: Corrected Center-Cropped T1 CN/MCI/AD Classification

## Objective

Repeat the CN/MCI/AD diagnostic experiment after rebuilding every cropped MRI
directly from its original `ADNI_Linear` T1 and mask. The experiment replaces
the invalid Phase 3K MRI results, whose cache contained all-zero intermediate
rows.

## Frozen design

- Cohort: the exact Phase 3G/3K original-training cohort, comprising 1,328
  subjects and 5,370 MRI-linked diagnostic visits.
- MRI: `193x229x193`, 1-mm registered T1 and mask read from the original path;
  deterministic center crop to `160x192x160`, without interpolation or padding.
- Cache acceptance: all 5,370 rows must have the expected shape, finite values,
  and more than 100 nonzero foreground voxels. Boundary-only validation is
  forbidden.
- Splits: the exact Phase 3G subject-level five folds.
- Training: five folds, three seeds per fold, the frozen SFCN and optimization
  settings from Phase 3K.
- Primary arms: MRI only, cognition only, cognition plus MRI, and cognition plus
  subject-shuffled MRI.
- Metrics: subject-balanced ACC, macro AUC, macro sensitivity, macro specificity,
  and macro F1, each with 2,000 subject-cluster bootstrap replicates.
- Isolation: official validation and test subjects are not loaded.

## Interpretation

MRI diagnostic signal and incremental value beyond cognition remain separate
claims. The corrected experiment supersedes all MRI-dependent Phase 3K metrics.
