# Phase 3C concurrent question-to-spatial grounding protocol

Date frozen: 2026-10-14

## Question

Does target-conditioned attention over spatial MRI tokens improve concurrent cognitive prediction beyond global pooling, and does the prediction depend reproducibly on the correct target query and spatial arrangement?

Phase 3B established stable concurrent information in both global MRI features and a 3D SFCN. It did not support longitudinal MRI fusion. Phase 3C therefore evaluates concurrent spatial grounding only. It does not load the test split and does not claim future forecasting improvement.

## Cohort and frozen inputs

- Reuse the exact 5,684/780 train/validation MRI anchors from Phase 3B.
- Reuse its train-only normalization and 48×56×48 masked MRI cache.
- Load the three Phase 3B SFCN checkpoints and freeze the first four convolutional blocks.
- Extract a 3×4×3 grid of 64-channel spatial tokens for every MRI. No backbone parameter is updated in Phase 3C.
- Predict the same concurrent ADAS13, CDR-SB, FAQ, MMSE, and MoCA targets. Test remains sealed.

## Matched heads

Both arms contain the same token projection, coordinate projection, five learned target queries, four-head attention module, demographic projection, decoder, and trainable-parameter count.

1. `global_pool`: mean-pool spatial tokens and combine the global vector with the target query and demographic context.
2. `target_spatial_attention`: use one learned query per target to attend to feature tokens with fixed normalized spatial coordinates.

Each arm is trained separately for seeds 2026, 2027, and 2028 using the corresponding frozen SFCN checkpoint.

## Negative controls

The trained attention model is evaluated without retraining under:

- `shuffled_subject`: replace each validation MRI with an MRI from a different validation subject;
- `spatial_permutation`: permute feature tokens while keeping coordinate positions fixed;
- `query_permutation`: assign a different learned target query to every output target.

These controls test subject identity, spatial arrangement, and target specificity separately.

## Statistics and stability

Primary metric is equal-target macro normalized MAE. Comparisons use 2,000-replicate subject-cluster bootstrap intervals on paired normalized absolute-error differences. Attention stability is measured across seed pairs using per-scan/per-target Spearman correlation and top-20% token overlap. Weighted attention centers are reported in approximate aligned-template millimeter coordinates; no atlas-region label is assigned because audited subject-level segmentations are unavailable.

## Frozen gate

Concurrent spatial grounding is supported only if:

1. attention improves over global pooling overall with CI upper bound below zero;
2. observed attention improves over both spatial-permutation and query-permutation controls overall;
3. at least two endpoints improve significantly over global pooling; and
4. median cross-seed attention Spearman is at least 0.30 and median top-token overlap is at least 0.40.

Subject-shuffle performance is reported as a required validity control but is not substituted for the spatial and query controls. Passing opens a separately frozen multimodal concurrent-grounding stage. Failure retains the SFCN as a global concurrent MRI encoder without an anatomical-attention claim.
