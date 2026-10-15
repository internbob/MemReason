# Phase 3E frozen spatial-grounding test-confirmation protocol

Date frozen: 2026-10-14, before loading spatial features or predictions for test subjects.

## Confirmatory question

Does the Phase 3C total-score target-conditioned MRI spatial-grounding result replicate in the sealed subject-disjoint test split without training, checkpoint selection, threshold revision, or architecture changes?

Phase 3D failed its validation gate and is not evaluated on test. Phase 3E confirms only the Phase 3C claim.

## Frozen cohort and preprocessing

- Read only `splits/test.csv` rows with a verified MRI/cognition date window of at most 90 days, valid image/mask paths, and a real scan date.
- If one scan is linked to more than one cognitive visit, retain the visit with the smallest absolute date difference, then the earlier nominal month.
- Apply the Phase 3B target-specific date rule: a total score is eligible only when its own instrument date is within 90 days of the MRI.
- Reuse Phase 3B train-only age and target normalization without refitting.
- Reproduce Phase 3B volume normalization/resampling and freeze the corresponding three SFCN backbones.
- Load the six frozen Phase 3C head checkpoints. No parameter is updated and no checkpoint is selected using test performance.

## Models and controls

For seeds 2026, 2027, and 2028, evaluate:

1. `global_pool` with the correct subject MRI;
2. `target_spatial_attention` with the correct subject MRI;
3. spatial attention with an MRI from a different test subject;
4. spatial attention after cyclic spatial-token permutation while coordinates remain fixed;
5. spatial attention after cyclic target-query permutation.

The primary metric is equal-target macro normalized MAE. All contrasts use paired 2,000-replicate subject-cluster bootstrap intervals. Cross-seed attention stability uses per-scan/per-target Spearman correlation and top-20% token overlap.

## Frozen confirmation gate

The Phase 3C finding is test-confirmed only when:

1. observed spatial attention improves over observed global pooling overall with CI upper bound below zero;
2. observed attention improves over subject shuffle, spatial permutation, and query permutation overall;
3. at least two individual endpoints significantly improve over global pooling; and
4. median cross-seed attention Spearman is at least 0.30 and top-20% overlap is at least 0.40.

All criteria are conjunctive and cannot be changed after test evaluation.

## Secondary robustness analyses

Without changing the primary gate, report frozen-model results by diagnosis (CN/MCI/AD), MRI–cognition gap (0–30/31–60/61–90 days), sex, age (<70/70–79/≥80), and concurrent endpoint availability (1–2/3–4/5 observed totals). Strata with fewer than 20 subjects are descriptive only and are excluded from bootstrap contrasts.

Attention remains a predictive weighting map over the aligned SFCN grid and is not interpreted as causal anatomy.
