# Phase 1E development protocol: gated semantics and metric-aligned reconstruction

Status at creation: frozen before Phase 1E validation predictions. This experiment never loads the test split.

## Motivation from Phase 1D

Typed semantics improved the equal-task validation score and significantly improved ADAS+CDR to MoCA, but masked-instrument and future forecasting did not meet the provisional gate. The two-stage schedule improved masked item while materially degrading masked instrument and future prediction. Phase 1E therefore keeps joint training and changes only two mechanisms identified before this run.

## Changes

1. The frozen task-semantic residual is multiplied by a trainable per-item gate initialized at zero. All arms contain the same gate parameters. The semantic arm therefore starts from the nonsemantic query and must learn to use semantics from training evidence.
2. Reconstruction loss is calculated per target item and then macro-averaged. Binary/ordinal items combine categorical cross-entropy and Huber loss on the expected score normalized by the train-only item standard deviation. This aligns optimization with macro normalized MAE and prevents frequently observed binary items from dominating masked-instrument training.

## Arms

- `numeric_query_aligned`: raw numeric values and learned item/type queries.
- `typed_response_aligned`: adds direction-aligned typed response encoders.
- `typed_semantic_gated_aligned`: adds zero-initialized gated frozen task semantics.

Architecture, parameter count, task-balanced sampling, seed, optimizer, tasks and validation selection are shared. Phase 1D test outputs are not used, and the Phase 1E test split is not read.

## Provisional gate

Relative to `numeric_query_aligned`, the semantic candidate must have a subject-bootstrap 95% CI upper bound below zero on masked item, masked instrument and both cross-scale tasks. Future forecasting must have a CI upper bound at most +0.005 normalized MAE. A pass only permits a new three-seed confirmation protocol; it does not open MRI Stage II.
