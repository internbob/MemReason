# Phase 1D development protocol: typed question-conditioned representation

Status at creation: development-only protocol written before Phase 1D validation predictions were computed. The test split is not loaded by this experiment.

## Purpose

Phase 1C showed that encoding the numeric response again as frozen mean-pooled text did not improve the response representation. Phase 1D tests whether explicit response types and an item-conditioned decoder improve partial-observation learning without changing the underlying subject split or targets.

## Shared data and tasks

- The same 68 verified scalar monotonic ADAS/CDR/MMSE/MoCA inputs.
- Diagnosis, FAQ items, totals, MRI and future observations are excluded from state inputs.
- The same masked-item, masked-instrument, two concurrent cross-scale and 6/12/24/36-month forecasting samples as Phase 1C.
- Train is used for optimization and validation for development selection. Test is never read.

## Parameter-matched arms

1. `numeric_query_joint`: raw normalized values, masks, time, learned item/type queries and joint training.
2. `typed_response_joint`: adds direction-aligned type-specific response-value encoders.
3. `typed_semantic_joint`: also adds frozen Bio_ClinicalBERT task semantics.
4. `typed_semantic_two_stage`: the same model as arm 3, first trained on reconstruction/cross-scale tasks and then fine-tuned jointly with a smaller reconstruction weight.

Every arm instantiates the same trainable modules. Disabled information paths are zeroed, so parameter counts remain identical.

## Model changes relative to Phase 1C

- Binary, ordinal and continuous responses receive explicit type IDs.
- Values are mapped to a verified direction-aligned severity coordinate before a type-specific value MLP.
- Masked targets are decoded with their item/type/task query rather than one unconditioned 68-output linear head.
- Binary/ordinal reconstruction uses categorical cross-entropy over verified valid values; continuous reconstruction uses normalized Huber loss.
- Task-balanced sampling remains active.

## Development decision

The primary development score is the equal-task macro normalized MAE across five task groups. Subject-cluster bootstrap compares every candidate with `numeric_query_joint`. A provisional Stage I signal requires the selected semantic candidate to improve both masked tasks and both cross-scale tasks with 95% CI upper bounds below zero; future forecasting must have CI upper bound at most +0.005.

This validation-only decision cannot open MRI Stage II. A successful candidate must be frozen under a new confirmation protocol and run with three seeds before any internal test re-evaluation.
