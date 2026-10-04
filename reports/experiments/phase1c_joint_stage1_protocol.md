# Phase 1C frozen protocol: joint response-aware Stage I

Status at creation: frozen before Phase 1C test samples, predictions or metrics are loaded or computed.

## Question

Does an item-level response-aware semantic encoder improve masked item reconstruction, masked instrument reconstruction, cross-scale generalization and future cognitive forecasting beyond the same parameter-matched numeric and fixed-task-semantic architectures?

## Data contract

- Existing subject-wise train/validation/test registry; no subject crosses splits.
- Inputs are the same 68 dictionary-verified, scalar, monotonic ADAS/CDR/MMSE/MoCA items used by Phase 1B and the formal baseline suite.
- Diagnosis, DXSUM, FAQ items, scale totals, MRI and future observations are excluded from inputs.
- Response vocabularies are built from valid training-split responses only. Validation/test responses absent from the training vocabulary fall back to the verified task descriptor.
- The existing ADNI test split is a locked internal re-evaluation because earlier experiments have already reported it. It is not presented as new external confirmation.

## Parameter-matched arms

1. `numeric_only`: descriptor buffers are zero; numeric values, masks, times and domain aggregation remain active.
2. `task_semantics`: every observed response uses the frozen Bio_ClinicalBERT embedding of its verified instrument/domain/construct/task description.
3. `response_aware`: every observed response uses a frozen Bio_ClinicalBERT embedding containing the verified task, the actual valid response/score, allowed range or category label, and score direction.

All arms share the same GRU-D dynamics, domain aggregation, reconstruction/forecast heads, optimizer, task sampling, seeds and trainable parameter count. Bio_ClinicalBERT is frozen and used only to create descriptor buffers before training.

## Joint tasks

### Masked item

For each eligible visit, one actually observed input item is deterministically selected and hidden. The model reconstructs only that hidden value. Mask selection is identical across arms.

### Masked instrument

For each eligible visit, one observed instrument is deterministically selected. All its observed items are hidden and reconstructed from the remaining instruments. Visits without another observed source instrument are excluded.

### Cross-scale

- ADAS + CDR verified items → concurrent MoCA total.
- MMSE + CDR verified items → concurrent ADAS13.

Only declared source-instrument items are visible.

### Future forecasting

Predict future ADAS13, CDR-SB, FAQ, MMSE and MoCA at 6/12/24/36 months using the same target windows as the formal baseline suite.

## Optimization and isolation

- Seeds 2026/2027/2028; AdamW; batch 256; maximum 40 epochs; patience 7.
- Inverse-frequency task sampling gives each task family equal expected sampling probability.
- Loss is the equal-weight mean of available task-specific normalized Huber losses in a batch.
- Checkpoints are selected by the equal-weight mean validation normalized MAE across the five declared task groups: masked item, masked instrument, both cross-scale pairs and future forecasting.
- Test samples are built only after all 3 arms × 3 seeds have completed validation training.

## Metrics and inference

- All tasks: target count, subject count, MAE, training-standard-deviation-normalized MAE and RMSE.
- Future/cross-scale: Pearson and Spearman.
- Reconstruction: binary accuracy/AUROC, ordinal MAE/weighted kappa and continuous MAE.
- Three-seed mean predictions are primary; seed variability is retained.
- Paired subject-cluster bootstrap with 2,000 replicates compares response-aware and fixed-task semantics with numeric-only separately for all five task groups.

## Gate 1 decision

The response-aware Stage I gate requires all of the following on the frozen evaluation:

1. masked-item response-aware minus numeric-only 95% CI upper bound < 0;
2. masked-instrument response-aware minus numeric-only 95% CI upper bound < 0;
3. both cross-scale response-aware minus numeric-only 95% CI upper bounds < 0;
4. missingness-only diagnosis balanced accuracy < 0.50;
5. future forecasting is reported separately; non-inferiority margin is +0.005 normalized MAE.

Failure is retained as a valid result and does not trigger changes under this run ID. MRI Stage II is not opened unless the Stage I semantic/cross-scale criteria pass.
