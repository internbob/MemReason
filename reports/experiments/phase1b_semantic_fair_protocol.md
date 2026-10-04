# Phase 1B frozen protocol: numeric versus semantic representations

Status at creation: frozen before Phase 1B test prediction or metric computation.

## Question

Does verified question/task semantics improve future cognitive forecasting and cross-scale prediction beyond the same numeric item values, masks and times?

## Dictionary gate

- An input must have an exact official ADNI field-name and source-table match, an official code/range specification consistent with every observed non-missing source value, a curated cognitive construct, and `verified_from_dictionary=true`.
- Diagnosis, FAQ items, totals and future values are not inputs.
- Non-monotonic MoCA delayed-recall cue categories are verified and retained in the source dictionary, but excluded from this scalar-response comparison. Raw MMSE spelling letters, MoCA serial responses and MoCA letter-vigilance responses remain excluded.
- Evidence is recorded in `reports/data_audit/adni_official_dictionary_rows.csv` and `adni_official_dictionary_validation.csv`.

## Three fair arms

1. `numeric_only`: normalized item values, item-specific masks and time; descriptor input is the zero vector.
2. `variable_name`: the same numeric input plus a frozen Bio_ClinicalBERT embedding of the source variable name.
3. `task_semantics`: the same numeric input plus a frozen Bio_ClinicalBERT embedding of official instrument, domain, construct and task description.

All arms use the same feature order, targets, subjects, samples, trainable architecture, descriptor projection size, optimizer, early stopping, batch size and three seeds. The descriptor matrix is the only arm-dependent input. The text encoder is frozen and run once to cache descriptors; it never sees outcome values, diagnosis, subject data or future observations. Trainable parameter counts must be identical.

## Tasks

### Primary: future total-score forecasting

Predict ADAS13, CDR-SB, FAQ, MMSE and MoCA at nominal 6/12/24/36-month horizons. The target is the nearest actual future measurement within ±3 months at 6/12 months and ±6 months at 24/36 months. Targets must be strictly later than the cutoff.

### Secondary: cross-scale prediction

- ADAS + CDR verified items at a visit → concurrent MoCA total.
- MMSE + CDR verified items at a visit → concurrent ADAS13.

Only the declared source-instrument items are visible in each cross-scale sample. Subject-wise train/validation/test assignments are unchanged.

## Optimization and test policy

- Seeds: 2026, 2027, 2028.
- AdamW, batch 256, learning rate 1e-3, weight decay 1e-4, maximum 40 epochs, patience 7.
- Each arm/seed checkpoint is selected only by validation macro normalized MAE.
- Engineering smoke tests do not load test samples.
- Test predictions are computed only after all 18 arm × seed × task checkpoints have completed validation training.

## Metrics and inference

- Per task/endpoint/horizon: n, coverage, MAE, normalized MAE, RMSE, Pearson and Spearman.
- Three-seed mean prediction is the primary prediction for each arm; seed variability is reported.
- Primary contrast: task-semantics ensemble minus numeric-only ensemble normalized absolute error.
- Variable-name minus numeric-only is a secondary contrast.
- A paired subject-cluster bootstrap with 2,000 replicates gives 95% confidence intervals. Negative differences favor the semantic arm.
- Cross-scale contrasts are reported separately for both declared instrument pairs.

## Interpretation

This experiment tests frozen text descriptors as item metadata. It does not test MRI, generated natural-language responses, an end-to-end fine-tuned language model, SFT or RL. Failure to improve is a valid result and will not trigger post-test hyperparameter changes under this run ID.
