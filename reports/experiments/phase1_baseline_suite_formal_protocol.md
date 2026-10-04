# Formal cognitive forecasting baseline suite

Status at creation: frozen before this suite loads the test split or computes test predictions.

## Question

How well do classical and strong recurrent/attention baselines forecast future ADAS13, CDR-SB, FAQ, MMSE and MoCA scores from irregular, partially observed cognitive item histories?

## Cohort, inputs and outcomes

- Use the existing subject-wise 70/10/20 train/validation/test split. Subjects cannot cross splits.
- Use the 68 dictionary-verified, scalar, monotonic state items used by Phase 1B.
- Inputs are numeric item values, observation masks, event order and the model-specific handling of nominal visit time.
- Diagnosis, DXSUM, FAQ items, scale totals, MRI, text semantics and future observations are excluded from inputs.
- Outcomes are the nearest actual future ADAS13, CDR-SB, FAQ, MMSE and MoCA observations at 6/12 months within ±3 months and 24/36 months within ±6 months. Every target must occur strictly after its cutoff.

## Predeclared methods

1. `LOCF`: last observed target value, with prediction coverage reported.
2. `LME`: target-specific linear mixed-effects model with population time trend and subject random intercept/slope when estimable.
3. `GRU`: masked numeric events in visit order without an explicit elapsed-time mechanism.
4. `LSTM`: masked numeric events in visit order without an explicit elapsed-time mechanism.
5. `T-LSTM`: elapsed-time decay of the short-term cell-memory component.
6. `GRU-D`: feature-wise decay based on time since each item was last observed.
7. `Transformer`: numeric/mask event tokens with explicit time and gap embeddings.

All neural models use hidden size 128, two recurrent/encoder layers, the same horizon head, AdamW settings, batch size, maximum epochs, early-stopping rule and seeds 2026/2027/2028. Parameter counts are reported rather than forced to match because the standard architectures have different gates.

## Test isolation

- Each neural checkpoint is selected only by validation macro normalized MAE.
- The test CSV and test samples are loaded only after all 15 neural checkpoints have completed validation training.
- Every predeclared method is evaluated once. Test results cannot change this run ID, architecture list or hyperparameters.

## Metrics and inference

- Per endpoint and horizon: n, coverage, MAE, normalized MAE, RMSE, Pearson and Spearman.
- Neural seed variability and the three-seed mean ensemble are reported.
- Validation macro normalized MAE identifies the strongest neural baseline for later method comparisons.
- Paired subject-cluster bootstrap with 2,000 replicates compares every neural ensemble with LOCF and LME. Negative normalized-MAE differences favor the neural model.

## Interpretation boundary

This suite establishes cognitive forecasting baselines. It does not test semantic value, MRI grounding, probabilistic-state calibration, SFT or GRPO.
