# Phase 1A formal numeric forecasting protocol

This protocol was fixed before test-set evaluation.

## Question

Can irregular, partially observed item-level numeric cognitive history predict future ADAS13, CDR-SB, FAQ, MMSE and MoCA totals at nominal 6, 12, 24 and 36 month horizons?

## Cohort and split

- Existing subject-wise split: 70% train, 10% validation, 20% test; seed 2026.
- No subject crosses splits.
- The test split is evaluated only after code, endpoints, horizons, tolerances and hyperparameters are fixed.
- Missing observations are masked. No complete-case filtering and no zero imputation as an observed value.

## Inputs and outcomes

- Neural inputs: numeric item values, observation masks and irregular nominal visit times.
- Explicitly excluded: diagnosis/DXSUM, FAQ outcomes, scale totals, future observations, MoCA DELW coding pending verification, raw MoCA serial/letter codes, ADAS q13, RAVLT and Logical Memory fields pending coding verification.
- Outcomes: ADAS13, CDR-SB, FAQ total, MMSE total and MoCA total.
- A future target is the nearest actually observed value within 6±3, 12±3, 24±6 or 36±6 months after a cutoff. Loss and metrics use only available targets.

## Predeclared methods

1. Last observation carried forward (LOCF), with coverage reported.
2. Linear mixed-effects model (LME): target-specific population time trend with subject random intercept and slope when estimable; empirical-Bayes random effects are inferred from evaluation-subject history only.
3. GRU-D numeric sequence model with learned feature-wise decay toward training means.
4. Time-aware Transformer using the same numeric values, masks and times as GRU-D.

Neural methods use three fixed seeds (2026, 2027, 2028), identical optimization budgets and no test-driven tuning.

## Metrics and inference

- Per endpoint and horizon: target count, prediction coverage, MAE, RMSE, Pearson r and Spearman rho.
- Model selection summary: macro MAE normalized by the training-set target standard deviation.
- Neural variability: mean and standard deviation across the three seeds; ensemble predictions are the seed mean.
- Paired subject-cluster bootstrap (1,000 replicates) estimates 95% confidence intervals for ensemble-neural minus classical-baseline MAE. Negative differences favor the neural model.
- LME convergence and residual diagnostics are saved. Normal residuals are not assumed for the primary MAE comparison.

## Interpretation boundary

This is an aggregate-score forecasting baseline experiment. It does not test question semantics, MRI value, latent-state calibration, SFT or GRPO. Nominal visit months are not actual MRI acquisition dates.
