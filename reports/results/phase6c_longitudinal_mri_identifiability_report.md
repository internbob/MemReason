# Phase 6C: longitudinal MRI temporal-signal identifiability

Original-train-only five-fold subject cross-fitting; validation and test were not loaded.

## Primary observable endpoint

| Arm | Memory composite MAE | Clinical NLL | 95% coverage | Interval width | Latent MAE |
|---|---:|---:|---:|---:|---:|
| cognition_only | 0.09002 | -0.7572 | 0.9458 | 0.4330 | 0.09332 |
| date_permuted_dynamics | 0.08765 | -0.7822 | 0.9480 | 0.4293 | 0.09272 |
| sign_flipped_dynamics | 0.08782 | -0.7793 | 0.9463 | 0.4291 | 0.09273 |
| static_mri | 0.08755 | -0.7829 | 0.9480 | 0.4290 | 0.09253 |
| subject_shuffled_dynamics | 0.08753 | -0.7831 | 0.9484 | 0.4286 | 0.09260 |
| true_longitudinal_mri | 0.08756 | -0.7836 | 0.9485 | 0.4291 | 0.09275 |

## Prespecified temporal-signal comparisons

- true longitudinal minus `static_mri` participant-mean MAE: **+0.000098** (95% participant-bootstrap CI **[-0.000084, +0.000290]**); folds better **2/5**; horizons better **1/4**.
- true longitudinal minus `date_permuted_dynamics` participant-mean MAE: **-0.000010** (95% participant-bootstrap CI **[-0.000177, +0.000165]**); folds better **4/5**; horizons better **3/4**.
- true longitudinal minus `sign_flipped_dynamics` participant-mean MAE: **+0.000002** (95% participant-bootstrap CI **[-0.000339, +0.000354]**); folds better **4/5**; horizons better **4/4**.
- true longitudinal minus `subject_shuffled_dynamics` participant-mean MAE: **+0.000157** (95% participant-bootstrap CI **[-0.000062, +0.000372]**); folds better **3/5**; horizons better **1/4**.

## Decision

Candidate clinical coverage was **0.9485**. Longitudinal MRI temporal signal supported: **`false`**. The next disagreement-aware filtering experiment is **closed**; MRI is **frozen as a static auxiliary covariate**.

This exploratory audit does not reopen the Phase 5F measurement gate and cannot establish a confirmatory multimodal or continuous-time claim.
