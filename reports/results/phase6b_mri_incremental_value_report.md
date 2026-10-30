# Phase 6B: MRI incremental-value and negative-control audit

Original-train-only five-fold subject cross-fitting; validation and test were not loaded.

## Observable clinical endpoint

| Arm | Memory composite MAE | Clinical NLL | 95% coverage | Interval width | Latent MAE |
|---|---:|---:|---:|---:|---:|
| cognition_only | 0.08561 | -0.7867 | 0.9470 | 0.4319 | 0.09128 |
| metadata_only | 0.08547 | -0.7869 | 0.9471 | 0.4317 | 0.09111 |
| mri_only | 0.14701 | -0.3007 | 0.9485 | 0.6681 | 0.18109 |
| real_mri | 0.08450 | -0.7974 | 0.9476 | 0.4298 | 0.09076 |
| subject_shuffled_mri | 0.08557 | -0.7856 | 0.9459 | 0.4314 | 0.09136 |
| time_shuffled_mri | 0.08453 | -0.7977 | 0.9477 | 0.4294 | 0.09072 |

## Matched real-MRI effects

- real MRI minus `cognition_only` participant-mean composite error: **-0.001087**; participant-bootstrap 95% CI **[-0.001498, -0.000670]**; folds better **5/5**; horizons non-inferior **4/4**.
- real MRI minus `metadata_only` participant-mean composite error: **-0.000740**; participant-bootstrap 95% CI **[-0.001084, -0.000392]**; folds better **5/5**; horizons non-inferior **4/4**.
- real MRI minus `subject_shuffled_mri` participant-mean composite error: **-0.000744**; participant-bootstrap 95% CI **[-0.001121, -0.000392]**; folds better **5/5**; horizons non-inferior **4/4**.
- real MRI minus `time_shuffled_mri` participant-mean composite error: **-0.000055**; participant-bootstrap 95% CI **[-0.000193, +0.000076]**; folds better **3/5**; horizons non-inferior **2/4**.

## Decision

The real-MRI candidate clinical coverage was **0.9476**. MRI-content support was `false` and the disagreement-aware filter remains `closed`.

This development audit does not reopen the Phase 5F measurement gate and cannot establish a confirmatory multimodal or continuous-time claim.
