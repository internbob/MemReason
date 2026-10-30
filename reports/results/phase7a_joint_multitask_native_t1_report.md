# Phase 7A: native-T1 CN/MCI/AD and Memory multitask learning

Original-train-only five-fold subject cross-fitting; official validation and test were not loaded.

## Cohort

- Scans: **4,263**
- Participants: **774**
- CN/MCI/AD visits: **{'MCI': 2664, 'CN': 816, 'AD': 783}**

## Subject-balanced diagnosis

| arm | accuracy | macro_auc | macro_sensitivity | macro_specificity | macro_f1 |
| --- | --- | --- | --- | --- | --- |
| classification_only | 0.2988 | 0.5958 | 0.4192 | 0.6858 | 0.3049 |
| joint_multitask | 0.2691 | 0.6085 | 0.432 | 0.6914 | 0.2763 |
| joint_subject_shuffled_mri | 0.2318 | 0.5499 | 0.3625 | 0.6782 | 0.2301 |

## Subject-balanced Memory regression

| arm | endpoint | mae | nll | coverage_95 |
| --- | --- | --- | --- | --- |
| joint_multitask | current | 0.1329 | -0.4248 | 0.9676 |
| joint_multitask | future | 0.08311 | -0.8291 | 0.9471 |
| joint_subject_shuffled_mri | current | 0.1435 | -0.3114 | 0.9412 |
| joint_subject_shuffled_mri | future | 0.08618 | -0.7979 | 0.9427 |
| regression_only | current | 0.1321 | -0.4208 | 0.966 |
| regression_only | future | 0.08349 | -0.8177 | 0.9438 |

## Prespecified contrasts

- Joint minus classification-only macro AUC: **+0.013100** (95% CI **[-0.007250, +0.032348]**).
- Joint minus regression-only future Memory MAE: **-0.000325** (95% CI **[-0.000644, -0.000034]**); folds better **3/5**, horizons noninferior **3/4**.

Joint multitask development gate passed: **`false`**.

This is internal original-train development evidence and does not open a confirmatory clinical claim.
