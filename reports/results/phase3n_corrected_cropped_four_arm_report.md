# Phase 3N corrected cropped-T1 three-class results

The corrected experiment contains 1,328 original-training subjects and 5,370
MRI-linked CN/MCI/AD visits. All 5,370 MRI volumes were rebuilt from the original
T1 and mask paths, center-cropped from `193x229x193` to `160x192x160` without
interpolation, and passed complete row-level cache validation.

Values are point estimates with 95% subject-cluster bootstrap confidence
intervals in brackets.

| Input | ACC | macro-AUC | macro-SEN | macro-SEC | macro-F1 |
|---|---:|---:|---:|---:|---:|
| MRI only | 0.4973 [0.4747, 0.5196] | 0.6854 [0.6659, 0.7043] | 0.4869 [0.4635, 0.5093] | 0.7318 [0.7195, 0.7431] | 0.4924 [0.4684, 0.5150] |
| Cognition only | 0.8826 [0.8699, 0.8951] | 0.9739 [0.9694, 0.9781] | 0.8924 [0.8802, 0.9046] | 0.9409 [0.9343, 0.9473] | 0.8828 [0.8702, 0.8953] |
| Cognition + MRI | 0.8403 [0.8247, 0.8558] | 0.9497 [0.9426, 0.9566] | 0.8302 [0.8135, 0.8469] | 0.9134 [0.9049, 0.9221] | 0.8387 [0.8231, 0.8540] |
| Cognition + shuffled MRI | 0.8226 [0.8071, 0.8375] | 0.9455 [0.9383, 0.9522] | 0.8029 [0.7853, 0.8199] | 0.9016 [0.8931, 0.9100] | 0.8188 [0.8025, 0.8340] |

`requested_four_arm_metrics_with_ci.csv` contains all point estimates and 95%
confidence intervals from 2,000 subject-cluster bootstrap replicates. Official
validation and test subjects were not loaded.
