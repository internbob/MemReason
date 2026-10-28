# Phase 3N corrected center-cropped T1 diagnostic cross-fitting

Status: completed on original-train subjects only with five-fold subject-wise cross-fitting. Official validation and test subjects were not loaded.

## Cohort

- MRI-linked diagnostic visits: 5370
- Subjects: 1328
- CN/MCI/AD visits: {'MCI': 2517, 'CN': 1877, 'AD': 976}
- CN/MCI/AD subjects: {'AD': 407, 'CN': 504, 'MCI': 708}
- Diagnosis-to-scan window: <= 90 days
- MRI input: interpolation-free 160x192x160 center crop at 1 mm.
- Cross-fitted SFCNs: 5 folds x 3 seeds = 15 checkpoints

## Subject-balanced out-of-fold performance

ACC is subject-balanced accuracy. AUC, SEN, SEC, and F1 are macro one-vs-rest values across CN/MCI/AD.

| arm | scans | subjects | accuracy | macro_auc | macro_sensitivity | macro_specificity | macro_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cognition | 5370 | 1328 | 0.8826 | 0.9739 | 0.8924 | 0.9409 | 0.8828 |
| cognition_global_mri | 5370 | 1328 | 0.7872 | 0.9227 | 0.7956 | 0.8889 | 0.7908 |
| cognition_sfcn_mri | 5370 | 1328 | 0.8403 | 0.9497 | 0.8302 | 0.9134 | 0.8387 |
| cognition_shuffled_sfcn | 5370 | 1328 | 0.8226 | 0.9455 | 0.8029 | 0.9016 | 0.8188 |
| demographics | 5370 | 1328 | 0.3087 | 0.5618 | 0.3775 | 0.6865 | 0.2846 |
| global_mri | 5370 | 1328 | 0.4675 | 0.6566 | 0.4814 | 0.7289 | 0.4688 |
| sfcn_mri | 5370 | 1328 | 0.4973 | 0.6854 | 0.4869 | 0.7318 | 0.4924 |

Per-class sensitivity, specificity, and F1 are available in `diagnostic_per_class_metrics.csv`; 2,000-replicate confidence intervals are in `diagnostic_metric_intervals.csv`.

## Paired subject-cluster bootstrap

| comparison | metric | difference | ci_low | ci_high | replicates | subjects |
| --- | --- | --- | --- | --- | --- | --- |
| sfcn_mri_minus_demographics | macro_auc | 0.1237 | 0.09734 | 0.1507 | 2000 | 1328 |
| sfcn_mri_minus_demographics | balanced_accuracy | 0.1095 | 0.07996 | 0.1389 | 2000 | 1328 |
| cognition_sfcn_mri_minus_cognition | macro_auc | -0.02412 | -0.02937 | -0.01936 | 2000 | 1328 |
| cognition_sfcn_mri_minus_cognition | balanced_accuracy | -0.06207 | -0.07775 | -0.04772 | 2000 | 1328 |
| cognition_sfcn_mri_minus_cognition_shuffled_sfcn | macro_auc | 0.004282 | -0.0008275 | 0.009552 | 2000 | 1328 |
| cognition_sfcn_mri_minus_cognition_shuffled_sfcn | balanced_accuracy | 0.0273 | 0.01142 | 0.04256 | 2000 | 1328 |

## Diagnostic performance figure

![Phase 3N diagnosis performance and incremental-value tests](figures/fig_phase3g_diagnosis_increment.png)

## Frozen decision

```json
{
  "run_id": "phase3n_corrected_cropped_mri_diagnosis_crossfit_20261018",
  "official_validation_loaded": false,
  "official_test_loaded": false,
  "training_subject_source": "original_train_only",
  "mri_concurrent_signal_supported": true,
  "mri_increment_beyond_cognition_supported": false,
  "criteria": {
    "mri_signal_effect": true,
    "mri_signal_ci": true,
    "fusion_increment_effect": false,
    "fusion_increment_ci": false,
    "mechanism_effect": false,
    "mechanism_ci": false,
    "balanced_accuracy_noninferiority": false,
    "brier_noninferiority": false,
    "decision_curve_gain": false
  },
  "ad_net_benefit_gain_thresholds": 0,
  "mci_conversion_open": false,
  "external_testing_required": true
}
```

MRI signal relative to demographics and MRI increment beyond cognition are separate claims. The conversion task opens only if every frozen increment criterion passes. Internal cross-fitting is not external testing and does not establish clinical deployment value.
