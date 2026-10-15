# Phase 4F frozen GRU-D + Phase 4C sealed-test confirmation

Status: completed once on the official subject-disjoint ADNI test cohort, with no training or checkpoint selection.

## Frozen system and cohort

- Numeric model: Phase 4A three-seed GRU-D ensemble with build-only normalization
- Structured policy: Phase 4B 1.5B LoRA with the frozen Phase 4C `grammar_model` decoder
- Forecast samples / available targets / subjects: 4951 / 20007 / 495
- Subject-balanced reasoning reports: 495 reports / 495 subjects
- Future truth, diagnosis, MRI, and GRPO adapters used as reasoning inputs: no

## Forecast confirmation

Macro normalized MAE was **0.37510** and pooled normalized MAE was 0.36288. On rows covered by LOCF, GRU-D/LOCF normalized MAE was 0.36200/0.37939; the subject-mean paired difference was -0.02144. Normalized uncertainty-error Spearman was 0.1584 (p=1.34e-112).

| target | horizon_months | n | subjects | mae | normalized_mae | mean_normalized_uncertainty |
| --- | --- | --- | --- | --- | --- | --- |
| adas_total13 | 6 | 751 | 334 | 3.464 | 0.303 | 0.05203 |
| adas_total13 | 12 | 1362 | 410 | 3.429 | 0.2999 | 0.06474 |
| adas_total13 | 24 | 1493 | 380 | 3.837 | 0.3356 | 0.07398 |
| adas_total13 | 36 | 1074 | 289 | 4.389 | 0.3839 | 0.09582 |
| cdr_cdrsb | 6 | 752 | 334 | 0.7983 | 0.2708 | 0.05077 |
| cdr_cdrsb | 12 | 1417 | 422 | 0.9371 | 0.3179 | 0.07038 |
| cdr_cdrsb | 24 | 1556 | 397 | 0.932 | 0.3162 | 0.06355 |
| cdr_cdrsb | 36 | 1116 | 299 | 1.209 | 0.4102 | 0.06375 |
| faq_faqtotal | 6 | 753 | 334 | 2.871 | 0.3603 | 0.0597 |
| faq_faqtotal | 12 | 1410 | 419 | 2.844 | 0.3569 | 0.07143 |
| faq_faqtotal | 24 | 1560 | 397 | 2.838 | 0.356 | 0.06606 |
| faq_faqtotal | 36 | 1113 | 299 | 3.347 | 0.42 | 0.06313 |
| mmse_mmscore | 6 | 759 | 335 | 1.386 | 0.357 | 0.05751 |
| mmse_mmscore | 12 | 1379 | 413 | 1.451 | 0.3737 | 0.06916 |
| mmse_mmscore | 24 | 1510 | 384 | 1.556 | 0.4009 | 0.05977 |
| mmse_mmscore | 36 | 1084 | 292 | 1.841 | 0.4742 | 0.07041 |
| moca_moca | 6 | 6 | 6 | 2.148 | 0.4429 | 0.09087 |
| moca_moca | 12 | 257 | 153 | 2.104 | 0.4337 | 0.07428 |
| moca_moca | 24 | 367 | 176 | 2.073 | 0.4274 | 0.08262 |
| moca_moca | 36 | 288 | 146 | 2.239 | 0.4615 | 0.09849 |

| comparison | metric | difference | ci_low | ci_high | replicates | paired_rows | subjects |
| --- | --- | --- | --- | --- | --- | --- | --- |
| grud_minus_locf | normalized_absolute_error | -0.02144 | -0.03913 | -0.004232 | 2000 | 19757 | 495 |

## Structured reasoning confirmation

| arm | format_validity | evidence_precision | evidence_recall | numeric_citation_accuracy | time_citation_accuracy | domain_accuracy | hallucinated_evidence_rate | future_evidence_leakage | current_state_consistency | temporal_consistency | forecast_preservation | uncertainty_preservation | exact_report_match | forbidden_claim |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grammar_executable | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 0 |
| grammar_model | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0.9952 | 0.9899 | 1 | 1 | 0.8141 | 0 |
| raw_sft | 0.9859 | 0.9857 | 0.9819 | 0.9855 | 0.9857 | 0.9857 | 0.01434 | 0 | 0.981 | 0.9794 | 0.9859 | 0.9859 | 0.8343 | 0 |
| teacher | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 0 |

## Grammar model minus raw SFT

| metric | grammar_minus_raw | ci_low | ci_high | replicates |
| --- | --- | --- | --- | --- |
| format_validity | 0.01414 | 0.00404 | 0.02429 | 2000 |
| evidence_precision | 0.01434 | 0.004646 | 0.02488 | 2000 |
| evidence_recall | 0.01812 | 0.007704 | 0.03031 | 2000 |
| current_state_consistency | 0.01414 | 0.005114 | 0.02505 | 2000 |
| temporal_consistency | 0.01051 | 0.00121 | 0.02182 | 2000 |
| forecast_preservation | 0.01414 | 0.00404 | 0.02429 | 2000 |
| uncertainty_preservation | 0.01414 | 0.00404 | 0.02429 | 2000 |
| exact_report_match | -0.0202 | -0.0404 | 0 | 2000 |

## Frozen decision

```json
{
  "run_id": "phase4f_frozen_grud_phase4c_test_20261015",
  "frozen_system": "phase4a_grud_plus_phase4c_grammar_model",
  "test_opened_once": true,
  "training_performed": false,
  "checkpoint_selection_performed": false,
  "test_forecast_subjects": 495,
  "test_reasoning_subjects": 495,
  "grammar_label_audit": {
    "current_state_label_mismatches": 36,
    "current_state_reports_affected": 32,
    "current_state_domain_labels": 2475,
    "trend_label_mismatches": 100,
    "trend_reports_affected": 67,
    "trend_domain_labels": 2475,
    "current_state_metric_definition": "mean nested agreement over status, severity_z, and items for five domains",
    "temporal_metric_definition": "mean nested agreement over label, start_severity_z, end_severity_z, and timepoints for five domains"
  },
  "criteria": {
    "forecast_macro_nmae": true,
    "forecast_locf_noninferiority": true,
    "positive_uncertainty_error_association": true,
    "format_validity": true,
    "evidence_precision": true,
    "evidence_recall": true,
    "numeric_citation_accuracy": true,
    "time_citation_accuracy": true,
    "domain_accuracy": true,
    "current_state_consistency": true,
    "temporal_consistency": false,
    "forecast_preservation": true,
    "uncertainty_preservation": true,
    "zero_hallucinated_evidence": true,
    "zero_future_evidence_leakage": true,
    "format_improvement_ci_vs_raw": true
  },
  "phase4f_frozen_system_confirmed": false,
  "paper_primary_method_ready": false,
  "grpo_supported": false,
  "multimodal_reasoning_supported": false,
  "further_test_tuning_permitted": false,
  "next_action": "report_confirmatory_failure_without_test_tuning"
}
```

The only failed prespecified criterion was temporal consistency. This metric averages four nested fields (label, start severity, end severity, and timepoints) across five domains per report. The grammar arm preserved every numeric and timepoint anchor but differed on 100 of 2475 domain-level trend labels, affecting 67 of 495 reports; its resulting nested temporal consistency was 0.98990, narrowly below the 0.99000 threshold. Current-state consistency used the analogous three-field nested score and included 36 of 2475 domain-level status-label differences across 32 reports. This threshold will not be changed after test access. Grammar decoding nevertheless improved temporal consistency over raw SFT by +0.01051, with a 95% subject-bootstrap interval [0.00121, 0.02182].

The result supports only the frozen cognitive forecasting and auditable structured-report pipeline. It does not revise the negative Phase 4D/4E GRPO results and does not support multimodal or clinical-validity claims.
