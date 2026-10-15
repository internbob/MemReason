# Phase 4B cognitive-only structured-reasoning SFT

Status: completed on the frozen Phase 4A build/dev/confirmation partition. Original ADNI validation and test subjects were not used.

## Data and model

- Local base model: `DeepSeek-R1-Distill-Qwen-1.5B`
- Source build/dev/confirmation reports: 13238/1582/1757
- Selected train/dev/confirmation reports: 2471/488/168
- Selected train/dev/confirmation subjects: 1336/167/168
- Maximum prompt + completion length: 3328 tokens; no examples truncated
- Optimization: one-epoch completion-only rank-16 LoRA, 4,358,144 trainable parameters
- Final train/dev loss: 0.0015/0.0011
- MRI, diagnosis, and future observed outcomes used as inputs: no

## Frozen confirmation results

| arm | format_validity | evidence_precision | evidence_recall | numeric_citation_accuracy | time_citation_accuracy | domain_accuracy | current_state_consistency | temporal_consistency | forecast_preservation | uncertainty_preservation | hallucinated_evidence_rate | future_evidence_leakage | exact_report_match | forbidden_claim |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 |
| sft | 0.9881 | 0.9881 | 0.9851 | 0.9881 | 0.9881 | 0.9881 | 0.9802 | 0.9771 | 0.9881 | 0.9881 | 0.0119 | 0 | 0.7798 | 0 |
| deterministic_teacher | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 1 | 0 |

All language-model metrics use raw greedy generations without JSON repair or numeric post-processing. The deterministic teacher is the Phase 4A executable report and defines the auditable ceiling.

## Failure analysis

Two of 168 SFT generations reached the frozen 1,280-token limit before closing the JSON object. Among the 166 valid reports, evidence precision, numeric/time/domain accuracy, forecast preservation, and uncertainty preservation were 1.0000; evidence recall was 0.9970, current-state consistency was 0.9920, and temporal consistency was 0.9889. This descriptive conditional analysis does not replace the all-report gate.

## Subject-cluster bootstrap: SFT minus base

| metric | sft_minus_base | ci_low | ci_high | bootstrap_replicates |
| --- | --- | --- | --- | --- |
| format_validity | 0.9881 | 0.9702 | 1 | 2000 |
| evidence_precision | 0.9881 | 0.9702 | 1 | 2000 |
| evidence_recall | 0.9851 | 0.9649 | 0.9994 | 2000 |
| current_state_consistency | 0.9802 | 0.9607 | 0.9944 | 2000 |
| temporal_consistency | 0.9771 | 0.9557 | 0.9911 | 2000 |
| forecast_preservation | 0.9881 | 0.9702 | 1 | 2000 |
| uncertainty_preservation | 0.9881 | 0.9702 | 1 | 2000 |

## Frozen decision

```json
{
  "run_id": "phase4b_cognitive_reasoning_sft_20261014",
  "base_model": "DeepSeek-R1-Distill-Qwen-1.5B",
  "confirmation_reports": 168,
  "confirmation_subjects": 168,
  "criteria": {
    "format_validity": false,
    "evidence_precision": false,
    "evidence_recall": false,
    "numeric_citation_accuracy": false,
    "time_citation_accuracy": false,
    "domain_accuracy": false,
    "current_state_consistency": true,
    "temporal_consistency": true,
    "forecast_preservation": false,
    "uncertainty_preservation": false,
    "hallucinated_evidence_rate": false,
    "future_evidence_leakage": true,
    "improvement_over_base": true
  },
  "base_composite": 0.0,
  "sft_composite": 0.9849631519274377,
  "sft_minus_base_composite": 0.9849631519274377,
  "phase4b_sft_passed": false,
  "cognitive_grpo_ready": false,
  "multimodal_reasoning_ready": false,
  "next_action": "retain_deterministic_teacher_and_audit_sft_failures"
}
```

Passing this gate opens cognitive-only GRPO preparation only. Multimodal reasoning remains closed because the preceding MRI gates failed.
