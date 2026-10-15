# Phase 4C dev-only grammar-constrained structured decoding

Status: completed with decoder selection on Phase 4A internal dev subjects and one-time confirmation on new original-validation subjects. The previous Phase 4B confirmation subjects were not reused.

## Cohorts and frozen components

- Dev: 167 reports / 167 subjects
- New confirmation: 234 reports / 234 subjects
- New confirmation source: original ADNI validation split
- Frozen components: Phase 4A GRU-D ensemble and normalization; Phase 4B DeepSeek-R1-Distill-Qwen-1.5B LoRA
- MRI, diagnosis, future truth, and original test used as reasoning inputs: no

## Dev-only decoder selection

| arm | format_validity | evidence_precision | evidence_recall | numeric_citation_accuracy | time_citation_accuracy | domain_accuracy | hallucinated_evidence_rate | future_evidence_leakage | current_state_consistency | temporal_consistency | forecast_preservation | uncertainty_preservation | exact_report_match | forbidden_claim |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grammar_executable | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 0 |
| grammar_model | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0.992 | 0.9943 | 1 | 1 | 0.8323 | 0 |
| raw_sft | 0.982 | 0.981 | 0.9802 | 0.981 | 0.981 | 0.981 | 0.01896 | 0 | 0.9764 | 0.9772 | 0.982 | 0.982 | 0.8204 | 0 |

Frozen selected arm: **grammar_model**. The model-constrained arm had priority over the executable compiler whenever it satisfied every gate.

## New-subject confirmation

| arm | format_validity | evidence_precision | evidence_recall | numeric_citation_accuracy | time_citation_accuracy | domain_accuracy | hallucinated_evidence_rate | future_evidence_leakage | current_state_consistency | temporal_consistency | forecast_preservation | uncertainty_preservation | exact_report_match | forbidden_claim |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grammar_model | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0.9946 | 0.994 | 1 | 1 | 0.859 | 0 |
| raw_sft | 0.9658 | 0.9645 | 0.9632 | 0.9645 | 0.9645 | 0.9645 | 0.03547 | 0 | 0.9615 | 0.9607 | 0.9658 | 0.9658 | 0.8205 | 0 |

## Subject-cluster bootstrap: selected minus raw SFT

| metric | selected_minus_raw | ci_low | ci_high | replicates |
| --- | --- | --- | --- | --- |
| format_validity | 0.03419 | 0.01282 | 0.05983 | 2000 |
| evidence_precision | 0.03547 | 0.0141 | 0.06111 | 2000 |
| evidence_recall | 0.03675 | 0.01624 | 0.06198 | 2000 |
| current_state_consistency | 0.03305 | 0.01282 | 0.05783 | 2000 |
| temporal_consistency | 0.03333 | 0.01303 | 0.05855 | 2000 |
| forecast_preservation | 0.03419 | 0.01282 | 0.05983 | 2000 |
| uncertainty_preservation | 0.03419 | 0.01282 | 0.05983 | 2000 |

## Decision

```json
{
  "run_id": "phase4c_grammar_constrained_decoding_20261015",
  "selected_arm": "grammar_model",
  "confirmation_examples": 234,
  "confirmation_subjects": 234,
  "criteria": {
    "format_validity": true,
    "evidence_precision": true,
    "evidence_recall": true,
    "numeric_citation_accuracy": true,
    "time_citation_accuracy": true,
    "domain_accuracy": true,
    "current_state_consistency": true,
    "temporal_consistency": true,
    "forecast_preservation": true,
    "uncertainty_preservation": true,
    "hallucinated_evidence_rate": true,
    "future_evidence_leakage": true
  },
  "phase4c_structured_decoding_passed": true,
  "stable_sft_policy_for_grpo": true,
  "cognitive_grpo_ready": true,
  "multimodal_reasoning_ready": false,
  "next_action": "freeze_constrained_cognitive_grpo_protocol"
}
```

`grammar_model` is token-trie-constrained model decoding. `grammar_executable` is a hybrid compiler using the documented thresholds and cannot by itself establish that the language-model policy is ready for GRPO.
