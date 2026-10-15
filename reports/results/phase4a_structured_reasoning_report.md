# Phase 4A frozen cognitive structured-reasoning baseline

Status: completed on a new subject-disjoint partition of original-train subjects. Original validation and test outcomes were not used.

## Data and frozen forecast

- Verified scalar cognitive inputs: 68
- Subject counts: {'build': 1336, 'confirmation': 168, 'dev': 167}
- Forecast-sample counts: {'build': 13238, 'dev': 1582, 'confirmation': 1757}
- GRU-D seeds/checkpoints: [2026, 2027, 2028] / 3
- Trainable parameters per checkpoint: [220173]
- LLM trained: no
- MRI/diagnosis used as reasoning input: no

## Frozen forecast description

| reasoning_split | normalized_mae | mean_normalized_uncertainty |
| --- | --- | --- |
| build | 0.321 | 0.06978 |
| confirmation | 0.3515 | 0.08153 |
| dev | 0.3751 | 0.07057 |

Confirmation uncertainty-error Spearman: 0.1869853748419199 (p=5.301823068083107e-57). This is descriptive and is not part of the mechanical report gate.

## Structured-report verification

| reasoning_split | reports | subjects | format_validity | evidence_precision | evidence_recall | numeric_citation_accuracy | time_citation_accuracy | hallucinated_evidence_rate | domain_accuracy | current_state_consistency | temporal_consistency | forecast_preservation | uncertainty_preservation | future_evidence_leakage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| build | 13238 | 1336 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 1 | 0 |
| confirmation | 1757 | 168 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 1 | 0 |
| dev | 1582 | 167 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 1 | 0 |

## Frozen decision

```json
{
  "run_id": "phase4a_frozen_cognitive_reasoning_20261014",
  "engineering_smoke_only": false,
  "original_train_only": true,
  "validation_evaluated": false,
  "test_evaluated": false,
  "llm_trained": false,
  "grpo_performed": false,
  "criteria": {
    "format_validity": true,
    "evidence_precision": true,
    "evidence_recall": true,
    "numeric_citation_accuracy": true,
    "time_citation_accuracy": true,
    "hallucinated_evidence_rate": true,
    "domain_accuracy": true,
    "current_state_consistency": true,
    "temporal_consistency": true,
    "forecast_preservation": true,
    "uncertainty_preservation": true,
    "zero_future_evidence_leakage": true
  },
  "confirmation_reports": 1757,
  "confirmation_subjects": 168,
  "uncertainty_error_spearman": 0.1869853748419199,
  "uncertainty_error_pvalue": 5.301823068083107e-57,
  "structured_reasoning_baseline_passed": true,
  "phase4b_sft_ready": true,
  "next_action": "train_cognitive_only_structured_reasoning_sft"
}
```

The JSONL completion is a deterministic, auditable teacher baseline rather than free chain-of-thought. Future truth is stored only in the forecast evaluation table and is absent from prompts and completions. A pass opens cognitive-only Phase 4B SFT development; it does not open multimodal reasoning or GRPO.
