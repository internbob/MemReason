# Phase 4D cognitive-only constrained GRPO

Status: completed and stopped at the frozen internal-development gate. Real on-policy rollout and LoRA optimization were performed; original-test reasoning confirmation was not opened.

## Frozen design and cohorts

- Rollout: 64 reports / 64 Phase 4A build subjects
- Development: 167 reports / 167 internal-dev subjects
- Group size / completed updates / on-policy candidates: 4 / 32 / 256
- Trainable parameters: 4,358,144 existing Phase 4B LoRA parameters
- Frozen components: SFT reference, GRU-D, evidence, numeric state statistics, forecast, uncertainty, and renderer
- Future outcomes entered prompts: no; they were reward/audit data only
- Original test loaded: no

## On-policy optimization audit

Mean first/last total reward was 0.9723/0.9597. Mean categorical KL across updates was 0.01235 and its maximum update mean was 0.05168, below the 0.15 stop threshold. Reward-variable groups comprised 0.7188 of rollout groups. Training completed all 32 updates without a safety abort.

The future-outcome, evidence, domain, uncertainty, and format components were recorded for every candidate. Numeric forecasts and renderer anchors were frozen, so only current-state and temporal actions supplied group-relative learning signal.

## Internal-development checkpoint selection

| arm | format_validity | evidence_precision | evidence_recall | numeric_citation_accuracy | time_citation_accuracy | domain_accuracy | hallucinated_evidence_rate | future_evidence_leakage | current_state_consistency | temporal_consistency | forecast_preservation | uncertainty_preservation | exact_report_match | forbidden_claim | state_label_accuracy | trend_label_accuracy | action_composite |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grpo_u16 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0.9832 | 0.9919 | 1 | 1 | 0.7246 | 0 | 0.9497 | 0.9677 | 0.9587 |
| grpo_u24 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0.9872 | 0.9922 | 1 | 1 | 0.7605 | 0 | 0.9617 | 0.9689 | 0.9653 |
| grpo_u32 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0.9916 | 0.9925 | 1 | 1 | 0.8084 | 0 | 0.9749 | 0.9701 | 0.9725 |
| grpo_u8 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0.9836 | 0.9919 | 1 | 1 | 0.7246 | 0 | 0.9509 | 0.9677 | 0.9593 |
| sft | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0.9952 | 0.9925 | 1 | 1 | 0.8443 | 0 | 0.9856 | 0.9701 | 0.9778 |

The best GRPO checkpoint was update 32. Its direct action composite was 0.97246, compared with 0.97784 for SFT, a difference of -0.00539. State and trend accuracy, KL control, reward variation, and safety criteria passed; the required improvement over SFT failed.

## Frozen decision

```json
{
  "run_id": "phase4d_cognitive_constrained_grpo_20261015",
  "real_on_policy_grpo_completed": true,
  "development_gate_passed": false,
  "confirmation_opened": false,
  "original_test_loaded": false,
  "selected_update_for_audit": 32,
  "sft_action_composite": 0.9778443113772456,
  "best_grpo_action_composite": 0.9724550898203593,
  "grpo_minus_sft_action_composite": -0.005389221556886259,
  "numeric_forecast_changed": false,
  "cognitive_rlvr_supported": false,
  "multimodal_grpo_supported": false,
  "next_action": "retain_phase4c_sft_policy_and_stop_grpo_tuning_on_current_dev"
}
```

The experiment therefore does not support a cognitive RLVR improvement claim. Phase 4C remains the supported constrained cognitive policy. The original-test reasoning cohort stays unopened, and the current internal-dev subjects must not be used for further GRPO hyperparameter search.
