# Phase 4E trajectory-action SFT and constrained GRPO

Status: development experiment completed and stopped at the frozen development gate. Real on-policy reward varied with bounded forecast and uncertainty actions.

## Frozen design

- Action-SFT / rollout / development: 256 / 96 / 256 mutually exclusive Phase 4A build subjects
- Phase 4D rollout subjects excluded: 64
- Future truth in prompts or action-SFT: no / no
- GRPO group size / completed updates / candidates: 4 / 32 / 256
- Trainable component: existing 1.5B-model LoRA parameters only; GRU-D, evidence, state/trend statistics, and renderer stayed frozen
- Original test loaded: no

## Development results

| arm | normalized_mae | normalized_crps | temporal_accuracy | raw_clinical_range_rate | clinical_range_rate | reward_outcome | reward_uncertainty | reward_temporal | reward_clinical_range | total_reward |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| action_sft | 0.3997 | 0.3569 | 0.5525 | 0.8094 | 1 | 0.6942 | 0.7218 | 0.5525 | 0.8094 | 0.6957 |
| frozen | 0.3511 | 0.319 | 0.5938 | 0.9297 | 1 | 0.7263 | 0.7483 | 0.5938 | 0.9297 | 0.7367 |
| grpo_u16 | 0.3997 | 0.3569 | 0.5525 | 0.8094 | 1 | 0.6942 | 0.7218 | 0.5525 | 0.8094 | 0.6957 |
| grpo_u24 | 0.3997 | 0.3569 | 0.5525 | 0.8094 | 1 | 0.6942 | 0.7218 | 0.5525 | 0.8094 | 0.6957 |
| grpo_u32 | 0.3997 | 0.3569 | 0.5525 | 0.8094 | 1 | 0.6942 | 0.7218 | 0.5525 | 0.8094 | 0.6957 |
| grpo_u8 | 0.3997 | 0.3569 | 0.5525 | 0.8094 | 1 | 0.6942 | 0.7218 | 0.5525 | 0.8094 | 0.6957 |

Selected checkpoint: update 8. Its normalized MAE was 0.39970, versus 0.35108 for frozen GRU-D and 0.39970 for matched action-SFT. Reward-variable groups comprised 0.3125 of rollout groups.

Greedy action diagnosis found that action-SFT and all four GRPO checkpoints selected `lower` residual plus `wide` uncertainty for every endpoint and every development report. Thus the one-epoch action-SFT did not recover its intended zero-residual/unit-uncertainty anchor under constrained first-token decoding, and GRPO did not alter that policy. This failure is part of the frozen result; training duration, temperature, and reward weights will not be retuned on these development subjects.

## Subject-bootstrap comparisons

| comparison | metric | difference | ci_low | ci_high | replicates |
| --- | --- | --- | --- | --- | --- |
| grpo_u8_minus_frozen | normalized_mae | 0.04862 | 0.0362 | 0.06064 | 2000 |
| grpo_u8_minus_frozen | normalized_crps | 0.03794 | 0.02655 | 0.04923 | 2000 |
| grpo_u8_minus_frozen | temporal_accuracy | -0.04134 | -0.07788 | -0.005265 | 2000 |
| grpo_u8_minus_action_sft | normalized_mae | 0 | 0 | 0 | 2000 |
| grpo_u8_minus_action_sft | normalized_crps | 0 | 0 | 0 | 2000 |
| grpo_u8_minus_action_sft | temporal_accuracy | 0 | 0 | 0 | 2000 |

## Frozen decision

```json
{
  "run_id": "phase4e_trajectory_action_grpo_20261015",
  "trajectory_action_sft_completed": true,
  "real_on_policy_grpo_completed": true,
  "development_gate_passed": false,
  "confirmation_opened": false,
  "original_test_loaded": false,
  "selected_update": 8,
  "grpo_minus_frozen_normalized_mae": 0.04862024960803846,
  "grpo_minus_action_sft_normalized_mae": 0.0,
  "action_sft_anchor_recovered": false,
  "greedy_action_policy": "lower_plus_wide_for_every_target",
  "failure_mode": "action_sft_policy_collapse_unchanged_by_grpo",
  "trajectory_action_rlvr_supported": false,
  "next_action": "retain_frozen_grud_and_phase4c_policy_stop_phase4e_tuning"
}
```

The result tests cognitive-only bounded trajectory actions. It does not support multimodal GRPO, and failure at this gate forbids further tuning on these development subjects.
