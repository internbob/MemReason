# Joint response-aware Stage I results

Status: completed under the frozen Phase 1C protocol.

## Fairness and data

- Verified scalar inputs: 68
- Cognitive domains: ['Attention', 'Executive', 'Language', 'Memory', 'Orientation']
- Arms: ['numeric_only', 'task_semantics', 'response_aware']
- Trainable parameter counts: {'numeric_only': 290993, 'task_semantics': 290993, 'response_aware': 290993}
- Samples by task/split: {'train': {'future_forecast': 16577, 'cross_adas_cdr_to_moca': 2781, 'cross_mmse_cdr_to_adas': 8954, 'masked_item': 10404, 'masked_instrument': 9806}, 'val': {'future_forecast': 2268, 'cross_adas_cdr_to_moca': 429, 'cross_mmse_cdr_to_adas': 1237, 'masked_item': 1452, 'masked_instrument': 1355}, 'test': {'future_forecast': 4951, 'cross_adas_cdr_to_moca': 797, 'cross_mmse_cdr_to_adas': 2666, 'masked_item': 3059, 'masked_instrument': 2892}}
- Validation scores: {'numeric_only': {'2026': 0.38299153167339556, '2027': 0.3818784369122904, '2028': 0.3837160022992635}, 'task_semantics': {'2026': 0.38434078162139607, '2027': 0.3810579924518186, '2028': 0.383000542013917}, 'response_aware': {'2026': 0.38439434139611073, '2027': 0.3812136147215482, '2028': 0.3829053600506412}}
- Test samples were loaded only after all nine validation-trained checkpoints completed.

## Test macro normalized MAE

| task | model | normalized_mae |
| --- | --- | --- |
| cross_adas_cdr_to_moca | numeric_only ensemble | 0.4213 |
| cross_adas_cdr_to_moca | response_aware ensemble | 0.4236 |
| cross_adas_cdr_to_moca | task_semantics ensemble | 0.4236 |
| cross_mmse_cdr_to_adas | numeric_only ensemble | 0.3683 |
| cross_mmse_cdr_to_adas | response_aware ensemble | 0.3663 |
| cross_mmse_cdr_to_adas | task_semantics ensemble | 0.3663 |
| future_forecast | numeric_only ensemble | 0.3804 |
| future_forecast | response_aware ensemble | 0.3828 |
| future_forecast | task_semantics ensemble | 0.3826 |
| masked_instrument | numeric_only ensemble | 0.3678 |
| masked_instrument | response_aware ensemble | 0.3746 |
| masked_instrument | task_semantics ensemble | 0.3746 |
| masked_item | numeric_only ensemble | 0.3423 |
| masked_item | response_aware ensemble | 0.3507 |
| masked_item | task_semantics ensemble | 0.3509 |

## Reconstruction metrics by value type

| task | model | value_type | n | mae | accuracy | auroc | weighted_kappa |
| --- | --- | --- | --- | --- | --- | --- | --- |
| masked_instrument | numeric_only ensemble | binary | 32694 | 0.1099 | 0.9131 | 0.8555 |  |
| masked_instrument | numeric_only ensemble | continuous | 11400 | 0.6689 |  |  |  |
| masked_instrument | numeric_only ensemble | ordinal | 2394 | 0.2179 |  |  | 0.7172 |
| masked_instrument | response_aware ensemble | binary | 32694 | 0.1127 | 0.9133 | 0.8607 |  |
| masked_instrument | response_aware ensemble | continuous | 11400 | 0.6675 |  |  |  |
| masked_instrument | response_aware ensemble | ordinal | 2394 | 0.2209 |  |  | 0.7166 |
| masked_instrument | task_semantics ensemble | binary | 32694 | 0.1126 | 0.9134 | 0.8608 |  |
| masked_instrument | task_semantics ensemble | continuous | 11400 | 0.6676 |  |  |  |
| masked_instrument | task_semantics ensemble | ordinal | 2394 | 0.221 |  |  | 0.7139 |
| masked_item | numeric_only ensemble | binary | 2073 | 0.1052 | 0.9257 | 0.8656 |  |
| masked_item | numeric_only ensemble | continuous | 712 | 0.6359 |  |  |  |
| masked_item | numeric_only ensemble | ordinal | 274 | 0.2034 |  |  | 0.8605 |
| masked_item | response_aware ensemble | binary | 2073 | 0.1086 | 0.9228 | 0.8631 |  |
| masked_item | response_aware ensemble | continuous | 712 | 0.6242 |  |  |  |
| masked_item | response_aware ensemble | ordinal | 274 | 0.2067 |  |  | 0.8623 |
| masked_item | task_semantics ensemble | binary | 2073 | 0.1086 | 0.9228 | 0.863 |  |
| masked_item | task_semantics ensemble | continuous | 712 | 0.6247 |  |  |  |
| masked_item | task_semantics ensemble | ordinal | 274 | 0.207 |  |  | 0.8679 |

## Paired subject bootstrap

Negative normalized-MAE differences favor the candidate.

| task | candidate | baseline | n_paired_targets | n_subjects | normalized_mae_difference | ci_lower | ci_upper | bootstrap_probability_candidate_better | replicates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cross_adas_cdr_to_moca | task_semantics ensemble | numeric_only ensemble | 797 | 336 | 0.002371 | -0.0008272 | 0.005878 | 0.076 | 2000 |
| cross_adas_cdr_to_moca | response_aware ensemble | numeric_only ensemble | 797 | 336 | 0.002368 | -0.0007911 | 0.005787 | 0.0725 | 2000 |
| cross_mmse_cdr_to_adas | task_semantics ensemble | numeric_only ensemble | 2666 | 612 | -0.001954 | -0.003453 | -0.0004428 | 0.994 | 2000 |
| cross_mmse_cdr_to_adas | response_aware ensemble | numeric_only ensemble | 2666 | 612 | -0.001936 | -0.003428 | -0.0004205 | 0.994 | 2000 |
| future_forecast | task_semantics ensemble | numeric_only ensemble | 20007 | 495 | 0.002753 | -0.001902 | 0.007282 | 0.1185 | 2000 |
| future_forecast | response_aware ensemble | numeric_only ensemble | 20007 | 495 | 0.002842 | -0.001885 | 0.007363 | 0.1115 | 2000 |
| masked_instrument | task_semantics ensemble | numeric_only ensemble | 46488 | 795 | 0.006456 | 0.005609 | 0.007272 | 0 | 2000 |
| masked_instrument | response_aware ensemble | numeric_only ensemble | 46488 | 795 | 0.00653 | 0.005682 | 0.007349 | 0 | 2000 |
| masked_item | task_semantics ensemble | numeric_only ensemble | 3059 | 860 | 0.007817 | 0.004683 | 0.0112 | 0 | 2000 |
| masked_item | response_aware ensemble | numeric_only ensemble | 3059 | 860 | 0.007631 | 0.004491 | 0.01102 | 0 | 2000 |

## Frozen Gate 1 decision

{
  "run_id": "phase1c_joint_stage1_20261004",
  "criteria": {
    "masked_item_improved": false,
    "masked_instrument_improved": false,
    "cross_scale_two_pairs_improved": false,
    "no_diagnosis_leakage": true,
    "future_noninferior": false
  },
  "stage1_gate_passed": false,
  "future_noninferiority_supportive": false,
  "missingness_only_validation_balanced_accuracy": 0.4196486931976516,
  "advance_to_mri_stage2": false,
  "note": "advance_to_mri_stage2 remains false unless all frozen Stage I semantic criteria pass."
}

## Interpretation limits

- Bio_ClinicalBERT is frozen; this experiment tests fixed response-text descriptors inside a trainable cognitive encoder.
- The internal ADNI test split has appeared in earlier project experiments and is not an external confirmation cohort.
- Cross-scale outcomes are concurrent. Future forecasting is temporally prospective under nominal visit months.
- MRI, diagnosis, SFT and RL are absent from model inputs.
- These test results cannot be used to modify this run ID.
