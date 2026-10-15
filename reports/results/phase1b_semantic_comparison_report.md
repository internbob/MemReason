# Phase 1B numeric versus semantic results

Status: completed under the frozen Phase 1B protocol.

## Fairness and data

- Dictionary-verified scalar inputs: 68
- Arms: ['numeric_only', 'variable_name', 'task_semantics']
- Frozen text encoder: `/home_data/home/zhouwl2023/project/New-AD/TAB-AD/models/Bio_ClinicalBERT`
- Samples: {'future_forecast': {'train': 16577, 'val': 2268, 'test': 4951}, 'cross_scale': {'train': 11735, 'val': 1666, 'test': 3463}}
- Trainable parameter counts: {'numeric_only': 269325, 'variable_name': 269325, 'task_semantics': 269325}
- All arms have identical trainable parameter counts and differ only in the fixed descriptor matrix.
- Validation checkpoint scores: {'future_forecast': {'numeric_only': {'2026': 0.4039585174108559, '2027': 0.40068158171361096, '2028': 0.3982279654144507}, 'variable_name': {'2026': 0.4031307052644764, '2027': 0.39984339859725954, '2028': 0.3985730824851551}, 'task_semantics': {'2026': 0.40283049927242753, '2027': 0.3998732911607351, '2028': 0.39875204296891364}}, 'cross_scale': {'numeric_only': {'2026': 0.37086011942248953, '2027': 0.36986326057353225, '2028': 0.3713537135453801}, 'variable_name': {'2026': 0.3699154144746779, '2027': 0.3684623510327335, '2028': 0.3729009378135114}, 'task_semantics': {'2026': 0.3700470854693055, '2027': 0.36866031398328236, '2028': 0.372965807631476}}}

## Test macro normalized MAE

| task | model | normalized_mae |
| --- | --- | --- |
| cross_adas_cdr_to_moca | numeric_only ensemble | 0.4261 |
| cross_adas_cdr_to_moca | task_semantics ensemble | 0.4226 |
| cross_adas_cdr_to_moca | variable_name ensemble | 0.4223 |
| cross_mmse_cdr_to_adas | numeric_only ensemble | 0.3684 |
| cross_mmse_cdr_to_adas | task_semantics ensemble | 0.3675 |
| cross_mmse_cdr_to_adas | variable_name ensemble | 0.3674 |
| future_forecast | numeric_only ensemble | 0.3766 |
| future_forecast | task_semantics ensemble | 0.3779 |
| future_forecast | variable_name ensemble | 0.3776 |

## Paired subject bootstrap

Negative normalized-MAE differences favor the candidate.

| task | candidate | baseline | n_paired_targets | n_subjects | normalized_mae_difference | ci_lower | ci_upper | bootstrap_probability_candidate_better | replicates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cross_adas_cdr_to_moca | variable_name ensemble | numeric_only ensemble | 797 | 336 | -0.003838 | -0.008692 | 0.0004544 | 0.9565 | 2000 |
| cross_adas_cdr_to_moca | task_semantics ensemble | numeric_only ensemble | 797 | 336 | -0.003509 | -0.008059 | 0.0005319 | 0.951 | 2000 |
| cross_mmse_cdr_to_adas | variable_name ensemble | numeric_only ensemble | 2666 | 612 | -0.001021 | -0.002657 | 0.0005963 | 0.8915 | 2000 |
| cross_mmse_cdr_to_adas | task_semantics ensemble | numeric_only ensemble | 2666 | 612 | -0.0009785 | -0.002707 | 0.0007251 | 0.876 | 2000 |
| future_forecast | variable_name ensemble | numeric_only ensemble | 20007 | 495 | 0.0005391 | 3.604e-05 | 0.001059 | 0.0165 | 2000 |
| future_forecast | task_semantics ensemble | numeric_only ensemble | 20007 | 495 | 0.0007403 | 0.0002447 | 0.001272 | 0.0025 | 2000 |

## Frozen-protocol decision

- Future forecasting: task semantics did not improve over numeric-only. A positive difference means worse error.
- Cross-scale: both declared pairs had small directionally favorable point estimates, but their 95% subject-bootstrap intervals crossed zero.
- The semantic-value gate is therefore **not passed**. These test results cannot be used to tune this run or claim stable semantic benefit.

## Complete test metrics

| task | model | target | horizon_months | n | n_subjects | mae | normalized_mae | rmse | spearman |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cross_adas_cdr_to_moca | numeric_only ensemble | moca_moca | 0 | 797 | 336 | 2.016 | 0.4261 | 2.575 | 0.718 |
| cross_adas_cdr_to_moca | task_semantics ensemble | moca_moca | 0 | 797 | 336 | 2 | 0.4226 | 2.552 | 0.7185 |
| cross_adas_cdr_to_moca | variable_name ensemble | moca_moca | 0 | 797 | 336 | 1.998 | 0.4223 | 2.55 | 0.7184 |
| cross_mmse_cdr_to_adas | numeric_only ensemble | adas_total13 | 0 | 2666 | 612 | 4.184 | 0.3684 | 5.309 | 0.7711 |
| cross_mmse_cdr_to_adas | task_semantics ensemble | adas_total13 | 0 | 2666 | 612 | 4.173 | 0.3675 | 5.304 | 0.7708 |
| cross_mmse_cdr_to_adas | variable_name ensemble | adas_total13 | 0 | 2666 | 612 | 4.173 | 0.3674 | 5.303 | 0.7708 |
| future_forecast | numeric_only ensemble | adas_total13 | 6 | 751 | 334 | 3.461 | 0.3047 | 4.514 | 0.902 |
| future_forecast | numeric_only ensemble | adas_total13 | 12 | 1362 | 410 | 3.424 | 0.3015 | 4.696 | 0.9094 |
| future_forecast | numeric_only ensemble | adas_total13 | 24 | 1493 | 380 | 3.838 | 0.338 | 5.455 | 0.8769 |
| future_forecast | numeric_only ensemble | adas_total13 | 36 | 1074 | 289 | 4.411 | 0.3884 | 6.671 | 0.8319 |
| future_forecast | numeric_only ensemble | cdr_cdrsb | 6 | 752 | 334 | 0.7471 | 0.2647 | 1.075 | 0.9012 |
| future_forecast | numeric_only ensemble | cdr_cdrsb | 12 | 1417 | 422 | 0.9019 | 0.3195 | 1.427 | 0.8834 |
| future_forecast | numeric_only ensemble | cdr_cdrsb | 24 | 1556 | 397 | 0.9244 | 0.3275 | 1.555 | 0.8516 |
| future_forecast | numeric_only ensemble | cdr_cdrsb | 36 | 1116 | 299 | 1.209 | 0.4283 | 2.122 | 0.7964 |
| future_forecast | numeric_only ensemble | faq_faqtotal | 6 | 753 | 334 | 2.722 | 0.348 | 3.993 | 0.8245 |
| future_forecast | numeric_only ensemble | faq_faqtotal | 12 | 1410 | 419 | 2.678 | 0.3423 | 4.133 | 0.8062 |
| future_forecast | numeric_only ensemble | faq_faqtotal | 24 | 1560 | 397 | 2.776 | 0.3549 | 4.551 | 0.7599 |
| future_forecast | numeric_only ensemble | faq_faqtotal | 36 | 1113 | 299 | 3.312 | 0.4234 | 5.456 | 0.7209 |
| future_forecast | numeric_only ensemble | mmse_mmscore | 6 | 759 | 335 | 1.379 | 0.3575 | 1.923 | 0.7915 |
| future_forecast | numeric_only ensemble | mmse_mmscore | 12 | 1379 | 413 | 1.451 | 0.3762 | 2.032 | 0.7852 |
| future_forecast | numeric_only ensemble | mmse_mmscore | 24 | 1510 | 384 | 1.565 | 0.4057 | 2.298 | 0.7507 |
| future_forecast | numeric_only ensemble | mmse_mmscore | 36 | 1084 | 292 | 1.848 | 0.4791 | 2.872 | 0.6708 |
| future_forecast | numeric_only ensemble | moca_moca | 6 | 6 | 6 | 2.078 | 0.4392 | 2.463 | 0.5885 |
| future_forecast | numeric_only ensemble | moca_moca | 12 | 257 | 153 | 2.031 | 0.4291 | 2.578 | 0.7257 |
| future_forecast | numeric_only ensemble | moca_moca | 24 | 367 | 176 | 2.054 | 0.4342 | 2.561 | 0.7143 |
| future_forecast | numeric_only ensemble | moca_moca | 36 | 288 | 146 | 2.227 | 0.4707 | 2.888 | 0.7061 |
| future_forecast | task_semantics ensemble | adas_total13 | 6 | 751 | 334 | 3.467 | 0.3053 | 4.525 | 0.9018 |
| future_forecast | task_semantics ensemble | adas_total13 | 12 | 1362 | 410 | 3.435 | 0.3024 | 4.707 | 0.9094 |
| future_forecast | task_semantics ensemble | adas_total13 | 24 | 1493 | 380 | 3.851 | 0.3391 | 5.453 | 0.8765 |
| future_forecast | task_semantics ensemble | adas_total13 | 36 | 1074 | 289 | 4.437 | 0.3907 | 6.669 | 0.831 |
| future_forecast | task_semantics ensemble | cdr_cdrsb | 6 | 752 | 334 | 0.7501 | 0.2657 | 1.085 | 0.901 |
| future_forecast | task_semantics ensemble | cdr_cdrsb | 12 | 1417 | 422 | 0.9071 | 0.3213 | 1.44 | 0.8831 |
| future_forecast | task_semantics ensemble | cdr_cdrsb | 24 | 1556 | 397 | 0.9258 | 0.328 | 1.561 | 0.8523 |
| future_forecast | task_semantics ensemble | cdr_cdrsb | 36 | 1116 | 299 | 1.213 | 0.4297 | 2.13 | 0.7973 |
| future_forecast | task_semantics ensemble | faq_faqtotal | 6 | 753 | 334 | 2.717 | 0.3474 | 3.994 | 0.8241 |
| future_forecast | task_semantics ensemble | faq_faqtotal | 12 | 1410 | 419 | 2.688 | 0.3437 | 4.144 | 0.8065 |
| future_forecast | task_semantics ensemble | faq_faqtotal | 24 | 1560 | 397 | 2.774 | 0.3546 | 4.559 | 0.7607 |
| future_forecast | task_semantics ensemble | faq_faqtotal | 36 | 1113 | 299 | 3.314 | 0.4236 | 5.461 | 0.7208 |
| future_forecast | task_semantics ensemble | mmse_mmscore | 6 | 759 | 335 | 1.379 | 0.3575 | 1.925 | 0.7913 |
| future_forecast | task_semantics ensemble | mmse_mmscore | 12 | 1379 | 413 | 1.454 | 0.377 | 2.033 | 0.7855 |
| future_forecast | task_semantics ensemble | mmse_mmscore | 24 | 1510 | 384 | 1.565 | 0.4058 | 2.291 | 0.7512 |
| future_forecast | task_semantics ensemble | mmse_mmscore | 36 | 1084 | 292 | 1.851 | 0.4798 | 2.865 | 0.6705 |
| future_forecast | task_semantics ensemble | moca_moca | 6 | 6 | 6 | 2.141 | 0.4524 | 2.525 | 0.5885 |
| future_forecast | task_semantics ensemble | moca_moca | 12 | 257 | 153 | 2.028 | 0.4286 | 2.575 | 0.7248 |
| future_forecast | task_semantics ensemble | moca_moca | 24 | 367 | 176 | 2.05 | 0.4334 | 2.559 | 0.7147 |
| future_forecast | task_semantics ensemble | moca_moca | 36 | 288 | 146 | 2.231 | 0.4715 | 2.89 | 0.7074 |
| future_forecast | variable_name ensemble | adas_total13 | 6 | 751 | 334 | 3.468 | 0.3054 | 4.526 | 0.9018 |
| future_forecast | variable_name ensemble | adas_total13 | 12 | 1362 | 410 | 3.435 | 0.3024 | 4.707 | 0.9094 |
| future_forecast | variable_name ensemble | adas_total13 | 24 | 1493 | 380 | 3.849 | 0.3389 | 5.452 | 0.8766 |
| future_forecast | variable_name ensemble | adas_total13 | 36 | 1074 | 289 | 4.433 | 0.3903 | 6.669 | 0.8311 |
| future_forecast | variable_name ensemble | cdr_cdrsb | 6 | 752 | 334 | 0.7487 | 0.2652 | 1.084 | 0.9011 |
| future_forecast | variable_name ensemble | cdr_cdrsb | 12 | 1417 | 422 | 0.9067 | 0.3212 | 1.438 | 0.8832 |
| future_forecast | variable_name ensemble | cdr_cdrsb | 24 | 1556 | 397 | 0.9248 | 0.3276 | 1.558 | 0.8523 |
| future_forecast | variable_name ensemble | cdr_cdrsb | 36 | 1116 | 299 | 1.212 | 0.4295 | 2.129 | 0.7974 |
| future_forecast | variable_name ensemble | faq_faqtotal | 6 | 753 | 334 | 2.716 | 0.3472 | 3.992 | 0.8244 |
| future_forecast | variable_name ensemble | faq_faqtotal | 12 | 1410 | 419 | 2.685 | 0.3432 | 4.139 | 0.8066 |
| future_forecast | variable_name ensemble | faq_faqtotal | 24 | 1560 | 397 | 2.772 | 0.3543 | 4.553 | 0.7611 |
| future_forecast | variable_name ensemble | faq_faqtotal | 36 | 1113 | 299 | 3.312 | 0.4234 | 5.456 | 0.7212 |
| future_forecast | variable_name ensemble | mmse_mmscore | 6 | 759 | 335 | 1.379 | 0.3576 | 1.926 | 0.7914 |
| future_forecast | variable_name ensemble | mmse_mmscore | 12 | 1379 | 413 | 1.455 | 0.3771 | 2.034 | 0.7855 |
| future_forecast | variable_name ensemble | mmse_mmscore | 24 | 1510 | 384 | 1.565 | 0.4058 | 2.292 | 0.751 |
| future_forecast | variable_name ensemble | mmse_mmscore | 36 | 1084 | 292 | 1.85 | 0.4795 | 2.865 | 0.6704 |
| future_forecast | variable_name ensemble | moca_moca | 6 | 6 | 6 | 2.129 | 0.45 | 2.51 | 0.5885 |
| future_forecast | variable_name ensemble | moca_moca | 12 | 257 | 153 | 2.029 | 0.4288 | 2.575 | 0.7255 |
| future_forecast | variable_name ensemble | moca_moca | 24 | 367 | 176 | 2.049 | 0.4332 | 2.557 | 0.7154 |
| future_forecast | variable_name ensemble | moca_moca | 36 | 288 | 146 | 2.227 | 0.4706 | 2.885 | 0.7078 |

## Interpretation limits

- This experiment tests frozen item metadata embeddings, not generated response text or a fine-tuned language model.
- MoCA delayed-recall cue categories were verified but excluded because they are non-monotonic categories in a scalar-response comparison.
- Cross-scale targets are concurrent and establish representation transfer, not temporal forecasting.
- MRI, diagnosis, SFT and RL are absent.
- Test results must not be used to modify this run ID. Any follow-up requires a new protocol and run identifier.
