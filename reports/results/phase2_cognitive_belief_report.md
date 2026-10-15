# Cognitive-only irregular-time belief dynamics and calibration

Status: completed validation-only development experiment. Test was not loaded.

The typed item/response encoder is used as a component. This experiment makes no universal semantic-superiority claim.

## Contract

- Sample counts: `{"calibration": {"forecast": 1179, "interpolation": 413, "samples": 1592, "subjects": 221}, "evaluation": {"forecast": 1089, "interpolation": 375, "samples": 1464, "subjects": 210}, "fit": {"forecast": 14915, "interpolation": 5150, "samples": 20065, "subjects": 2734}, "tune": {"forecast": 1662, "interpolation": 585, "samples": 2247, "subjects": 287}}`
- Trainable parameter counts: `{"grud_gaussian": 265554, "typed_belief_dynamics": 82379, "typed_direct_gaussian": 290130}`
- Three-seed ensemble variance combines within-model and between-seed uncertainty.
- Variance temperatures were fit on subject-disjoint validation-calibration cases and evaluated on validation-evaluation cases.

## Evaluation summary

| model | calibrated | n_targets | n_subjects | normalized_mae | gaussian_nll | gaussian_crps | coverage_95 | width_95_normalized | multilevel_calibration_error | uncertainty_error_spearman | uncertainty_missingness_spearman |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grud_gaussian | False | 5936 | 114 | 0.3715 | 0.6591 | 0.2684 | 0.9431 | 1.745 | 0.02885 | 0.5805 | 0.1843 |
| grud_gaussian | True | 5936 | 114 | 0.3715 | 0.6163 | 0.2697 | 0.9468 | 1.833 | 0.02914 | 0.5624 | 0.1596 |
| typed_belief_dynamics | False | 5936 | 114 | 0.3813 | 0.7171 | 0.2801 | 0.9311 | 1.798 | 0.02792 | 0.5511 | 0.1614 |
| typed_belief_dynamics | True | 5936 | 114 | 0.3813 | 0.6919 | 0.2805 | 0.9316 | 1.832 | 0.02859 | 0.5405 | 0.142 |
| typed_direct_gaussian | False | 5936 | 114 | 0.3609 | 0.6421 | 0.2649 | 0.9387 | 1.736 | 0.03736 | 0.5723 | 0.1846 |
| typed_direct_gaussian | True | 5936 | 114 | 0.3609 | 0.6061 | 0.2659 | 0.9387 | 1.78 | 0.035 | 0.5568 | 0.1588 |

## Interpretation

The explicit typed belief arm reached calibrated 95% coverage 0.9316 and multilevel calibration error 0.0286, but its overall normalized-MAE difference versus GRU-D was +0.00983 (95% CI [-0.00074, +0.02187]) and its NLL difference was +0.07564 (95% CI [+0.01566, +0.13653]). It therefore fails the frozen dynamics gate. The typed direct Gaussian arm had the best point estimates (normalized MAE 0.3609; NLL 0.6061), but its overall paired intervals still crossed zero (MAE difference -0.01060, 95% CI [-0.03355, +0.00577]). Its interpolation MAE improvement was supported (-0.01437, 95% CI [-0.02971, -0.00419]), which is a task-specific result rather than evidence for the explicit belief mechanism.

## Calibrated task/target/horizon results

| model | task | target | horizon_months | n_targets | normalized_mae | gaussian_nll | coverage_95 | multilevel_calibration_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grud_gaussian | forecast | adas_total13 | 6 | 172 | 0.2797 | 0.4291 | 0.9593 | 0.02093 |
| grud_gaussian | forecast | adas_total13 | 12 | 288 | 0.3425 | 0.6088 | 0.9271 | 0.0401 |
| grud_gaussian | forecast | adas_total13 | 24 | 317 | 0.3536 | 0.7546 | 0.9401 | 0.01321 |
| grud_gaussian | forecast | adas_total13 | 36 | 211 | 0.4661 | 1.317 | 0.9479 | 0.003851 |
| grud_gaussian | forecast | cdr_cdrsb | 6 | 174 | 0.2697 | 0.1275 | 0.9885 | 0.08894 |
| grud_gaussian | forecast | cdr_cdrsb | 12 | 310 | 0.3401 | 0.3864 | 0.9548 | 0.03992 |
| grud_gaussian | forecast | cdr_cdrsb | 24 | 352 | 0.3862 | 0.7231 | 0.9489 | 0.09517 |
| grud_gaussian | forecast | cdr_cdrsb | 36 | 238 | 0.4921 | 1.199 | 0.9748 | 0.1138 |
| grud_gaussian | forecast | faq_faqtotal | 6 | 175 | 0.3486 | 0.5413 | 0.9829 | 0.1239 |
| grud_gaussian | forecast | faq_faqtotal | 12 | 311 | 0.3848 | 0.5546 | 0.9228 | 0.05406 |
| grud_gaussian | forecast | faq_faqtotal | 24 | 348 | 0.3701 | 0.5503 | 0.9569 | 0.09037 |
| grud_gaussian | forecast | faq_faqtotal | 36 | 238 | 0.4469 | 0.7761 | 0.9538 | 0.09905 |
| grud_gaussian | forecast | mmse_mmscore | 6 | 175 | 0.4139 | 0.743 | 0.9314 | 0.06464 |
| grud_gaussian | forecast | mmse_mmscore | 12 | 290 | 0.4434 | 0.8063 | 0.9241 | 0.009914 |
| grud_gaussian | forecast | mmse_mmscore | 24 | 319 | 0.434 | 0.9067 | 0.9342 | 0.01195 |
| grud_gaussian | forecast | mmse_mmscore | 36 | 209 | 0.5099 | 1.124 | 0.9378 | 0.02362 |
| grud_gaussian | forecast | moca_moca | 6 | 4 | 0.4277 | 0.694 | 1 | 0.1125 |
| grud_gaussian | forecast | moca_moca | 12 | 78 | 0.4896 | 1.01 | 0.8718 | 0.1465 |
| grud_gaussian | forecast | moca_moca | 24 | 100 | 0.4797 | 0.8645 | 0.94 | 0.0375 |
| grud_gaussian | forecast | moca_moca | 36 | 86 | 0.4808 | 0.8797 | 0.9767 | 0.04099 |
| grud_gaussian | interpolation | adas_total13 | 0 | 352 | 0.2654 | 0.3429 | 0.9375 | 0.02756 |
| grud_gaussian | interpolation | cdr_cdrsb | 0 | 370 | 0.2259 | -0.1413 | 0.9703 | 0.05304 |
| grud_gaussian | interpolation | faq_faqtotal | 0 | 368 | 0.2999 | 0.2922 | 0.9511 | 0.0413 |
| grud_gaussian | interpolation | mmse_mmscore | 0 | 355 | 0.3503 | 0.5253 | 0.938 | 0.01496 |
| grud_gaussian | interpolation | moca_moca | 0 | 96 | 0.4141 | 0.7585 | 0.9375 | 0.1 |
| typed_belief_dynamics | forecast | adas_total13 | 6 | 172 | 0.2866 | 0.3946 | 0.9651 | 0.02209 |
| typed_belief_dynamics | forecast | adas_total13 | 12 | 288 | 0.3506 | 0.6349 | 0.9097 | 0.04618 |
| typed_belief_dynamics | forecast | adas_total13 | 24 | 317 | 0.3644 | 0.8179 | 0.9401 | 0.02188 |
| typed_belief_dynamics | forecast | adas_total13 | 36 | 211 | 0.4869 | 1.423 | 0.9289 | 0.02518 |
| typed_belief_dynamics | forecast | cdr_cdrsb | 6 | 174 | 0.2864 | 0.2409 | 0.931 | 0.0625 |
| typed_belief_dynamics | forecast | cdr_cdrsb | 12 | 310 | 0.3474 | 0.6042 | 0.9194 | 0.03589 |
| typed_belief_dynamics | forecast | cdr_cdrsb | 24 | 352 | 0.4003 | 0.9442 | 0.9375 | 0.06534 |
| typed_belief_dynamics | forecast | cdr_cdrsb | 36 | 238 | 0.4969 | 1.513 | 0.9412 | 0.08876 |
| typed_belief_dynamics | forecast | faq_faqtotal | 6 | 175 | 0.3891 | 0.5835 | 0.9371 | 0.07179 |
| typed_belief_dynamics | forecast | faq_faqtotal | 12 | 311 | 0.3987 | 0.627 | 0.91 | 0.03943 |
| typed_belief_dynamics | forecast | faq_faqtotal | 24 | 348 | 0.3974 | 0.62 | 0.9511 | 0.06595 |
| typed_belief_dynamics | forecast | faq_faqtotal | 36 | 238 | 0.476 | 0.8769 | 0.9496 | 0.07931 |
| typed_belief_dynamics | forecast | mmse_mmscore | 6 | 175 | 0.3836 | 0.7054 | 0.9143 | 0.03321 |
| typed_belief_dynamics | forecast | mmse_mmscore | 12 | 290 | 0.4494 | 0.8675 | 0.9138 | 0.02284 |
| typed_belief_dynamics | forecast | mmse_mmscore | 24 | 319 | 0.445 | 0.9063 | 0.9373 | 0.02998 |
| typed_belief_dynamics | forecast | mmse_mmscore | 36 | 209 | 0.5285 | 1.097 | 0.933 | 0.043 |
| typed_belief_dynamics | forecast | moca_moca | 6 | 4 | 0.4094 | 0.6654 | 1 | 0.0875 |
| typed_belief_dynamics | forecast | moca_moca | 12 | 78 | 0.4654 | 0.9012 | 0.8974 | 0.08558 |
| typed_belief_dynamics | forecast | moca_moca | 24 | 100 | 0.4565 | 0.8787 | 0.95 | 0.0175 |
| typed_belief_dynamics | forecast | moca_moca | 36 | 86 | 0.4795 | 0.9239 | 0.9884 | 0.05843 |
| typed_belief_dynamics | interpolation | adas_total13 | 0 | 352 | 0.2717 | 0.3751 | 0.9034 | 0.04602 |
| typed_belief_dynamics | interpolation | cdr_cdrsb | 0 | 370 | 0.2371 | 0.05729 | 0.9541 | 0.04223 |
| typed_belief_dynamics | interpolation | faq_faqtotal | 0 | 368 | 0.2932 | 0.214 | 0.9293 | 0.06182 |
| typed_belief_dynamics | interpolation | mmse_mmscore | 0 | 355 | 0.365 | 0.6124 | 0.9211 | 0.01567 |
| typed_belief_dynamics | interpolation | moca_moca | 0 | 96 | 0.4076 | 0.7384 | 0.9375 | 0.1104 |
| typed_direct_gaussian | forecast | adas_total13 | 6 | 172 | 0.2917 | 0.4268 | 0.936 | 0.01163 |
| typed_direct_gaussian | forecast | adas_total13 | 12 | 288 | 0.3348 | 0.6021 | 0.9271 | 0.0349 |
| typed_direct_gaussian | forecast | adas_total13 | 24 | 317 | 0.3464 | 0.7565 | 0.9432 | 0.01479 |
| typed_direct_gaussian | forecast | adas_total13 | 36 | 211 | 0.4455 | 1.279 | 0.9336 | 0.007405 |
| typed_direct_gaussian | forecast | cdr_cdrsb | 6 | 174 | 0.2552 | 0.06531 | 0.9713 | 0.07601 |
| typed_direct_gaussian | forecast | cdr_cdrsb | 12 | 310 | 0.3261 | 0.3244 | 0.9484 | 0.04315 |
| typed_direct_gaussian | forecast | cdr_cdrsb | 24 | 352 | 0.3811 | 0.7675 | 0.9489 | 0.09162 |
| typed_direct_gaussian | forecast | cdr_cdrsb | 36 | 238 | 0.4772 | 1.254 | 0.9664 | 0.1064 |
| typed_direct_gaussian | forecast | faq_faqtotal | 6 | 175 | 0.3447 | 0.4708 | 0.9714 | 0.09107 |
| typed_direct_gaussian | forecast | faq_faqtotal | 12 | 311 | 0.3629 | 0.5414 | 0.9357 | 0.05004 |
| typed_direct_gaussian | forecast | faq_faqtotal | 24 | 348 | 0.3645 | 0.5145 | 0.9569 | 0.08606 |
| typed_direct_gaussian | forecast | faq_faqtotal | 36 | 238 | 0.4404 | 0.7668 | 0.9622 | 0.1148 |
| typed_direct_gaussian | forecast | mmse_mmscore | 6 | 175 | 0.4114 | 0.8218 | 0.88 | 0.0575 |
| typed_direct_gaussian | forecast | mmse_mmscore | 12 | 290 | 0.4383 | 0.8532 | 0.9103 | 0.02974 |
| typed_direct_gaussian | forecast | mmse_mmscore | 24 | 319 | 0.419 | 0.9216 | 0.9248 | 0.03484 |
| typed_direct_gaussian | forecast | mmse_mmscore | 36 | 209 | 0.4897 | 1.091 | 0.933 | 0.04061 |
| typed_direct_gaussian | forecast | moca_moca | 6 | 4 | 0.4752 | 0.8374 | 1 | 0.125 |
| typed_direct_gaussian | forecast | moca_moca | 12 | 78 | 0.4886 | 0.9919 | 0.859 | 0.1144 |
| typed_direct_gaussian | forecast | moca_moca | 24 | 100 | 0.4745 | 0.8818 | 0.94 | 0.04 |
| typed_direct_gaussian | forecast | moca_moca | 36 | 86 | 0.4864 | 0.9178 | 0.9651 | 0.03169 |
| typed_direct_gaussian | interpolation | adas_total13 | 0 | 352 | 0.2622 | 0.3267 | 0.9205 | 0.03466 |
| typed_direct_gaussian | interpolation | cdr_cdrsb | 0 | 370 | 0.2082 | -0.1945 | 0.9595 | 0.05777 |
| typed_direct_gaussian | interpolation | faq_faqtotal | 0 | 368 | 0.2765 | 0.2157 | 0.9511 | 0.05353 |
| typed_direct_gaussian | interpolation | mmse_mmscore | 0 | 355 | 0.3366 | 0.5449 | 0.9183 | 0.02835 |
| typed_direct_gaussian | interpolation | moca_moca | 0 | 96 | 0.4037 | 0.7474 | 0.9062 | 0.09479 |

## Paired subject-cluster bootstrap

Negative differences favor the candidate.

| task | candidate | baseline | n_paired_targets | n_subjects | replicates | mae_difference | mae_difference_ci_lower | mae_difference_ci_upper | mae_difference_probability_better | nll_difference | nll_difference_ci_lower | nll_difference_ci_upper | nll_difference_probability_better |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | typed_direct_gaussian | grud_gaussian | 5936 | 114 | 2000 | -0.0106 | -0.03355 | 0.005774 | 0.7805 | -0.01014 | -0.04423 | 0.01838 | 0.6835 |
| forecast | typed_direct_gaussian | grud_gaussian | 4395 | 114 | 2000 | -0.009282 | -0.03877 | 0.01017 | 0.7015 | -0.002853 | -0.04203 | 0.0278 | 0.532 |
| interpolation | typed_direct_gaussian | grud_gaussian | 1541 | 95 | 2000 | -0.01437 | -0.02971 | -0.00419 | 1 | -0.03093 | -0.06668 | 0.003189 | 0.9645 |
| overall | typed_belief_dynamics | grud_gaussian | 5936 | 114 | 2000 | 0.009834 | -0.0007356 | 0.02187 | 0.038 | 0.07564 | 0.01566 | 0.1365 | 0.006 |
| forecast | typed_belief_dynamics | grud_gaussian | 4395 | 114 | 2000 | 0.01134 | -0.001476 | 0.02485 | 0.048 | 0.08283 | 0.02773 | 0.144 | 0.0015 |
| interpolation | typed_belief_dynamics | grud_gaussian | 1541 | 95 | 2000 | 0.005529 | -0.003791 | 0.01607 | 0.127 | 0.05515 | -0.03329 | 0.1443 | 0.1185 |

## Calibration temperatures

| model | level | task | target | horizon_months | n | temperature |
| --- | --- | --- | --- | --- | --- | --- |
| grud_gaussian | exact | forecast | adas_total13 | 6 | 180 | 0.9553 |
| grud_gaussian | exact | forecast | adas_total13 | 12 | 331 | 0.9003 |
| grud_gaussian | exact | forecast | adas_total13 | 24 | 353 | 0.9799 |
| grud_gaussian | exact | forecast | adas_total13 | 36 | 235 | 1.085 |
| grud_gaussian | exact | forecast | cdr_cdrsb | 6 | 179 | 0.9816 |
| grud_gaussian | exact | forecast | cdr_cdrsb | 12 | 348 | 0.9179 |
| grud_gaussian | exact | forecast | cdr_cdrsb | 24 | 379 | 1.198 |
| grud_gaussian | exact | forecast | cdr_cdrsb | 36 | 263 | 1.458 |
| grud_gaussian | exact | forecast | faq_faqtotal | 6 | 181 | 1.263 |
| grud_gaussian | exact | forecast | faq_faqtotal | 12 | 348 | 1.019 |
| grud_gaussian | exact | forecast | faq_faqtotal | 24 | 377 | 1.195 |
| grud_gaussian | exact | forecast | faq_faqtotal | 36 | 260 | 1.435 |
| grud_gaussian | exact | forecast | mmse_mmscore | 6 | 181 | 0.8665 |
| grud_gaussian | exact | forecast | mmse_mmscore | 12 | 332 | 1.041 |
| grud_gaussian | exact | forecast | mmse_mmscore | 24 | 355 | 1.019 |
| grud_gaussian | exact | forecast | mmse_mmscore | 36 | 238 | 1.176 |
| grud_gaussian | exact | forecast | moca_moca | 12 | 76 | 0.7896 |
| grud_gaussian | exact | forecast | moca_moca | 24 | 101 | 0.984 |
| grud_gaussian | exact | forecast | moca_moca | 36 | 77 | 1.164 |
| grud_gaussian | exact | interpolation | adas_total13 | 0 | 392 | 0.8763 |
| grud_gaussian | exact | interpolation | cdr_cdrsb | 0 | 409 | 0.8447 |
| grud_gaussian | exact | interpolation | faq_faqtotal | 0 | 408 | 0.982 |
| grud_gaussian | exact | interpolation | mmse_mmscore | 0 | 394 | 0.944 |
| grud_gaussian | exact | interpolation | moca_moca | 0 | 97 | 0.8037 |
| grud_gaussian | task_target | forecast | adas_total13 | -1 | 1099 | 0.9766 |
| grud_gaussian | task_target | forecast | cdr_cdrsb | -1 | 1169 | 1.158 |
| grud_gaussian | task_target | forecast | faq_faqtotal | -1 | 1166 | 1.216 |
| grud_gaussian | task_target | forecast | mmse_mmscore | -1 | 1106 | 1.039 |
| grud_gaussian | task_target | forecast | moca_moca | -1 | 257 | 0.9906 |
| grud_gaussian | task_target | interpolation | adas_total13 | -1 | 392 | 0.8763 |
| grud_gaussian | task_target | interpolation | cdr_cdrsb | -1 | 409 | 0.8447 |
| grud_gaussian | task_target | interpolation | faq_faqtotal | -1 | 408 | 0.982 |
| grud_gaussian | task_target | interpolation | mmse_mmscore | -1 | 394 | 0.944 |
| grud_gaussian | task_target | interpolation | moca_moca | -1 | 97 | 0.8037 |
| grud_gaussian | global | * | * | -1 | 6497 | 1.052 |
| typed_belief_dynamics | exact | forecast | adas_total13 | 6 | 180 | 0.9862 |
| typed_belief_dynamics | exact | forecast | adas_total13 | 12 | 331 | 0.9237 |
| typed_belief_dynamics | exact | forecast | adas_total13 | 24 | 353 | 1.03 |
| typed_belief_dynamics | exact | forecast | adas_total13 | 36 | 235 | 1.134 |
| typed_belief_dynamics | exact | forecast | cdr_cdrsb | 6 | 179 | 0.9906 |
| typed_belief_dynamics | exact | forecast | cdr_cdrsb | 12 | 348 | 0.9175 |
| typed_belief_dynamics | exact | forecast | cdr_cdrsb | 24 | 379 | 1.034 |
| typed_belief_dynamics | exact | forecast | cdr_cdrsb | 36 | 263 | 1.16 |
| typed_belief_dynamics | exact | forecast | faq_faqtotal | 6 | 181 | 1.019 |
| typed_belief_dynamics | exact | forecast | faq_faqtotal | 12 | 348 | 0.9106 |
| typed_belief_dynamics | exact | forecast | faq_faqtotal | 24 | 377 | 1.048 |
| typed_belief_dynamics | exact | forecast | faq_faqtotal | 36 | 260 | 1.183 |
| typed_belief_dynamics | exact | forecast | mmse_mmscore | 6 | 181 | 0.8998 |
| typed_belief_dynamics | exact | forecast | mmse_mmscore | 12 | 332 | 1.052 |
| typed_belief_dynamics | exact | forecast | mmse_mmscore | 24 | 355 | 1.061 |
| typed_belief_dynamics | exact | forecast | mmse_mmscore | 36 | 238 | 1.21 |
| typed_belief_dynamics | exact | forecast | moca_moca | 12 | 76 | 0.8213 |
| typed_belief_dynamics | exact | forecast | moca_moca | 24 | 101 | 1.003 |
| typed_belief_dynamics | exact | forecast | moca_moca | 36 | 77 | 1.199 |
| typed_belief_dynamics | exact | interpolation | adas_total13 | 0 | 392 | 0.9749 |
| typed_belief_dynamics | exact | interpolation | cdr_cdrsb | 0 | 409 | 0.8622 |
| typed_belief_dynamics | exact | interpolation | faq_faqtotal | 0 | 408 | 0.911 |
| typed_belief_dynamics | exact | interpolation | mmse_mmscore | 0 | 394 | 1.06 |
| typed_belief_dynamics | exact | interpolation | moca_moca | 0 | 97 | 0.8695 |
| typed_belief_dynamics | task_target | forecast | adas_total13 | -1 | 1099 | 1.016 |
| typed_belief_dynamics | task_target | forecast | cdr_cdrsb | -1 | 1169 | 1.025 |
| typed_belief_dynamics | task_target | forecast | faq_faqtotal | -1 | 1166 | 1.037 |
| typed_belief_dynamics | task_target | forecast | mmse_mmscore | -1 | 1106 | 1.068 |
| typed_belief_dynamics | task_target | forecast | moca_moca | -1 | 257 | 1.016 |
| typed_belief_dynamics | task_target | interpolation | adas_total13 | -1 | 392 | 0.9749 |
| typed_belief_dynamics | task_target | interpolation | cdr_cdrsb | -1 | 409 | 0.8622 |
| typed_belief_dynamics | task_target | interpolation | faq_faqtotal | -1 | 408 | 0.911 |
| typed_belief_dynamics | task_target | interpolation | mmse_mmscore | -1 | 394 | 1.06 |
| typed_belief_dynamics | task_target | interpolation | moca_moca | -1 | 97 | 0.8695 |
| typed_belief_dynamics | global | * | * | -1 | 6497 | 1.014 |
| typed_direct_gaussian | exact | forecast | adas_total13 | 6 | 180 | 0.9476 |
| typed_direct_gaussian | exact | forecast | adas_total13 | 12 | 331 | 0.8923 |
| typed_direct_gaussian | exact | forecast | adas_total13 | 24 | 353 | 0.9798 |
| typed_direct_gaussian | exact | forecast | adas_total13 | 36 | 235 | 1.076 |
| typed_direct_gaussian | exact | forecast | cdr_cdrsb | 6 | 179 | 0.9238 |
| typed_direct_gaussian | exact | forecast | cdr_cdrsb | 12 | 348 | 0.9008 |
| typed_direct_gaussian | exact | forecast | cdr_cdrsb | 24 | 379 | 1.133 |
| typed_direct_gaussian | exact | forecast | cdr_cdrsb | 36 | 263 | 1.4 |
| typed_direct_gaussian | exact | forecast | faq_faqtotal | 6 | 181 | 1.102 |
| typed_direct_gaussian | exact | forecast | faq_faqtotal | 12 | 348 | 0.9051 |
| typed_direct_gaussian | exact | forecast | faq_faqtotal | 24 | 377 | 1.069 |
| typed_direct_gaussian | exact | forecast | faq_faqtotal | 36 | 260 | 1.381 |
| typed_direct_gaussian | exact | forecast | mmse_mmscore | 6 | 181 | 0.8464 |
| typed_direct_gaussian | exact | forecast | mmse_mmscore | 12 | 332 | 1.093 |
| typed_direct_gaussian | exact | forecast | mmse_mmscore | 24 | 355 | 1.067 |
| typed_direct_gaussian | exact | forecast | mmse_mmscore | 36 | 238 | 1.214 |
| typed_direct_gaussian | exact | forecast | moca_moca | 12 | 76 | 0.8283 |
| typed_direct_gaussian | exact | forecast | moca_moca | 24 | 101 | 0.9751 |
| typed_direct_gaussian | exact | forecast | moca_moca | 36 | 77 | 1.153 |
| typed_direct_gaussian | exact | interpolation | adas_total13 | 0 | 392 | 0.8906 |
| typed_direct_gaussian | exact | interpolation | cdr_cdrsb | 0 | 409 | 0.839 |
| typed_direct_gaussian | exact | interpolation | faq_faqtotal | 0 | 408 | 0.8937 |
| typed_direct_gaussian | exact | interpolation | mmse_mmscore | 0 | 394 | 0.9886 |
| typed_direct_gaussian | exact | interpolation | moca_moca | 0 | 97 | 0.7971 |
| typed_direct_gaussian | task_target | forecast | adas_total13 | -1 | 1099 | 0.9709 |
| typed_direct_gaussian | task_target | forecast | cdr_cdrsb | -1 | 1169 | 1.109 |
| typed_direct_gaussian | task_target | forecast | faq_faqtotal | -1 | 1166 | 1.108 |
| typed_direct_gaussian | task_target | forecast | mmse_mmscore | -1 | 1106 | 1.076 |
| typed_direct_gaussian | task_target | forecast | moca_moca | -1 | 257 | 0.9914 |
| typed_direct_gaussian | task_target | interpolation | adas_total13 | -1 | 392 | 0.8906 |
| typed_direct_gaussian | task_target | interpolation | cdr_cdrsb | -1 | 409 | 0.839 |
| typed_direct_gaussian | task_target | interpolation | faq_faqtotal | -1 | 408 | 0.8937 |
| typed_direct_gaussian | task_target | interpolation | mmse_mmscore | -1 | 394 | 0.9886 |
| typed_direct_gaussian | task_target | interpolation | moca_moca | -1 | 97 | 0.7971 |
| typed_direct_gaussian | global | * | * | -1 | 6497 | 1.024 |

## Development decision

```json
{
  "run_id": "phase2_cognitive_belief_dev_20261010",
  "engineering_smoke_only": false,
  "development_only": true,
  "test_evaluated": false,
  "candidate": "typed_belief_dynamics",
  "baseline": "grud_gaussian",
  "criteria": {
    "normalized_mae_noninferior": false,
    "calibrated_nll_improved": false,
    "multilevel_calibration_error_within_bound": true,
    "coverage_95_within_bounds": true
  },
  "cognitive_dynamics_gate_passed": false,
  "advance_to_mri": false,
  "semantic_superiority_claim": false,
  "strongest_point_estimate_arm": "typed_direct_gaussian",
  "typed_direct_secondary_finding": {
    "overall_normalized_mae_difference": -0.010601499073091485,
    "overall_normalized_mae_ci": [
      -0.033553837503301785,
      0.005774014716366026
    ],
    "interpolation_normalized_mae_difference": -0.014365278009299882,
    "interpolation_normalized_mae_ci": [
      -0.02970799916139634,
      -0.004190447768639178
    ],
    "interpretation": "task_specific_interpolation_signal_without_overall_confirmed_superiority"
  },
  "next_action": "retain_grud_as_confirmed_primary_and_typed_direct_as_interpolation_component_candidate"
}
```

This decision concerns the cognitive dynamics component only. MRI remains closed pending real scan dates, QC/provenance and a separately frozen MRI protocol.
