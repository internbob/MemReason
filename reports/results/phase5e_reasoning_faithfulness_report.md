# Phase 5E frozen reasoning-faithfulness and renderer-ablation audit

Status: completed as a post-test frozen audit. No checkpoint, decoder, threshold, or gate was selected from these results.

## Cohort and interventions

- Phase 4F reasoning cohort: 495 reports / 495 subjects
- Leave-one-visit candidates evaluated with the frozen three-seed GRU-D ensemble: 1013
- Subjects eligible for paired highest-influence versus random deletion: 278
- Additional counterfactuals: reversed time gaps and reflected verified Memory scores
- Future truth, diagnosis, MRI, and GRPO inputs: not used

## Counterfactual response

| condition | metric | mean | ci_low | ci_high | subjects | replicates |
| --- | --- | --- | --- | --- | --- | --- |
| highest_influence_visit_deletion | normalized_forecast_displacement | 0.2244 | 0.1957 | 0.2554 | 278 | 2000 |
| highest_influence_visit_deletion | state_label_change_fraction | 0.1712 | 0.1489 | 0.1928 | 278 | 2000 |
| highest_influence_visit_deletion | trend_label_change_fraction | 0.5295 | 0.4784 | 0.5791 | 278 | 2000 |
| highest_influence_visit_deletion | forecast_direction_change_fraction | 0.3101 | 0.2813 | 0.3381 | 278 | 2000 |
| highest_influence_visit_deletion | evidence_exact_jaccard | 0.3494 | 0.2943 | 0.4046 | 278 | 2000 |
| highest_influence_visit_deletion | compiled_report_changed | 1 | 1 | 1 | 278 | 2000 |
| highest_influence_visit_deletion | verifier_exact_report_match | 1 | 1 | 1 | 278 | 2000 |
| highest_influence_visit_deletion | verifier_evidence_precision | 1 | 1 | 1 | 278 | 2000 |
| highest_influence_visit_deletion | verifier_future_evidence_leakage | 0 | 0 | 0 | 278 | 2000 |
| highest_influence_visit_deletion | deleted_visit_stale_citations | 0 | 0 | 0 | 278 | 2000 |
| matched_random_visit_deletion | normalized_forecast_displacement | 0.08113 | 0.07302 | 0.08991 | 278 | 2000 |
| matched_random_visit_deletion | state_label_change_fraction | 0.05899 | 0.04245 | 0.07626 | 278 | 2000 |
| matched_random_visit_deletion | trend_label_change_fraction | 0.4568 | 0.4022 | 0.5086 | 278 | 2000 |
| matched_random_visit_deletion | forecast_direction_change_fraction | 0.1748 | 0.1518 | 0.1971 | 278 | 2000 |
| matched_random_visit_deletion | evidence_exact_jaccard | 0.7959 | 0.7501 | 0.8401 | 278 | 2000 |
| matched_random_visit_deletion | compiled_report_changed | 1 | 1 | 1 | 278 | 2000 |
| matched_random_visit_deletion | verifier_exact_report_match | 1 | 1 | 1 | 278 | 2000 |
| matched_random_visit_deletion | verifier_evidence_precision | 1 | 1 | 1 | 278 | 2000 |
| matched_random_visit_deletion | verifier_future_evidence_leakage | 0 | 0 | 0 | 278 | 2000 |
| matched_random_visit_deletion | deleted_visit_stale_citations | 0 | 0 | 0 | 278 | 2000 |
| memory_score_reflection | normalized_forecast_displacement | 1.507 | 1.444 | 1.568 | 495 | 2000 |
| memory_score_reflection | state_label_change_fraction | 0.141 | 0.1333 | 0.1491 | 495 | 2000 |
| memory_score_reflection | trend_label_change_fraction | 0.03475 | 0.02788 | 0.04202 | 495 | 2000 |
| memory_score_reflection | forecast_direction_change_fraction | 0.4149 | 0.3927 | 0.4372 | 495 | 2000 |
| memory_score_reflection | evidence_exact_jaccard | 0.6738 | 0.671 | 0.6765 | 495 | 2000 |
| memory_score_reflection | compiled_report_changed | 1 | 1 | 1 | 495 | 2000 |
| memory_score_reflection | verifier_exact_report_match | 1 | 1 | 1 | 495 | 2000 |
| memory_score_reflection | verifier_evidence_precision | 1 | 1 | 1 | 495 | 2000 |
| memory_score_reflection | verifier_future_evidence_leakage | 0 | 0 | 0 | 495 | 2000 |
| memory_score_reflection | deleted_visit_stale_citations | 0 | 0 | 0 | 495 | 2000 |
| reversed_time_gaps | normalized_forecast_displacement | 0.002679 | 0.001974 | 0.003548 | 495 | 2000 |
| reversed_time_gaps | state_label_change_fraction | 0 | 0 | 0 | 495 | 2000 |
| reversed_time_gaps | trend_label_change_fraction | 0 | 0 | 0 | 495 | 2000 |
| reversed_time_gaps | forecast_direction_change_fraction | 0.004848 | 0.00202 | 0.008081 | 495 | 2000 |
| reversed_time_gaps | evidence_exact_jaccard | 0.9947 | 0.9883 | 1 | 495 | 2000 |
| reversed_time_gaps | compiled_report_changed | 0.1919 | 0.1576 | 0.2283 | 495 | 2000 |
| reversed_time_gaps | verifier_exact_report_match | 1 | 1 | 1 | 495 | 2000 |
| reversed_time_gaps | verifier_evidence_precision | 1 | 1 | 1 | 495 | 2000 |
| reversed_time_gaps | verifier_future_evidence_leakage | 0 | 0 | 0 | 495 | 2000 |
| reversed_time_gaps | deleted_visit_stale_citations | 0 | 0 | 0 | 495 | 2000 |

## Primary paired deletion contrast

| comparison | metric | difference | ci_low | ci_high | subjects | replicates |
| --- | --- | --- | --- | --- | --- | --- |
| highest_influence_minus_matched_random_deletion | normalized_forecast_displacement | 0.1432 | 0.1191 | 0.1706 | 278 | 2000 |

## Renderer ablation on the identical Phase 4F reports

| arm | format_validity | evidence_precision | evidence_recall | numeric_citation_accuracy | time_citation_accuracy | domain_accuracy | hallucinated_evidence_rate | future_evidence_leakage | current_state_consistency | temporal_consistency | forecast_preservation | uncertainty_preservation | exact_report_match | forbidden_claim |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grammar_executable | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 0 |
| grammar_model | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0.9952 | 0.9899 | 1 | 1 | 0.8141 | 0 |
| raw_sft | 0.9859 | 0.9857 | 0.9819 | 0.9855 | 0.9857 | 0.9857 | 0.01434 | 0 | 0.981 | 0.9794 | 0.9859 | 0.9859 | 0.8343 | 0 |
| teacher | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 0 |

## Decision

The executable compiler reproduced the teacher report exactly and mechanically preserved evidence, forecasts, and uncertainty. The LLM grammar renderer did not add factual fidelity over the executable renderer and retained the previously documented state/trend label errors. The defensible method is therefore the frozen GRU-D forecaster plus typed executable reasoning IR, compiler, and independent verifier; the 1.5B LLM is an optional renderer rather than the scientific core.

Counterfactual sensitivity is evidence of input dependence and traceability only. It does not validate a clinical latent state or repair the failed Phase 4F confirmatory gate.
