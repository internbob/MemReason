# Phase 3E frozen spatial-grounding test confirmation

Status: completed one-time evaluation on the sealed subject-disjoint test split. No training or checkpoint selection was performed.

## Frozen contract

- Test MRI anchors: 1905
- Test subjects: 404
- Test targets: 7843
- Frozen SFCN/head seeds: [2026, 2027, 2028]
- Head trainable parameters in Phase 3C: {'global_pool': 76673, 'target_spatial_attention': 76673}
- Duplicate-scan resolutions: 2 source rows

## Test summary

| model | pairing | normalized_mae | mae |
| --- | --- | --- | --- |
| target_spatial_attention | observed | 0.6351 | 3.424 |
| global_pool | observed | 0.6454 | 3.475 |
| target_spatial_attention | spatial_permutation | 0.684 | 3.68 |
| target_spatial_attention | query_permutation | 0.8242 | 4.724 |
| global_pool | shuffled_subject | 0.843 | 4.628 |
| target_spatial_attention | shuffled_subject | 0.8913 | 4.884 |

## Paired subject-cluster bootstrap

Negative differences favor observed target-conditioned spatial attention.

| target | horizon_months | candidate | candidate_pairing | baseline | baseline_pairing | n | subjects | normalized_mae_difference | ci_lower | ci_upper | probability_candidate_better | replicates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall |  | target_spatial_attention | observed | global_pool | observed | 7843 | 404 | -0.01088 | -0.02522 | 0.004273 | 0.9135 | 2000 |
| adas_total13 |  | target_spatial_attention | observed | global_pool | observed | 1875 | 403 | -0.007831 | -0.02559 | 0.009812 | 0.791 | 2000 |
| cdr_cdrsb |  | target_spatial_attention | observed | global_pool | observed | 1880 | 404 | -0.009952 | -0.02723 | 0.0072 | 0.8625 | 2000 |
| faq_faqtotal |  | target_spatial_attention | observed | global_pool | observed | 1875 | 403 | -0.008673 | -0.02646 | 0.01033 | 0.8135 | 2000 |
| mmse_mmscore |  | target_spatial_attention | observed | global_pool | observed | 1901 | 404 | -0.01753 | -0.03506 | -6.477e-05 | 0.9755 | 2000 |
| moca_moca |  | target_spatial_attention | observed | global_pool | observed | 312 | 150 | -0.007574 | -0.04668 | 0.03262 | 0.642 | 2000 |
| overall |  | target_spatial_attention | observed | target_spatial_attention | shuffled_subject | 7843 | 404 | -0.2579 | -0.2991 | -0.2154 | 1 | 2000 |
| adas_total13 |  | target_spatial_attention | observed | target_spatial_attention | shuffled_subject | 1875 | 403 | -0.3058 | -0.3548 | -0.2574 | 1 | 2000 |
| cdr_cdrsb |  | target_spatial_attention | observed | target_spatial_attention | shuffled_subject | 1880 | 404 | -0.2523 | -0.2997 | -0.2042 | 1 | 2000 |
| faq_faqtotal |  | target_spatial_attention | observed | target_spatial_attention | shuffled_subject | 1875 | 403 | -0.2549 | -0.3049 | -0.2035 | 1 | 2000 |
| mmse_mmscore |  | target_spatial_attention | observed | target_spatial_attention | shuffled_subject | 1901 | 404 | -0.2207 | -0.2634 | -0.1809 | 1 | 2000 |
| moca_moca |  | target_spatial_attention | observed | target_spatial_attention | shuffled_subject | 312 | 150 | -0.2473 | -0.3314 | -0.1641 | 1 | 2000 |
| overall |  | target_spatial_attention | observed | target_spatial_attention | spatial_permutation | 7843 | 404 | -0.05011 | -0.06634 | -0.03283 | 1 | 2000 |
| adas_total13 |  | target_spatial_attention | observed | target_spatial_attention | spatial_permutation | 1875 | 403 | -0.04114 | -0.06061 | -0.02237 | 1 | 2000 |
| cdr_cdrsb |  | target_spatial_attention | observed | target_spatial_attention | spatial_permutation | 1880 | 404 | -0.05221 | -0.07196 | -0.03065 | 1 | 2000 |
| faq_faqtotal |  | target_spatial_attention | observed | target_spatial_attention | spatial_permutation | 1875 | 403 | -0.05445 | -0.07517 | -0.033 | 1 | 2000 |
| mmse_mmscore |  | target_spatial_attention | observed | target_spatial_attention | spatial_permutation | 1901 | 404 | -0.05377 | -0.07433 | -0.0344 | 1 | 2000 |
| moca_moca |  | target_spatial_attention | observed | target_spatial_attention | spatial_permutation | 312 | 150 | -0.04296 | -0.08017 | -0.001815 | 0.9805 | 2000 |
| overall |  | target_spatial_attention | observed | target_spatial_attention | query_permutation | 7843 | 404 | -0.2272 | -0.2611 | -0.1951 | 1 | 2000 |
| adas_total13 |  | target_spatial_attention | observed | target_spatial_attention | query_permutation | 1875 | 403 | -0.4681 | -0.5427 | -0.3923 | 1 | 2000 |
| cdr_cdrsb |  | target_spatial_attention | observed | target_spatial_attention | query_permutation | 1880 | 404 | -0.007106 | -0.01274 | -0.001212 | 0.99 | 2000 |
| faq_faqtotal |  | target_spatial_attention | observed | target_spatial_attention | query_permutation | 1875 | 403 | -0.001876 | -0.0033 | -0.000314 | 0.9915 | 2000 |
| mmse_mmscore |  | target_spatial_attention | observed | target_spatial_attention | query_permutation | 1901 | 404 | -0.4664 | -0.532 | -0.3975 | 1 | 2000 |
| moca_moca |  | target_spatial_attention | observed | target_spatial_attention | query_permutation | 312 | 150 | -0.001814 | -0.01852 | 0.01687 | 0.588 | 2000 |

## Cross-seed attention stability

```json
{
  "median_spearman": 0.40180180180180186,
  "median_top_token_overlap": 0.5
}
```

## Frozen decision

```json
{
  "run_id": "phase3e_frozen_spatial_grounding_test_20261014",
  "confirmation_only": true,
  "training_performed": false,
  "test_evaluated": true,
  "criteria": {
    "attention_better_than_global_pool": false,
    "attention_better_than_subject_shuffle": true,
    "attention_better_than_spatial_permutation": true,
    "attention_better_than_query_permutation": true,
    "minimum_significant_endpoints": false,
    "attention_spearman_stable": true,
    "top_token_overlap_stable": true
  },
  "significant_endpoints_vs_global": 1,
  "attention_stability": {
    "median_spearman": 0.40180180180180186,
    "median_top_token_overlap": 0.5
  },
  "phase3c_test_confirmed": false,
  "next_action": "retain_validation_only_phase3c_without_confirmatory_claim"
}
```

Secondary diagnosis, timing-gap, sex, age, and endpoint-availability results are stored in `stratum_metrics.csv` and `stratum_bootstrap_vs_global.csv`. They do not alter the primary gate. Attention is a predictive weighting map and is not causal anatomical evidence.
