# Phase 5B: Frozen Reasoning Error Decomposition

## Status

Completed as a post-test descriptive analysis of the unchanged Phase 4F grammar-model outputs. No training, decoding, prompt editing, threshold selection, or report regeneration was performed. The Phase 4F confirmation decision remains failed because temporal consistency was 0.98990 against the prespecified 0.99000 threshold.

## Cohort and denominators

- 495 subject-disjoint test reports, one per subject.
- 2,475 current-state decisions and 2,475 trend decisions across five cognitive domains.
- Current-state mismatches: **36/2,475 (1.45%; 95% subject-bootstrap CI 0.97–1.98%)**.
- Trend mismatches: **100/2,475 (4.04%; 95% CI 2.99–5.17%)**.

## Main decomposition

The highest current-state mismatch rate occurred in **Attention** (10/495, 2.02%). The highest trend mismatch rate occurred in **Memory** (40/495, 8.08%). Across both fields, the highest horizon-level rate was at **36 months** (22/760, 2.89%). Expected `improve` trends had 39/130 mismatches (30.00%), and the most frequent directed confusion was **trend: improve → stable** (26 decisions).

For current state, the near-boundary (<0.10 z from ±0.5) mismatch rate was 17/869 (1.96%) versus 18/1605 (1.12%) far from the boundary; the near-minus-far difference was 0.83 percentage points (95% CI -0.17 to 1.85). For trend, only 8 decisions were near the boundary: 5/8 (62.50%) versus 77/1369 (5.62%) far from it, a difference of 56.88 points (95% CI 13.99 to 94.19). This trend contrast is therefore imprecise despite its large point estimate. Insufficient-history trend decisions are retained as a separate structural category and are excluded from that contrast.

## Interpretation

These results localize the already observed Phase 4F label discrepancies; they do not revise the confirmatory outcome. All numeric severity values, timepoints, evidence citations, forecasts, and uncertainty fields remained preserved in Phase 4F. The error pattern therefore concerns categorical state/trend selection within otherwise valid structured reports. Because this decomposition was conducted after opening the test set, subgroup confidence intervals are descriptive and cannot justify test-set repair or a new claim of gate success.

The next admissible experiment is a prospectively frozen external-cohort validation of the unchanged **GRU-D cognitive forecasting + grammar-constrained structured reporting** system. Any revised label policy must be developed without these 495 subjects and evaluated on another untouched cohort.

## Artifacts

- `decision_level_decomposition.csv`: all 4,950 domain-field decisions and their frozen covariates.
- `mismatch_rate_summary.csv`: denominator-aware rates with subject-cluster bootstrap intervals.
- `confusion_matrix.csv`: full expected-to-predicted counts.
- `near_far_boundary_contrast.csv`: prespecified near-versus-far descriptive contrasts.
- `figures/phase5b_error_decomposition.pdf` and `.png`: publication-ready summary figure.
- `analysis_decision.json`: scope and next-action record.
- `verification.json`: independent source and metric verification.
