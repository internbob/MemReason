# Phase 5B: Frozen reasoning error decomposition

Status: post-test descriptive analysis frozen before inspecting subgroup results.

## Question

Where do the 36 current-state and 100 trend-label discrepancies from the frozen Phase 4F grammar model occur, and are they concentrated near the deterministic label boundaries, in particular domains or horizons, or in sparse histories?

## Frozen inputs

This analysis reads only the 495 subject-disjoint Phase 4F test examples and their already generated `grammar_model` reports. It does not train a model, change a prompt, alter the state/trend thresholds, select a checkpoint, or regenerate any report. SHA-256 hashes of both inputs and the Phase 4F decision are recorded. The Phase 4F decision remains unchanged.

The current-state reference uses the frozen boundaries at severity z = -0.5 and +0.5. The trend reference uses end-minus-start severity boundaries at -0.25 and +0.25; histories with fewer than two timepoints remain `insufficient_history`.

## Unit of analysis and denominators

Each of 495 reports contributes five cognitive domains for current state and five for trend, yielding 2,475 decisions per field and 4,950 decisions overall. Every rate table retains its full denominator. Subject IDs define clusters; one report was selected per subject in Phase 4F.

## Prespecified descriptive factors

- Field: current state or trend.
- Cognitive domain: Memory, Executive, Attention, Language, Orientation.
- Forecast horizon: 6, 12, 24, or 36 months.
- Expected label and expected-to-predicted confusion.
- Maximum domain-history length: 1, 2–3, or 4+ timepoints.
- Missing last-target count: 0–1 or 2+ unavailable targets.
- Distance to the nearest frozen decision boundary: `[0,0.05)`, `[0.05,0.10)`, `[0.10,0.25)`, or `>=0.25`. Unavailable state and insufficient-history trend decisions are reported separately.

Near-boundary decisions have distance below 0.10; far-boundary decisions have distance at least 0.10. The near-minus-far mismatch-rate contrast and all subgroup rates use 2,000 subject-cluster bootstrap replicates with percentile 95% confidence intervals.

## Interpretation and stopping rule

This is an exploratory post-test error analysis. Confidence intervals quantify sampling uncertainty but do not convert subgroup observations into confirmatory claims; no p-values or multiplicity-adjusted claims are made. Results cannot be used to repair Phase 4C/4F on these subjects. The only allowed next model evaluation is a prospectively frozen external-cohort validation.
