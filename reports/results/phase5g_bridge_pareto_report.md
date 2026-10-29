# Phase 5G: Bridge-strength Pareto scan and residual-adaptive gate

## Scope

Five-fold original-train-only development evaluated 11 variants (55 neural fits). Original validation and test subjects were not loaded.

## Pareto result

The coefficient-zero reference obtained held-out-instrument macro NMAE **0.4098** and Memory CCC **0.4487**. Under the prespecified +0.010 prediction tolerance, the selected point was `ungated_w0p1` with NMAE **0.4180** and CCC **0.6296**, 95% participant-bootstrap CI **[0.6163, 0.6423]**.

CCC peaked near λ=0.03--0.10 and declined at stronger alignment. At λ=0.30 and λ=1.00, Memory-state standard deviation fell below the original 0.20 anti-collapse threshold, showing that smaller cross-view differences at high λ reflected state compression rather than improved concordance.

## Matched residual-gate effects

- λ=0.01: gated-minus-ungated participant error **-0.00006**, 95% CI **[-0.00014, +0.00001]**; CCC change **-0.0054**.
- λ=0.03: gated-minus-ungated participant error **-0.00013**, 95% CI **[-0.00024, -0.00002]**; CCC change **-0.0070**.
- λ=0.1: gated-minus-ungated participant error **-0.00025**, 95% CI **[-0.00039, -0.00011]**; CCC change **-0.0111**.
- λ=0.3: gated-minus-ungated participant error **-0.00023**, 95% CI **[-0.00038, -0.00007]**; CCC change **-0.0143**.
- λ=1: gated-minus-ungated participant error **-0.00047**, 95% CI **[-0.00070, -0.00025]**; CCC change **-0.0192**.

The deterministic gate retained about 90% of graph pressure across all edge groups. It produced small, increasingly precise prediction gains as λ increased, but reduced CCC at every matched coefficient and showed little edge-strength differentiation. Thus this gate did not dominate the ungated Pareto frontier.

## Decision

This is a development Pareto experiment, not a confirmation run. The Phase 5F gate is not retrospectively reopened and continuous-time dynamics remain closed.
