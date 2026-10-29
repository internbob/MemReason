# Phase 5G: Bridge-strength Pareto scan and residual-adaptive gating

## Question

Phase 5F showed that graph alignment increased cross-instrument Memory-state
concordance but worsened held-out-instrument prediction relative to the
residual-only hierarchy. Phase 5G tests whether this trade-off is monotone in
the bridge coefficient and whether a residual-evidence gate can preserve
instrument-specific information without discarding supported graph alignment.

## Frozen scope

- Use only the original ADNI training split and the Phase 5F subject folds.
- Preserve the Phase 5F item contract, scaling, masks, graph construction,
  hierarchy, residual penalties, optimizer, and inner-development procedure.
- Do not load original validation or test subjects, diagnosis, FAQ, MRI, or
  future outcomes.
- Use identical initialization and masking seeds within each fold so variants
  differ only in bridge coefficient and gate status.

## Experiment 1: graph-strength scan

Train the hierarchical residual model with bridge coefficients
`0, 0.01, 0.03, 0.10, 0.30, 1.00`. The zero coefficient is the internal
residual-only reference. For every coefficient report held-out-item and
held-out-instrument macro normalized MAE, fold range, graph-supported Memory
CCC with participant-bootstrap 95% CI, mean absolute cross-view difference,
state variance, posterior SD, and residual-loading fraction.

The empirical Pareto frontier minimizes held-out-instrument macro NMAE and
maximizes Memory CCC. Frontier membership is descriptive development evidence;
it is not a new frozen gate and cannot open continuous-time dynamics.

## Experiment 2: residual-adaptive bridge gate

For every nonzero bridge coefficient, repeat training with

`g = floor + (1 - floor) * exp(-temperature * residual_evidence)`,

where `floor=0.25`, `temperature=1`, and residual evidence is the mean absolute
standardized residual posterior across both instrument views. The effective
dimension-specific graph weight is `w_effective = w_graph * g`.

The gate is detached from gradient flow. Thus residual encoders cannot reduce
the bridge objective merely by inflating residual means. The positive floor
ensures that a supported edge always retains at least one quarter of its
alignment pressure. Instrument residual states remain unaligned.

## Comparisons

- At every matched nonzero coefficient: gated minus ungated participant-level
  prediction error and CCC difference.
- Against coefficient zero: predictive degradation and coherence gain.
- Against the original Phase 5F residual-only and bridge results as external
  implementation checks; these references are not mixed into model selection.

The preferred development point, if any, is the highest-CCC Pareto point whose
held-out-instrument macro NMAE is no more than 0.01 worse than the coefficient
zero reference. The original Phase 5F measurement gate remains closed unless
separately rerun under a newly frozen confirmation protocol.

