# Phase 5D: Anchored Bifactor MIRT and Measurement Invariance

## Scope

Five-fold nested subject-wise development used only the original ADNI train split. The model used a positive-loading global impairment factor plus a positive-loading Memory-specific factor, typed item likelihoods, and explicit time-DIF parameters. Original validation and test subjects were not loaded. Diagnosis and ADNI phase were audited for residual DIF but were not inputs to state inference.

## Concurrent measurement audit

All five instruments were connected by nominal-visit co-observation. Actual date coverage was available for ADAS, CDR, MMSE, and MoCA; the Logical Memory exam-date columns were empty in the frozen split, so Logical Memory linkage remained visit-code based and is reported separately in `instrument_connectivity.csv`.

## Predictive measurement results

The partial-invariant candidate obtained held-out-item/held-out-instrument macro normalized MAE **0.5181/0.5273**. Fixed domain scoring obtained **0.4866/0.5212**, and the unconstrained autoencoder obtained **0.4446/0.4539**.

The candidate-minus-fixed held-out-instrument participant difference was **0.00081**, 95% CI **[-0.00300, 0.00465]**. Candidate-minus-autoencoder was **0.07476**, 95% CI **[0.07014, 0.07957]**.

## State coherence and invariance

The frozen practical rule selected **0** fold-specific time-DIF item decisions; when none were selected, the partial-invariant arm was exactly the invariant arm rather than a stochastic refit. Memory cross-view concordance was **0.3713**, participant-bootstrap 95% CI **[0.3547, 0.3873]**. There were **8** stable Memory anchors from **2** instruments in at least 4 of 5 folds. The candidate beat fixed scoring in **1/5** outer folds.

## Gate

- improvement over fixed score: `false`;
- non-inferiority to the unconstrained autoencoder within +0.01: `false`;
- cross-view concordance lower bound above 0.70: `false`;
- stable Memory anchor requirement: `true`;
- outer-fold reproducibility: `false`.

Therefore `phase5d_measurement_gate_passed=false` and `continuous_time_dynamics_open=false`. A named and anchored state is accepted only when identification, cross-instrument coherence, and predictive measurement all pass together.
