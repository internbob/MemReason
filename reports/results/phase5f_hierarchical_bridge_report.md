# Phase 5F: Hierarchical Shared State, Instrument Residuals, and Bridge Graph

## Scope

Five-fold nested subject-wise development used only the original ADNI train split. The model separated a global-plus-domain shared state, three prespecified Memory subconstructs, and shrinkage-controlled instrument residuals. Bridge consistency was applied only to availability/date-supported instrument edges. Original validation and test subjects, diagnosis, FAQ, MRI, and future outcomes were not loaded.

## Predictive measurement

The candidate obtained held-out-item/held-out-instrument macro normalized MAE **0.3989/0.4166**. Fixed scoring obtained **0.4866/0.5212**, and the unconstrained autoencoder obtained **0.4446/0.4539**.

The candidate-minus-fixed held-out-instrument participant difference was **-0.09920**, 95% CI **[-0.10317, -0.09516]**. Candidate-minus-autoencoder was **-0.02525**, 95% CI **[-0.02818, -0.02241]**.

Relative to the residual-only ablation, the bridge candidate changed held-out-instrument error by **0.00861**, 95% CI **[0.00665, 0.01038]**. Thus the bridge is evaluated as a state-coherence/prediction trade-off rather than assumed to improve both objectives.

## Bridge state

Graph-supported Memory cross-view CCC was **0.6340**, participant-bootstrap 95% CI **[0.6207, 0.6467]**. Cross-view Memory-state standard deviation was **0.2294** and the mean residual share of decoder loading was **0.0854**.

Logical Memory had no usable assessment dates and every LOGMEM edge fell below the frozen graph-support threshold. The graph-supported CCC therefore primarily assesses ADAS/CDR/MMSE Memory bridges and does not establish LOGMEM transport validity.

## Frozen development gate

- improvement over fixed score: `true`;
- non-inferiority to the unconstrained autoencoder: `true`;
- cross-view CCC lower bound: `false`;
- non-collapse state variance: `true`;
- residual-loading control: `true`;
- outer-fold reproducibility: `true`.

Therefore `phase5f_measurement_gate_passed=false` and `continuous_time_dynamics_open=false`. Passing would be development evidence only; confirmation still requires a new untouched or external cohort.
