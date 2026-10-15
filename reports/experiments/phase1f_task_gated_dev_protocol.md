# Phase 1F terminal development protocol: task-specific information gates

Status at creation: frozen before Phase 1F validation predictions. Test is never loaded.

Phase 1E established significant masked-item improvement and favorable future point estimates, but one gate shared across all objectives could not improve masked-instrument and both cross-scale tasks simultaneously. Phase 1F gives each of the five predeclared task groups its own per-item value and task-semantic residual gate. Both gates start at zero, so all candidate arms begin at the matched numeric-query representation and must learn task-specific departures from training evidence.

The three arms are numeric, typed response, and typed response plus frozen task semantics. Every arm instantiates identical gate, encoder and decoder parameters. Optimization retains Phase 1E's per-item macro reconstruction loss and categorical expected-score alignment.

The provisional gate is unchanged: subject-bootstrap CI upper bounds below zero for both masked tasks and both cross-scale tasks, plus future non-inferiority at +0.005 normalized MAE. This is the final representation revision on the current validation split. If it fails, the project stops optimizing this semantic claim on the same validation set and records one of two conclusions: retain typed question-conditioned encoding as an engineering component without a superiority claim, or obtain a new development/confirmation cohort before further architecture search.
