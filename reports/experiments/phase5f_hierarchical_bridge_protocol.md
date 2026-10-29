# Phase 5F: Hierarchical Shared State, Instrument Residuals, and Bridge Graph

Status: protocol frozen before the Phase 5F development run. This is a new
measurement-development stage after the negative Phase 5C/5D results. It does
not reopen either prior gate and does not use the consumed validation or test
subjects.

## Question

Can a hierarchical cognitive state preserve cross-instrument meaning without
the predictive loss of a single anchored Memory factor, when instrument method
effects are isolated in a shrinkage-controlled residual and cross-view
agreement is imposed only over empirically supported instrument bridges?

## Data boundary

- Use only the original ADNI train subjects with five outer subject folds and
  a subject-disjoint inner development subset.
- Reuse the Phase 5C item contract, fold seed, deterministic held-out-item and
  held-out-instrument masks, fixed-score baseline, and unconstrained
  autoencoder predictions.
- Diagnosis, FAQ, MRI, future outcomes, validation subjects, and test subjects
  are excluded from state inference and model selection.
- Instrument connectivity uses observation availability and assessment dates,
  not response values or future outcomes. Logical Memory links without usable
  exam dates receive the prespecified nominal-visit downweight.

## Model

The shared posterior contains one global state and five domain states. Memory
also contains three prespecified subconstruct states: objective recall,
acquisition/recognition, and informant-rated memory. Item decoders have positive
loadings only on the global, matching domain, and matching Memory-subconstruct
states. This fixes the clinical direction: larger states denote worse
performance.

The residual arms additionally infer one state per observed instrument. An
instrument residual can decode only items from that instrument, is exactly zero
when the instrument is absent, and receives a stronger standard-normal
shrinkage penalty than shared states. Thus a fully held-out instrument must be
predicted from shared states rather than an unavailable method residual.

The bridge arm partitions observed instruments into disjoint views. It aligns
only shared posterior dimensions observed in both views, using an edge weight
defined by domain overlap, concurrent-subject support, and actual 30-day date
agreement. Date-unavailable nominal-visit links are downweighted; unsupported
links receive zero weight. Residual states are never aligned across
instruments.

## Arms and ablations

1. `fixed_domain_score`: frozen Phase 5C baseline.
2. `unconstrained_autoencoder`: frozen Phase 5C predictive baseline.
3. `hierarchical_shared`: hierarchy without instrument residuals or graph loss.
4. `hierarchical_residual`: adds shrinkage-controlled instrument residuals.
5. `hierarchical_bridge`: adds the supported bridge-graph posterior loss and is
   the prespecified candidate.

## Evaluation and gate

Primary predictive metrics are macro normalized MAE for deterministic
held-out-item and held-out-instrument targets. The participant-cluster
bootstrap compares the candidate with both frozen baselines. Cross-view
evaluation reports Memory-state concordance, Spearman correlation, posterior
standard deviation, and absolute difference only for graph-supported pairs.
State standard deviation and the residual share of decoder loading are explicit
anti-collapse diagnostics.

The candidate opens preparation of a separately frozen Phase 6 continuous-time
protocol only if all conditions hold:

1. held-out-instrument error improves over fixed scoring with a 95% participant
   bootstrap upper bound below zero;
2. it is non-inferior to the unconstrained autoencoder within +0.01;
3. the participant-bootstrap lower bound for Memory cross-view CCC exceeds
   0.70;
4. cross-view Memory-state standard deviation is at least 0.20;
5. residual decoder loading is at most half of total candidate loading;
6. held-out-instrument error is below fixed scoring in at least four folds.

Failure is retained as a formal measurement result. Passing is development
evidence only and requires a new untouched or external cohort for confirmation.

