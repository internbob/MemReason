# Phase 5D: Anchored Bifactor MIRT and Measurement-Invariance Development

Status: completed on 2026-10-16. The measurement gate failed and continuous-time dynamics remain closed.

## Question

Can an identified global-plus-memory item-response model recover the same memory-specific state from different concurrent cognitive instruments, without materially losing held-out-instrument prediction relative to the Phase 5C unconstrained autoencoder?

## Data boundary

- Use only subjects in the original ADNI train split.
- Use five outer subject folds and an inner subject-disjoint development subset in every outer fit partition.
- Do not use the consumed original validation or test subjects for model selection or confirmation.
- A row is a nominal ADNI visit. Instrument-date agreement is audited separately; no future visit is merged into the row.
- Diagnosis, FAQ, MRI, and future outcomes are excluded from latent-state inference.

## Measurement model

The primary state has two identified dimensions: global cognitive impairment and memory-specific impairment. Every item has a positive global discrimination after responses are oriented so that higher values mean greater impairment. The 14 audited Memory items additionally have a positive memory-specific discrimination. Binary items use Bernoulli likelihoods, bounded continuous scores use beta likelihoods, and CDR ordinal items use a graded-response likelihood.

The latent prior fixes both dimensions to a standard normal reference scale. Positive discriminations, the simple loading structure, and time-invariant item parameters identify the axes. This replaces the named but statistically unidentified five-dimensional neural state evaluated in Phase 5C.

## Arms

1. `fixed_domain_score`: frozen Phase 5C baseline.
2. `unconstrained_autoencoder`: frozen Phase 5C predictive baseline.
3. `invariant_mirt`: item parameters shared across all visit-time bins.
4. `partial_invariant_mirt`: time offsets only for items with inner-fit residual DIF above the frozen practical threshold.
5. `no_anchor_mirt`: time offsets for every item, used as an identification ablation.

## Measurement-invariance audit

Residual DIF is evaluated within the outer fit data across baseline, months 1-12, months 13-24, and later than 24 months. Diagnosis and ADNI phase are audited as secondary grouping variables but are not supplied to the state estimator. An anchor candidate must have at least 200 fit observations and no practical residual DIF in time, diagnosis, or ADNI phase. Anchor selection is repeated independently in every outer fold.

## Evaluation

The frozen Phase 5C deterministic held-out-item and held-out-instrument masks are reused. Primary metrics are macro item normalized MAE and participant-cluster bootstrap differences. Cross-view evaluation partitions instruments into disjoint evidence views, infers memory state independently, and reports concordance correlation, Spearman correlation, absolute state difference, and posterior standard deviation. Item-level negative log likelihood and predictive coverage are secondary diagnostics.

## Gate

The candidate advances to continuous-time dynamics only if all conditions hold:

1. held-out-instrument normalized MAE improves over fixed domain scoring with a 95% participant-bootstrap upper bound below zero;
2. it is non-inferior to the unconstrained autoencoder within +0.01, using the 95% upper bound;
3. the participant-bootstrap lower bound for memory cross-view concordance exceeds 0.70;
4. at least three Memory anchors from at least two instruments are stable in four of five folds;
5. held-out-instrument error is lower than fixed scoring in at least four outer folds.

Failure is retained as a formal measurement result. Passing opens a separately frozen continuous-time dynamics protocol but does not permit reuse of the consumed ADNI validation or test cohorts.

## Recorded decision

Five-fold cross-fitting produced 15 checkpoints and 1,348,975 prediction rows across the three MIRT arms and two frozen Phase 5C baselines. No item reached the prespecified practical time-DIF threshold in any fold, so the partial-invariant arm reduced exactly to the invariant arm. Diagnosis-related residual DIF remained common and was reported without supplying diagnosis to state inference.

The partial/invariant model obtained held-out-item and held-out-instrument macro normalized MAE 0.5181 and 0.5273. Its held-out-instrument participant difference versus fixed scoring was +0.00081, 95% CI [-0.00300, 0.00465], and versus the unconstrained autoencoder was +0.07476, 95% CI [0.07014, 0.07957]. Memory cross-view concordance was 0.3713, 95% participant-bootstrap CI [0.3547, 0.3873]. Eight stable Memory anchors from ADAS and MMSE met the anchor criterion, but the predictive and cross-view conditions failed. Therefore `phase5d_measurement_gate_passed=false` and `continuous_time_dynamics_open=false`.
