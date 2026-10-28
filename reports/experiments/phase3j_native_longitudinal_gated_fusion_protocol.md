# Phase 3J: Native-resolution longitudinal gated fusion

## Objective

Phase 3J tests whether a slowly varying structural MRI state provides
incremental information beyond the frozen irregular cognitive trajectory. It
does not retune the Phase 3I image resolution or the Phase 4A cognitive
encoder.

## Frozen inputs

- MRI is loaded at the complete `193 x 229 x 193`, 1-mm affine grid. There is
  no interpolation, cropping, padding, or slice selection.
- Five Phase 3I outer-fold SFCNs are frozen. Their AD probabilities are
  averaged into a scan-level structural-burden score; cross-model variation is
  retained as uncertainty.
- At cutoff time `t`, only MRI scans with real `scan_date <= t` are available.
- Structural history contains the most recent burden, ensemble disagreement,
  count, recency, temporal mean/standard deviation, last-minus-first change,
  per-year slope, and observed time span.
- Cognitive forecasts come from the frozen three-seed Phase 4A GRU-D.

## Future cognitive forecasting

The original-training Phase 4A `dev` subjects fit the fusion layer and
`confirmation` subjects select ridge strength or stop the gated model. The
original validation subjects form the evaluation cohort and are not used for
fusion fitting or selection. Evaluation is restricted to forecast cutoffs
with at least one prior MRI; a prespecified subgroup has at least two prior
MRI scans. Arms are frozen cognitive GRU-D, cognitive recalibration,
recent-MRI residual fusion, longitudinal linear residual fusion,
availability-aware longitudinal gated residual fusion, and subject-shuffled
MRI control.

Primary outcomes are macro normalized MAE over five endpoints and four
horizons, endpoint/horizon MAE, and subject-cluster bootstrap differences.
Sparse cognitive history is prespecified as at most two history visits or at
least 50% missing verified item responses.

## MCI-to-AD progression

The Phase 3H original-training cohort fits 24- and 36-month progression
models. New original-validation cohorts use the same earliest eligible
MRI-linked MCI baseline, 90-day concurrent diagnosis window, and complete
follow-up definition. Arms are cognition, native MRI, direct concatenation,
longitudinal gated residual fusion, and shuffled-MRI fusion. ACC, ROC AUC,
sensitivity, specificity, and F1 are reported with subject bootstrap
intervals.

## Gate

Opening multimodal reasoning requires all of the following:

1. Gated fusion improves evaluation macro NMAE by at least 0.005 over frozen
   cognition and the 95% paired-subject CI excludes zero.
2. Correct longitudinal MRI is better than shuffled MRI.
3. The sparse-history subgroup is noninferior to frozen cognition.
4. At least one conversion horizon improves AUC by at least 0.01.

Failure retains MRI as an auxiliary structural signal and keeps the final
reasoning method cognitive-first. Official test data are not loaded.
