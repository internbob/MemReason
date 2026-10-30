# Phase 6C: longitudinal MRI temporal-signal identifiability

Status: frozen before execution. Phase 6B found a small benefit from MRI but
could not distinguish correctly ordered MRI from within-subject time-shuffled
MRI. Phase 6C asks the narrower question: does MRI *change over time* add
forecast information beyond the same participant's latest static MRI state?

## Cohort and leakage boundary

- Use only original ADNI training subjects and the existing five subject folds.
  Validation and test are not loaded.
- The cognitive cutoff is `cognition_date_end`, the end of the observed
  cognitive assessment window. Only MRI scans with an actual acquisition date
  on or before that date are eligible.
- The primary cohort requires at least two pre-cutoff MRI scans spanning at
  least 0.5 years. A prespecified sensitivity cohort requires at least 1 year.
- The primary endpoint is future observed Memory-composite absolute error at
  6, 12, 24, and 36 months. Phase 5G latent states remain auxiliary targets.

## MRI representation

The five static features are latest burden, latest burden disagreement,
cutoff-to-latest-scan recency, log scan count, and log observation span. The
five longitudinal features are first-to-last burden change, actual-date OLS
burden slope, last-two-scan burden change and slope, and first-to-last
disagreement change. MRI normalization is fitted on each outer-fit partition
only and shared by all arms in that fold.

## Frozen arms and interventions

1. `cognition_only`: cognitive history without MRI.
2. `static_mri`: cognitive history plus only the five static MRI features.
3. `true_longitudinal_mri`: static plus correctly dated longitudinal features.
4. `date_permuted_dynamics`: preserve the subject's latest static MRI,
   availability, dates, recency, count, and span, but cyclically permute which
   scan values are assigned to the dates before recomputing dynamics.
5. `sign_flipped_dynamics`: train exactly as `true_longitudinal_mri`, then
   reverse only the five normalized dynamic features at inference. This is a
   directional intervention. Training a separate model on consistently
   sign-flipped dynamics would be invalid because that transformation is
   invertible.
6. `subject_shuffled_dynamics`: preserve all static features and exchange only
   dynamic features between different subjects within scan-count and span
   strata, separately in fit, development, and evaluation partitions.

All models use the same reference-aware Gaussian filter, train-fold clinical
decoder, optimizer settings, and subject partitions. Controls never change the
same subject's latest burden, latest disagreement, availability, recency, scan
count, or span.

## Frozen decision rule

Temporal MRI signal is supported only if `true_longitudinal_mri` has lower
aggregate Memory-composite MAE than static MRI and all three dynamics
interventions, is better in at least four of five outer folds and three of four
horizons against every comparator, and its empirical 95% clinical coverage is
within [0.90, 0.98]. Participant-cluster bootstrap intervals are descriptive
and are not an additional gate.

Passing opens a later disagreement-aware filtering experiment. Failure freezes
MRI as a static auxiliary covariate for the current representation. Neither
outcome reopens the Phase 5F measurement gate or constitutes confirmatory
multimodal evidence.
