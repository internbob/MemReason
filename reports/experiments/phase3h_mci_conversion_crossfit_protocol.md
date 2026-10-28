# Phase 3H MCI-to-AD conversion cross-fitting protocol

## Status and purpose

This is an exploratory, user-requested follow-up to Phase 3G. Phase 3G's
prespecified automatic progression gate failed, so Phase 3H does not revise or
override that confirmatory decision. It estimates whether baseline structural
MRI contains prognostic information for conversion from MCI to AD and reports
the requested ACC, AUC, sensitivity, specificity, and F1 metrics.

## Cohort and outcome

- Use original-train subjects only. Do not load official validation or test.
- Begin with the real-date, MRI-linked Phase 3G cohort and retain visits whose
  concurrent DXSUM diagnosis is MCI.
- For each horizon, choose the earliest eligible MCI MRI per subject.
- A positive outcome is the first subsequent DXSUM AD diagnosis occurring
  strictly after the baseline diagnosis date and within 24 or 36 months.
- A nonconverter must have at least one subsequent DXSUM diagnosis at or after
  the complete horizon and no AD diagnosis within that horizon.
- Exclude right-censored baselines for which neither condition can be proven.
- Diagnosis fields and post-baseline cognitive observations are forbidden as
  predictors.

## Predictors and arms

Demographics are age and sex. Cognitive history uses ADAS13, CDR-SB, FAQ,
MMSE, and MoCA observations at or before the baseline MRI cutoff, represented
by last value, mean, slope, log count, recency, and availability. Global MRI
features are deterministic volume/intensity summaries. The 3D SFCN consumes
the 48 x 56 x 48 masked baseline T1 volume.

The seven arms are demographics, global MRI, SFCN MRI, cognition, cognition +
global MRI, cognition + SFCN MRI, and cognition + subject-shuffled SFCN MRI.
The shuffled arm tests whether any fusion gain depends on the correct patient's
image.

## Cross-fitting

- Analyze 24- and 36-month outcomes separately.
- Use five subject-wise stratified outer folds and an 85/15 subject-wise inner
  fit/tune split.
- Train a new binary SFCN from scratch in each horizon/fold/seed using only
  outer-fit baseline scans. Select epochs by inner-tune ROC-AUC.
- Use seeds 2026, 2027, and 2028 and average seed probabilities.
- Fit all final classifiers only on outer-fit subjects. Outer-test predictions
  are used once for evaluation.
- Use the frozen probability threshold 0.5 for ACC, sensitivity, specificity,
  and F1. AUC remains threshold-free.

## Metrics and statistics

For every arm and horizon report ACC, ROC-AUC, sensitivity for conversion,
specificity for nonconversion, F1 for conversion, balanced accuracy, and
confusion counts. Report 2,000 subject-bootstrap 95% confidence intervals for
ACC, AUC, sensitivity, specificity, and F1. Also report paired bootstrap AUC
contrasts for SFCN MRI versus demographics, fusion versus cognition, and fusion
versus shuffled-MRI fusion.

## Interpretation boundary

This experiment is internal train-only cross-fitting. It evaluates prognostic
classification, not etiologic AD biomarker status, time-to-event risk under
competing risks, or clinical deployment. Any positive finding requires an
untouched cohort or external confirmation.
