# Phase 3G MRI diagnostic incremental-value cross-fitting protocol

## Clinical question

At an MRI-linked cognitive visit, does structural T1 MRI contain concurrent
CN/MCI/AD diagnostic information, and does it improve diagnosis beyond the
available longitudinal cognitive history and demographics?

The intended use is supportive risk stratification for patients undergoing
cognitive evaluation. The experiment does not evaluate autonomous diagnosis,
etiologic Alzheimer biomarker status, treatment selection, or clinical
deployment.

## Data and reference standard

- Use only original-train subjects and never load the official validation or
  test subjects.
- Start from the real-date, 90-day MRI/cognition anchors and the cached
  48 x 56 x 48 masked T1 volumes created by Phase 3B.
- Use the harmonized `dxsum_diagnosis_label` at the MRI-linked visit as the
  reference standard. Include only CN, MCI, and AD.
- Require the DXSUM date and scan date to differ by no more than 90 days.
- Do not use diagnosis fields as model inputs.
- Use every eligible longitudinal scan, but keep all scans from a subject in
  one fold and give each subject equal total weight.

## Predictors

Demographics comprise age and sex. Cognitive history is constructed strictly
at or before the MRI-linked cutoff and contains, for ADAS13, CDR-SB, FAQ, MMSE,
and MoCA, the last value, historical mean, least-squares slope, observed count,
and recency. Missing histories remain explicit.

Global MRI features are deterministic Phase 3A image/mask summaries. The SFCN
encoder consumes the cached 3D T1 volume. A new SFCN is trained from scratch in
each outer-fold/seed using only outer-fit subjects; Phase 3B checkpoints are not
reused. The selected fold-local encoder produces a 128-dimensional embedding.

## Arms

1. `demographics`
2. `global_mri` = demographics + global MRI
3. `sfcn_mri` = demographics + cross-fitted SFCN embedding
4. `cognition` = demographics + longitudinal cognitive history
5. `cognition_global_mri`
6. `cognition_sfcn_mri` (primary candidate)
7. `cognition_shuffled_sfcn` (same fitted fusion classifier, with test MRI
   embeddings permuted across different subjects)

All final heads are fixed-C multinomial logistic regression models fitted only
on outer-fit subjects. A scalar temperature is selected on the inner-tune
subjects. No hyperparameter uses outer-test predictions.

## Cross-fitting and evaluation

- Five subject-wise stratified outer folds.
- An inner subject-wise 85/15 fit/tune split in every outer fold.
- SFCN seeds 2026, 2027, and 2028; probabilities are averaged across seeds.
- Primary metrics use equal total weight per subject: macro one-vs-rest AUROC,
  balanced accuracy, macro F1, multiclass log loss, multiclass Brier score, and
  expected calibration error.
- Report per-class sensitivity, confusion matrices, AD-vs-non-AD decision
  curves, diagnosis/sex/age/ADNI-phase/history-missingness strata, and 2,000
  subject-cluster bootstrap intervals.

## Frozen gates

The concurrent diagnostic increment gate requires all of the following:

1. `sfcn_mri - demographics` macro AUROC >= 0.01 with bootstrap CI low > 0.
2. `cognition_sfcn_mri - cognition` macro AUROC >= 0.01 with CI low > 0.
3. `cognition_sfcn_mri - cognition_shuffled_sfcn` macro AUROC >= 0.01 with CI low > 0.
4. Fusion balanced accuracy is no more than 0.01 below cognition-only.
5. Fusion Brier score is no more than 0.01 above cognition-only.
6. Fusion has greater AD-vs-non-AD net benefit than cognition-only at at least
   two prespecified probability thresholds.

The 24/36-month MCI-to-AD conversion task opens only if this entire concurrent
increment gate passes. A failure may still support cross-sectional MRI signal,
but it cannot support diagnostic increment beyond cognition or clinical utility.

## Interpretation boundary

ADNI clinical diagnosis partly depends on cognitive assessments. Accordingly,
`MRI versus demographics` measures imaging signal, while `fusion versus
cognition` is the primary incremental-value comparison. Internal cross-fitting
does not replace external testing. No result from this experiment changes the
negative Phase 3A/3E longitudinal and spatial-confirmation decisions.
