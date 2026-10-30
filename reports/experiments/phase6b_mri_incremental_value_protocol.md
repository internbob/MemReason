# Phase 6B: MRI incremental-value and negative-control audit

Status: frozen before execution. Phase 6A produced a small real-versus-shuffled
MRI latent-MAE difference and under-covered predictive intervals. Phase 6B
tests whether correctly matched pre-cutoff MRI content improves an observable
future Memory outcome beyond cognitive history and MRI acquisition metadata.
It remains original-train-only exploratory work and cannot reopen the failed
Phase 5F measurement gate.

## Data boundary and primary outcome

- Reuse the Phase 6A five outer subject folds and subject-disjoint inner
  development rule.
- Use only original ADNI train subjects. Original validation and test subjects
  are not loaded.
- Cognitive posteriors and MRI summaries are restricted to the prediction
  cutoff or earlier. Future diagnosis and future MRI are never model inputs.
- The primary endpoint is future observed Memory composite absolute error. The
  composite is the mean train-contract severity fraction over available Memory
  items and therefore lies on a common 0--1 scale.
- The Phase 5G Memory posterior remains an auxiliary distributional target;
  latent MAE cannot by itself establish clinical incremental value.

## Frozen arms

1. `cognition_only`: reference-aware Gaussian filtering of cognitive
   posteriors without an MRI update.
2. `metadata_only`: add only MRI recency and scan-count metadata.
3. `real_mri`: add correctly matched structural burden, burden disagreement,
   burden slope, recency, and scan count.
4. `subject_shuffled_mri`: preserve each sample's MRI availability, recency,
   and scan count while permuting the three structural features across subjects
   within the same fit/dev/evaluation partition.
5. `time_shuffled_mri`: preserve availability and acquisition metadata while
   permuting structural features among cutoffs of the same subject.
6. `mri_only`: use the train-fold reference trajectory and real MRI evidence
   without cognitive posterior evidence.

All arms use identical fold membership and optimizer settings. MRI
normalization is fitted only on outer-fit samples. Negative controls are
constructed independently inside fit, development, and evaluation partitions.

## Training and uncertainty

The continuous-time model predicts the Phase 5G Gaussian teacher and is
optimized with Gaussian NLL plus a 2-Wasserstein term. A train-fold-only linear
decoder maps latent state to the observed Memory composite. Its residual
variance is included in a clinical Gaussian NLL term and in clinical prediction
intervals. No evaluation-fold outcome is used for decoder fitting, early
stopping, normalization, or interval construction.

## Evaluation and development decision

Report primary Memory-composite MAE, clinical NLL, 95% coverage and interval
width, plus auxiliary latent MAE/NLL/Wasserstein distance. Report results
overall, by horizon, by fold, and for MRI-available samples. Participant-level
paired bootstrap intervals are descriptive.

The MRI-content signal is considered supported only if `real_mri` has lower
primary MAE than cognition-only, metadata-only, subject-shuffled, and
time-shuffled controls; each comparison favors real MRI in at least four of
five folds; real MRI is non-inferior in at least three of four horizons for
each comparator; and clinical 95% coverage lies in [0.90, 0.98]. Failure is a
valid negative result and blocks escalation to a multimodal disagreement-aware
filter until MRI representation or alignment is revised.
