# Phase 6A: Exploratory distributional latent-state dynamics

Status: frozen before execution. Phase 5F did not pass its prespecified
measurement gate, so this experiment is explicitly exploratory and cannot
retrospectively open continuous-time claims.

## Teacher state and data boundary

The target is the selected Phase 5G graph-supported OOF Memory posterior. Its
mean is the average of the two disjoint instrument views; its variance includes
both posterior variance and view disagreement. Only original-train subjects are
used. Dynamic models use five subject-wise outer folds with a subject-disjoint
inner development split. Original validation and test subjects are not loaded.

## Sequential arms

1. `latent_locf`: carry the latest posterior mean forward.
2. `exact_gaussian`: analytic diagonal local-linear Gaussian propagation from
   the latest posterior.
3. `precision_filter`: propagate and precision-fuse every observed posterior.
4. `latent_wasserstein`: add Gaussian Wasserstein alignment to the future
   posterior teacher.
5. `reference_filter`: express state as a deviation from a train-fold-only CN
   reference trajectory before filtering.
6. `reference_mri_filter`: add a pre-cutoff native-MRI evidence factor.
7. `shuffled_mri_filter`: identical model with deterministically permuted MRI
   features as a negative control.

MRI summaries use only scans at or before the cognitive cutoff. No future MRI,
diagnosis, validation outcome, or test outcome is available to the model.

## Evaluation

Report latent MAE, distributional NLL, Gaussian 2-Wasserstein distance, 95%
coverage, uncertainty growth, and a train-fold-calibrated observed Memory
composite MAE. Participant-cluster bootstrap comparisons are descriptive.

Even a positive result is hypothesis-generating because the Phase 5G teacher
did not meet the original 0.70 cross-view CCC threshold.
