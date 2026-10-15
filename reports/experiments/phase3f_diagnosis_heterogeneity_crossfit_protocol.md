# Phase 3F diagnosis-heterogeneity cross-fit protocol

Status: frozen before the Phase 3F out-of-fold models were trained.

## Question and data boundary

Phase 3E did not confirm the overall Phase 3C advantage, but its prespecified secondary analysis suggested opposite effects in AD and MCI. Phase 3F tests whether this heterogeneity reproduces without reusing validation or test outcomes. Only subjects assigned to the original `train` split are eligible. Diagnosis labels are used after prediction for stratified statistics and are never model inputs.

## Cross-fitting and models

Subjects are assigned once to five mutually exclusive folds, stratified by their maximum observed diagnosis severity and MRI cohort where counts permit. Each fold is predicted by a model trained on the other four folds. Every scan therefore receives an out-of-fold prediction from a model that did not see that subject.

The two parameter-matched Phase 3C heads are compared: global pooling and target-conditioned spatial attention. SFCN spatial tokens, optimizer, architecture, seeds, and preprocessing are frozen from Phase 3C. Epoch counts are fixed independently for each arm and seed from the Phase 3C development curves; Phase 3F performs no early stopping or hyperparameter selection.

## Primary estimands and controls

The primary estimand is the paired normalized absolute-error difference, spatial attention minus global pooling. Negative values favor spatial attention. Subject-cluster bootstrap intervals are computed overall and within visit-level CN, MCI, and AD groups. A subject-cluster bootstrap interaction estimates the AD effect minus the MCI effect. Fold-wise effects assess directional replication.

Observed spatial attention is also compared with subject-shuffled MRI tokens, a fixed spatial-token permutation, and a fixed target-query permutation. Cross-seed attention stability and cross-fold diagnosis/target mean-map stability are evaluated with Spearman correlation and top-20% token overlap.

## Frozen gate

The diagnosis-heterogeneity gate passes only if all conditions hold:

1. AD has a significant benefit (95% CI upper bound below zero).
2. MCI has significant harm (95% CI lower bound above zero).
3. The AD-minus-MCI interaction has a 95% CI upper bound below zero.
4. AD benefit and MCI harm each have the expected direction in at least four of five folds.
5. Observed attention significantly outperforms all three mechanism controls overall.
6. Median cross-seed Spearman is at least 0.30 and median top-token overlap is at least 0.40.
7. Median cross-fold mean-map Spearman is at least 0.25 and median top-token overlap is at least 0.35.

This is a train-only hypothesis replication study prompted by a test-set subgroup observation. Even a passing result is developmental evidence and cannot restore the failed Phase 3E confirmatory claim. A new external or prospectively untouched cohort would be required for confirmation.
