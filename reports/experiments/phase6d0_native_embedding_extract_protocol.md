# Phase 6D0: fold-aligned native-T1 embedding extraction

Status: frozen before execution. This stage creates the leakage-safe input for
the embedding experiments; it does not test an outcome hypothesis.

## Fold contract

- The five Phase 5G/6C subject folds are retained.
- For encoder fold `k`, every Phase 6 subject assigned to fold `k` is excluded
  from SFCN fitting and inner tuning.
- Phase 3I AD/CN subjects outside the Phase 5G state cohort retain their
  original stratified Phase 3I fold assignment.
- A separate SFCN is trained for each fold. Downstream fold `k` uses only
  encoder `k`; embeddings from independently trained encoders are never
  averaged coordinate-wise.

## Images and representation

Native T1 volumes remain on the complete `193 x 229 x 193`, 1-mm grid with the
existing brain-mask normalization. The SFCN is trained on the original-train
AD-vs-CN task. Its frozen 128-dimensional penultimate embedding and AD
probability are cached for every original-train longitudinal scan under each
fold-specific encoder.

## Required audit

The stage passes only if all five checkpoints and embedding arrays exist, every
array is finite and exactly 128 dimensional, probabilities lie in `[0,1]`, and
no Phase 6 evaluation subject occurs in the corresponding encoder's fit or
tune partition. Original validation and test data are not loaded.
