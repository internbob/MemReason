# Phase 5H: Measurement-identifiability and reliability-ceiling audit

Status: frozen before execution. This audit uses only the original ADNI train
split and does not train or select a new latent-state model.

## Question

Is the Phase 5F cross-instrument Memory CCC threshold of 0.70 empirically
supported by the reliability and overlap of the available instruments, or is
the observed ceiling dominated by non-parallel constructs and measurement
noise?

## Analyses

1. Build severity-oriented Memory composites separately for each instrument
   using only items in the frozen Phase 5C contract.
2. Estimate deterministic split-half reliability and Spearman-Brown correction
   for instruments with at least two Memory items.
3. Estimate consecutive-visit stability over 3--18 months. This is reported as
   temporal stability, not pure test-retest reliability, because true disease
   change may occur.
4. Estimate concurrent cross-instrument Pearson, Spearman, and CCC with
   participant-cluster bootstrap intervals.
5. Compute an attenuation ceiling `sqrt(reliability_a * reliability_b)` when
   both instruments have estimable positive reliability. This is a diagnostic,
   not proof that the corresponding latent state is identifiable.
6. Recompute the selected Phase 5G graph-supported CCC as an implementation
   reference.

The audit cannot retrospectively reopen the Phase 5F gate. A revised threshold
would require a new protocol and independent confirmation.
