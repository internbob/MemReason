# Phase 6D1: static 128-D embedding incremental value

Status: frozen before execution. Phase 6D0 must pass before this stage starts.

For outer fold `k`, all MRI scans are encoded only by the fold-aligned encoder
`k`. A StandardScaler and 16-component PCA are fitted using scans from the
outer-fit, non-development subjects only. Only scans acquired on or before the
cognitive cutoff are eligible; the latest embedding, AD probability, recency,
and scan count are exposed to the downstream arms.

The six arms isolate cognition, acquisition metadata, the fold-aligned AD
probability, the embedding, their combination, and a subject-shuffled embedding
that preserves availability and acquisition metadata. The primary endpoint is
future observed Memory-composite MAE at 6/12/24/36 months.

The embedding gate requires `embedding_static` to beat cognition-only,
probability-static, and subject-shuffled embedding in aggregate with a
participant-bootstrap upper bound below zero, in at least four of five folds,
with at least three of four horizons noninferior and 95% clinical coverage in
`[0.90, 0.98]`. Failure stops the embedding sequence; passing opens Phase 6D2.
