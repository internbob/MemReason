# Phase 3D multimodal item-to-MRI grounding protocol

Date frozen: 2026-10-14

## Question

Can verified cognitive item queries combine with concurrent MRI spatial tokens to reconstruct deliberately hidden cognitive responses more accurately than cognition alone or global MRI pooling? Does any gain depend on the correct subject, spatial arrangement, and item query?

This validation-only experiment follows the passed Phase 3C spatial-grounding gate. It does not evaluate future forecasting or load the test split.

## Cohort and fixed inputs

- Reuse the exact Phase 3B/3C train and validation MRI anchors within the verified 90-day cognition–MRI window.
- Use the same 68 scalar, monotonic cognitive fields whose item definitions and response directions were verified for Phase 1B.
- Use frozen Bio_ClinicalBERT task-descriptor embeddings as item queries. The language encoder is not updated.
- Reuse the three frozen Phase 3C SFCN spatial grids (3×4×3 tokens, 64 channels). No MRI backbone parameter is updated.
- Fit item normalization on eligible train observations only. Test remains sealed.

## Concurrent partial-observation tasks

For every MRI anchor, deterministic masks are generated before model training:

1. `masked_item`: hide up to three observed verified items and predict each response separately;
2. `masked_instrument`: hide all available responses from one observed instrument and reconstruct them jointly.

Only item responses whose instrument assessment date lies within 90 days of the MRI are eligible. Naturally missing values remain missing and never contribute to the loss.

## Parameter-matched arms

All arms contain the same descriptor projection, response encoder, two-layer cognitive self-attention encoder, MRI projection, four-head cross-attention module, fusion layer, and shared item decoder.

1. `cognition_only`: reconstruct from the remaining cognitive items and demographics; MRI context is zero.
2. `global_mri`: add the mean-pooled MRI token representation to each item query.
3. `item_spatial_attention`: use every response-aware item query to attend over MRI spatial tokens before reconstruction.

Each arm is trained with seeds 2026, 2027, and 2028. Model selection uses the mean of the two validation task macro normalized MAEs.

## Negative controls and statistics

The trained spatial model is evaluated with a different validation subject's MRI, with feature tokens permuted while coordinates remain fixed, and with semantic item queries cyclically permuted while response values remain in their native fields. Primary contrasts use 2,000 subject-cluster bootstrap replicates of paired normalized absolute errors. Cross-seed MRI-attention stability uses Spearman correlation and top-20% token overlap for each hidden item.

## Frozen gate

Full multimodal item grounding is supported only when:

1. spatial attention improves both masked tasks over cognition-only with bootstrap upper bounds below zero;
2. it improves overall over global pooling;
3. observed pairing improves over subject shuffle, spatial permutation, and query permutation;
4. median cross-seed attention Spearman is at least 0.25, top-token overlap is at least 0.35, and mean normalized attention entropy is below 0.98.

Failure retains Phase 3C as a total-score concurrent MRI result and prevents a claim of item-level multimodal grounding. Attention is treated as a predictive grounding map, not causal neuroanatomy.
