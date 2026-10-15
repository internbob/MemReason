# Phase 4F: Frozen GRU-D + Phase 4C sealed-test confirmation

Status: frozen before loading the original ADNI test split for structured reasoning.

## Primary question

Does the completely frozen cognitive system preserve useful multi-horizon forecast accuracy and produce valid, evidence-grounded structured reports on the official subject-disjoint test cohort?

## Frozen system

The numeric component is the three-checkpoint Phase 4A GRU-D ensemble trained with build-only normalization and selected before any original validation or test evaluation. The report policy is the Phase 4B DeepSeek-R1-Distill-Qwen-1.5B rank-16 LoRA decoded with the Phase 4C `grammar_model` token trie. The verified 68-item dictionary, evidence compiler, domain statistics, state/trend enums, forecast and confidence renderers, prompt, and all decoding settings are unchanged. Phase 4D and Phase 4E adapters are excluded.

Checkpoint and adapter SHA-256 hashes are recorded in `pretest_lock.json` before test data construction. Phase 4F performs no training, checkpoint selection, prompt repair, threshold selection, or model update.

## Test cohort and outputs

All valid 6-, 12-, 24-, and 36-month forecast samples from `splits/test.csv` are evaluated numerically. For structured-report metrics, one sample per eligible test subject is selected by SHA-256 stable hashing with seed 20261022 without inspecting outcomes. All selected histories must contain only verified observations at or before the cutoff. Future truth is stored only in the forecast evaluation table and never enters prompts, completions, or decoding.

The frozen `grammar_model` is the primary arm. Raw greedy Phase 4B SFT is the prespecified generation control. The deterministic teacher and executable threshold renderer are audit controls. MRI and diagnosis are not reasoning inputs.

## Metrics and statistical analysis

Forecast evaluation reports MAE and normalized MAE by target and horizon, macro normalized MAE over available target-horizon cells, paired GRU-D versus LOCF normalized error on rows with an observed last target, and the Spearman association between normalized ensemble uncertainty and normalized absolute error. The paired comparison uses 2,000 subject-cluster bootstrap replicates.

Structured-report evaluation reports JSON validity, evidence precision/recall, numeric/time/domain citation accuracy, hallucinated evidence, future-evidence leakage, state consistency, trend consistency, forecast preservation, uncertainty preservation, and exact report match. Grammar-minus-raw differences use 2,000 subject-cluster bootstrap replicates. Frozen robustness summaries stratify by horizon, maximum observed domain timepoints (1, 2–3, or 4+), and number of unavailable last targets (0–1 or 2+). These strata are descriptive and cannot change the primary decision.

## Confirmation gate

The complete system passes only if:

1. GRU-D macro normalized MAE is at most 0.42.
2. On paired covered rows, GRU-D normalized MAE is no more than 0.02 worse than LOCF.
3. Normalized ensemble uncertainty has a positive Spearman association with normalized absolute error.
4. Grammar decoding achieves 100% valid JSON, forecast preservation, and uncertainty preservation.
5. Evidence precision/recall and numeric/time/domain accuracy are at least 0.99.
6. State and trend consistency are at least 0.99.
7. Hallucinated evidence and future-evidence leakage are exactly zero.
8. The 95% subject-bootstrap confidence interval for grammar-minus-raw JSON validity has a lower bound above zero.

The test is opened once. Regardless of outcome, Phase 4F does not permit further adjustment on test subjects. A failed gate is reported as a confirmatory negative result; a passed gate supports the frozen cognitive-only system but does not support GRPO, multimodal reasoning, or clinical validity claims.
