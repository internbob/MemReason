# Phase 4A frozen cognitive structured-reasoning baseline protocol

Status: frozen before Phase 4A training and report generation.

## Scope

Phase 4A establishes a machine-verifiable cognitive-only structured-report baseline. It does not train an LLM and does not claim free-form clinical reasoning. MRI and diagnosis are excluded. The experiment uses only subjects from the original train split; the original validation and consumed test splits are not used.

Eligible original-train subjects are divided 80%/10%/10% into reasoning-build, reasoning-dev, and reasoning-confirmation partitions at the subject level, stratified by maximum observed diagnosis severity only for split balance. Diagnosis is not a model or report input.

## Frozen cognitive forecast

Three GRU-D models use the 68 dictionary-verified monotonic scalar items from the formal baseline suite. Normalization is fitted on reasoning-build subjects only. Checkpoints are selected on reasoning-dev macro normalized MAE and then frozen. The three-seed mean is the point forecast; across-seed standard deviation is the reported epistemic uncertainty. Confirmation subjects do not participate in fitting, early stopping, normalization, or report construction rules.

## Structured report

Each report contains exactly five JSON objects: `current_state`, `evidence`, `trend`, `forecast`, and `confidence`. Evidence is limited to verified cognitive items observed at or before the cutoff. Up to two item trajectories per cognitive domain are selected deterministically by recency. Current state and trend are deterministic functions of those cited histories and reasoning-build normalization. Forecast and confidence are copied from the frozen GRU-D ensemble. Future outcomes are retained in a separate evaluation table and never appear in prompts or completions.

The generated JSONL is an auditable teacher baseline and candidate Phase 4B SFT corpus. It contains structured evidence summaries rather than hidden chain-of-thought.

## Evaluation and gate

The frozen reasoning-confirmation partition determines readiness for Phase 4B. The gate requires:

- JSON validity 100%;
- evidence precision/recall and numeric/time citation accuracy at least 99%;
- hallucinated evidence at most 1%;
- domain accuracy and temporal consistency at least 95%;
- current-state consistency at least 99%;
- forecast and uncertainty preservation at least 99.99%;
- zero future-evidence leakage.

Forecast MAE and uncertainty-error association are reported descriptively and do not alter the mechanical reasoning-readiness gate. Passing Phase 4A opens cognitive-only SFT development. It does not open multimodal reasoning or GRPO and does not override failed MRI/dynamics gates.
