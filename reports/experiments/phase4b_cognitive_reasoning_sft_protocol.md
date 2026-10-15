# Phase 4B cognitive-only structured-reasoning SFT protocol

Status: frozen before LoRA training and before any reasoning-confirmation generation.

## Scope and isolation

Phase 4B trains a local causal language model to emit the five-section JSON contract established in Phase 4A. It uses the unchanged Phase 4A subject-disjoint build/dev/confirmation registry. The source corpus contains 13,238/1,582/1,757 build/dev/confirmation reports. To keep the fixed 3.3k-token experiment within a single-GPU budget while retaining subject diversity, training uses a deterministic subject-balanced subset of at most two reports per build subject. Dev uses at most one stable-hash-selected report per subject and forecast horizon; confirmation uses one stable-hash-selected report for every confirmation subject and retains all four horizons across the cohort. Selection is frozen before any confirmation generation. Original ADNI validation and test subjects remain excluded. MRI, diagnosis, and future observed outcomes are not model inputs.

## Model and optimization

The local `DeepSeek-R1-Distill-Qwen-1.5B` checkpoint is the fixed base model. Only rank-16 LoRA parameters on query/key/value/output attention projections are optimized. Training uses completion-only cross entropy: prompt tokens and padding receive label `-100`. The budget is one epoch over the selected build subset, batch size 4, gradient accumulation 4, learning rate 2e-4, BF16, and SDPA. Gradient checkpointing is disabled because the A100 80GB fits the batch directly. There is no hyperparameter search and no selection based on confirmation outcomes.

## Input and target contract

For every verified item trajectory, the prompt retains the first and latest observations. These endpoints preserve the evidence needed for current state and direction while bounding context length. The prompt also supplies label-free frozen domain statistics, the Phase 4A GRU-D forecast, and its ensemble uncertainty. These are model outputs or deterministic summaries available at the cutoff; they are not future truth. The LLM must classify the state/trend labels, select the latest evidence citations, and copy all supplied numeric forecast and confidence anchors exactly. Targets are the Phase 4A reports with exactly `current_state`, `evidence`, `trend`, `forecast`, and `confidence`.

All prompt-plus-completion sequences must fit within 3,584 tokens. Any overflow is a hard error; no target truncation is permitted.

## Frozen comparison and gate

The confirmation set compares the unmodified base model, the LoRA SFT model, and the deterministic Phase 4A teacher. Greedy decoding is used for both language-model arms. Raw generations are evaluated without JSON repair or numeric post-processing. Subject-cluster bootstrap intervals compare SFT with the base model.

The SFT gate requires 100% JSON/schema validity; at least 99% evidence precision/recall, numeric/time/domain accuracy, forecast preservation, and uncertainty preservation; at least 95% current-state and temporal consistency; hallucinated evidence at most 1%; zero future-evidence leakage; and a positive improvement in the mean of format, evidence, state, trend, forecast, and uncertainty scores over the base model. Passing opens cognitive-only GRPO preparation. It does not open multimodal reasoning.
