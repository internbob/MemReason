# Phase 4C dev-only grammar-constrained structured-decoding protocol

Status: frozen before Phase 4C dev decoding and before constructing or decoding the new confirmation cohort.

## Objective and isolation

Phase 4C tests whether the frozen Phase 4B LoRA can be converted into a reliable structured decoder without additional parameter updates. Decoder selection uses only the Phase 4A internal dev subjects. The previous Phase 4B confirmation subjects are excluded from selection and confirmation. After selection, the frozen decoder is evaluated once on eligible subjects from the original ADNI validation split, which were absent from Phase 4A/B training, dev, and confirmation.

The frozen Phase 4A GRU-D ensemble and build-only normalization are applied to the new subjects. Future observed outcomes remain outside prompts and reports. MRI and diagnosis are excluded.

## Decoder arms

`raw_sft` is the unchanged Phase 4B greedy decoder with a 1,280-token limit. `grammar_model` fixes the five-section JSON grammar and copies deterministic evidence, numeric state statistics, forecast, and uncertainty anchors from the verified input. At each state or trend label, the frozen LoRA selects greedily from a token trie containing only the documented enum values. `grammar_executable` uses the same renderer but derives state and trend labels from the predeclared Phase 4A thresholds; it is an executable control rather than evidence of improved language-model reasoning.

No malformed JSON repair, retry, or access to target reports is allowed. The model-constrained arm has priority if it passes every dev gate. The executable arm is selected only if the model-constrained arm fails and the executable arm passes. This hierarchy is frozen before dev results.

## Gate and interpretation

The selected decoder requires 100% JSON validity, forecast and uncertainty preservation, and zero hallucinated or future evidence. Evidence precision/recall, numeric/time/domain accuracy, current-state consistency, and temporal consistency must be at least 99%.

Passing with `grammar_model` establishes a stable cognitive-only SFT decoder and may open a separately frozen constrained-GRPO experiment. Passing only with `grammar_executable` establishes a reliable hybrid report compiler but does not establish that the raw SFT policy is ready for GRPO. Multimodal reasoning remains closed.
