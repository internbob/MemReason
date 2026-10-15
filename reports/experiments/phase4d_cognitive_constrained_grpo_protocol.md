# Phase 4D cognitive-only constrained GRPO protocol

Status: frozen on 15 October 2026 before Phase 4D rollout generation, parameter updates, checkpoint evaluation, and original-test reasoning-example construction.

## Scope and data isolation

Phase 4D tests real on-policy GRPO updates to the Phase 4B rank-16 LoRA inside the Phase 4C token-trie action space. The cognitive forecaster, normalization, evidence rows, numeric state statistics, forecast means, and uncertainty anchors remain frozen. MRI and diagnosis remain excluded.

Rollouts use one stable-hash sample from each of 64 Phase 4A build subjects. Checkpoint selection uses one sample from each of the 167 Phase 4A internal-dev subjects. The 168 Phase 4B confirmation subjects and 234 Phase 4C confirmation subjects are excluded from rollout and checkpoint selection. Only after the development decision is written may the pipeline load the original ADNI test split and construct a one-time reasoning confirmation cohort of at most 256 subjects. These test subjects are new to structured-reasoning training and development, although the original test split has been used by earlier non-reasoning experiments.

## Constrained policy and rewards

Each on-policy candidate contains ten model-controlled actions: five current-state labels and five temporal-trend labels. At every action, sampling is restricted to the frozen Phase 4C enum trie. All enum alternatives have unique first tokens under the local tokenizer, so the exact constrained categorical policy is the softmax over those permitted first-token logits; subsequent tokens are deterministic trie continuations.

Each prompt produces four candidates at temperature 1.2. Rewards are availability-normalized with frozen weights: current state 0.35, temporal consistency 0.35, future-outcome audit 0.10, evidence 0.05, domain 0.05, uncertainty preservation 0.05, and format 0.05. Current-state and temporal components vary across candidates and supply the learning signal. Evidence, domain, uncertainty, and format are deterministic renderer checks. The outcome component uses real future observations and the frozen GRU-D forecast, but is identical within a candidate group because Phase 4D does not permit the language model to alter numeric forecasts. It is recorded as an RLVR audit component and cannot create a group-relative gradient. MRI and cross-modal rewards are unavailable and omitted rather than set to zero.

## Optimization and stopping

The policy starts from the Phase 4B SFT adapter; a separate frozen copy is the reference policy. Only existing LoRA parameters are optimized. The fixed settings are group size 4, rollout batch size 2, activation gradient checkpointing, learning rate 1e-5, clipped ratio epsilon 0.2, exact categorical reference KL beta 0.02, gradient norm 0.5, and at most 32 updates. Adapters are saved at updates 8, 16, 24, and 32. Gradient checkpointing was enabled after a pre-update allocation attempt exceeded A100 memory; no rollout update, metric, or test data existed before this hardware-only adjustment.

Training aborts on non-finite loss/gradients, three consecutive updates with mean categorical KL above 0.15, or eight consecutive updates with mean group reward standard deviation below 0.005. Every update records the complete reward decomposition, reward dispersion, exact KL, entropy, clip fraction, gradient norm, and sampled actions.

## Selection and final gate

The development set selects the checkpoint with the highest mean of direct state-label and trend-label accuracy; ties choose the earlier checkpoint. Selection requires strict improvement over the SFT action composite, state and trend accuracy of at least 0.97, controlled KL, at least 10% reward-variable rollout groups, and no safety abort.

After selection, SFT and the single frozen GRPO checkpoint are evaluated once on the new reasoning confirmation cohort. The final gate requires 100% JSON validity and anchor preservation, zero future-evidence leakage, state and trend paired-bootstrap lower bounds of at least -0.01, and positive 95% confidence-interval lower bounds for both the action-composite and exact-report improvements. Forecast values must remain byte-equivalent to the frozen anchors. Passing supports only cognitive-only constrained GRPO. It does not support multimodal GRPO or claim that numeric forecasting improved.
