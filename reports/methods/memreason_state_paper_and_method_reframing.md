# MemReason-State: Paper and Method Reframing

## Central research question

**Can a model infer an interpretable, continuously evolving cognitive memory state from sparse, asynchronous, and partially observed cognitive assessments, and use that state to forecast and explain individualized change across the Alzheimer disease continuum?**

This question should become the paper's organizing principle. Forecasting is one test of whether the state is useful; structured reporting is one way to expose the state. Neither the GRU-D forecaster nor JSON generation is itself the central contribution.

## Recommended paper identity

**Working title:** *MemReason-State: Interpretable Continuous-Time Memory-State Inference from Asynchronous Longitudinal Cognitive Assessments*

**Two-sentence pitch:** Longitudinal Alzheimer cohorts observe different cognitive items and instruments at irregular times, so a recurrent predictor can be accurate without representing a clinically coherent patient state. MemReason-State separates item measurement, continuous-time progression, event-driven Bayesian updating, and evidence attribution to produce an instrument-independent state whose validity can be tested through cross-instrument agreement, future prediction, uncertainty behavior, and downstream clinical associations.

The term `reasoning` should mean a verifiable computation over observations and state transitions:

1. what was believed before an observation;
2. which observation changed the belief;
3. how much and in which domain it changed;
4. how the state evolves without new evidence;
5. what future observations the updated state predicts.

A language model may verbalize this trace, but the trace must exist and be correct without the language model.

## Position relative to prior work

Continuous-time latent ODE models already establish that irregular observations can be represented by a latent process defined between visits; irregular-time handling alone is therefore not the novelty ([Rubanova et al., 2019](https://openreview.net/pdf?id=HygCYNSlLB)). Alzheimer disease progression models have also forecast subject-specific multi-biomarker trajectories across several cohorts, so individualized trajectory prediction alone is insufficient ([Vogel et al., 2023](https://www.nature.com/articles/s41467-022-35712-5)). Recent ADNI studies derive latent cognitive constructs or unsupervised cognitive/brain-health trajectories, demonstrating the value of construct-level modeling but typically without item-event Bayesian updates and executable evidence reasoning ([Mofrad et al., 2024](https://www.nature.com/articles/s41598-024-63301-7); [Subramaniapillai et al., 2024](https://www.nature.com/articles/s41598-024-60914-w)).

The defensible novelty target is the combination of: domain-identified typed item measurement, state inference from asynchronous cross-instrument evidence, explicit prior-to-posterior updates, uncertainty behavior, and a faithful reasoning trace whose statements are computed from those updates. A 2026 personalized Bayesian monitoring preprint is especially close in its emphasis on individualized baselines, sequential updating, and uncertainty; MemReason-State must distinguish itself through item-level cross-instrument state identification and state-update faithfulness rather than claim that Bayesian updating itself is new ([Sahakyan and Elvevåg, 2026](https://arxiv.org/abs/2608.03877)).

## What changes relative to the current system

| Current component | New role |
|---|---|
| GRU-D | Strong forecasting baseline and optional residual encoder; no longer the claimed memory-state mechanism. |
| Phase 2 `typed_belief_dynamics` | Engineering precursor. Its free latent dimensions and learned gates are not sufficiently identified or interpretable, and its frozen gate failed. |
| Phase 4A/4C reports | Reliable renderer and audit schema for an independently computed state-update trace. |
| Phase 4B LLM | Optional surface realizer. It cannot define the state or supply unobserved reasoning. |
| Phase 4D/4E GRPO | Negative appendix/ablation. Remove RL from the main method name unless a new external experiment supports it. |
| MRI Phase 3 | Optional construct-validity anchor. Do not make MRI a required state input because longitudinal and item-level MRI gates failed. |

## Proposed method: Interpretable Dynamic Memory State Model

### 1. Typed item measurement model

For subject (i), item (j), and observation time (t), model the observed response (y_{ijt}) through an item-specific likelihood:

\[
y_{ijt} \sim p_j\!\left(y\mid \alpha_j^\top z_i(t)+b_j\right).
\]

Use Bernoulli likelihoods for binary items, ordered-logit likelihoods for ordinal items, and bounded Gaussian/Beta or count likelihoods for continuous/count items. Fix each loading's direction using the audited scoring direction. Restrict primary loadings to the verified cognitive domain map, which gives the latent axes a stable meaning and prevents arbitrary rotation.

The state is

\[
z_i(t)=[m_i(t),e_i(t),a_i(t),l_i(t),o_i(t)],
\]

where (m) is the memory core and the remaining dimensions capture executive, attention, language, and orientation support. FAQ and diagnosis are excluded from the state encoder and used only for downstream validity. Memory evidence must include the audited memory items from ADAS, CDR, MMSE, and MoCA. Logical Memory and AVLT fields may be added only after source-level coding verification.

### 2. Progressive burden plus transient cognitive variation

Separate the slowly evolving impairment burden from short-term practice effects, noise, fatigue, and reversible variation:

\[
z_i(t)=p_i(t)+r_i(t),\qquad
\frac{dp_i(t)}{dt}=\operatorname{softplus}(f_\theta(p_i(t),t)),
\]

\[
dr_i(t)=-\kappa\,r_i(t)dt+\sigma dW_t.
\]

The positive impairment direction makes (p_i(t)) a slowly non-decreasing burden, while (r_i(t)) may improve or worsen locally. An unconstrained-dynamics ablation must test whether the monotone burden assumption improves interpretability without harming forecasting. If it fails, the main model should retain smooth but non-monotone dynamics rather than force an unsupported disease assumption.

### 3. Predict-update belief dynamics

Between observations, propagate the posterior through the continuous dynamics:

\[
q(z_i(t^-))=\operatorname{Predict}\left(q(z_i(t_k^+)),t-t_k\right).
\]

At every observed item or instrument event, update the prior with the typed likelihood:

\[
q(z_i(t^+))\propto q(z_i(t^-))\prod_{j\in\mathcal O_t}p_j(y_{ijt}\mid z_i(t)).
\]

Implement the update with a diagonal or low-rank Gaussian variational filter. The model must expose prior mean/variance, posterior mean/variance, and event innovation. Uncertainty should increase across long gaps and contract after informative observations; these are evaluated behaviors, not assumed claims.

### 4. Instrument-independent state constraints

The same patient state should not depend strongly on which instrument happened to be administered. Train with paired views of the same history:

- randomly partition observed instruments/items into two evidence views;
- infer (q_A(z_i(t))) and (q_B(z_i(t)));
- minimize a symmetric distribution distance only where the views contain adequate evidence;
- decode held-out items and instruments from both states.

This converts masked-item, masked-instrument, and cross-scale tasks into tests of one shared state rather than unrelated auxiliary heads.

### 5. State-grounded reasoning trace

For each event (e), compute:

- **innovation:** observed response minus prior predictive expectation;
- **state shift:** (\Delta\mu_e=\mu^+_e-\mu^-_e);
- **uncertainty reduction:** (\Delta H_e=H[q^-]-H[q^+]);
- **information gain:** (D_{KL}(q^+\|q^-));
- **forecast contribution:** change in a future prediction after incorporating the event.

The structured report should cite the largest verified contributions and render:

```json
{
  "memory_state": {"posterior_mean": "...", "posterior_interval": "..."},
  "state_update": [{"event_id": "...", "innovation": "...", "delta_state": "..."}],
  "trajectory": {"progressive_rate": "...", "transient_component": "..."},
  "forecast": {"6m": "...", "12m": "...", "24m": "...", "36m": "..."},
  "uncertainty": {"current": "...", "future": "..."}
}
```

Grammar-constrained decoding may guarantee schema validity, but every number and categorical statement must be executable from the state model. This is evidence-grounded state reasoning rather than free-form chain-of-thought.

## Training objectives

The development objective should combine:

\[
\mathcal L =
\mathcal L_{item\ likelihood}
+\lambda_f\mathcal L_{future\ likelihood}
+\lambda_h\mathcal L_{heldout\ instrument}
+\lambda_c\mathcal L_{cross\ view\ consistency}
+\lambda_u\mathcal L_{uncertainty\ calibration}
+\lambda_s\mathcal L_{smoothness}.
\]

Use proper likelihood losses for each response type. Avoid optimizing only total-score MAE because that allows the state to ignore item structure. Tune all weights within nested train-only folds; the consumed ADNI validation and test subjects cannot be used for method selection.

## Evidence required before calling the representation a memory state

1. **Measurement validity:** held-out memory items are decoded better than LOCF, population mean, multidimensional IRT, and a matched autoencoder.
2. **Cross-instrument coherence:** states inferred from different instrument subsets agree for the same patient/time and decode the omitted instrument.
3. **Temporal validity:** the state forecasts future memory, global cognition, and function across 6/12/24/36 months; GRU-D remains the predictive baseline.
4. **Downstream validity:** without using them as encoder inputs, the state and its slope associate with future FAQ, diagnosis transition, and MRI representations.
5. **Uncertainty validity:** longer evidence gaps increase uncertainty; informative events reduce it; intervals have acceptable coverage after calibration.
6. **Explanation faithfulness:** deleting high-information events changes the inferred state and forecast more than deleting matched low-information events; score-direction counterfactuals move the state in the expected direction.
7. **External transport:** the frozen measurement/dynamics structure reproduces on a separate cohort, with instrument mapping specified before evaluation.

## Formal baselines and ablations

- LOCF and linear mixed-effects trajectories.
- Multidimensional IRT/factor model without dynamics.
- GRU-D and typed direct Gaussian.
- Current Phase 2 typed belief model.
- Continuous-time state model without domain-constrained loadings.
- Model without progressive/transient decomposition.
- Model without cross-view consistency.
- Deterministic state versus probabilistic state.
- State-derived executable report versus Phase 4C grammar model.

The core success condition should require state validity plus predictive non-inferiority. A small MAE improvement alone is insufficient, and improved interpretability cannot excuse materially worse forecasting.

## How to write the paper

### Introduction

1. Longitudinal cognitive evidence is distributed across instruments, items, and irregular times.
2. Total-score forecasting hides whether a model has learned a coherent patient state.
3. Generic latent states are rotationally ambiguous and therefore hard to interpret clinically.
4. MemReason-State uses typed measurement constraints and continuous predict-update dynamics to make the state testable.
5. State-grounded reasoning exposes how evidence updates the posterior and forecast.

### Claimed contributions

1. An item-level, instrument-aware measurement model that identifies a shared memory-centered cognitive state.
2. A continuous-time probabilistic state transition that separates long-term burden from transient variation.
3. A verifiable reasoning trace based on Bayesian innovation, information gain, state shift, and forecast contribution.
4. A validation framework covering cross-instrument coherence, future prediction, calibration, faithfulness, and external transport.

### Results order

1. Cohort and asynchronous observation pattern.
2. Measurement validity and cross-instrument coherence.
3. Interpolation and future forecasting against GRU-D.
4. State trajectories, uncertainty dynamics, and downstream validity.
5. Reasoning faithfulness and representative patient trajectories.
6. External validation.
7. MRI, semantic encoder, and GRPO findings as secondary or negative ablations.

### Claims to avoid

- Do not call the current GRU-D hidden vector an interpretable memory state.
- Do not call template/grammar correctness clinical reasoning.
- Do not claim MRI improves longitudinal prediction or item grounding.
- Do not claim GRPO improves reasoning or forecasts.
- Do not use the consumed ADNI test set to select the redesigned method.

## Existing evidence that can remain in the paper

- GRU-D is a strong and test-confirmed cognitive forecasting baseline.
- Typed direct Gaussian modeling has a supported interpolation-specific benefit.
- Grammar constraints reliably preserve evidence, forecasts, and uncertainty fields.
- Phase 5B identifies categorical trend selection, rather than evidence or numeric-anchor preservation, as the main reporting failure.
- Semantic descriptors, explicit Phase 2 belief dynamics, MRI longitudinal fusion, multimodal item grounding, and two GRPO formulations did not pass their complete gates. These results motivate the simpler, state-centered redesign and belong in ablations or limitations.

## Immediate data issue

The current dictionary marks all AVLT variables and both Logical Memory scores as unverified. In the train split, memory-specific fields are present for 531 subjects and 2,392 visits, but `AVLT_Trial1Int` through `AVLT_Trial7Int` are mostly 0–2 and include 999, which is consistent with intrusion counts rather than recall totals. They cannot be treated as recall performance. `LogMemIIA_DelayedScore` includes the special code 99 and also requires source-level verification. The first implementation task is therefore a formal AVLT/Logical Memory coding audit and missing-code correction.

## External validation candidates

NACC is the strongest cognitive-state validation candidate because its UDS supplies longitudinal, standardized, item-documented multi-domain neurocognitive assessments across many centers. OASIS-3 is useful for longitudinal cognitive/MRI validation but offers narrower cross-instrument coverage. AIBL maps part of its clinical data to ADNI-style tables, which may simplify endpoint transport, although the overlap is incomplete. Dataset access and the instrument mapping must be finalized before freezing the external protocol.
