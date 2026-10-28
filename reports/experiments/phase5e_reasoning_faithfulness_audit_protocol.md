# Phase 5E frozen reasoning-faithfulness and renderer-ablation audit

## Status and scope

This is a post-test audit of the already frozen Phase 4F system. The official
ADNI test cohort has already been opened. Consequently, this analysis cannot
select checkpoints, decoders, thresholds, or repair the failed Phase 4F gate.
All interventions and summary metrics below are fixed before their outputs are
computed.

## Source cohort and frozen artifacts

- Use exactly the 495 one-per-subject reasoning examples stored by Phase 4F.
- Reconstruct the corresponding test forecasting samples without changing the
  subject split, target horizons, normalization, or three GRU-D checkpoints.
- Use the frozen Phase 4A seeds 2026, 2027, and 2028.
- Do not use future truth, diagnosis, MRI, or GRPO adapters as model inputs.

## Visit-deletion intervention

For every report with at least two observed input visits, create one candidate
for each visit by removing that complete visit from the GRU-D history and from
the evidence ledger. Run the frozen ensemble on all candidates. Define the
highest-influence visit as the candidate with the largest mean absolute change
in the five forecasts after normalization by the build-set target standard
deviations. Break ties by the latest visit and then by row index.

Select a matched random deletion from the remaining eligible visits using seed
20261016. The primary contrast is the paired subject-level difference in
normalized forecast displacement between highest-influence and random deletion.

## Additional counterfactuals

1. **Reversed time gaps:** preserve the order and values of visits but reverse
   the sequence of inter-visit gaps while holding the first and final visit
   months fixed. This tests whether forecasts respond to irregular timing.
2. **Memory score reflection:** for verified Memory inputs, replace every
   observed value `x` by `minimum + maximum - x`, using build-only observed
   bounds. This is an explicit score counterfactual, not a plausible patient
   trajectory and not an accuracy experiment.

Each perturbed evidence ledger is recompiled into the typed report. The audit
records forecast displacement, forecast-direction changes, state-label changes,
trend-label changes, evidence changes, and independent verifier scores. It does
not require a change for every patient; null responses are reported rather than
removed.

## Renderer ablation

Reuse the matched Phase 4F outputs for `raw_sft`, `grammar_model`,
`grammar_executable`, and `teacher`. Compare factual preservation and exact
report match. Because `grammar_executable` and `teacher` are identical under the
current schema, that equality is an explicit result rather than evidence for an
LLM advantage.

## Statistical analysis

- Report subject-level means and 2,000-replicate subject bootstrap intervals.
- For the primary deletion contrast, report the 95% bootstrap interval for
  highest-influence minus random forecast displacement.
- All other intervention summaries are descriptive post-test audits.

## Interpretation rule

The audit may support sensitivity, traceability, and mechanical verification.
It cannot establish a clinically valid latent cognitive state, clinical utility,
or a new confirmatory test result. If the executable renderer matches or exceeds
the LLM renderer, the paper method must identify the compiler/verifier as the
core contribution and describe the LLM as optional.
