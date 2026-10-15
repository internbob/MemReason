# Cognitive-only irregular-time probabilistic belief dynamics protocol

Date frozen: 2026-10-10. This protocol is frozen before the validation evaluation partition is scored.

## Scientific scope

This experiment treats the verified typed item/response encoder as a component. It does not test or claim universal semantic superiority. It tests whether an explicit continuous-time Gaussian predict-update state improves uncertainty-aware cognitive trajectory modelling over a strong irregular-time GRU-D Gaussian baseline.

Only cognitive scale data are used. Diagnosis, MRI, FAQ items, unverified fields, text semantics, test subjects and future observations outside each task's allowed context are excluded from inputs. The five outcomes are ADAS13, CDR-SB, FAQ, MMSE and MoCA.

## Subject-disjoint development partitions

- The existing train subjects are deterministically hash-partitioned into fit (90%) and inner tune (10%). Inner tune controls early stopping only.
- Normalization means and standard deviations are estimated from fit subjects only and are frozen for inner tune, calibration and evaluation.
- The existing validation subjects are deterministically hash-partitioned into calibration (50%) and evaluation (50%).
- Variance temperatures are fit only on validation-calibration subjects. All reported final development metrics and paired comparisons use validation-evaluation subjects.
- The existing test split is never loaded by this experiment.

## Tasks

1. Extrapolation uses all observations through a cutoff and predicts real observations nearest 6, 12, 24 or 36 months within the frozen tolerance windows.
2. Interpolation requires at least three visits. An entire middle visit is hidden; earlier and later visits may be used to reconstruct the five observed outcomes at the hidden month. The hidden visit is absent from model inputs.

## Arms

1. `grud_gaussian`: numeric values, missingness masks and exact irregular intervals; direct heteroscedastic Gaussian output.
2. `typed_direct_gaussian`: adds verified item identity, response type, direction-aware severity and domain aggregation, while retaining a direct Gaussian output.
3. `typed_belief_dynamics`: uses the same typed component and an explicit diagonal-Gaussian latent state. Between events it applies a learned non-negative-time drift/diffusion prediction; at observed events it applies gated cognitive mean/variance updates. Interpolation fuses forward and reverse beliefs by diagonal Gaussian product of experts.

All neural arms use identical samples, targets, optimizer budget and three fixed seeds. Parameter counts are reported rather than asserted equal because explicit belief uncertainty is itself the tested mechanism. Training minimizes normalized Gaussian NLL plus a small normalized Huber term and balances interpolation and forecast-horizon groups.

## Calibration and outcomes

The three-seed ensemble uses total predictive variance (within-model variance plus between-seed mean variance). A positive multiplicative standard-deviation temperature is fit for each task/target/horizon group when at least 20 calibration targets exist; sparse groups fall back to task/target and then global estimates. Temperatures are clipped to [0.25, 4.0].

Reported outcomes are normalized MAE, Gaussian NLL in training-normalized outcome space, Gaussian CRPS, 50/80/90/95% coverage and width, multilevel calibration error, and PIT diagnostics. Results are also stratified by history length and observed-feature fraction. Subject-cluster bootstrap uses 2,000 replicates.

## Development decision

`typed_belief_dynamics` advances as the cognitive dynamics component only if, after calibration on disjoint subjects:

- its paired normalized-MAE difference versus `grud_gaussian` has a 95% CI upper bound no larger than +0.005;
- its paired Gaussian-NLL difference has a 95% CI upper bound below zero;
- its multilevel absolute calibration error is no greater than 0.05; and
- empirical 95% coverage lies in [0.90, 0.98].

The decision concerns cognitive-only dynamics. It does not open MRI experiments, establish external confirmation, or restore a universal semantic-superiority claim.
