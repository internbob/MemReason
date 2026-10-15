# Phase 4E: Frozen trajectory-action SFT and constrained GRPO

Status: protocol frozen before action-SFT and on-policy rollout.

## Question

Can a constrained cognitive policy improve five-endpoint future forecasts when its actions modify the frozen GRU-D anchor, so that future-outcome reward varies within each GRPO candidate group?

## Cohorts and leakage controls

Phase 4A build subjects are deterministically repartitioned into mutually exclusive action-SFT, rollout, and internal-development cohorts. All Phase 4D rollout subjects are excluded. One sample is selected per subject without inspecting future outcomes. The existing Phase 4B adapter initializes action-SFT, but its training targets contained only frozen forecasts and never future observations. Future truth is absent from every prompt. It is used only as the supervised verifier for on-policy reward and evaluation. Original test data remain closed unless the complete development gate passes.

## Action space

For each of ADAS13, CDR-SB, FAQ, MMSE, and MoCA, the policy chooses one bounded residual action (`drop`, `lower`, `anchor`, `raise`, or `boost`) corresponding to -0.5, -0.25, 0, +0.25, or +0.5 training standard deviations relative to the frozen forecast. It also chooses an uncertainty multiplier (`compact`, `baseline`, or `wide`) of 0.75, 1.0, or 1.5. Adjusted forecasts are clipped to the observed training range. Evidence, current-state statistics, trends, and the report renderer remain frozen.

Action-SFT teaches only the safe anchor action (zero residual and unit uncertainty). It does not use future truth. GRPO is therefore the only stage that can learn outcome-dependent forecast changes.

## Frozen optimization

GRPO uses four candidates per prompt, temperature 1.3, clipping 0.2, KL coefficient 0.02, learning rate 1e-5, and at most 32 updates. Rewards combine normalized future accuracy (0.65), Gaussian CRPS-based uncertainty quality (0.15), future-direction agreement (0.10), and clinical-range validity (0.10). Stop rules cover nonfinite optimization, three consecutive update-level KL violations above 0.15, and eight consecutive low-variance updates.

## Evaluation and stopping rule

Checkpoints are selected only on the new internal-development subjects. GRPO must improve normalized MAE over both the frozen forecast and the matched action-SFT policy, with a subject-bootstrap 95% confidence interval below zero versus frozen. Normalized CRPS and temporal-direction accuracy must remain within 0.01 of frozen, all adjusted values must stay in clinical range, reward-variable groups must be at least 10%, KL must remain controlled, and training must finish without a safety abort. Failure stops Phase 4E without loading original test data or tuning again on these development subjects.
