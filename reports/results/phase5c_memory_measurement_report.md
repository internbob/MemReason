# Phase 5C Stage 1: Interpretable Memory-State Measurement

## Scope

Five-fold nested subject-wise development used only the original ADNI train split. The experiment evaluated 70 frozen state items: the prior 68 verified cognitive items plus audited Logical Memory immediate and delayed recall. AVLT intrusion fields were excluded from state input. Original validation and test data were not loaded.

## Results

The probabilistic domain model achieved held-out-item macro normalized MAE **0.6782** and held-out-instrument macro normalized MAE **0.5507**. Its corresponding Memory-item values were **0.7969** and **0.5872**. Cross-view state Spearman was **-0.0776** over 43095 paired state dimensions.

The prespecified held-out-instrument improvement condition was **false** and the cross-view coherence condition was **false**. Therefore `stage1_measurement_gate_passed=false`. This is development evidence only and does not reopen the consumed ADNI validation or test cohorts.

## Interpretation

The gate requires both predictive reconstruction and state coherence; a constrained state is not accepted solely because its axes are named. Full Phase 5C additionally requires continuous-time forecasting, calibration, downstream validity, reasoning faithfulness, and external confirmation, so `phase5c_complete=false` regardless of this Stage 1 outcome.
