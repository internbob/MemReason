# Stage I representation development terminal decision

Date: 2026-10-09

## Decision

Stop semantic architecture tuning on the current train/validation split. Do not evaluate Phase 1D/1E/1F on test and do not open MRI Stage II under the original Gate 1.

## Evidence

- Phase 1C formal test: frozen response-text semantics failed both masked criteria, the two-pair cross-scale criterion and future non-inferiority.
- Phase 1D validation: typed item-conditioned encoding improved the equal-task score and ADAS+CDR to MoCA, but did not satisfy the joint gate.
- Phase 1E validation: metric-aligned per-item loss produced significant masked-item improvement, but masked-instrument and cross-scale criteria remained unmet.
- Phase 1F validation: task-specific gates improved the overall semantic-arm score from 0.38826 to 0.38334 and significantly improved masked item. Masked-instrument had a favorable but inconclusive difference of -0.00284 (95% CI [-0.01127, 0.00453]); future had -0.00447 (95% CI [-0.01457, 0.00623]), narrowly missing the +0.005 non-inferiority bound. Both cross-scale confidence intervals crossed zero and their point estimates were unfavorable.

## Scientific interpretation

Typed question-conditioned representations can improve selected partial-observation tasks, especially masked-item reconstruction, but the available evidence does not support a general claim that frozen Bio_ClinicalBERT task semantics improve all required cross-instrument and longitudinal objectives. Repeated tuning on the same validation subjects would overfit the gate.

## Allowed next routes

1. Reframe the method so typed item identity, response type and partial-observation dynamics are components without claiming universal semantic superiority. Continue cognitive-only irregular-time belief/calibration experiments under a new protocol; MRI remains conditional on its own data readiness and a revised scientific gate.
2. Obtain a new development/confirmation cohort or a prospectively locked phase/site cohort before further semantic architecture search.

The existing ADNI test results remain internal re-evaluation evidence and cannot become a clean external confirmation set.
