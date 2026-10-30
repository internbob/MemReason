# Public aggregate results

This directory contains aggregate, de-identified artifacts exported from the
git-ignored experiment output directories. It intentionally excludes ADNI
participant rows, prompts, generations, scan paths, model checkpoints, and
subject-level predictions.

The currently supported primary system is **GRU-D cognitive forecasting plus
grammar-constrained structured reporting**. Its one-time sealed-test result is
reported in `phase4f_frozen_test_report.md`. The forecast criteria passed, but
the complete structured-reasoning gate narrowly failed because nested temporal
consistency was 0.98990 versus the frozen 0.99000 threshold.

Selected later results are also exported here as compact aggregate artifacts:

- `phase3i_*` contains the native-resolution AD-vs-CN MRI classification result.
- `phase3m_*` contains the matched MRI preprocessing sensitivity analysis.
- `phase3n_*` contains the corrected center-cropped three-class MRI result.
- `phase5e_*` contains the frozen reasoning-faithfulness and renderer ablation.
- `phase5f_*` contains the train-only hierarchical shared-state, instrument-residual,
  and bridge-graph measurement development result. Predictive gates passed, but
  the frozen cross-view CCC threshold did not, so dynamics remain closed.
- `phase5g_*` contains the bridge-strength Pareto scan and residual-adaptive gate
  study. The selected development point improves the prediction/coherence trade-off,
  but does not reach the frozen measurement threshold.
- `phase5h_*` contains the train-only measurement-identifiability audit. The
  attenuation ceiling supports a 0.70 CCC for only four of six instrument
  pairs, so the universal cross-view threshold and measurement gate remain closed.
- `phase6a_*` contains the explicitly exploratory train-only latent-state dynamics
  study. The reference+MRI filter improves latent MAE over latent LOCF, but its
  95% interval coverage fails calibration; it does not reopen the measurement gate.
- `phase6b_*` contains the train-only MRI incremental-value audit. Real MRI
  improves the observed Memory endpoint over cognition-only, acquisition metadata,
  and subject-shuffled controls, but not over within-subject time shuffling; the
  multimodal disagreement-aware filter therefore remains closed.
- `phase6c_*` contains the actual-date longitudinal MRI identifiability audit on
  participants with at least two pre-cutoff scans spanning six months. Correct
  dynamics do not improve over the same latest static MRI state or subject-shuffled
  dynamics, so MRI is frozen as a static auxiliary covariate for this representation.
- `phase6d0_*` contains the fold-aligned native-resolution SFCN extraction audit.
  Five separately trained encoders produce 128-dimensional embeddings for 4,616
  original-train scans, with zero overlap between each encoder's fit subjects and
  its Phase 6 evaluation fold. Embedding coordinates are never averaged across encoders.
- `phase6d1_*` contains the static 128-dimensional embedding incremental-value audit.
  The embedding arm improves the observed Memory endpoint over cognition-only and
  subject-shuffled embeddings, but not reproducibly over its own encoder's AD
  probability. The prespecified gate is therefore false, so Phase 6D2 and Phase 6E
  were not opened.

These exports contain reports, aggregate metrics, decisions, and verification
records only. Model weights, MRI arrays, caches, participant-level predictions,
and source datasets remain outside Git.

Run the following command from the repository root after a new experiment:

```bash
bash scripts/git_update.sh "Describe the update"
```

The script refreshes these artifacts using an explicit allowlist, runs the test
suite, scans the staged files, creates a commit, and pushes the current branch.
