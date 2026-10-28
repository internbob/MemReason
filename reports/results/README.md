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

These exports contain reports, aggregate metrics, decisions, and verification
records only. Model weights, MRI arrays, caches, participant-level predictions,
and source datasets remain outside Git.

Run the following command from the repository root after a new experiment:

```bash
bash scripts/git_update.sh "Describe the update"
```

The script refreshes these artifacts using an explicit allowlist, runs the test
suite, scans the staged files, creates a commit, and pushes the current branch.
