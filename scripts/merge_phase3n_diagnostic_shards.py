#!/usr/bin/env python3
"""Merge disjoint diagnostic fold shards and run the frozen summary."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memreason_rl.experiments.phase3g_mri_diagnosis import load_config, summarize


CONFIG = ROOT / "configs/phase3n_corrected_cropped_mri_diagnosis_crossfit.yaml"
MAIN = ROOT / "outputs/phase3n_corrected_cropped_mri_diagnosis_crossfit_20261018"
SHARD = ROOT / "outputs/phase3n_corrected_cropped_mri_diagnosis_crossfit_shard_b_20261018"


def canonical(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame[columns].astype(str).sort_values(columns).reset_index(drop=True)


def main() -> None:
    manifests = [pd.read_csv(path / "diagnostic_manifest.csv", low_memory=False) for path in [MAIN, SHARD]]
    manifest_columns = ["scan_id", "subject_id", "label", "fold", "volume_index"]
    pd.testing.assert_frame_equal(
        canonical(manifests[0], manifest_columns), canonical(manifests[1], manifest_columns)
    )
    folds = [pd.read_csv(path / "subject_folds.csv") for path in [MAIN, SHARD]]
    pd.testing.assert_frame_equal(
        canonical(folds[0], ["subject_id", "fold"]),
        canonical(folds[1], ["subject_id", "fold"]),
    )

    seed_blocks = [pd.read_csv(path / "crossfit_seed_predictions.csv") for path in [MAIN, SHARD]]
    if set(seed_blocks[0].fold.unique()) != {0, 1, 2}:
        raise RuntimeError("Main diagnostic shard does not contain folds 0, 1, and 2")
    if set(seed_blocks[1].fold.unique()) != {3, 4}:
        raise RuntimeError("Second diagnostic shard does not contain folds 3 and 4")
    seed_predictions = pd.concat(seed_blocks, ignore_index=True)
    identity = ["row_index", "arm", "fold", "seed"]
    if seed_predictions.duplicated(identity).any():
        raise RuntimeError("Duplicate diagnostic prediction after shard merge")
    seed_predictions.sort_values(identity).to_csv(MAIN / "crossfit_seed_predictions.csv", index=False)

    ensembles = [pd.read_csv(path / "crossfit_predictions.csv") for path in [MAIN, SHARD]]
    ensemble = pd.concat(ensembles, ignore_index=True)
    ensemble_identity = ["row_index", "arm", "fold"]
    if ensemble.duplicated(ensemble_identity).any():
        raise RuntimeError("Duplicate ensemble prediction after shard merge")
    if ensemble.row_index.nunique() != len(manifests[0]):
        raise RuntimeError("Merged prediction cohort is incomplete")
    ensemble.sort_values(ensemble_identity).to_csv(MAIN / "crossfit_predictions.csv", index=False)

    curves = pd.concat(
        [pd.read_csv(path / "sfcn_training_curves.csv") for path in [MAIN, SHARD]],
        ignore_index=True,
    )
    curves.to_csv(MAIN / "sfcn_training_curves.csv", index=False)
    for source in (SHARD / "checkpoints").glob("fold*/seed*.pt"):
        destination = MAIN / "checkpoints" / source.parent.name / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    checkpoints = list((MAIN / "checkpoints").glob("fold*/seed*.pt"))
    if len(checkpoints) != 15:
        raise RuntimeError(f"Expected 15 diagnostic checkpoints, found {len(checkpoints)}")

    config = load_config(CONFIG)
    summarize(config, MAIN)
    audit = {
        "passed": True,
        "folds": [0, 1, 2, 3, 4],
        "neural_seeds": config["neural_seeds"],
        "checkpoints": len(checkpoints),
        "scans": int(len(manifests[0])),
        "subjects": int(manifests[0].subject_id.nunique()),
    }
    (MAIN / "shard_merge_verification.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
