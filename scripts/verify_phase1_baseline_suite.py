#!/usr/bin/env python3
"""Verify the frozen formal baseline-suite artifact bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "outputs" / "phase1_baseline_suite_formal_20261004"


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def verify_predictions(path: Path, expected_rows: int) -> dict:
    frame = pd.read_csv(path)
    assert len(frame) == expected_rows
    assert frame.truth.notna().all()
    assert frame.loc[~frame.model.eq("LOCF"), "prediction"].notna().all()
    key = ["sample_id", "subject_id", "target", "horizon_months"]
    assert frame.groupby(key, dropna=False).truth.nunique().max() == 1
    expected_models = {
        "LOCF", "LME", "GRU", "LSTM", "T-LSTM", "GRU-D", "Transformer",
        "GRU ensemble", "LSTM ensemble", "T-LSTM ensemble", "GRU-D ensemble",
        "Transformer ensemble",
    }
    assert set(frame.model.unique()) == expected_models
    raw = frame[frame.model.isin({"GRU", "LSTM", "T-LSTM", "GRU-D", "Transformer"})]
    assert set(raw.seed.astype(int).unique()) == {2026, 2027, 2028}
    assert raw.groupby(key, dropna=False).size().eq(15).all()
    ensemble = frame[frame.model.str.endswith(" ensemble")]
    assert ensemble.seed.isna().all()
    assert ensemble.groupby(key, dropna=False).size().eq(5).all()
    return {
        "rows": len(frame),
        "unique_targets": int(frame.groupby(key, dropna=False).ngroups),
        "subjects": int(frame.subject_id.nunique()),
    }


def main(run_dir: Path) -> None:
    manifest = load_json(run_dir / "manifest.json")
    status = load_json(run_dir / "status.json")
    selection = load_json(run_dir / "selection.json")
    assert status["completed"] and status["test_evaluated"]
    assert manifest["dictionary_verified"] and manifest["test_evaluated"]
    assert not manifest["diagnosis_input"]
    assert len(manifest["features"]) == 68
    assert manifest["architectures"] == ["GRU", "LSTM", "T-LSTM", "GRU-D", "Transformer"]
    assert selection["selected_architecture"] == "GRU-D"
    assert len(list((run_dir / "checkpoints").glob("**/best.pt"))) == 15
    assert len(list((run_dir / "checkpoints").glob("**/curves.csv"))) == 15

    bootstrap = pd.read_csv(run_dir / "paired_subject_bootstrap.csv")
    assert len(bootstrap) == 10
    assert bootstrap.replicates.eq(2000).all()
    assert bootstrap[["normalized_mae_difference", "ci_lower", "ci_upper"]].notna().all().all()

    result = {
        "run_id": manifest["run_id"],
        "features": len(manifest["features"]),
        "checkpoints": 15,
        "validation_predictions": verify_predictions(run_dir / "validation_predictions.csv", 202224),
        "test_predictions": verify_predictions(run_dir / "test_predictions.csv", 440154),
        "selected_architecture": selection["selected_architecture"],
        "bootstrap_contrasts": len(bootstrap),
        "bootstrap_replicates_per_contrast": 2000,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    main(parser.parse_args().run_dir.resolve())
