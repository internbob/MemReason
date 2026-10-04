#!/usr/bin/env python3
"""Verify the frozen Phase 1B result bundle without retraining."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "outputs" / "phase1b_semantic_fair_20261003"


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def verify_predictions(path: Path, expected_rows: int) -> dict:
    frame = pd.read_csv(path)
    assert len(frame) == expected_rows, (path, len(frame), expected_rows)
    assert frame[["truth", "prediction"]].notna().all().all()

    truth_key = ["task", "sample_id", "target", "horizon_months"]
    assert frame.groupby(truth_key, dropna=False)["truth"].nunique().max() == 1

    expected_models = {
        "numeric_only",
        "variable_name",
        "task_semantics",
        "numeric_only ensemble",
        "variable_name ensemble",
        "task_semantics ensemble",
    }
    assert set(frame.model.unique()) == expected_models
    raw = frame[~frame.model.str.endswith(" ensemble")]
    assert set(raw.seed.astype(int).unique()) == {2026, 2027, 2028}
    assert raw.groupby(truth_key, dropna=False).size().eq(9).all()
    ensemble = frame[frame.model.str.endswith(" ensemble")]
    assert ensemble.seed.isna().all()
    assert ensemble.groupby(truth_key, dropna=False).size().eq(3).all()
    return {
        "rows": len(frame),
        "unique_targets": int(frame.groupby(truth_key, dropna=False).ngroups),
        "subjects": int(frame.subject_id.nunique()),
    }


def main(run_dir: Path) -> None:
    manifest = load_json(run_dir / "manifest.json")
    status = load_json(run_dir / "status.json")
    gate = load_json(run_dir / "gate_decision.json")

    assert status["completed"] and status["test_evaluated"]
    assert manifest["dictionary_verified"] and manifest["test_evaluated"]
    assert manifest["deterministic_algorithms"]
    assert len(manifest["features"]) == 68
    assert set(manifest["parameter_counts"].values()) == {269325}
    assert not gate["semantic_value_gate_passed"]
    assert not gate["advance_semantic_claim"]

    checkpoints = sorted((run_dir / "checkpoints").glob("**/best.pt"))
    curves = sorted((run_dir / "checkpoints").glob("**/curves.csv"))
    assert len(checkpoints) == len(curves) == 18

    validation = pd.read_csv(ROOT / "reports/data_audit/adni_official_dictionary_validation.csv")
    unmatched = pd.read_csv(ROOT / "reports/data_audit/adni_official_dictionary_unmatched.csv")
    assert len(validation) == 107
    assert validation.matched_official_rows.gt(0).all()
    assert int(validation.verified_from_dictionary.sum()) == 99
    assert unmatched.empty

    bootstrap = pd.read_csv(run_dir / "paired_subject_bootstrap.csv")
    assert len(bootstrap) == 6
    assert bootstrap.replicates.eq(2000).all()
    assert bootstrap[["normalized_mae_difference", "ci_lower", "ci_upper"]].notna().all().all()

    log_text = (run_dir / "run.log").read_text(errors="replace")
    assert "UserWarning" not in log_text
    result = {
        "run_id": manifest["run_id"],
        "dictionary_fields_matched": len(validation),
        "dictionary_fields_verified": int(validation.verified_from_dictionary.sum()),
        "features": len(manifest["features"]),
        "checkpoints": len(checkpoints),
        "validation_predictions": verify_predictions(
            run_dir / "validation_predictions.csv", 130296
        ),
        "test_predictions": verify_predictions(run_dir / "test_predictions.csv", 281640),
        "bootstrap_replicates_per_contrast": 2000,
        "semantic_value_gate_passed": gate["semantic_value_gate_passed"],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    main(parser.parse_args().run_dir.resolve())
