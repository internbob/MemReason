from pathlib import Path

import numpy as np
import pandas as pd

from memreason_rl.experiments.phase3e_test_confirmation import prepare_arrays, stratum_tables


def test_prepare_arrays_reuses_frozen_normalization(tmp_path: Path):
    phase3b = tmp_path / "phase3b"
    phase3b.mkdir()
    (phase3b / "normalization.json").write_text(
        '{"target_mean":{"score":10.0},"target_std":{"score":2.0},'
        '"age_mean":70.0,"age_std":5.0}'
    )
    frame = pd.DataFrame({"age": [75.0], "sex_numeric": [1.0], "score": [14.0]})
    demo, target, mask, mean, std = prepare_arrays(frame, ["score"], phase3b)
    assert demo.tolist() == [[1.0, 1.0]]
    assert target.tolist() == [[2.0]]
    assert mask.tolist() == [[True]]
    assert mean.tolist() == [10.0] and std.tolist() == [2.0]


def test_stratum_analysis_does_not_change_primary_predictions():
    frame = pd.DataFrame({
        "scan_id": ["a", "b"], "diagnosis_group": ["CN", "MCI"],
        "gap_group": ["0-30d", "0-30d"], "sex_group": ["F", "M"],
        "age_group": ["70-79", ">=80"], "availability_group": ["5", "3-4"],
    })
    rows = []
    for model, predictions in [("global_pool", [1.0, 2.0]), ("target_spatial_attention", [0.5, 1.5])]:
        for scan_id, subject, prediction in zip(["a", "b"], ["p1", "p2"], predictions):
            rows.append({
                "scan_id": scan_id, "source_scan_id": scan_id, "subject_id": subject,
                "split": "test", "model": model, "pairing": "observed", "target": "score",
                "truth": 0.0, "prediction": prediction, "seed": pd.NA,
            })
    predictions = pd.DataFrame(rows)
    config = {"statistics": {"minimum_stratum_subjects": 20, "bootstrap_replicates": 20, "bootstrap_seed": 1}}
    metric, bootstrap = stratum_tables(predictions, frame, {"score": 1.0}, config)
    assert not metric.empty
    assert bootstrap.empty
    assert len(predictions) == 4


def test_confirmation_source_contains_no_training_loop():
    source = (Path(__file__).resolve().parents[1] / "memreason_rl/experiments/phase3e_test_confirmation.py").read_text()
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert "splits/test.csv" in source
