#!/usr/bin/env python
"""Independent integrity checks for Phase 3B MRI representation validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/phase3b_mri_representation_dev_20261014"
MODELS = ["demographics_only", "global_mri", "sfcn_3d"]
SEEDS = [2026, 2027, 2028]


def verify_ensemble(frame: pd.DataFrame, model: str, pairing: str) -> None:
    seeds = frame[
        frame.model.eq(model) & frame.pairing.eq(pairing) & frame.seed.notna()
    ].copy()
    combined = frame[
        frame.model.eq(model) & frame.pairing.eq(pairing) & frame.seed.isna()
    ].copy()
    assert set(seeds.seed.astype(int)) == set(SEEDS)
    keys = [
        key
        for key in [
            "sample_id", "scan_id", "source_scan_id", "subject_id", "split", "target",
            "horizon_months", "current", "truth", "future_value",
        ]
        if key in frame.columns
    ]
    recalculated = seeds.groupby(keys, as_index=False).prediction.mean().sort_values(keys).reset_index(drop=True)
    stored = combined[keys + ["prediction"]].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(recalculated, stored, rtol=1e-7, atol=1e-7)


def main() -> None:
    decision = json.loads((RUN / "development_decision.json").read_text())
    run = json.loads((RUN / "run_manifest.json").read_text())
    assert decision["test_evaluated"] is False and run["test_evaluated"] is False
    source = (ROOT / "memreason_rl/experiments/phase3b_mri_representation.py").read_text()
    assert "splits/test.csv" not in source and 'splits/test' not in source

    checkpoints = sorted((RUN / "checkpoints").glob("*/*/best.pt"))
    assert len(checkpoints) == len(MODELS) * len(SEEDS) == 9
    seen = set()
    for checkpoint in checkpoints:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        seen.add((payload["representation"], int(payload["seed"])))
        assert payload["trainable_parameters"] == run["parameter_counts"][payload["representation"]]
    assert seen == {(model, seed) for model in MODELS for seed in SEEDS}

    manifest = pd.read_csv(RUN / "concurrent_manifest.csv")
    assert set(manifest.split) == {"train", "val"}
    assert manifest.scan_id.nunique() == len(manifest) == 6464
    assert manifest.loc[manifest.split.eq("train"), "subject_id"].nunique() == 1349
    assert manifest.loc[manifest.split.eq("val"), "subject_id"].nunique() == 185
    assert len(manifest[manifest.split.eq("val")]) == 780
    volume_index = pd.read_csv(RUN / "volume_index.csv")
    assert volume_index.scan_id.astype(str).tolist() == manifest.scan_id.astype(str).tolist()
    volumes = np.load(RUN / "volumes_48x56x48.npy", mmap_mode="r")
    assert volumes.shape == (6464, 48, 56, 48)
    probe = np.linspace(0, len(volumes) - 1, 25, dtype=int)
    assert np.isfinite(np.asarray(volumes[probe])).all()

    concurrent = pd.read_csv(RUN / "concurrent_predictions.csv")
    for model in MODELS:
        verify_ensemble(concurrent, model, "observed")
        if model != "demographics_only":
            verify_ensemble(concurrent, model, "shuffled_subject")
    scan_subject = manifest.set_index("scan_id").subject_id.astype(str)
    shuffled = concurrent[concurrent.pairing.eq("shuffled_subject")]
    assert (
        shuffled.source_scan_id.map(scan_subject).astype(str).to_numpy()
        != shuffled.subject_id.astype(str).to_numpy()
    ).all()

    normalization = json.loads((RUN / "normalization.json").read_text())
    scales = normalization["target_std"]
    ensemble = concurrent[concurrent.seed.isna()]
    metrics = pd.read_csv(RUN / "concurrent_metrics.csv")
    for row in metrics.itertuples(index=False):
        group = ensemble[
            ensemble.model.eq(row.model)
            & ensemble.pairing.eq(row.pairing)
            & ensemble.target.eq(row.target)
        ]
        value = np.abs(group.prediction - group.truth).mean() / scales[row.target]
        assert np.isclose(value, row.normalized_mae)

    delta_manifest = pd.read_csv(RUN / "future_change_manifest.csv")
    scan_dates = pd.to_datetime(manifest.set_index("scan_id").scan_date)
    assert (
        pd.to_datetime(delta_manifest.target_date).to_numpy()
        > delta_manifest.scan_id.map(scan_dates).to_numpy()
    ).all()
    delta_predictions = pd.read_csv(RUN / "future_change_predictions.csv")
    verify_ensemble(delta_predictions, "delta_sfcn_3d", "observed")
    verify_ensemble(delta_predictions, "delta_sfcn_3d", "shuffled_subject")

    concurrent_boot = pd.read_csv(RUN / "concurrent_bootstrap.csv")
    delta_boot = pd.read_csv(RUN / "future_change_bootstrap.csv")
    for name in ["global_mri", "sfcn_3d"]:
        versus_demo = concurrent_boot[
            concurrent_boot.candidate.eq(name)
            & concurrent_boot.baseline.eq("demographics_only")
        ]
        versus_shuffle = concurrent_boot[
            concurrent_boot.candidate.eq(name)
            & concurrent_boot.baseline.eq(name)
            & concurrent_boot.baseline_pairing.eq("shuffled_subject")
        ]
        stored = decision["concurrent_representation_decisions"][name]
        assert stored["overall_better_than_demographics"] == bool(
            versus_demo.loc[versus_demo.target.eq("overall"), "ci_upper"].iloc[0] < 0
        )
        assert stored["overall_better_than_subject_shuffle"] == bool(
            versus_shuffle.loc[versus_shuffle.target.eq("overall"), "ci_upper"].iloc[0] < 0
        )
    for name in ["delta_global_mri", "delta_sfcn_3d"]:
        comparison = delta_boot[
            delta_boot.candidate.eq(name) & delta_boot.baseline.eq("delta_control")
        ]
        stored = decision["future_change_decisions"][name]
        assert stored["overall_better_than_control"] == bool(
            comparison.loc[comparison.horizon_months.isna(), "ci_upper"].iloc[0] < 0
        )

    assert decision["phase3b_concurrent_grounding_ready"] is True
    assert decision["phase3b_longitudinal_fusion_ready"] is False
    result = {
        "verified": True,
        "checkpoints": len(checkpoints),
        "mri_volumes": len(manifest),
        "concurrent_validation_targets": int(
            concurrent[
                concurrent.model.eq("sfcn_3d")
                & concurrent.pairing.eq("observed")
                & concurrent.seed.eq(2026)
            ].shape[0]
        ),
        "future_change_validation_targets": int(len(delta_manifest[delta_manifest.split.eq("val")])),
        "target_after_scan_verified": True,
        "subject_shuffle_verified": True,
        "concurrent_grounding_ready": True,
        "longitudinal_fusion_ready": False,
        "test_evaluated": False,
    }
    (RUN / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
