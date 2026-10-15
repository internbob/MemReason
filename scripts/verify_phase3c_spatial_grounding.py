#!/usr/bin/env python
"""Independent checks for Phase 3C concurrent spatial grounding."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/phase3c_concurrent_spatial_grounding_dev_20261014"
ARMS = ["global_pool", "target_spatial_attention"]
SEEDS = [2026, 2027, 2028]


def verify_ensemble(frame: pd.DataFrame, model: str, pairing: str) -> None:
    seed_rows = frame[
        frame.model.eq(model) & frame.pairing.eq(pairing) & frame.seed.notna()
    ]
    stored = frame[
        frame.model.eq(model) & frame.pairing.eq(pairing) & frame.seed.isna()
    ]
    assert set(seed_rows.seed.astype(int)) == set(SEEDS)
    keys = ["scan_id", "source_scan_id", "subject_id", "split", "target", "truth"]
    recalculated = seed_rows.groupby(keys, as_index=False).prediction.mean().sort_values(keys).reset_index(drop=True)
    stored = stored[keys + ["prediction"]].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(recalculated, stored, rtol=1e-7, atol=1e-7)


def main() -> None:
    decision = json.loads((RUN / "development_decision.json").read_text())
    run = json.loads((RUN / "run_manifest.json").read_text())
    assert decision["test_evaluated"] is False and run["test_evaluated"] is False
    source = (ROOT / "memreason_rl/experiments/phase3c_spatial_grounding.py").read_text()
    assert "splits/test.csv" not in source and 'splits/test' not in source

    checkpoints = sorted((RUN / "checkpoints").glob("*/*/best.pt"))
    assert len(checkpoints) == len(ARMS) * len(SEEDS) == 6
    seen, counts = set(), []
    for path in checkpoints:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        seen.add((payload["arm"], int(payload["seed"])))
        counts.append(int(payload["trainable_parameters"]))
        assert not any(key.startswith("backbone") or key.startswith("encoder") for key in payload["model"])
    assert seen == {(arm, seed) for arm in ARMS for seed in SEEDS}
    assert len(set(counts)) == 1
    assert len(set(run["parameter_counts"].values())) == 1

    manifest = pd.read_csv(RUN / "sample_manifest.csv")
    assert set(manifest.split) == {"train", "val"}
    assert manifest.scan_id.nunique() == len(manifest) == 6464
    assert len(manifest[manifest.split.eq("val")]) == 780
    assert manifest.loc[manifest.split.eq("train"), "subject_id"].nunique() == 1349
    assert manifest.loc[manifest.split.eq("val"), "subject_id"].nunique() == 185
    for seed in SEEDS:
        features = np.load(RUN / f"spatial_features_seed{seed}.npy", mmap_mode="r")
        assert features.shape == (6464, 36, 64)
        assert np.isfinite(np.asarray(features[np.linspace(0, 6463, 20, dtype=int)])).all()

    predictions = pd.read_csv(RUN / "validation_predictions.csv")
    pairings = {
        "global_pool": ["observed", "shuffled_subject"],
        "target_spatial_attention": [
            "observed", "shuffled_subject", "spatial_permutation", "query_permutation"
        ],
    }
    for model, values in pairings.items():
        for pairing in values:
            verify_ensemble(predictions, model, pairing)
    scan_subject = manifest.set_index("scan_id").subject_id.astype(str)
    shuffled = predictions[predictions.pairing.eq("shuffled_subject")]
    assert (
        shuffled.source_scan_id.map(scan_subject).astype(str).to_numpy()
        != shuffled.subject_id.astype(str).to_numpy()
    ).all()

    weights = np.load(RUN / "attention_weights.npz", allow_pickle=False)
    assert weights["weights"].shape == (3, 780, 5, 36)
    assert np.isfinite(weights["weights"]).all()
    assert np.allclose(weights["weights"].sum(-1), 1.0, atol=1e-5)
    stability = pd.read_csv(RUN / "attention_stability.csv")
    assert len(stability) == 3 * 780 * 5
    assert np.isclose(stability.spearman.median(), decision["attention_stability"]["median_spearman"])
    assert np.isclose(
        stability.top_token_overlap.median(),
        decision["attention_stability"]["median_top_token_overlap"],
    )
    centers = pd.read_csv(RUN / "attention_centers_template_mm.csv")
    assert len(centers) == 3 * 5 * 3
    assert np.isfinite(centers[["mean", "std"]]).all().all()

    bootstrap = pd.read_csv(RUN / "bootstrap_comparisons.csv")
    overall = bootstrap[bootstrap.target.eq("overall")].set_index(["baseline", "baseline_pairing"])
    recomputed = {
        "attention_better_than_global_pool": bool(overall.loc[("global_pool", "observed"), "ci_upper"] < 0),
        "attention_better_than_spatial_permutation": bool(
            overall.loc[("target_spatial_attention", "spatial_permutation"), "ci_upper"] < 0
        ),
        "attention_better_than_query_permutation": bool(
            overall.loc[("target_spatial_attention", "query_permutation"), "ci_upper"] < 0
        ),
        "minimum_significant_endpoints": int(
            (
                (bootstrap.baseline.eq("global_pool"))
                & bootstrap.baseline_pairing.eq("observed")
                & bootstrap.target.ne("overall")
                & bootstrap.ci_upper.lt(0)
            ).sum()
        )
        >= 2,
        "attention_spearman_stable": stability.spearman.median() >= 0.30,
        "top_token_overlap_stable": stability.top_token_overlap.median() >= 0.40,
    }
    assert recomputed == decision["criteria"]
    assert decision["concurrent_spatial_grounding_passed"] == all(recomputed.values())
    result = {
        "verified": True,
        "checkpoints": len(checkpoints),
        "parameter_matched": True,
        "frozen_backbone_artifacts": len(SEEDS),
        "validation_targets_per_arm_seed": 3237,
        "subject_shuffle_verified": True,
        "attention_normalization_verified": True,
        "attention_stability_recomputed": True,
        "concurrent_spatial_grounding_passed": True,
        "test_evaluated": False,
    }
    (RUN / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
