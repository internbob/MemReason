#!/usr/bin/env python
"""Independent verification for the one-time Phase 3E test confirmation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/phase3e_frozen_spatial_grounding_test_20261014"
PHASE3C = ROOT / "outputs/phase3c_concurrent_spatial_grounding_dev_20261014"
ARMS = ["global_pool", "target_spatial_attention"]
SEEDS = [2026, 2027, 2028]


def verify_ensemble(frame: pd.DataFrame, model: str, pairing: str) -> None:
    seeded = frame[frame.model.eq(model) & frame.pairing.eq(pairing) & frame.seed.notna()]
    stored = frame[frame.model.eq(model) & frame.pairing.eq(pairing) & frame.seed.isna()]
    assert set(seeded.seed.astype(int)) == set(SEEDS)
    keys = ["scan_id", "source_scan_id", "subject_id", "split", "model", "pairing", "target", "truth"]
    recalculated = seeded.groupby(keys, as_index=False).prediction.mean().sort_values(keys).reset_index(drop=True)
    stored = stored[keys + ["prediction"]].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(recalculated, stored, rtol=1e-7, atol=1e-7)


def main() -> None:
    decision = json.loads((RUN / "test_decision.json").read_text())
    run = json.loads((RUN / "run_manifest.json").read_text())
    assert decision["test_evaluated"] is True and run["test_evaluated"] is True
    assert decision["training_performed"] is False and run["training_performed"] is False
    source = (ROOT / "memreason_rl/experiments/phase3e_test_confirmation.py").read_text()
    assert "torch.optim" not in source and ".backward(" not in source

    manifest = pd.read_csv(RUN / "test_manifest.csv")
    assert len(manifest) == manifest.scan_id.nunique() == 1905
    assert manifest.subject_id.nunique() == 404 and set(manifest.split) == {"test"}
    registered_test = set(pd.read_csv(ROOT / "splits/test.csv", usecols=["subject_id"]).subject_id.astype(str))
    registered_train = set(pd.read_csv(ROOT / "splits/train.csv", usecols=["subject_id"]).subject_id.astype(str))
    registered_val = set(pd.read_csv(ROOT / "splits/val.csv", usecols=["subject_id"]).subject_id.astype(str))
    observed = set(manifest.subject_id.astype(str))
    assert observed <= registered_test and not observed & (registered_train | registered_val)
    assert manifest[["image_path", "mask_path", "scan_date"]].notna().all().all()
    assert manifest.absolute_match_days.between(0, 90).all()

    volume = np.load(RUN / "volumes_48x56x48.npy", mmap_mode="r")
    assert volume.shape == (1905, 48, 56, 48)
    assert np.isfinite(np.asarray(volume[np.linspace(0, 1904, 20, dtype=int)])).all()
    for seed in SEEDS:
        features = np.load(RUN / f"spatial_features_seed{seed}.npy", mmap_mode="r")
        assert features.shape == (1905, 36, 64)
        assert np.isfinite(np.asarray(features[np.linspace(0, 1904, 20, dtype=int)])).all()

    for arm in ARMS:
        for seed in SEEDS:
            source_checkpoint = torch.load(
                PHASE3C / "checkpoints" / arm / str(seed) / "best.pt",
                map_location="cpu", weights_only=False,
            )
            assert source_checkpoint["arm"] == arm and int(source_checkpoint["seed"]) == seed
            assert int(source_checkpoint["trainable_parameters"]) == int(run["parameter_counts"][arm])

    predictions = pd.read_csv(RUN / "test_predictions.csv")
    pairings = {
        "global_pool": ["observed", "shuffled_subject"],
        "target_spatial_attention": ["observed", "shuffled_subject", "spatial_permutation", "query_permutation"],
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

    attention = np.load(RUN / "attention_weights.npz", allow_pickle=False)
    assert attention["weights"].shape == (3, 1905, 5, 36)
    assert np.isfinite(attention["weights"]).all()
    assert np.allclose(attention["weights"].sum(-1), 1.0, atol=1e-5)
    stability = pd.read_csv(RUN / "attention_stability.csv")
    assert len(stability) == 3 * 1905 * 5
    summary = decision["attention_stability"]
    assert np.isclose(stability.spearman.median(), summary["median_spearman"])
    assert np.isclose(stability.top_token_overlap.median(), summary["median_top_token_overlap"])

    bootstrap = pd.read_csv(RUN / "bootstrap_comparisons.csv")
    overall = bootstrap[bootstrap.target.eq("overall")].set_index(["baseline", "baseline_pairing"])
    versus_global = bootstrap[
        bootstrap.baseline.eq("global_pool") & bootstrap.baseline_pairing.eq("observed")
    ]
    recomputed = {
        "attention_better_than_global_pool": bool(overall.loc[("global_pool", "observed"), "ci_upper"] < 0),
        "attention_better_than_subject_shuffle": bool(overall.loc[("target_spatial_attention", "shuffled_subject"), "ci_upper"] < 0),
        "attention_better_than_spatial_permutation": bool(overall.loc[("target_spatial_attention", "spatial_permutation"), "ci_upper"] < 0),
        "attention_better_than_query_permutation": bool(overall.loc[("target_spatial_attention", "query_permutation"), "ci_upper"] < 0),
        "minimum_significant_endpoints": int(((versus_global.target.ne("overall")) & versus_global.ci_upper.lt(0)).sum()) >= 2,
        "attention_spearman_stable": summary["median_spearman"] >= 0.30,
        "top_token_overlap_stable": summary["median_top_token_overlap"] >= 0.40,
    }
    assert recomputed == decision["criteria"]
    assert decision["phase3c_test_confirmed"] is False and not all(recomputed.values())

    strata = pd.read_csv(RUN / "stratum_bootstrap_vs_global.csv")
    assert set(strata.dimension) == {
        "diagnosis_group", "gap_group", "sex_group", "age_group", "availability_group"
    }
    assert (strata.subjects >= 20).all()
    result = {
        "verified": True,
        "training_performed": False,
        "test_subject_isolation_verified": True,
        "frozen_phase3c_checkpoints_verified": 6,
        "test_prediction_ensembles_recomputed": True,
        "subject_shuffle_verified": True,
        "attention_normalization_verified": True,
        "gate_recomputed": True,
        "phase3c_test_confirmed": False,
    }
    (RUN / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
