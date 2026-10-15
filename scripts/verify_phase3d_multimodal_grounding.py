#!/usr/bin/env python
"""Independent artifact checks for Phase 3D multimodal item grounding."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/phase3d_multimodal_item_grounding_dev_20261014"
ARMS = ["cognition_only", "global_mri", "item_spatial_attention"]
SEEDS = [2026, 2027, 2028]


def verify_ensemble(frame: pd.DataFrame, model: str, pairing: str) -> None:
    seeded = frame[frame.model.eq(model) & frame.pairing.eq(pairing) & frame.seed.notna()]
    stored = frame[frame.model.eq(model) & frame.pairing.eq(pairing) & frame.seed.isna()]
    assert set(seeded.seed.astype(int)) == set(SEEDS)
    keys = [
        "sample_id", "scan_id", "source_scan_id", "subject_id", "split", "model", "pairing",
        "task", "instrument", "target", "truth", "normalized_truth",
    ]
    recalculated = seeded.groupby(keys, as_index=False)[["prediction", "normalized_prediction"]].mean()
    recalculated = recalculated.sort_values(keys).reset_index(drop=True)
    stored = stored[keys + ["prediction", "normalized_prediction"]].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(recalculated, stored, rtol=1e-7, atol=1e-7)


def main() -> None:
    decision = json.loads((RUN / "development_decision.json").read_text())
    manifest = json.loads((RUN / "run_manifest.json").read_text())
    assert decision["test_evaluated"] is False and manifest["test_evaluated"] is False
    source = (ROOT / "memreason_rl/experiments/phase3d_multimodal_grounding.py").read_text()
    assert "splits/test.csv" not in source and 'splits/test' not in source

    checkpoints = sorted((RUN / "checkpoints").glob("*/*/best.pt"))
    assert len(checkpoints) == 9
    seen, counts = set(), []
    for path in checkpoints:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        seen.add((payload["arm"], int(payload["seed"])))
        counts.append(int(payload["trainable_parameters"]))
        assert len(payload["features"]) == 68
    assert seen == {(arm, seed) for arm in ARMS for seed in SEEDS}
    assert len(set(counts)) == 1 and len(set(manifest["parameter_counts"].values())) == 1

    anchors = pd.read_csv(RUN / "anchor_manifest.csv")
    samples = pd.read_csv(RUN / "sample_manifest.csv")
    assert len(anchors) == anchors.scan_id.nunique() == 6464
    assert anchors.split.value_counts().to_dict() == {"train": 5684, "val": 780}
    assert samples.split.value_counts().to_dict() == {"train": 22728, "val": 3120}
    assert set(samples.task) == {"masked_item", "masked_instrument"}
    assert samples.sample_id.is_unique
    assert not samples.subject_id.astype(str).isin(pd.read_csv(ROOT / "splits/test.csv").subject_id.astype(str)).any()

    predictions = pd.read_csv(RUN / "validation_predictions.csv")
    pairings = {
        "cognition_only": ["observed"],
        "global_mri": ["observed", "shuffled_subject"],
        "item_spatial_attention": ["observed", "shuffled_subject", "spatial_permutation", "query_permutation"],
    }
    for model, values in pairings.items():
        for pairing in values:
            verify_ensemble(predictions, model, pairing)
    scan_subject = anchors.set_index("scan_id").subject_id.astype(str)
    shuffled = predictions[predictions.pairing.eq("shuffled_subject")]
    assert (
        shuffled.source_scan_id.map(scan_subject).astype(str).to_numpy()
        != shuffled.subject_id.astype(str).to_numpy()
    ).all()
    assert np.isfinite(predictions[["truth", "prediction", "normalized_truth", "normalized_prediction"]]).all().all()

    attention = np.load(RUN / "attention_weights.npz", allow_pickle=False)
    assert attention["weights"].shape == (3, 14755, 36)
    assert len(np.unique(attention["keys"])) == 14755
    assert np.isfinite(attention["weights"]).all()
    assert np.allclose(attention["weights"].sum(-1), 1.0, atol=1e-5)
    stability = pd.read_csv(RUN / "attention_stability.csv")
    assert len(stability) == 3 * 14755
    summary = decision["attention_stability"]
    assert np.isclose(stability.spearman.median(), summary["median_spearman"])
    assert np.isclose(stability.top_token_overlap.median(), summary["median_top_token_overlap"])
    entropy = -(attention["weights"] * np.log(attention["weights"].clip(1e-12))).sum(-1) / np.log(36)
    assert np.isclose(entropy.mean(), summary["mean_normalized_entropy"], atol=1e-6)

    bootstrap = pd.read_csv(RUN / "bootstrap_comparisons.csv").set_index(
        ["baseline", "baseline_pairing", "task"]
    )
    recomputed = {
        "both_tasks_better_than_cognition": bool(all(
            bootstrap.loc[("cognition_only", "observed", task), "ci_upper"] < 0
            for task in ["masked_item", "masked_instrument"]
        )),
        "overall_better_than_global_mri": bool(
            bootstrap.loc[("global_mri", "observed", "overall"), "ci_upper"] < 0
        ),
        "better_than_all_negative_controls": bool(all(
            bootstrap.loc[("item_spatial_attention", pairing, "overall"), "ci_upper"] < 0
            for pairing in ["shuffled_subject", "spatial_permutation", "query_permutation"]
        )),
        "attention_spearman_stable": summary["median_spearman"] >= 0.25,
        "top_token_overlap_stable": summary["median_top_token_overlap"] >= 0.35,
        "attention_noncollapsed": summary["mean_normalized_entropy"] < 0.98,
    }
    assert recomputed == decision["criteria"]
    assert decision["multimodal_item_grounding_passed"] == all(recomputed.values()) is False
    result = {
        "verified": True,
        "checkpoints": 9,
        "parameter_matched": True,
        "masked_response_input_zeroing_tested": True,
        "validation_prediction_ensembles_recomputed": True,
        "subject_shuffle_verified": True,
        "attention_normalization_verified": True,
        "gate_recomputed": True,
        "multimodal_item_grounding_passed": False,
        "test_evaluated": False,
    }
    (RUN / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
