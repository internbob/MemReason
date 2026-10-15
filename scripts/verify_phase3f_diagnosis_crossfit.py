"""Independent artifact checks for Phase 3F diagnosis cross-fitting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/phase3f_diagnosis_heterogeneity_crossfit.yaml"


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    output = ROOT / "outputs" / config["run_id"]
    manifest = pd.read_csv(output / "sample_manifest.csv")
    folds = pd.read_csv(output / "subject_folds.csv")
    predictions = pd.read_csv(output / "crossfit_predictions.csv")
    bootstrap = pd.read_csv(output / "bootstrap_comparisons.csv")
    fold_effects = pd.read_csv(output / "fold_diagnosis_effects.csv")
    interaction = json.loads((output / "diagnosis_interaction.json").read_text())
    decision = json.loads((output / "development_decision.json").read_text())

    train_subjects = set(pd.read_csv(ROOT / "splits/train.csv", usecols=["subject_id"]).subject_id.astype(str))
    val_subjects = set(pd.read_csv(ROOT / "splits/val.csv", usecols=["subject_id"]).subject_id.astype(str))
    test_subjects = set(pd.read_csv(ROOT / "splits/test.csv", usecols=["subject_id"]).subject_id.astype(str))
    used = set(manifest.subject_id.astype(str))
    assert used <= train_subjects
    assert not used & (val_subjects | test_subjects)
    assert manifest.scan_id.is_unique and len(manifest) == 5684
    assert manifest.subject_id.nunique() == 1349
    assert set(manifest.fold) == set(range(int(config["folds"])))
    assert folds.subject_id.is_unique
    assert manifest.groupby("subject_id").fold.nunique().max() == 1

    expected_checkpoints = len(config["arms"]) * len(config["seeds"]) * int(config["folds"])
    checkpoints = list((output / "checkpoints").glob("*/*/fold*/fixed_epoch.pt"))
    assert len(checkpoints) == expected_checkpoints == 30
    counts = {}
    for path in checkpoints:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        arm, seed, fold = checkpoint["arm"], int(checkpoint["seed"]), int(checkpoint["fold"])
        assert int(checkpoint["epochs"]) == int(config["training"]["frozen_epochs"][arm][seed])
        assert len(pd.read_csv(path.parent / "curves.csv")) == int(checkpoint["epochs"])
        counts[arm] = int(checkpoint["trainable_parameters"])
    assert counts == {"global_pool": 76673, "target_spatial_attention": 76673}

    seeds = predictions[predictions.seed.notna()].copy()
    ensembles = predictions[predictions.seed.isna()].copy()
    expected_seed_pairings = {
        ("global_pool", "observed"), ("global_pool", "shuffled_subject"),
        ("target_spatial_attention", "observed"),
        ("target_spatial_attention", "shuffled_subject"),
        ("target_spatial_attention", "spatial_permutation"),
        ("target_spatial_attention", "query_permutation"),
    }
    assert set(map(tuple, seeds[["model", "pairing"]].drop_duplicates().to_numpy())) == expected_seed_pairings
    assert seeds.seed.nunique() == 3
    key = ["scan_id", "subject_id", "model", "pairing", "target", "truth", "seed"]
    assert not seeds.duplicated(key).any()
    assert ensembles.scan_id.isin(manifest.scan_id).all()
    assert ensembles.fold.notna().all()

    weights = np.load(output / "attention_weights.npz", allow_pickle=False)
    assert weights["weights"].shape == (3, 5684, 5, 36)
    assert weights["scan_ids"].astype(str).tolist() == manifest.scan_id.astype(str).tolist()
    np.testing.assert_allclose(weights["weights"].sum(-1), 1.0, atol=2e-5)

    scales_json = json.loads((ROOT / "outputs" / config["sources"]["phase3b_run"] / "normalization.json").read_text())
    scales = scales_json["target_std"]
    candidate = ensembles[(ensembles.model == "target_spatial_attention") & (ensembles.pairing == "observed")]
    baseline = ensembles[(ensembles.model == "global_pool") & (ensembles.pairing == "observed")]
    keys = ["scan_id", "subject_id", "target", "truth"]
    paired = candidate[keys + ["prediction"]].merge(
        baseline[keys + ["prediction"]], on=keys, suffixes=("_candidate", "_baseline"), validate="one_to_one"
    )
    paired["difference"] = (
        (paired.prediction_candidate - paired.truth).abs()
        - (paired.prediction_baseline - paired.truth).abs()
    ) / paired.target.map(scales)
    diagnosis = manifest.set_index("scan_id").diagnosis_group
    paired["diagnosis_group"] = paired.scan_id.map(diagnosis)
    for group in ["overall", "CN", "MCI", "AD"]:
        expected = paired.difference.mean() if group == "overall" else paired.loc[paired.diagnosis_group == group, "difference"].mean()
        actual = bootstrap.loc[
            (bootstrap.diagnosis_group == group) & (bootstrap.baseline == "global_pool"),
            "normalized_mae_difference",
        ].iloc[0]
        assert np.isclose(expected, actual)
    expected_interaction = (
        paired.loc[paired.diagnosis_group == "AD", "difference"].mean()
        - paired.loc[paired.diagnosis_group == "MCI", "difference"].mean()
    )
    assert np.isclose(expected_interaction, interaction["interaction"])
    assert int((fold_effects.loc[fold_effects.diagnosis_group == "AD", "normalized_mae_difference"] < 0).sum()) == decision["ad_direction_folds"]
    assert int((fold_effects.loc[fold_effects.diagnosis_group == "MCI", "normalized_mae_difference"] > 0).sum()) == decision["mci_direction_folds"]
    assert decision["original_train_only"] and not decision["validation_evaluated"] and not decision["test_evaluated"]
    assert decision["diagnosis_heterogeneity_replicated"] == all(decision["criteria"].values())
    result = {"status": "passed", "checkpoints": len(checkpoints),
              "subjects": len(used), "scans": len(manifest),
              "targets": len(candidate), "decision": decision["diagnosis_heterogeneity_replicated"]}
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
