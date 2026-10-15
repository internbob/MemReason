#!/usr/bin/env python
"""Independent integrity checks for the Phase 3A MRI-anchor experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/phase3a_global_mri_anchor_dev_20261013"
TARGET_DATES = {
    "adas_total13": "adas_visit_date",
    "cdr_cdrsb": "cdr_visit_date",
    "faq_faqtotal": "faq_visit_date",
    "mmse_mmscore": "mmse_visit_date",
    "moca_moca": "moca_visit_date",
}
ARMS = [
    "cognition_only",
    "observed_mri",
    "shuffled_subject_mri",
    "shuffled_time_mri",
]
SEEDS = [2026, 2027, 2028]


def main() -> None:
    config = yaml.safe_load((ROOT / "configs/phase3a_global_mri_anchor_dev.yaml").read_text())
    decision = json.loads((RUN / "development_decision.json").read_text())
    run = json.loads((RUN / "run_manifest.json").read_text())
    assert decision["test_evaluated"] is False and run["test_evaluated"] is False
    source = (ROOT / "memreason_rl/experiments/phase3a_global_mri_anchor.py").read_text()
    assert "splits/test.csv" not in source and 'splits/test' not in source

    checkpoints = sorted((RUN / "checkpoints").glob("*/*/best.pt"))
    assert len(checkpoints) == len(ARMS) * len(SEEDS) == 12
    checkpoint_counts = []
    for path in checkpoints:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint_counts.append(payload["trainable_parameters"])
        assert payload["arm"] in ARMS and payload["seed"] in SEEDS
    assert len(set(checkpoint_counts)) == 1
    assert len(set(run["parameter_counts"].values())) == 1

    samples = pd.read_csv(RUN / "sample_manifest.csv")
    assert set(samples.split) == {"train", "val"}
    assert samples.sample_id.nunique() == len(samples)
    assert samples.date_difference_days.abs().max() <= config["mri"]["date_window_days"]
    assert (samples.n_targets > 0).all()
    assert samples.loc[samples.split.eq("train"), "subject_id"].nunique() == 1349
    assert samples.loc[samples.split.eq("val"), "subject_id"].nunique() == 185

    qc = pd.read_csv(RUN / "project_qc.csv")
    assert qc.scan_id.nunique() == len(qc) == run["unique_mri"]
    assert np.isclose(qc.project_qc_pass.mean(), run["project_qc_pass_rate"])
    assert qc.loc[qc.project_qc_pass, "feature_dim"].nunique() == 1
    archive = np.load(RUN / "mri_features.npz", allow_pickle=False)
    assert archive["features"].shape == (int(qc.project_qc_pass.sum()), int(qc.feature_dim.dropna().iloc[0]))
    assert np.isfinite(archive["features"]).all()

    predictions = pd.read_csv(RUN / "validation_predictions.csv")
    seed_predictions = predictions[predictions.seed.notna()].copy()
    assert set(seed_predictions.model) == set(ARMS)
    assert set(seed_predictions.seed.astype(int)) == set(SEEDS)
    keys = ["sample_id", "subject_id", "target", "horizon_months", "cutoff_month", "target_month", "truth"]
    reference = seed_predictions[
        seed_predictions.model.eq(ARMS[0]) & seed_predictions.seed.eq(SEEDS[0])
    ][keys].sort_values(keys).reset_index(drop=True)
    for arm in ARMS:
        for seed in SEEDS:
            current = seed_predictions[
                seed_predictions.model.eq(arm) & seed_predictions.seed.eq(seed)
            ][keys].sort_values(keys).reset_index(drop=True)
            pd.testing.assert_frame_equal(current, reference)

    # Independently check the target-date-after-MRI rule on every validation target.
    val = pd.read_csv(ROOT / "splits/val.csv", low_memory=False)
    val["subject_id"] = val.subject_id.astype(str)
    lookup = val.set_index(["subject_id", "visit_month"])
    scan_dates = samples.set_index("sample_id").scan_date
    one_arm = seed_predictions[
        seed_predictions.model.eq("observed_mri") & seed_predictions.seed.eq(SEEDS[0])
    ]
    for row in one_arm.itertuples(index=False):
        target_row = lookup.loc[(str(row.subject_id), float(row.target_month))]
        target_date = pd.to_datetime(target_row[TARGET_DATES[row.target]])
        scan_date = pd.to_datetime(scan_dates.loc[row.sample_id])
        assert pd.notna(target_date) and target_date > scan_date

    boot = pd.read_csv(RUN / "bootstrap_comparisons.csv")
    overall = boot[boot.scope.eq("overall")].set_index("baseline")
    horizon = boot[
        boot.baseline.eq("cognition_only_ensemble") & boot.horizon_months.isin([12, 24])
    ]
    recomputed = {
        "technical_qc_pass_rate": float(qc.project_qc_pass.mean())
        >= config["gate"]["minimum_technical_qc_pass_rate"],
        "observed_better_than_cognition": overall.loc["cognition_only_ensemble", "ci_upper"] < 0,
        "observed_better_than_shuffled_subject": overall.loc[
            "shuffled_subject_mri_ensemble", "ci_upper"
        ]
        < 0,
        "observed_better_than_shuffled_time": overall.loc[
            "shuffled_time_mri_ensemble", "ci_upper"
        ]
        < 0,
        "supported_12_or_24_month_improvement": bool((horizon.ci_upper < 0).any()),
    }
    assert recomputed == decision["criteria"]
    assert decision["global_mri_anchor_gate_passed"] == all(recomputed.values())
    result = {
        "verified": True,
        "checkpoints": len(checkpoints),
        "validation_targets_per_arm_seed": len(reference),
        "unique_mri": len(qc),
        "technical_qc_pass_rate": float(qc.project_qc_pass.mean()),
        "target_after_scan_verified": True,
        "parameter_matched": True,
        "gate_passed": decision["global_mri_anchor_gate_passed"],
        "test_evaluated": False,
    }
    (RUN / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
