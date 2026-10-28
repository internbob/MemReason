from __future__ import annotations

import numpy as np

from memreason_rl.experiments.phase3h_mci_conversion import (
    MONTH_DAYS,
    CONFIG,
    build_horizon_frame,
    load_config,
    metric_values,
    shuffled_indices,
)


def test_binary_metric_contract() -> None:
    truth = np.asarray([0, 0, 1, 1])
    probability = np.asarray([0.1, 0.8, 0.9, 0.2])
    metrics = metric_values(truth, probability, 0.5)
    assert metrics["accuracy"] == 0.5
    assert metrics["sensitivity"] == 0.5
    assert metrics["specificity"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["auc"] == 0.75
    assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (1, 1, 1, 1)


def test_real_conversion_cohorts_are_temporally_valid() -> None:
    config = load_config(CONFIG)
    expected = {24: (609, 128), 36: (554, 183)}
    for horizon, (subjects, converters) in expected.items():
        frame, volume_path = build_horizon_frame(config, horizon)
        assert volume_path.exists()
        assert len(frame) == subjects
        assert frame.subject_id.nunique() == subjects
        assert int(frame.label.sum()) == converters
        positive = frame.label.eq(1)
        assert frame.loc[positive, "conversion_date"].notna().all()
        assert (frame.loc[positive, "conversion_time_months"] > 0).all()
        assert (frame.loc[positive, "conversion_time_months"] <= horizon).all()
        assert frame.loc[~positive, "conversion_date"].isna().all()
        assert (frame.loc[~positive, "followup_months"] >= horizon).all()
        observed_days = (
            frame.loc[positive, "conversion_date"]
            - frame.loc[positive, "index_diagnosis_date"]
        ).dt.days
        np.testing.assert_allclose(
            observed_days / MONTH_DAYS,
            frame.loc[positive, "conversion_time_months"],
        )


def test_shuffle_has_no_identity_pairing() -> None:
    import pandas as pd

    frame = pd.DataFrame({"subject_id": [f"S{i}" for i in range(8)]})
    indices = np.arange(len(frame))
    shuffled = shuffled_indices(frame, indices, 2026)
    assert set(shuffled) == set(indices)
    assert np.all(shuffled != indices)
