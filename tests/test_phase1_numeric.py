import numpy as np
import pandas as pd
import torch

from memreason_rl.data.common import ROOT
from memreason_rl.data.dictionary import load_dictionary
from memreason_rl.experiments.phase1_numeric import (
    ForecastSample,
    GRUDForecaster,
    NumericCollator,
    TimeAwareTransformerForecaster,
    build_samples,
    load_config,
    select_numeric_features,
)


def test_formal_feature_contract_excludes_diagnosis_outcomes_totals_and_unresolved_items():
    config = load_config(ROOT / "configs/phase1_numeric_formal.yaml")
    features = select_numeric_features(config, load_dictionary())
    forbidden = {
        "adas_total13",
        "adas_totscore",
        "cdr_cdrsb",
        "cdr_cdglobal",
        "faq_faqtotal",
        "mmse_mmscore",
        "moca_moca",
        "moca_letters",
        "adas_q13score",
    }
    assert forbidden.isdisjoint(features)
    assert not any(item.startswith(("dxsum_", "faq_", "AVLT_", "LogMem")) for item in features)
    assert not any(item.startswith(("moca_delw", "moca_serial")) for item in features)


def test_formal_samples_use_strict_future_nearest_observation_within_tolerance():
    frame = pd.DataFrame(
        {
            "subject_id": ["s1"] * 4,
            "visit_month": [0.0, 5.0, 8.0, 13.0],
            "feature": [1.0, 2.0, 3.0, 4.0],
            "target": [10.0, 11.0, 99.0, 13.0],
        }
    )
    samples = build_samples(frame, "train", ["feature"], ["target"], [6], {6: 3})
    first = next(sample for sample in samples if sample.cutoff_month == 0)
    assert first.target_months[0] == 5.0
    assert first.targets[0] == 11.0
    assert first.target_months[0] > first.cutoff_month
    second = next(sample for sample in samples if sample.cutoff_month == 5)
    assert second.target_months[0] == 13.0
    assert second.target_months[0] > second.cutoff_month


def _sample(subject: str, months: list[float]) -> ForecastSample:
    n = len(months)
    features = np.arange(n * 3, dtype=np.float32).reshape(n, 3)
    return ForecastSample(
        sample_id=subject,
        subject_id=subject,
        split="val",
        cutoff_month=months[-1],
        horizon_months=12,
        months=np.asarray(months, dtype=float),
        features=features,
        feature_mask=np.ones_like(features, dtype=bool),
        history_targets=np.zeros((n, 2), dtype=np.float32),
        history_target_mask=np.ones((n, 2), dtype=bool),
        targets=np.asarray([1.0, 2.0], dtype=np.float32),
        target_mask=np.ones(2, dtype=bool),
        target_months=np.asarray([months[-1] + 12] * 2, dtype=np.float32),
    )


def test_neural_forecasters_ignore_padded_values_and_return_target_shape():
    normalization = {name: {"mean": 0.0, "std": 1.0} for name in ["a", "b", "c", "y1", "y2"]}
    batch = NumericCollator(["a", "b", "c"], ["y1", "y2"], normalization)(
        [_sample("short", [0.0]), _sample("long", [0.0, 6.0])]
    )
    for model in [GRUDForecaster(3, 2, 8, 2, 0.0), TimeAwareTransformerForecaster(3, 2, 8, 1, 0.0)]:
        model.eval()
        with torch.no_grad():
            before = model(batch)
            changed = dict(batch)
            changed["values"] = batch["values"].clone()
            changed["values"][~batch["event_mask"]] = 12345
            after = model(changed)
        assert before.shape == (2, 2)
        assert torch.isfinite(before).all()
        torch.testing.assert_close(before, after)
