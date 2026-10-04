import numpy as np
import torch

from memreason_rl.data.common import ROOT
from memreason_rl.data.dictionary import load_dictionary
from memreason_rl.experiments.phase1_baseline_suite import build_model, load_config
from memreason_rl.experiments.phase1_numeric import ForecastSample, NumericCollator
from memreason_rl.experiments.phase1b_semantic import select_verified_scalar_features


def _sample(subject: str, months: list[float]) -> ForecastSample:
    n = len(months)
    values = np.arange(n * 3, dtype=np.float32).reshape(n, 3)
    return ForecastSample(
        sample_id=subject,
        subject_id=subject,
        split="val",
        cutoff_month=months[-1],
        horizon_months=12,
        months=np.asarray(months, dtype=float),
        features=values,
        feature_mask=np.ones_like(values, dtype=bool),
        history_targets=np.zeros((n, 2), dtype=np.float32),
        history_target_mask=np.ones((n, 2), dtype=bool),
        targets=np.asarray([1.0, 2.0], dtype=np.float32),
        target_mask=np.ones(2, dtype=bool),
        target_months=np.asarray([months[-1] + 12] * 2, dtype=np.float32),
    )


def test_baseline_suite_uses_verified_scalar_contract():
    config = load_config(ROOT / "configs/phase1_baseline_suite_formal.yaml")
    features = select_verified_scalar_features(config, load_dictionary())
    assert len(features) == 68
    assert config["architectures"] == ["GRU", "LSTM", "T-LSTM", "GRU-D", "Transformer"]
    assert not any(name.startswith(("dxsum_", "faq_", "AVLT_", "LogMem")) for name in features)


def test_all_deep_baselines_ignore_padding_and_return_finite_targets():
    names = ["a", "b", "c", "y1", "y2"]
    normalization = {name: {"mean": 0.0, "std": 1.0} for name in names}
    batch = NumericCollator(["a", "b", "c"], ["y1", "y2"], normalization)(
        [_sample("short", [0.0]), _sample("long", [0.0, 6.0])]
    )
    for architecture in ["GRU", "LSTM", "T-LSTM", "GRU-D", "Transformer"]:
        model = build_model(architecture, 3, 2, 8, 2, 0.0).eval()
        with torch.no_grad():
            before = model(batch)
            changed = dict(batch)
            changed["values"] = batch["values"].clone()
            changed["values"][~batch["event_mask"]] = 12345.0
            after = model(changed)
        assert before.shape == (2, 2)
        assert torch.isfinite(before).all()
        torch.testing.assert_close(before, after)
