from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from memreason_rl.experiments.phase3c_spatial_grounding import (
    SpatialGrounder,
    coordinate_grid,
    stability_table,
)


def test_heads_are_parameter_matched_and_finite() -> None:
    coordinates = coordinate_grid((3, 4, 3))
    tokens = torch.randn(3, 36, 64)
    demo = torch.randn(3, 2)
    counts = []
    for arm in ["global_pool", "target_spatial_attention"]:
        model = SpatialGrounder(arm, 64, 128, 5, 4, 0.0, coordinates)
        prediction, weights = model(tokens, demo, need_weights=True)
        assert prediction.shape == (3, 5)
        assert torch.isfinite(prediction).all()
        counts.append(sum(parameter.numel() for parameter in model.parameters()))
        if arm == "target_spatial_attention":
            assert weights.shape == (3, 5, 36)
    assert len(set(counts)) == 1


def test_spatial_and_query_permutations_change_attention_output() -> None:
    torch.manual_seed(4)
    coordinates = coordinate_grid((3, 4, 3))
    model = SpatialGrounder("target_spatial_attention", 64, 32, 5, 4, 0.0, coordinates).eval()
    tokens = torch.randn(2, 36, 64)
    demo = torch.randn(2, 2)
    observed, _ = model(tokens, demo)
    spatial, _ = model(tokens, demo, spatial_permutation=torch.roll(torch.arange(36), 1))
    query, _ = model(tokens, demo, query_permutation=torch.roll(torch.arange(5), 1))
    assert not torch.allclose(observed, spatial)
    assert not torch.allclose(observed, query)


def test_attention_stability_identical_maps_is_one() -> None:
    rng = np.random.default_rng(3)
    weights = rng.random((4, 5, 36)).astype(np.float32)
    frame, summary = stability_table({2026: weights, 2027: weights.copy()}, [str(i) for i in range(5)], 0.2)
    assert np.allclose(frame.spearman, 1.0)
    assert np.allclose(frame.top_token_overlap, 1.0)
    assert summary == {"median_spearman": 1.0, "median_top_token_overlap": 1.0}


def test_phase3c_source_does_not_load_test_split() -> None:
    source = (
        Path(__file__).parents[1]
        / "memreason_rl"
        / "experiments"
        / "phase3c_spatial_grounding.py"
    ).read_text()
    assert "splits/test.csv" not in source
    assert 'splits/test' not in source
