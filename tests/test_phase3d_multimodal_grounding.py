import numpy as np
import pandas as pd

from memreason_rl.experiments.phase3d_multimodal_grounding import (
    GroundingDataset,
    bootstrap_difference,
    coordinate_grid,
)


def test_grounding_dataset_hides_supervision_value(tmp_path):
    tokens = np.zeros((1, 2, 3), dtype=np.float16)
    path = tmp_path / "tokens.npy"
    np.save(path, tokens)
    samples = pd.DataFrame(
        [{"sample_id": "s", "row": 0, "targets": "1", "task": "masked_item",
          "instrument": "X", "split": "val", "subject_id": "p", "scan_id": "m"}]
    )
    dataset = GroundingDataset(
        samples,
        np.asarray([[1.0, 7.0, 3.0]], dtype=np.float32),
        np.asarray([[True, True, False]]),
        np.zeros((1, 2), dtype=np.float32),
        path,
    )
    item = dataset[0]
    assert item["value"].tolist() == [1.0, 0.0, 0.0]
    assert item["target"].tolist() == [1.0, 7.0, 3.0]
    assert item["input_mask"].tolist() == [True, False, False]
    assert item["output_mask"].tolist() == [False, True, False]


def test_coordinate_grid_has_one_coordinate_per_spatial_token():
    grid = coordinate_grid((3, 4, 3))
    assert grid.shape == (36, 3)
    assert np.all(grid >= -1) and np.all(grid <= 1)


def test_bootstrap_difference_pairs_by_sample_and_target():
    common = {
        "sample_id": ["a", "b"], "scan_id": ["m1", "m2"],
        "subject_id": ["p1", "p2"], "task": ["masked_item", "masked_item"],
        "target": ["x", "x"], "normalized_truth": [0.0, 0.0],
    }
    candidate = pd.DataFrame({**common, "model": "spatial", "pairing": "observed",
                              "normalized_prediction": [0.1, 0.1]})
    baseline = pd.DataFrame({**common, "model": "cognitive", "pairing": "observed",
                             "normalized_prediction": [0.3, 0.3]})
    result = bootstrap_difference(candidate, baseline, 100, 7, "masked_item")
    assert np.isclose(result["difference"], -0.2)
    assert result["ci_upper"] < 0
