import torch

from memreason_rl.experiments.phase6d1_static_embedding_incremental import feature_mask


CONFIG = {"mri": {"features": [str(i) for i in range(19)], "embedding_indices": [0, 16],
                  "probability_index": 16, "metadata_indices": [17, 18]}}


def test_embedding_mask_excludes_probability():
    mask = feature_mask("embedding_static", CONFIG)
    assert torch.all(mask[:16] == 1) and mask[16] == 0 and torch.all(mask[17:] == 1)


def test_probability_mask_excludes_embedding():
    mask = feature_mask("probability_static", CONFIG)
    assert torch.all(mask[:16] == 0) and mask[16] == 1 and torch.all(mask[17:] == 1)
