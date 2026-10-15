import numpy as np
import pandas as pd
import torch

from memreason_rl.data.common import ROOT
from memreason_rl.data.dictionary import load_dictionary
from memreason_rl.experiments.phase1b_semantic import select_verified_scalar_features
from memreason_rl.experiments.phase1c_joint_stage1 import (
    JointCollator,
    ResponseAwareStage1,
    build_domain_assignment,
    build_reconstruction_samples,
    load_config,
    reconstruction_type_metrics,
    response_text,
)


def test_response_text_contains_task_actual_response_and_direction():
    entry = load_dictionary()["mmse_mmyear"]
    text = response_text("mmse_mmyear", entry, 1.0)
    assert "Orientation" in text
    assert "Correct" in text
    assert "higher=better" in text
    assert "diagnosis" not in text.lower()


def test_reconstruction_masks_hide_only_declared_targets():
    features = ["adas_q1score", "cdr_cdmemory", "mmse_mmyear", "moca_trails"]
    targets = ["adas_total13", "cdr_cdrsb", "faq_faqtotal", "mmse_mmscore", "moca_moca"]
    frame = pd.DataFrame(
        [
            {
                "subject_id": "s1",
                "visit_month": 12.0,
                "adas_q1score": 2.0,
                "cdr_cdmemory": 0.5,
                "mmse_mmyear": 1.0,
                "moca_trails": 1.0,
            }
        ]
    )
    samples = build_reconstruction_samples(frame, "train", features, targets, load_dictionary(), 2026)
    item = samples["masked_item"][0]
    instrument = samples["masked_instrument"][0]
    assert item.reconstruction_mask.sum() == 1
    assert not np.any(item.feature_mask[0] & item.reconstruction_mask)
    assert instrument.reconstruction_mask.any()
    assert not np.any(instrument.feature_mask[0] & instrument.reconstruction_mask)
    assert instrument.feature_mask.any()


def test_phase1c_arms_are_parameter_matched_and_ignore_padding():
    config = load_config(ROOT / "configs/phase1c_joint_stage1.yaml")
    dictionary = load_dictionary()
    features = select_verified_scalar_features(config, dictionary)
    assert len(features) == 68
    _, domains = build_domain_assignment(features, dictionary)
    counts = []
    for seed in [0, 1, 2]:
        generator = torch.Generator().manual_seed(seed)
        task = torch.randn(len(features), 768, generator=generator)
        response = torch.randn(len(features), 4, 768, generator=generator)
        model = ResponseAwareStage1(task, response, domains, 5, config).eval()
        counts.append(sum(parameter.numel() for parameter in model.parameters()))
        batch = {
            "values": torch.randn(2, 2, len(features), generator=generator),
            "mask": torch.ones(2, 2, len(features), dtype=torch.bool),
            "response_index": torch.zeros(2, 2, len(features), dtype=torch.long),
            "times": torch.tensor([[0.0, 0.0], [0.0, 0.5]]),
            "event_mask": torch.tensor([[True, False], [True, True]]),
            "horizon": torch.ones(2),
            "task_id": torch.tensor([0, 2]),
        }
        batch["mask"][0, 1] = False
        batch["response_index"][0, 1] = -1
        with torch.no_grad():
            before = model(batch)
            changed = dict(batch)
            changed["values"] = batch["values"].clone()
            changed["values"][0, 1] = 12345.0
            after = model(changed)
        torch.testing.assert_close(before[0], after[0])
        torch.testing.assert_close(before[1], after[1])
    assert len(set(counts)) == 1


def test_ordinal_reconstruction_metrics_accept_float_encoded_truth():
    dictionary = load_dictionary()
    predictions = pd.DataFrame(
        {
            "task": ["masked_item"] * 4,
            "model": ["response_aware ensemble"] * 4,
            "target": ["cdr_cdmemory"] * 4,
            "truth": [0.0, 0.5, 1.0, 2.0],
            "prediction": [0.1, 0.6, 1.2, 1.8],
        }
    )
    result = reconstruction_type_metrics(predictions, dictionary)
    assert result.loc[0, "weighted_kappa"] == 1.0
