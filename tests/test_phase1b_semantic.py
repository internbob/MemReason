import numpy as np
import pandas as pd
import torch

from memreason_rl.data.common import ROOT
from memreason_rl.data.dictionary import load_dictionary
from memreason_rl.experiments.phase1b_semantic import (
    DescriptorGRUDForecaster,
    build_cross_scale_samples,
    descriptor_text,
    load_config,
    select_verified_scalar_features,
)


def test_phase1b_uses_only_verified_monotonic_nonoutcome_inputs():
    config = load_config(ROOT / "configs/phase1b_semantic_fair.yaml")
    dictionary = load_dictionary()
    features = select_verified_scalar_features(config, dictionary)
    assert len(features) == 68
    assert all(dictionary[key]["verified_from_dictionary"] for key in features)
    assert all(dictionary[key]["direction"] in {"higher=better", "higher=worse"} for key in features)
    assert not any(key.startswith(("dxsum_", "faq_")) for key in features)
    assert not set(config["targets"]).intersection(features)
    assert not any(key.startswith("moca_delw") for key in features)


def test_semantic_descriptor_contains_verified_task_but_no_outcome_or_subject_data():
    dictionary = load_dictionary()
    text = descriptor_text("adas_q1score", dictionary["adas_q1score"], "task_semantics")
    assert "Word Recall" in text
    assert "Memory" in text
    assert "diagnosis" not in text.lower()
    assert "subject" not in text.lower()


def test_all_arms_have_identical_trainable_architecture_size():
    counts = []
    for descriptor in [torch.zeros(5, 768), torch.randn(5, 768), torch.randn(5, 768)]:
        model = DescriptorGRUDForecaster(descriptor, 3, 8, 16, 2, 0.1)
        counts.append(sum(parameter.numel() for parameter in model.parameters()))
    assert len(set(counts)) == 1


def test_cross_scale_samples_hide_non_source_instruments_and_totals():
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
                "adas_total13": 10.0,
                "cdr_cdrsb": 1.0,
                "faq_faqtotal": 2.0,
                "mmse_mmscore": 28.0,
                "moca_moca": 24.0,
            }
        ]
    )
    pairs = [
        {"name": "adas_cdr_to_moca", "source_prefixes": ["adas_", "cdr_"], "target": "moca_moca"},
        {"name": "mmse_cdr_to_adas", "source_prefixes": ["mmse_", "cdr_"], "target": "adas_total13"},
    ]
    samples = build_cross_scale_samples(frame, "train", features, targets, pairs)
    assert len(samples) == 2
    first, second = samples
    np.testing.assert_array_equal(first.feature_mask[0], [True, True, False, False])
    np.testing.assert_array_equal(second.feature_mask[0], [False, True, True, False])
    assert first.targets[4] == 24.0 and first.target_mask.sum() == 1
    assert second.targets[0] == 10.0 and second.target_mask.sum() == 1
