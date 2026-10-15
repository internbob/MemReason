import torch

from memreason_rl.data.common import ROOT
from memreason_rl.data.dictionary import load_dictionary
from memreason_rl.experiments.phase1b_semantic import select_verified_scalar_features
from memreason_rl.experiments.phase1c_joint_stage1 import build_domain_assignment, load_config
from memreason_rl.experiments.phase1d_typed_representation import (
    TASK_GROUPS,
    TYPE_IDS,
    TypedQuestionStage1,
    feature_metadata,
    severity_coordinate,
    typed_loss,
)


def test_verified_direction_aligned_severity_coordinates():
    dictionary = load_dictionary()
    features = ["adas_q1score", "mmse_mmyear", "cdr_cdmemory"]
    types, bounds, directions, _, counts = feature_metadata(features, dictionary)
    raw = torch.tensor([[[0.0, 0.0, 0.0], [10.0, 1.0, 3.0]]])
    severity = severity_coordinate(raw, bounds, directions)
    torch.testing.assert_close(severity[0, 0], torch.tensor([0.0, 1.0, 0.0]))
    torch.testing.assert_close(severity[0, 1], torch.tensor([1.0, 0.0, 1.0]))
    assert types.tolist() == [TYPE_IDS["continuous"], TYPE_IDS["binary"], TYPE_IDS["ordinal"]]
    assert counts.tolist() == [0, 2, 5]


def test_phase1d_arms_are_parameter_matched_and_typed_path_is_active():
    config = load_config(ROOT / "configs/phase1d_typed_representation_dev.yaml")
    dictionary = load_dictionary()
    features = select_verified_scalar_features(config, dictionary)
    _, domains = build_domain_assignment(features, dictionary)
    types, _, _, category_values, category_counts = feature_metadata(features, dictionary)
    descriptors = torch.randn(len(features), 768, generator=torch.Generator().manual_seed(4))
    batch = {
        "values": torch.randn(2, 2, len(features)),
        "mask": torch.ones(2, 2, len(features), dtype=torch.bool),
        "severity": torch.rand(2, 2, len(features)),
        "times": torch.tensor([[0.0, 1.0], [0.0, 2.0]]),
        "event_mask": torch.ones(2, 2, dtype=torch.bool),
        "horizon": torch.tensor([1.0, 2.0]),
        "task_group_id": torch.tensor([0, 3]),
    }
    counts, outputs = [], {}
    for name, arm in config["arms"].items():
        model = TypedQuestionStage1(
            descriptors,
            domains,
            types,
            category_values,
            category_counts,
            5,
            arm,
            config,
        ).eval()
        counts.append(sum(parameter.numel() for parameter in model.parameters()))
        with torch.no_grad():
            outputs[name] = model(batch)
        assert outputs[name][0].shape == (2, 5)
        assert outputs[name][1].shape == (2, len(features))
        assert outputs[name][2].shape[:2] == (2, len(features))
    assert len(set(counts)) == 1

    numeric = TypedQuestionStage1(
        descriptors,
        domains,
        types,
        category_values,
        category_counts,
        5,
        config["arms"]["numeric_query_joint"],
        config,
    ).eval()
    changed = dict(batch)
    changed["severity"] = 1.0 - batch["severity"]
    with torch.no_grad():
        before = numeric(batch)
        after = numeric(changed)
    for left, right in zip(before, after):
        torch.testing.assert_close(left, right)


def test_typed_loss_combines_continuous_and_categorical_targets():
    feature_types = torch.tensor([TYPE_IDS["continuous"], TYPE_IDS["binary"]])
    outputs = (
        torch.zeros(2, 5, requires_grad=True),
        torch.zeros(2, 2, requires_grad=True),
        torch.zeros(2, 2, 2, requires_grad=True),
    )
    batch = {
        "task_group_id": torch.tensor([TASK_GROUPS.index("masked_item")] * 2),
        "total_mask": torch.zeros(2, 5, dtype=torch.bool),
        "total_targets": torch.zeros(2, 5),
        "reconstruction_mask": torch.tensor([[True, False], [False, True]]),
        "reconstruction_targets": torch.tensor([[0.5, 0.0], [0.0, 1.0]]),
        "reconstruction_category": torch.tensor([[-1, -1], [-1, 1]]),
    }
    loss = typed_loss(outputs, batch, feature_types, reconstruction_weight=1.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert outputs[1].grad is not None
    assert outputs[2].grad is not None
