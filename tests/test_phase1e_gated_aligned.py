import torch

from memreason_rl.data.common import ROOT
from memreason_rl.data.dictionary import load_dictionary
from memreason_rl.experiments.phase1b_semantic import select_verified_scalar_features
from memreason_rl.experiments.phase1c_joint_stage1 import build_domain_assignment, load_config
from memreason_rl.experiments.phase1d_typed_representation import TYPE_IDS, feature_metadata
from memreason_rl.experiments.phase1e_gated_aligned import (
    GatedTypedQuestionStage1,
    metric_aligned_loss,
)


def _model_inputs():
    config = load_config(ROOT / "configs/phase1e_gated_aligned_dev.yaml")
    dictionary = load_dictionary()
    features = select_verified_scalar_features(config, dictionary)
    _, domains = build_domain_assignment(features, dictionary)
    types, _, _, category_values, category_counts = feature_metadata(features, dictionary)
    descriptors = torch.randn(len(features), 768, generator=torch.Generator().manual_seed(7))
    return config, features, domains, types, category_values, category_counts, descriptors


def test_zero_semantic_gate_starts_from_typed_response_model():
    config, features, domains, types, values, counts, descriptors = _model_inputs()
    response = GatedTypedQuestionStage1(
        descriptors,
        domains,
        types,
        values,
        counts,
        5,
        config["arms"]["typed_response_aligned"],
        config,
    ).eval()
    semantic = GatedTypedQuestionStage1(
        descriptors,
        domains,
        types,
        values,
        counts,
        5,
        config["arms"]["typed_semantic_gated_aligned"],
        config,
    ).eval()
    semantic.load_state_dict(response.state_dict())
    torch.testing.assert_close(response._static_query(), semantic._static_query())
    with torch.no_grad():
        semantic.semantic_gate.fill_(0.5)
    assert not torch.allclose(response._static_query(), semantic._static_query())
    assert sum(p.numel() for p in response.parameters()) == sum(
        p.numel() for p in semantic.parameters()
    )
    assert len(features) == 68


def test_metric_aligned_loss_backpropagates_per_item_targets():
    config, _, domains, types, values, counts, descriptors = _model_inputs()
    model = GatedTypedQuestionStage1(
        descriptors,
        domains,
        types,
        values,
        counts,
        5,
        config["arms"]["typed_semantic_gated_aligned"],
        config,
    )
    batch_size, n_features = 3, len(types)
    outputs = (
        torch.zeros(batch_size, 5, requires_grad=True),
        torch.zeros(batch_size, n_features, requires_grad=True),
        torch.zeros(batch_size, n_features, values.shape[1], requires_grad=True),
    )
    continuous_index = int(torch.nonzero(types.eq(TYPE_IDS["continuous"]))[0])
    binary_index = int(torch.nonzero(types.eq(TYPE_IDS["binary"]))[0])
    reconstruction_mask = torch.zeros(batch_size, n_features, dtype=torch.bool)
    reconstruction_mask[0, continuous_index] = True
    reconstruction_mask[:, binary_index] = True
    categories = torch.full((batch_size, n_features), -1, dtype=torch.long)
    categories[:, binary_index] = torch.tensor([0, 1, 1])
    targets = torch.zeros(batch_size, n_features)
    targets[0, continuous_index] = 0.5
    targets[:, binary_index] = torch.tensor([0.0, 1.0, 1.0])
    batch = {
        "task_group_id": torch.tensor([3, 3, 3]),
        "total_mask": torch.zeros(batch_size, 5, dtype=torch.bool),
        "total_targets": torch.zeros(batch_size, 5),
        "reconstruction_mask": reconstruction_mask,
        "reconstruction_targets": targets,
        "reconstruction_category": categories,
    }
    loss = metric_aligned_loss(
        outputs,
        batch,
        model,
        torch.zeros(n_features),
        torch.ones(n_features),
        0.5,
        0.5,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert outputs[1].grad is not None
    assert outputs[2].grad is not None
