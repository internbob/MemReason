import torch

from memreason_rl.data.common import ROOT
from memreason_rl.data.dictionary import load_dictionary
from memreason_rl.experiments.phase1b_semantic import select_verified_scalar_features
from memreason_rl.experiments.phase1c_joint_stage1 import build_domain_assignment, load_config
from memreason_rl.experiments.phase1d_typed_representation import feature_metadata
from memreason_rl.experiments.phase1f_task_gated import TaskGatedTypedQuestionStage1


def test_task_specific_gates_are_matched_and_start_at_numeric_representation():
    config = load_config(ROOT / "configs/phase1f_task_gated_dev.yaml")
    dictionary = load_dictionary()
    features = select_verified_scalar_features(config, dictionary)
    _, domains = build_domain_assignment(features, dictionary)
    types, _, _, values, counts = feature_metadata(features, dictionary)
    descriptors = torch.randn(len(features), 768, generator=torch.Generator().manual_seed(11))
    numeric = TaskGatedTypedQuestionStage1(
        descriptors,
        domains,
        types,
        values,
        counts,
        5,
        config["arms"]["numeric_task_gated"],
        config,
    ).eval()
    semantic = TaskGatedTypedQuestionStage1(
        descriptors,
        domains,
        types,
        values,
        counts,
        5,
        config["arms"]["typed_semantic_task_gated"],
        config,
    ).eval()
    semantic.load_state_dict(numeric.state_dict())
    batch = {
        "values": torch.randn(2, 2, len(features)),
        "mask": torch.ones(2, 2, len(features), dtype=torch.bool),
        "severity": torch.rand(2, 2, len(features)),
        "times": torch.tensor([[0.0, 1.0], [0.0, 2.0]]),
        "event_mask": torch.ones(2, 2, dtype=torch.bool),
        "horizon": torch.tensor([1.0, 2.0]),
        "task_group_id": torch.tensor([0, 3]),
    }
    with torch.no_grad():
        numeric_output = numeric(batch)
        semantic_output = semantic(batch)
    for left, right in zip(numeric_output, semantic_output):
        torch.testing.assert_close(left, right)
    with torch.no_grad():
        semantic.semantic_gate[0].fill_(0.5)
        semantic.value_gate[0].fill_(0.5)
        changed = semantic(batch)
    assert not torch.allclose(numeric_output[0][0], changed[0][0])
    assert torch.allclose(numeric_output[0][1], changed[0][1])
    assert sum(p.numel() for p in numeric.parameters()) == sum(
        p.numel() for p in semantic.parameters()
    )
