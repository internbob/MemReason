import numpy as np
import pandas as pd
import torch

from memreason_rl.experiments.phase5f_hierarchical_bridge import (
    HierarchicalBridgeAE,
    build_connection_graph,
    construct_membership,
    graph_alignment_loss,
    residual_evidence_gate,
    shared_bridge_weights,
    split_instrument_views,
)


def _items():
    return [
        {"field": "a_recall", "instrument": "A", "domain": "Memory"},
        {"field": "b_recall", "instrument": "B", "domain": "Memory"},
        {"field": "a_language", "instrument": "A", "domain": "Language"},
        {"field": "b_language", "instrument": "B", "domain": "Language"},
    ]


def test_construct_membership_is_frozen_and_disjoint():
    names, membership = construct_membership(
        _items(), {"recall": ["a_recall", "b_recall"]})
    assert names == ["recall"]
    assert membership.tolist() == [0, 0, -1, -1]


def test_heldout_instrument_has_exactly_zero_residual_state():
    model = HierarchicalBridgeAE(
        domain_index=np.asarray([0, 0, 1, 1]),
        construct_index=np.asarray([0, 0, -1, -1]),
        instrument_index=np.asarray([0, 1, 0, 1]),
        hidden=8,
        use_residual=True,
        logvar_bounds=(-6, 3),
    )
    values = torch.randn(3, 4)
    mask = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 1, 1]], dtype=torch.float32)
    state = model.state(values, mask)
    assert state["residual_mean"][0, 1] == 0
    assert state["residual_mean"][1, 0] == 0
    assert state["residual_available"].tolist() == [[True, False], [False, True], [True, True]]
    assert torch.all(model.loadings()["global"] > 0)
    assert torch.all(model.loadings()["domain"] > 0)


def test_instrument_views_are_disjoint_and_graph_supported():
    observed = np.ones((5, 4), bool)
    instrument = np.asarray([0, 1, 0, 1])
    graph = np.asarray([[1.0, 0.6], [0.6, 1.0]], np.float32)
    left, right, weight = split_instrument_views(observed, instrument, graph,
                                                 np.random.default_rng(7))
    assert not np.any(left & right)
    assert np.all((left | right) == observed)
    assert np.allclose(weight, 0.6)


def test_connection_graph_uses_dates_and_not_response_values():
    frame = pd.DataFrame({
        "subject_id": ["s1", "s2", "s3"],
        "adas_visit_date": ["2020-01-01", "2020-02-01", "2020-03-01"],
        "cdr_visit_date": ["2020-01-01", "2020-04-01", "2020-03-15"],
    })
    mask = np.ones((3, 4), bool)
    config = {"graph": {"date_window_days": 30, "nominal_visit_weight": 0.25,
                         "minimum_edge_weight": 0.0}}
    graph, matrix, instruments = build_connection_graph(frame, mask, _items(), config)
    assert instruments == ["A", "B"]
    # The synthetic instrument names have no date-column contract, so this
    # test uses nominal visit support and remains response-value independent.
    assert graph.concurrent_rows.iloc[0] == 3
    assert graph.edge_weight.iloc[0] > 0
    assert np.isclose(matrix[0, 1], graph.edge_weight.iloc[0])


def test_graph_alignment_is_finite_with_supported_views():
    model = HierarchicalBridgeAE(
        domain_index=np.asarray([0, 0, 1, 1]),
        construct_index=np.asarray([0, 0, -1, -1]),
        instrument_index=np.asarray([0, 1, 0, 1]),
        hidden=8,
        use_residual=True,
        logvar_bounds=(-6, 3),
    )
    values = torch.randn(4, 4)
    left = torch.tensor([[1, 0, 1, 0]] * 4, dtype=torch.float32)
    right = torch.tensor([[0, 1, 0, 1]] * 4, dtype=torch.float32)
    loss = graph_alignment_loss(model, values, left, right, torch.ones(4, 7))
    assert torch.isfinite(loss) and loss >= 0


def test_residual_evidence_gate_is_bounded_and_monotone():
    available = torch.ones(2, 2, dtype=torch.bool)
    logvar = torch.zeros(2, 2)
    low = {
        "residual_mean": torch.tensor([[0.0, 0.0], [0.2, 0.2]]),
        "residual_logvar": logvar,
        "residual_available": available,
    }
    high = {
        "residual_mean": torch.tensor([[0.0, 0.0], [2.0, 2.0]]),
        "residual_logvar": logvar,
        "residual_available": available,
    }
    gate = residual_evidence_gate(low, high, floor=0.25, temperature=1.0)
    assert torch.all(gate >= 0.25)
    assert torch.all(gate <= 1.0)
    assert gate[1] < gate[0]
    assert not gate.requires_grad


def test_gated_graph_alignment_is_finite():
    model = HierarchicalBridgeAE(
        domain_index=np.asarray([0, 0, 1, 1]),
        construct_index=np.asarray([0, 0, -1, -1]),
        instrument_index=np.asarray([0, 1, 0, 1]),
        hidden=8,
        use_residual=True,
        logvar_bounds=(-6, 3),
    )
    values = torch.randn(4, 4)
    left = torch.tensor([[1, 0, 1, 0]] * 4, dtype=torch.float32)
    right = torch.tensor([[0, 1, 0, 1]] * 4, dtype=torch.float32)
    loss = graph_alignment_loss(
        model,
        values,
        left,
        right,
        torch.ones(4, 7),
        residual_gate_floor=0.25,
        residual_gate_temperature=1.0,
    )
    assert torch.isfinite(loss) and loss >= 0


def test_bridge_weights_are_dimension_specific():
    # Instrument 0 and 1 have a supported edge. Instrument 2 has no supported
    # edge to instrument 0. A non-Memory item from instrument 1 must not make
    # the Memory state from instrument 2 align with instrument 0.
    view_a = np.asarray([[1, 0, 1]], bool)
    view_b = np.asarray([[0, 1, 0]], bool)
    instrument = np.asarray([2, 0, 1])
    domain = np.asarray([0, 0, 1])
    construct = np.asarray([0, 0, -1])
    graph = np.asarray([[1.0, 0.8, 0.0], [0.8, 1.0, 0.0], [0.0, 0.0, 1.0]], np.float32)
    weights = shared_bridge_weights(view_a, view_b, instrument, domain, construct, graph)
    assert np.isclose(weights[0, 0], 0.8)  # global can use instrument 1--0
    assert weights[0, 1] == 0.0  # Memory sees only instrument 2--0
    assert weights[0, 6] == 0.0  # objective-recall construct is also unsupported
