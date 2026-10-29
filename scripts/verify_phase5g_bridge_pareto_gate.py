#!/usr/bin/env python3
"""Verify completeness and invariants of the Phase 5G experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from memreason_rl.data.common import ROOT
from memreason_rl.experiments.phase5g_bridge_pareto_gate import make_variants, pareto_membership


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/phase5g_bridge_pareto_gate_dev.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/phase5g_bridge_pareto_gate_dev_20261029")
    parser.add_argument("--shard-root", type=Path, default=ROOT / "outputs/phase5g_bridge_pareto_gate_dev_20261029_shards")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    variants = make_variants(config)
    names = {variant["arm"] for variant in variants}
    output = args.output_dir

    aggregate = pd.read_csv(output / "aggregate_metrics.csv")
    frontier = pd.read_csv(output / "pareto_frontier.csv")
    states = pd.read_csv(output / "cross_view_memory_states.csv")
    folds = pd.read_csv(output / "fold_manifest.csv")
    decision = json.loads((output / "development_decision.json").read_text())

    assert set(aggregate.arm) == names
    assert set(frontier.arm) == names
    assert set(states.arm) == names
    assert set(folds.arm) == names
    assert aggregate.groupby("arm").task.nunique().eq(2).all()
    assert aggregate["items"].eq(70).all()
    assert folds.groupby("arm").fold.nunique().eq(int(config["folds"])).all()
    assert len(folds) == len(variants) * int(config["folds"])
    assert states.groupby("arm").size().nunique() == 1
    assert states.loc[~states.gated, "residual_gate"].eq(1).all()
    gated = states[states.gated]
    floor = float(config["residual_adaptive_gate"]["floor"])
    assert gated.residual_gate.between(floor - 1e-7, 1 + 1e-7).all()
    assert np.allclose(gated.effective_edge_weight, gated.edge_weight * gated.residual_gate)
    assert frontier.pareto_optimal.to_numpy(bool).tolist() == pareto_membership(frontier).tolist()
    assert decision["validation_loaded"] is False
    assert decision["test_loaded"] is False
    assert decision["continuous_time_dynamics_open"] is False
    checkpoints = list(args.shard_root.glob("fold_*/checkpoints/*/fold_*.pt"))
    assert len(checkpoints) == len(variants) * int(config["folds"])
    verification = {
        "status": "passed",
        "variants": len(variants),
        "folds": int(config["folds"]),
        "checkpoints": len(checkpoints),
        "selected": decision["selected_development_point"],
    }
    (output / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
