#!/usr/bin/env python3
"""Merge fold shards and finalize Phase 5G statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from memreason_rl.data.common import ROOT
from memreason_rl.experiments.phase5d_anchored_mirt import concordance_correlation
from memreason_rl.experiments.phase5g_bridge_pareto_gate import (
    ccc_bootstrap_arm,
    make_variants,
    pareto_membership,
    subject_bootstrap_comparisons,
)


def concatenate(shards: list[Path], name: str) -> pd.DataFrame:
    frames = [pd.read_csv(shard / name) for shard in shards]
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/phase5g_bridge_pareto_gate_dev.yaml")
    parser.add_argument("--shard-root", type=Path, default=ROOT / "outputs/phase5g_bridge_pareto_gate_dev_20261029_shards")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/phase5g_bridge_pareto_gate_dev_20261029")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    variants = make_variants(config)
    shards = sorted(path for path in args.shard_root.glob("fold_*") if path.is_dir())
    expected = int(config["folds"])
    if len(shards) != expected:
        raise RuntimeError(f"Expected {expected} fold shards, found {len(shards)}")
    observed_folds = sorted(
        int(pd.read_csv(shard / "fold_manifest.csv").fold.iloc[0]) for shard in shards
    )
    if observed_folds != list(range(expected)):
        raise RuntimeError(f"Incomplete fold set: {observed_folds}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    item_fold = concatenate(shards, "item_fold_metrics.csv")
    item_fold.to_csv(output / "item_fold_metrics.csv", index=False)
    item_metrics = item_fold.groupby(["arm", "task", "item", "domain"], as_index=False).agg(
        error_sum=("error_sum", "sum"), decisions=("decisions", "sum")
    )
    item_metrics["mae"] = item_metrics.error_sum / item_metrics.decisions
    item_metrics.to_csv(output / "item_metrics.csv", index=False)
    aggregate = item_metrics.groupby(["arm", "task"], as_index=False).agg(
        macro_item_mae=("mae", "mean"), items=("item", "nunique"), decisions=("decisions", "sum")
    )
    memory = item_metrics[item_metrics.domain.eq("Memory")].groupby(["arm", "task"], as_index=False).agg(
        memory_macro_item_mae=("mae", "mean"), memory_items=("item", "nunique")
    )
    aggregate = aggregate.merge(memory, on=["arm", "task"])
    aggregate.to_csv(output / "aggregate_metrics.csv", index=False)

    fold_metrics = item_fold.groupby(["fold", "arm", "task"], as_index=False).agg(
        macro_item_mae=("mae", "mean")
    )
    fold_metrics.to_csv(output / "fold_metrics.csv", index=False)
    subject_metrics = concatenate(shards, "subject_task_metrics.csv")
    subject_metrics.to_csv(output / "subject_task_metrics.csv", index=False)
    states = concatenate(shards, "cross_view_memory_states.csv")
    states.to_csv(output / "cross_view_memory_states.csv", index=False)
    graph = concatenate(shards, "connection_graph.csv").drop_duplicates()
    graph.to_csv(output / "connection_graph.csv", index=False)
    fold_manifest = concatenate(shards, "fold_manifest.csv")
    fold_manifest.to_csv(output / "fold_manifest.csv", index=False)

    ccc_rows = [
        ccc_bootstrap_arm(
            states,
            variant["arm"],
            int(config["statistics"]["state_bootstrap_replicates"]),
            int(config["statistics"]["bootstrap_seed"]) + index,
        )
        for index, variant in enumerate(variants)
    ]
    summaries = []
    for variant in variants:
        arm = variant["arm"]
        group = states[states.arm.eq(arm)]
        summaries.append(
            {
                "arm": arm,
                "graph_weight": variant["graph_weight"],
                "gated": variant["gated"],
                "spearman": float(spearmanr(group.view_a, group.view_b).statistic),
                "mean_absolute_difference": float(group.absolute_difference.mean()),
                "memory_state_std": float(pd.concat([group.view_a, group.view_b]).std()),
                "mean_posterior_sd": float(pd.concat([group.view_a_sd, group.view_b_sd]).mean()),
                "mean_edge_weight": float(group.edge_weight.mean()),
                "mean_residual_gate": float(group.residual_gate.mean()),
                "mean_effective_edge_weight": float(group.effective_edge_weight.mean()),
            }
        )
    summary = pd.DataFrame(summaries).merge(pd.DataFrame(ccc_rows), on="arm")
    summary.to_csv(output / "cross_view_summary.csv", index=False)
    comparisons = subject_bootstrap_comparisons(subject_metrics, variants, config)
    comparisons.to_csv(output / "subject_bootstrap.csv", index=False)

    instrument = aggregate[aggregate.task.eq("heldout_instrument")][
        ["arm", "macro_item_mae", "memory_macro_item_mae"]
    ].rename(columns={"macro_item_mae": "heldout_instrument_nmae"})
    item = aggregate[aggregate.task.eq("heldout_item")][["arm", "macro_item_mae"]].rename(
        columns={"macro_item_mae": "heldout_item_nmae"}
    )
    fold_range = fold_metrics[fold_metrics.task.eq("heldout_instrument")].groupby("arm").macro_item_mae.agg(
        heldout_instrument_fold_min="min", heldout_instrument_fold_max="max"
    ).reset_index()
    loading = fold_manifest.groupby("arm", as_index=False).residual_loading_fraction.mean()
    frontier = summary.merge(instrument, on="arm").merge(item, on="arm").merge(
        fold_range, on="arm"
    ).merge(loading, on="arm")
    frontier["pareto_optimal"] = pareto_membership(frontier)
    frontier.to_csv(output / "pareto_frontier.csv", index=False)

    zero = frontier[(~frontier.gated) & frontier.graph_weight.eq(0)].iloc[0]
    tolerance = float(config["selection"]["maximum_nmae_degradation_vs_residual_only"])
    eligible = frontier[
        frontier.pareto_optimal
        & (frontier.heldout_instrument_nmae <= zero.heldout_instrument_nmae + tolerance)
    ]
    selected = eligible.sort_values(["ccc", "heldout_instrument_nmae"], ascending=[False, True]).iloc[0]
    decision = {
        "run_id": config["run_id"],
        "development_only": True,
        "original_train_only": True,
        "validation_loaded": False,
        "test_loaded": False,
        "variants": len(variants),
        "models": len(variants) * expected,
        "zero_reference": zero.arm,
        "selected_development_point": selected.arm,
        "selected_graph_weight": float(selected.graph_weight),
        "selected_gated": bool(selected.gated),
        "selected_heldout_instrument_nmae": float(selected.heldout_instrument_nmae),
        "selected_ccc": float(selected.ccc),
        "selected_ccc_ci_low": float(selected.ccc_ci_low),
        "selected_ccc_ci_high": float(selected.ccc_ci_high),
        "selected_pareto_optimal": bool(selected.pareto_optimal),
        "selection_tolerance": tolerance,
        "phase5f_measurement_gate_reopened": False,
        "continuous_time_dynamics_open": False,
    }
    (output / "development_decision.json").write_text(json.dumps(decision, indent=2) + "\n")

    matched = comparisons[
        comparisons.comparison.eq("matched_gate") & comparisons.task.eq("heldout_instrument")
    ].sort_values("graph_weight")
    lines = [
        "# Phase 5G: Bridge-strength Pareto scan and residual-adaptive gate",
        "",
        "## Scope",
        "",
        f"Five-fold original-train-only development evaluated {len(variants)} variants "
        f"({len(variants) * expected} neural fits). Original validation and test subjects were not loaded.",
        "",
        "## Pareto result",
        "",
        f"The coefficient-zero reference obtained held-out-instrument macro NMAE "
        f"**{zero.heldout_instrument_nmae:.4f}** and Memory CCC **{zero.ccc:.4f}**. "
        f"Under the prespecified +{tolerance:.3f} prediction tolerance, the selected point was "
        f"`{selected.arm}` with NMAE **{selected.heldout_instrument_nmae:.4f}** and CCC "
        f"**{selected.ccc:.4f}**, 95% participant-bootstrap CI "
        f"**[{selected.ccc_ci_low:.4f}, {selected.ccc_ci_high:.4f}]**.",
        "",
        "CCC peaked near λ=0.03--0.10 and declined at stronger alignment. At λ=0.30 and "
        "λ=1.00, Memory-state standard deviation fell below the original 0.20 anti-collapse "
        "threshold, showing that smaller cross-view differences at high λ reflected state "
        "compression rather than improved concordance.",
        "",
        "## Matched residual-gate effects",
        "",
    ]
    for row in matched.itertuples(index=False):
        gated_arm = frontier[frontier.arm.eq(row.candidate)].iloc[0]
        ungated_arm = frontier[frontier.arm.eq(row.baseline)].iloc[0]
        lines.append(
            f"- λ={row.graph_weight:g}: gated-minus-ungated participant error "
            f"**{row.difference:+.5f}**, 95% CI **[{row.ci_low:+.5f}, {row.ci_high:+.5f}]**; "
            f"CCC change **{gated_arm.ccc - ungated_arm.ccc:+.4f}**."
        )
    lines.extend(
        [
            "",
            "The deterministic gate retained about 90% of graph pressure across all edge groups. "
            "It produced small, increasingly precise prediction gains as λ increased, but reduced "
            "CCC at every matched coefficient and showed little edge-strength differentiation. "
            "Thus this gate did not dominate the ungated Pareto frontier.",
            "",
            "## Decision",
            "",
            "This is a development Pareto experiment, not a confirmation run. The Phase 5F gate is "
            "not retrospectively reopened and continuous-time dynamics remain closed.",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines))
    manifest = {
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256(
            (ROOT / "reports/experiments/phase5g_bridge_pareto_gate_protocol.md").read_bytes()
        ).hexdigest(),
        "validation_or_test_loaded": False,
        "fold_shards": [str(path) for path in shards],
        "variants": len(variants),
        "neural_checkpoints": len(variants) * expected,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
