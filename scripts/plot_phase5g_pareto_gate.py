#!/usr/bin/env python3
"""Plot standalone publication figures for the Phase 5G experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#E69F00"
PURPLE = "#CC79A7"
BLACK = "#111111"


def configure() -> None:
    sns.set_theme(style="ticks", context="paper")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{stem}.png", dpi=450, bbox_inches="tight")
    plt.close(fig)


def title(ax: plt.Axes, heading: str, subtitle: str) -> None:
    ax.set_title(heading, loc="left", fontweight="bold", pad=24)
    ax.text(0, 1.01, subtitle, transform=ax.transAxes, va="bottom", fontsize=7, color="#555555")


def plot_pareto(frontier: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.1))
    for gated, group in frontier.groupby("gated"):
        marker, color, label = ("D", GREEN, "Residual-adaptive gate") if gated else ("o", BLUE, "Ungated bridge")
        ax.errorbar(
            group.heldout_instrument_nmae,
            group.ccc,
            yerr=[group.ccc - group.ccc_ci_low, group.ccc_ci_high - group.ccc],
            fmt=marker,
            color=color,
            ecolor=color,
            ms=5.5,
            capsize=2,
            lw=0,
            elinewidth=0.9,
            label=label,
            zorder=3,
        )
        for row in group.itertuples(index=False):
            offset = (4, -11) if gated else (4, 4)
            ax.annotate(f"λ={row.graph_weight:g}", (row.heldout_instrument_nmae, row.ccc), xytext=offset, textcoords="offset points", fontsize=6.2, color=color)
    pareto = frontier[frontier.pareto_optimal].sort_values("heldout_instrument_nmae")
    ax.plot(pareto.heldout_instrument_nmae, pareto.ccc, color=BLACK, lw=1.0, ls="--", label="Empirical Pareto frontier", zorder=2)
    ax.axhline(0.70, color="#777777", lw=0.8, ls=(0, (4, 3)))
    ax.text(ax.get_xlim()[1], 0.704, "CCC target", ha="right", va="bottom", fontsize=6.5, color="#666666")
    ax.set_xlabel("Held-out-instrument macro NMAE (lower is better)")
    ax.set_ylabel("Cross-view Memory CCC (higher is better)")
    title(ax, "Bridge-strength Pareto frontier", "Points are five-fold out-of-fold estimates; vertical bars are 95% participant-bootstrap CIs")
    ax.legend(frameon=False, loc="lower right")
    sns.despine(ax=ax)
    save(fig, output, "figure_01_bridge_strength_pareto")


def plot_gate_prediction(comparisons: pd.DataFrame, output: Path) -> None:
    data = comparisons[(comparisons.comparison == "matched_gate") & (comparisons.task == "heldout_instrument")].sort_values("graph_weight")
    fig, ax = plt.subplots(figsize=(4.8, 3.7))
    ax.errorbar(
        data.graph_weight,
        data.difference,
        yerr=[data.difference - data.ci_low, data.ci_high - data.difference],
        marker="D",
        ms=5,
        color=GREEN,
        ecolor="#555555",
        capsize=3,
        lw=1.2,
    )
    ax.axhline(0, color="#777777", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xticks(data.graph_weight, [f"{value:g}" for value in data.graph_weight])
    ax.set_xlabel("Bridge coefficient λ (log scale)")
    ax.set_ylabel("Gated − ungated subject MAE")
    title(ax, "Predictive effect of residual-adaptive gating", "Negative values favor gating; bars are 95% subject-bootstrap CIs")
    sns.despine(ax=ax)
    save(fig, output, "figure_02_gate_prediction_effect")


def plot_gate_coherence(frontier: pd.DataFrame, output: Path) -> None:
    data = frontier[frontier.graph_weight > 0].sort_values("graph_weight")
    fig, ax = plt.subplots(figsize=(4.8, 3.7))
    for gated, group in data.groupby("gated"):
        label = "Residual-adaptive gate" if gated else "Ungated bridge"
        color, marker = (GREEN, "D") if gated else (BLUE, "o")
        ax.errorbar(
            group.graph_weight,
            group.ccc,
            yerr=[group.ccc - group.ccc_ci_low, group.ccc_ci_high - group.ccc],
            color=color,
            marker=marker,
            ms=4.5,
            lw=1.1,
            capsize=2,
            label=label,
        )
    ax.axhline(0.70, color="#777777", lw=0.8, ls="--", label="CCC target")
    ax.set_xscale("log")
    weights = sorted(data.graph_weight.unique())
    ax.set_xticks(weights, [f"{value:g}" for value in weights])
    ax.set_xlabel("Bridge coefficient λ (log scale)")
    ax.set_ylabel("Cross-view Memory CCC")
    title(ax, "Coherence response to bridge strength", "Vertical bars are 95% participant-bootstrap CIs")
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    save(fig, output, "figure_03_gate_coherence_response")


def plot_gate_heatmap(states: pd.DataFrame, output: Path) -> None:
    data = states[states.gated].copy()
    data["edge_group"] = pd.cut(
        data.edge_weight,
        bins=[0, 0.42, 0.52, 1],
        labels=["Low (~0.395)", "Medium (~0.444)", "High (~0.600)"],
        include_lowest=True,
    )
    matrix = data.pivot_table(index="graph_weight", columns="edge_group", values="residual_gate", aggfunc="mean", observed=True).sort_index()
    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    sns.heatmap(matrix, cmap="cividis", vmin=0.25, vmax=1.0, annot=True, fmt=".3f", linewidths=0.7, cbar_kws={"label": "Mean residual gate"}, ax=ax)
    ax.set_xlabel("Supported connection-strength group")
    ax.set_ylabel("Bridge coefficient λ")
    title(ax, "Residual-evidence gating pattern", "A value of 1 retains full graph pressure; the frozen minimum is 0.25")
    save(fig, output, "figure_04_residual_gate_heatmap")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("outputs/phase5g_bridge_pareto_gate_dev_20261029"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    results = args.results_dir.resolve()
    output = (args.output_dir or results / "figures").resolve()
    configure()
    frontier = pd.read_csv(results / "pareto_frontier.csv")
    comparisons = pd.read_csv(results / "subject_bootstrap.csv")
    states = pd.read_csv(results / "cross_view_memory_states.csv")
    plot_pareto(frontier, output)
    plot_gate_prediction(comparisons, output)
    plot_gate_coherence(frontier, output)
    plot_gate_heatmap(states, output)
    print(f"Created 4 standalone figures in {output}")


if __name__ == "__main__":
    main()
