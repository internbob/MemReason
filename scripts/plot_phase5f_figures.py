#!/usr/bin/env python3
"""Create standalone publication figures for the Phase 5F experiment.

Every figure is exported independently as vector PDF and 450-dpi PNG.  The
script only consumes frozen Phase 5F result artifacts and does not refit any
model or alter experiment outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import seaborn as sns


OKABE_ITO = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
}

ARM_LABELS = {
    "fixed_domain_score": "Fixed score",
    "unconstrained_autoencoder": "Unconstrained AE",
    "hierarchical_shared": "Shared only",
    "hierarchical_residual": "Residual only",
    "hierarchical_bridge": "Shared + residual + graph",
}

ARM_COLORS = {
    "fixed_domain_score": OKABE_ITO["black"],
    "unconstrained_autoencoder": OKABE_ITO["orange"],
    "hierarchical_shared": OKABE_ITO["sky"],
    "hierarchical_residual": OKABE_ITO["purple"],
    "hierarchical_bridge": OKABE_ITO["green"],
}

TASK_LABELS = {
    "heldout_item": "Held-out item",
    "heldout_instrument": "Held-out instrument",
}


def configure_style() -> None:
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
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=450, bbox_inches="tight")
    plt.close(fig)


def add_panel_title(ax: plt.Axes, title: str, subtitle: str | None = None) -> None:
    ax.set_title(title, loc="left", fontweight="bold", pad=24)
    if subtitle:
        ax.text(0, 1.01, subtitle, transform=ax.transAxes, va="bottom", color="#555555", fontsize=7)


def figure_connection_network(graph: pd.DataFrame, out_dir: Path) -> None:
    mean_graph = (
        graph.groupby(["instrument_a", "instrument_b"], as_index=False)
        .agg(edge_weight=("edge_weight", "mean"), concurrent_subjects=("concurrent_subjects", "mean"))
    )
    positions = {
        "ADAS": (0.20, 0.72),
        "CDR": (0.20, 0.25),
        "MMSE": (0.56, 0.16),
        "MOCA": (0.78, 0.63),
        "LOGMEM": (0.53, 0.88),
    }
    fig, ax = plt.subplots(figsize=(5.2, 4.1))
    positive = mean_graph[mean_graph["edge_weight"] > 0]
    maximum = positive["edge_weight"].max()
    for row in mean_graph.itertuples(index=False):
        x1, y1 = positions[row.instrument_a]
        x2, y2 = positions[row.instrument_b]
        if row.edge_weight > 0:
            width = 0.8 + 6.0 * row.edge_weight / maximum
            alpha = 0.28 + 0.65 * row.edge_weight / maximum
            ax.plot([x1, x2], [y1, y2], color=OKABE_ITO["blue"], lw=width, alpha=alpha, zorder=1)
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(
                mx,
                my,
                f"{row.edge_weight:.3f}",
                ha="center",
                va="center",
                fontsize=6.5,
                bbox={"boxstyle": "round,pad=0.14", "fc": "white", "ec": "none", "alpha": 0.9},
                zorder=3,
            )
        elif "LOGMEM" in (row.instrument_a, row.instrument_b):
            ax.plot([x1, x2], [y1, y2], color="#B8B8B8", lw=0.6, ls=(0, (2, 3)), alpha=0.55, zorder=0)

    for name, (x, y) in positions.items():
        isolated = name == "LOGMEM"
        circle = patches.Circle(
            (x, y),
            0.075,
            facecolor="white" if isolated else OKABE_ITO["sky"],
            edgecolor="#777777" if isolated else OKABE_ITO["blue"],
            linewidth=1.4,
            linestyle="--" if isolated else "-",
            zorder=4,
        )
        ax.add_patch(circle)
        ax.text(x, y, name, ha="center", va="center", fontweight="bold", fontsize=8, zorder=5)
    add_panel_title(
        ax,
        "Instrument connection graph",
        "Edge width and labels show mean fold-specific bridge weight; dashed edges are unsupported",
    )
    ax.text(
        positions["LOGMEM"][0],
        positions["LOGMEM"][1] + 0.105,
        "no usable dates",
        ha="center",
        fontsize=6.5,
        color="#666666",
    )
    ax.set_xlim(0.05, 0.93)
    ax.set_ylim(0.02, 1.02)
    ax.set_aspect("equal")
    ax.axis("off")
    save_figure(fig, out_dir, "figure_01_connection_network")


def figure_prediction_coherence_pareto(
    aggregate: pd.DataFrame, cross_summary: pd.DataFrame, out_dir: Path
) -> None:
    prediction = aggregate[aggregate["task"] == "heldout_instrument"][
        ["arm", "macro_item_mae"]
    ]
    data = prediction.merge(cross_summary[["arm", "ccc"]], on="arm", how="inner")
    fig, ax = plt.subplots(figsize=(4.8, 3.7))
    for row in data.itertuples(index=False):
        ax.scatter(
            row.macro_item_mae,
            row.ccc,
            s=76 if row.arm == "hierarchical_bridge" else 52,
            color=ARM_COLORS[row.arm],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        offsets = {
            "hierarchical_shared": (5, -13),
            "hierarchical_residual": (5, 7),
            "hierarchical_bridge": (6, 5),
        }
        ax.annotate(
            ARM_LABELS[row.arm],
            (row.macro_item_mae, row.ccc),
            xytext=offsets[row.arm],
            textcoords="offset points",
            fontsize=7,
        )
    ax.axhline(0.70, color="#777777", lw=0.9, ls="--")
    ax.text(ax.get_xlim()[1], 0.705, "Frozen CCC gate = 0.70", ha="right", va="bottom", fontsize=6.5, color="#666666")
    ax.annotate(
        "preferred direction",
        xy=(0.404, 0.665),
        xytext=(0.422, 0.535),
        arrowprops={"arrowstyle": "->", "lw": 0.8},
        fontsize=7,
    )
    ax.set_xlabel("Held-out-instrument macro NMAE (lower is better)")
    ax.set_ylabel("Cross-view CCC (higher is better)")
    ax.set_xlim(data["macro_item_mae"].min() - 0.008, data["macro_item_mae"].max() + 0.012)
    ax.set_ylim(0.38, 0.73)
    add_panel_title(ax, "Prediction–coherence trade-off", "Only hierarchical arms define cross-view latent states")
    sns.despine(ax=ax)
    save_figure(fig, out_dir, "figure_02_prediction_coherence_pareto")


def figure_cross_view_hexbin(states: pd.DataFrame, out_dir: Path) -> None:
    data = states[states["arm"] == "hierarchical_bridge"]
    ccc = data[["view_a", "view_b"]].cov().iloc[0, 1]
    ccc = 2 * ccc / (data["view_a"].var() + data["view_b"].var() + (data["view_a"].mean() - data["view_b"].mean()) ** 2)
    lo = min(data["view_a"].min(), data["view_b"].min())
    hi = max(data["view_a"].max(), data["view_b"].max())
    fig, ax = plt.subplots(figsize=(4.3, 4.0))
    hb = ax.hexbin(data["view_a"], data["view_b"], gridsize=52, mincnt=1, bins="log", cmap="cividis")
    ax.plot([lo, hi], [lo, hi], color="white", lw=2.2, alpha=0.9)
    ax.plot([lo, hi], [lo, hi], color="#333333", lw=0.8, ls="--")
    cb = fig.colorbar(hb, ax=ax, pad=0.02)
    cb.set_label("Observation count (log scale)")
    ax.text(0.04, 0.94, f"CCC = {ccc:.3f}\nn = {len(data):,} paired views", transform=ax.transAxes, va="top", bbox={"boxstyle": "round", "fc": "white", "ec": "#BBBBBB", "alpha": 0.92})
    ax.set_xlabel("Memory state from view A")
    ax.set_ylabel("Memory state from view B")
    add_panel_title(ax, "Cross-view latent-state agreement", "Hierarchical bridge; held-out subjects across five folds")
    sns.despine(ax=ax)
    save_figure(fig, out_dir, "figure_03_cross_view_posterior_hexbin")


def figure_bland_altman(states: pd.DataFrame, out_dir: Path) -> None:
    data = states[states["arm"] == "hierarchical_bridge"].copy()
    mean_state = (data["view_a"] + data["view_b"]) / 2
    difference = data["view_a"] - data["view_b"]
    bias = difference.mean()
    loa_low, loa_high = bias - 1.96 * difference.std(ddof=1), bias + 1.96 * difference.std(ddof=1)
    fig, ax = plt.subplots(figsize=(5.0, 3.7))
    hb = ax.hexbin(mean_state, difference, gridsize=55, mincnt=1, bins="log", cmap="cividis")
    cb = fig.colorbar(hb, ax=ax, pad=0.02)
    cb.set_label("Observation count (log scale)")
    ax.axhline(bias, color=OKABE_ITO["vermillion"], lw=1.4, label=f"Mean bias = {bias:.3f}")
    ax.axhline(loa_low, color="#555555", lw=1.0, ls="--", label=f"95% limits = [{loa_low:.3f}, {loa_high:.3f}]")
    ax.axhline(loa_high, color="#555555", lw=1.0, ls="--")
    ax.set_xlabel("Mean of paired memory states")
    ax.set_ylabel("View A − view B")
    ax.legend(frameon=False, loc="lower right")
    add_panel_title(ax, "Bland–Altman agreement map", f"Hierarchical bridge; n = {len(data):,} paired views")
    sns.despine(ax=ax)
    save_figure(fig, out_dir, "figure_04_bland_altman")


def figure_item_benefit_heatmap(
    item_metrics: pd.DataFrame, predictions: pd.DataFrame, out_dir: Path
) -> None:
    item_map = predictions[["item", "instrument", "domain"]].drop_duplicates("item")
    selected = item_metrics[item_metrics["arm"].isin(["fixed_domain_score", "hierarchical_bridge"])]
    pivot = selected.pivot_table(index=["item", "domain"], columns=["task", "arm"], values="mae")
    benefit = pd.DataFrame(index=pivot.index)
    for task in TASK_LABELS:
        benefit[task] = pivot[(task, "fixed_domain_score")] - pivot[(task, "hierarchical_bridge")]
    benefit = benefit.reset_index().merge(item_map, on=["item", "domain"], how="left")
    benefit = benefit.sort_values(["instrument", "domain", "item"]).reset_index(drop=True)
    matrix = benefit[list(TASK_LABELS)].to_numpy()
    max_abs = np.nanmax(np.abs(matrix))
    fig = plt.figure(figsize=(6.1, 12.2))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 0.09, 0.045], wspace=0.08)
    ax = fig.add_subplot(grid[0, 0])
    group_ax = fig.add_subplot(grid[0, 1], sharey=ax)
    color_ax = fig.add_subplot(grid[0, 2])
    image = ax.imshow(matrix, aspect="auto", cmap="RdBu", norm=TwoSlopeNorm(vcenter=0, vmin=-max_abs, vmax=max_abs))
    ax.set_xticks(range(2), [TASK_LABELS[t] for t in TASK_LABELS])
    ax.set_yticks(np.arange(len(benefit)), benefit["item"], fontsize=5.4)
    ax.tick_params(axis="x", top=False, labeltop=False, bottom=True, labelbottom=True)
    boundaries = benefit.groupby("instrument", sort=False).size().cumsum().to_numpy()[:-1] - 0.5
    for y in boundaries:
        ax.axhline(y, color="white", lw=1.6)
    starts = benefit.groupby("instrument", sort=False).size()
    start = 0
    for instrument, count in starts.items():
        center = start + (count - 1) / 2
        group_ax.text(0.5, center, instrument, rotation=90, va="center", ha="center", fontweight="bold", fontsize=7)
        start += count
    group_ax.set_xlim(0, 1)
    group_ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    for spine in group_ax.spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(image, cax=color_ax)
    cb.set_label("NMAE reduction vs fixed score")
    add_panel_title(ax, "Item-level benefit map", "Positive values favor the hierarchical bridge; all 70 frozen items")
    ax.set_xlabel("")
    ax.set_ylabel("")
    save_figure(fig, out_dir, "figure_05_item_benefit_heatmap")


def subject_level_differences(predictions: pd.DataFrame) -> pd.DataFrame:
    data = predictions[predictions["arm"].isin(["fixed_domain_score", "hierarchical_bridge"])]
    subject = (
        data.groupby(["task", "fold", "subject_id", "arm"], as_index=False)["absolute_error"].mean()
        .pivot(index=["task", "fold", "subject_id"], columns="arm", values="absolute_error")
        .dropna()
        .reset_index()
    )
    subject["benefit"] = subject["fixed_domain_score"] - subject["hierarchical_bridge"]
    return subject


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, replicates: int = 2000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    means = np.empty(replicates)
    for start in range(0, replicates, 200):
        stop = min(start + 200, replicates)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return values.mean(), low, high


def figure_fold_forest(subject_diff: pd.DataFrame, out_dir: Path) -> None:
    rng = np.random.default_rng(20261029)
    rows = []
    for task in TASK_LABELS:
        task_data = subject_diff[subject_diff["task"] == task]
        for fold in sorted(task_data["fold"].unique()):
            values = task_data.loc[task_data["fold"] == fold, "benefit"].to_numpy()
            mean, low, high = bootstrap_mean_ci(values, rng)
            rows.append({"task": task, "fold": fold, "mean": mean, "low": low, "high": high, "subjects": len(values)})
    summary = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharex=True, sharey=True)
    for ax, task in zip(axes, TASK_LABELS):
        data = summary[summary["task"] == task].sort_values("fold", ascending=False)
        y = np.arange(len(data))
        ax.errorbar(data["mean"], y, xerr=[data["mean"] - data["low"], data["high"] - data["mean"]], fmt="o", color=OKABE_ITO["green"], ecolor="#555555", capsize=2.5, lw=1.0)
        ax.axvline(0, color="#777777", ls="--", lw=0.8)
        ax.set_yticks(y, [f"Fold {fold}  (n={n:,})" for fold, n in zip(data["fold"], data["subjects"])])
        ax.set_title(TASK_LABELS[task], fontweight="bold")
        ax.set_xlabel("Mean NMAE reduction vs fixed score")
        sns.despine(ax=ax)
    fig.suptitle("Cross-validation stability", x=0.07, ha="left", fontweight="bold", fontsize=10)
    fig.text(0.07, 0.92, "Points are subject-level means; bars are 95% subject-bootstrap CIs", fontsize=7, color="#555555")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save_figure(fig, out_dir, "figure_06_fold_forest")


def half_violin(ax: plt.Axes, values: np.ndarray, position: float, color: str, rng: np.random.Generator) -> None:
    violin = ax.violinplot(values, positions=[position], widths=0.72, showmeans=False, showmedians=False, showextrema=False)
    body = violin["bodies"][0]
    body.set_facecolor(color)
    body.set_edgecolor("none")
    body.set_alpha(0.55)
    vertices = body.get_paths()[0].vertices
    vertices[:, 0] = np.minimum(vertices[:, 0], position)
    quartiles = np.quantile(values, [0.25, 0.5, 0.75])
    ax.plot([position + 0.06, position + 0.06], [quartiles[0], quartiles[2]], color="#333333", lw=1.4)
    ax.scatter(position + 0.06, quartiles[1], s=13, color="#111111", zorder=4)
    sampled = values if len(values) <= 500 else rng.choice(values, 500, replace=False)
    jitter = rng.uniform(0.10, 0.34, size=len(sampled))
    ax.scatter(position + jitter, sampled, s=4, alpha=0.18, color=color, linewidth=0, rasterized=True)


def figure_subject_raincloud(subject_diff: pd.DataFrame, out_dir: Path) -> None:
    rng = np.random.default_rng(20261029)
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    pooled = subject_diff["benefit"].to_numpy()
    display_low, display_high = np.quantile(pooled, [0.005, 0.995])
    outside = int(((pooled < display_low) | (pooled > display_high)).sum())
    for position, task in enumerate(TASK_LABELS):
        values = subject_diff.loc[subject_diff["task"] == task, "benefit"].to_numpy()
        display_values = np.clip(values, display_low, display_high)
        half_violin(ax, display_values, position, OKABE_ITO["green"] if position else OKABE_ITO["blue"], rng)
    ax.axhline(0, color="#777777", lw=0.8, ls="--")
    ax.set_xticks([0, 1], [f"{TASK_LABELS[t]}\n(n={sum(subject_diff['task'] == t):,})" for t in TASK_LABELS])
    ax.set_ylabel("Subject-level NMAE reduction vs fixed score")
    ax.set_ylim(display_low - 0.03, display_high + 0.03)
    add_panel_title(
        ax,
        "Who benefits from the hierarchical bridge?",
        f"Half violins show the 0.5th–99.5th percentile ({outside} tail values clipped); bars show IQR and median",
    )
    sns.despine(ax=ax)
    save_figure(fig, out_dir, "figure_07_subject_raincloud")


def subject_bin_ci(data: pd.DataFrame, value: str) -> pd.DataFrame:
    per_subject = data.groupby(["arm", "bin", "subject_id"], observed=True, as_index=False)[value].mean()
    return (
        per_subject.groupby(["arm", "bin"], observed=True)[value]
        .agg(mean="mean", sem="sem", subjects="size")
        .reset_index()
    )


def figure_uncertainty_consistency(states: pd.DataFrame, out_dir: Path) -> None:
    data = states.copy()
    data["combined_sd"] = np.sqrt(data["view_a_sd"] ** 2 + data["view_b_sd"] ** 2)
    data["bin"] = data.groupby("arm")["combined_sd"].transform(lambda x: pd.qcut(x, 8, duplicates="drop"))
    x_summary = data.groupby(["arm", "bin"], observed=True)["combined_sd"].mean().rename("combined_sd").reset_index()
    y_summary = subject_bin_ci(data, "absolute_difference")
    summary = x_summary.merge(y_summary, on=["arm", "bin"])
    fig, ax = plt.subplots(figsize=(5.1, 3.8))
    for arm in ["hierarchical_shared", "hierarchical_residual", "hierarchical_bridge"]:
        arm_data = summary[summary["arm"] == arm].sort_values("combined_sd")
        ax.errorbar(arm_data["combined_sd"], arm_data["mean"], yerr=1.96 * arm_data["sem"], marker="o", ms=3.8, lw=1.1, capsize=2, color=ARM_COLORS[arm], label=ARM_LABELS[arm])
    x = np.linspace(summary["combined_sd"].min(), summary["combined_sd"].max(), 100)
    ax.plot(x, np.sqrt(2 / np.pi) * x, color="#555555", ls="--", lw=0.9, label="Gaussian calibration reference")
    ax.set_xlabel("Combined posterior SD")
    ax.set_ylabel("Mean absolute cross-view difference")
    ax.legend(frameon=False, ncol=2)
    add_panel_title(ax, "Posterior uncertainty consistency", "Up to eight within-arm uncertainty bins; error bars are 95% subject-level CIs")
    sns.despine(ax=ax)
    save_figure(fig, out_dir, "figure_08_uncertainty_consistency")


def figure_edge_sensitivity(states: pd.DataFrame, out_dir: Path) -> None:
    data = states.copy()
    data["edge_group"] = pd.cut(data["edge_weight"], bins=[0, 0.42, 0.52, 1], labels=["Low (~0.395)", "Medium (~0.444)", "High (~0.600)"], include_lowest=True)
    per_subject = data.groupby(["arm", "edge_group", "subject_id"], observed=True, as_index=False)["absolute_difference"].mean()
    summary = per_subject.groupby(["arm", "edge_group"], observed=True)["absolute_difference"].agg(mean="mean", sem="sem", subjects="size").reset_index()
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    x = np.arange(3)
    offsets = [-0.08, 0, 0.08]
    for offset, arm in zip(offsets, ["hierarchical_shared", "hierarchical_residual", "hierarchical_bridge"]):
        arm_data = summary[summary["arm"] == arm].set_index("edge_group").reindex(data["edge_group"].cat.categories)
        ax.errorbar(x + offset, arm_data["mean"], yerr=1.96 * arm_data["sem"], marker="o", ms=4, lw=1.1, capsize=2, color=ARM_COLORS[arm], label=ARM_LABELS[arm])
    ax.set_xticks(x, data["edge_group"].cat.categories)
    ax.set_xlabel("Connection strength group")
    ax.set_ylabel("Mean absolute cross-view difference")
    ax.legend(frameon=False)
    add_panel_title(ax, "Sensitivity to graph connection strength", "Lower disagreement is better; error bars are 95% subject-level CIs")
    sns.despine(ax=ax)
    save_figure(fig, out_dir, "figure_09_edge_weight_sensitivity")


def rounded_box(ax: plt.Axes, xy: tuple[float, float], width: float, height: float, text: str, fc: str, ec: str, fontsize: float = 8) -> None:
    box = patches.FancyBboxPatch(xy, width, height, boxstyle="round,pad=0.018,rounding_size=0.025", facecolor=fc, edgecolor=ec, linewidth=1.2)
    ax.add_patch(box)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=fontsize, fontweight="bold")


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], label: str | None = None, color: str = "#555555") -> None:
    ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "-|>", "lw": 1.1, "color": color, "shrinkA": 3, "shrinkB": 3})
    if label:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.025, label, ha="center", fontsize=6.5, color=color)


def figure_method_schematic(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    instruments = [("ADAS", 0.08), ("CDR", 0.25), ("MMSE", 0.42), ("MOCA", 0.59), ("LOGMEM", 0.76)]
    for name, x in instruments:
        rounded_box(ax, (x, 0.79), 0.12, 0.10, name, "#F3F3F3", "#888888", 7.5)
        arrow(ax, (x + 0.06, 0.79), (0.42, 0.70))
    rounded_box(ax, (0.33, 0.58), 0.30, 0.12, "Item responses\n+ frozen domain map", "#DCEEFF", OKABE_ITO["blue"])
    rounded_box(ax, (0.08, 0.35), 0.22, 0.12, "Shared latent state\n$z_{subject,visit,domain}$", "#DDF3EA", OKABE_ITO["green"])
    rounded_box(ax, (0.39, 0.35), 0.22, 0.12, "Instrument residual\n$r_{instrument,domain}$", "#F7E4F1", OKABE_ITO["purple"])
    rounded_box(ax, (0.70, 0.35), 0.22, 0.12, "Connection graph\n$w_{A,B}$", "#FFF0D7", OKABE_ITO["orange"])
    arrow(ax, (0.42, 0.58), (0.19, 0.47), "hierarchy")
    arrow(ax, (0.48, 0.58), (0.50, 0.47), "scale-specific")
    arrow(ax, (0.55, 0.58), (0.81, 0.47), "overlap × timing × support")
    rounded_box(ax, (0.23, 0.12), 0.50, 0.13, "Bridge posterior\nshared state + graph-weighted residual", "#E8F4EF", OKABE_ITO["green"], 9)
    arrow(ax, (0.19, 0.35), (0.36, 0.25))
    arrow(ax, (0.50, 0.35), (0.48, 0.25))
    arrow(ax, (0.81, 0.35), (0.62, 0.25))
    ax.text(0.48, 0.055, "Held-out item prediction   •   held-out instrument prediction   •   cross-view coherence", ha="center", fontsize=7.5)
    ax.set_title("Hierarchical shared-state bridge", loc="left", fontweight="bold", fontsize=11, pad=4)
    ax.text(0.0, 0.96, "Fold-fitted measurement model; graph weights never use evaluation outcomes", transform=ax.transAxes, color="#555555", fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    save_figure(fig, out_dir, "figure_10_method_schematic")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("outputs/phase5f_hierarchical_bridge_dev_20261028"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    results_dir = args.results_dir.resolve()
    out_dir = (args.output_dir or results_dir / "figures").resolve()
    configure_style()

    aggregate = pd.read_csv(results_dir / "aggregate_metrics.csv")
    graph = pd.read_csv(results_dir / "connection_graph.csv")
    cross_summary = pd.read_csv(results_dir / "cross_view_summary.csv")
    states = pd.read_csv(results_dir / "cross_view_memory_states.csv")
    predictions = pd.read_csv(results_dir / "crossfit_predictions.csv")
    item_metrics = pd.read_csv(results_dir / "item_metrics.csv")

    figure_connection_network(graph, out_dir)
    figure_prediction_coherence_pareto(aggregate, cross_summary, out_dir)
    figure_cross_view_hexbin(states, out_dir)
    figure_bland_altman(states, out_dir)
    figure_item_benefit_heatmap(item_metrics, predictions, out_dir)
    subject_diff = subject_level_differences(predictions)
    figure_fold_forest(subject_diff, out_dir)
    figure_subject_raincloud(subject_diff, out_dir)
    figure_uncertainty_consistency(states, out_dir)
    figure_edge_sensitivity(states, out_dir)
    figure_method_schematic(out_dir)
    print(f"Created 10 standalone figures in {out_dir}")


if __name__ == "__main__":
    main()
