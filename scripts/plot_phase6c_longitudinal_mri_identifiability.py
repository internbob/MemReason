#!/usr/bin/env python3
"""Create separate publication-quality Phase 6C figures."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from memreason_rl.data.common import ROOT


OUTPUT = ROOT / "outputs/phase6c_longitudinal_mri_identifiability_dev_20261030"
FIGURES = OUTPUT / "figures"
ORDER = ["cognition_only", "static_mri", "true_longitudinal_mri", "date_permuted_dynamics",
         "sign_flipped_dynamics", "subject_shuffled_dynamics"]
LABELS = {
    "cognition_only": "Cognition only", "static_mri": "Static MRI",
    "true_longitudinal_mri": "True longitudinal MRI",
    "date_permuted_dynamics": "Date-permuted dynamics",
    "sign_flipped_dynamics": "Sign-flipped intervention",
    "subject_shuffled_dynamics": "Subject-shuffled dynamics",
}
COLORS = {
    "cognition_only": "#0072B2", "static_mri": "#56B4E9",
    "true_longitudinal_mri": "#009E73", "date_permuted_dynamics": "#D55E00",
    "sign_flipped_dynamics": "#CC79A7", "subject_shuffled_dynamics": "#E69F00",
}
MARKERS = dict(zip(ORDER, ["o", "s", "D", "^", "v", "P"]))


def style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 10,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "lines.linewidth": 1.5, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf")
    fig.savefig(FIGURES / f"{name}.png", dpi=450)
    plt.close(fig)


def grouped_effects(predictions: pd.DataFrame, group: pd.Series, name: str) -> pd.DataFrame:
    candidate = predictions[predictions.arm.eq("true_longitudinal_mri")][
        ["sample_id", "subject_id"]].copy()
    candidate[name] = group.to_numpy()
    errors = predictions.pivot_table(index=["sample_id", "subject_id"], columns="arm",
                                     values="composite_absolute_error").reset_index()
    frame = errors.merge(candidate, on=["sample_id", "subject_id"], validate="one_to_one")
    rows = []
    rng = np.random.default_rng(20261030)
    for comparator in ["static_mri", "date_permuted_dynamics"]:
        frame["difference"] = frame.true_longitudinal_mri - frame[comparator]
        values = frame.groupby([name, "subject_id"], observed=True).difference.mean().reset_index()
        for category, local in values.groupby(name, observed=True):
            array = local.difference.to_numpy(float)
            draws = array[rng.integers(0, len(array), size=(2000, len(array)))].mean(1)
            rows.append({
                name: str(category), "comparator": comparator,
                "difference": float(array.mean()), "ci_low": float(np.quantile(draws, .025)),
                "ci_high": float(np.quantile(draws, .975)), "subjects": len(array),
            })
    return pd.DataFrame(rows)


def main() -> None:
    style()
    aggregate = pd.read_csv(OUTPUT / "aggregate_metrics.csv")
    folds = pd.read_csv(OUTPUT / "metrics_by_fold.csv")
    horizon = pd.read_csv(OUTPUT / "metrics_by_horizon.csv")
    bootstrap = pd.read_csv(OUTPUT / "paired_subject_bootstrap.csv")
    predictions = pd.read_csv(OUTPUT / "crossfit_predictions.csv")
    aggregate_table = aggregate.set_index("arm")

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for y, arm in enumerate(ORDER):
        local = folds[folds.arm.eq(arm)].sort_values("fold")
        ax.scatter(local.composite_mae, np.full(len(local), y), color=COLORS[arm], alpha=.42,
                   s=22, marker=MARKERS[arm], edgecolor="none")
        ax.scatter(aggregate_table.loc[arm, "composite_mae"], y, color=COLORS[arm],
                   marker=MARKERS[arm], s=56, edgecolor="white", linewidth=.7, zorder=3)
    ax.set_yticks(range(len(ORDER)), [LABELS[arm] for arm in ORDER]); ax.invert_yaxis()
    ax.set_xlabel("Future observed Memory composite MAE (lower is better)")
    ax.set_title("Longitudinal MRI identifiability across outer folds", loc="left")
    ax.grid(axis="x", color="#dddddd", linewidth=.5)
    save(fig, "figure_01_primary_clinical_endpoint")

    comparators = ["static_mri", "date_permuted_dynamics", "sign_flipped_dynamics",
                   "subject_shuffled_dynamics"]
    primary = bootstrap[bootstrap.metric.eq("composite_absolute_error")].set_index("comparator").loc[comparators]
    fig, ax = plt.subplots(figsize=(6.5, 3.5)); y = np.arange(len(primary))
    colors = [COLORS[arm] for arm in comparators]
    ax.hlines(y, primary.ci_low, primary.ci_high, color=colors, linewidth=2)
    ax.scatter(primary.difference, y, color=colors, s=38, zorder=3)
    ax.axvline(0, color="black", linewidth=.8, linestyle="--")
    ax.set_yticks(y, [f"vs {LABELS[arm]}" for arm in comparators]); ax.invert_yaxis()
    ax.set_xlabel("True longitudinal MRI minus comparator MAE (95% participant-bootstrap CI)")
    ax.set_title("Paired temporal-signal effects", loc="left")
    ax.grid(axis="x", color="#dddddd", linewidth=.5)
    save(fig, "figure_02_paired_temporal_effects")

    fig, ax = plt.subplots(figsize=(6.5, 3.9))
    for arm in ORDER:
        local = horizon[horizon.arm.eq(arm)].sort_values("horizon_months")
        ax.plot(local.horizon_months, local.composite_mae, color=COLORS[arm],
                marker=MARKERS[arm], markersize=4.8, label=LABELS[arm])
    ax.set_xticks([6, 12, 24, 36]); ax.set_xlabel("Forecast horizon (months)")
    ax.set_ylabel("Future observed Memory composite MAE")
    ax.set_title("Temporal-signal comparison by forecast horizon", loc="left")
    ax.legend(frameon=False, ncol=2); ax.grid(color="#e5e5e5", linewidth=.5)
    save(fig, "figure_03_horizon_response")

    candidate = predictions[predictions.arm.eq("true_longitudinal_mri")].copy()
    span_categories = ["0.5–1 y", "1–2 y", "2–4 y", ">4 y"]
    span = pd.cut(candidate.mri_span_years, [.5, 1, 2, 4, np.inf], labels=span_categories,
                  include_lowest=True)
    effects = grouped_effects(predictions, span, "span")
    fig, ax = plt.subplots(figsize=(6.2, 3.7)); offsets = {"static_mri": -.10, "date_permuted_dynamics": .10}
    for comparator, offset in offsets.items():
        local = effects[effects.comparator.eq(comparator)].set_index("span").reindex(span_categories)
        x = np.arange(len(span_categories)) + offset
        ax.errorbar(x, local.difference, yerr=[local.difference-local.ci_low,
                    local.ci_high-local.difference], fmt=MARKERS[comparator],
                    color=COLORS[comparator], capsize=2.5, label=f"vs {LABELS[comparator]}")
    ax.axhline(0, color="black", linewidth=.8, linestyle="--")
    ax.set_xticks(range(len(span_categories)), span_categories)
    ax.set_xlabel("Pre-cutoff MRI observation span"); ax.set_ylabel("Paired MAE difference")
    ax.set_title("Temporal effect by MRI observation span", loc="left")
    ax.legend(frameon=False); ax.grid(axis="y", color="#e5e5e5", linewidth=.5)
    save(fig, "figure_04_effect_by_mri_span")

    count_categories = ["2", "3", "4–5", "≥6"]
    count = pd.cut(candidate.mri_scan_count, [1, 2, 3, 5, np.inf], labels=count_categories)
    effects = grouped_effects(predictions, count, "scan_count")
    fig, ax = plt.subplots(figsize=(6.0, 3.7))
    for comparator, offset in offsets.items():
        local = effects[effects.comparator.eq(comparator)].set_index("scan_count").reindex(count_categories)
        x = np.arange(len(count_categories)) + offset
        ax.errorbar(x, local.difference, yerr=[local.difference-local.ci_low,
                    local.ci_high-local.difference], fmt=MARKERS[comparator],
                    color=COLORS[comparator], capsize=2.5, label=f"vs {LABELS[comparator]}")
    ax.axhline(0, color="black", linewidth=.8, linestyle="--")
    ax.set_xticks(range(len(count_categories)), count_categories)
    ax.set_xlabel("Pre-cutoff MRI scans (n)"); ax.set_ylabel("Paired MAE difference")
    ax.set_title("Temporal effect by longitudinal scan count", loc="left")
    ax.legend(frameon=False); ax.grid(axis="y", color="#e5e5e5", linewidth=.5)
    save(fig, "figure_05_effect_by_scan_count")

    fig, ax = plt.subplots(figsize=(6.1, 4.1))
    ax.axhspan(.90, .98, color="#E6F4EA", alpha=.8, label="Prespecified coverage band")
    for arm in ORDER:
        row = aggregate_table.loc[arm]
        ax.scatter(row.interval_width, row.coverage_95, color=COLORS[arm],
                   marker=MARKERS[arm], s=52, label=LABELS[arm])
    ax.set_xlabel("Mean 95% prediction-interval width"); ax.set_ylabel("Empirical 95% coverage")
    ax.set_title("Clinical interval calibration", loc="left")
    ax.set_ylim(.89, .99); ax.legend(frameon=False, ncol=2, loc="best")
    ax.grid(color="#e5e5e5", linewidth=.5)
    save(fig, "figure_06_calibration_tradeoff")

    (FIGURES / "README.md").write_text(
        "# Phase 6C figures\n\nEach result is an independent vector PDF and 450-DPI PNG; "
        "none is a screenshot or image montage.\n\n"
        "1. `figure_01_primary_clinical_endpoint`: aggregate and five outer-fold MAE.\n"
        "2. `figure_02_paired_temporal_effects`: participant-bootstrap primary effects.\n"
        "3. `figure_03_horizon_response`: 6/12/24/36-month endpoint response.\n"
        "4. `figure_04_effect_by_mri_span`: descriptive effects by observation span.\n"
        "5. `figure_05_effect_by_scan_count`: descriptive effects by scan count.\n"
        "6. `figure_06_calibration_tradeoff`: coverage versus interval width.\n\n"
        "Negative differences favor true longitudinal MRI. Span and count analyses are "
        "mechanism diagnostics, not selection criteria.\n"
    )


if __name__ == "__main__":
    main()
