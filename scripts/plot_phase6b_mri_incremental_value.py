#!/usr/bin/env python3
"""Create separate publication-quality Phase 6B figures."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from memreason_rl.data.common import ROOT


OUTPUT = ROOT / "outputs/phase6b_mri_incremental_value_dev_20261029"
FIGURES = OUTPUT / "figures"
ORDER = ["cognition_only", "metadata_only", "real_mri", "subject_shuffled_mri",
         "time_shuffled_mri", "mri_only"]
LABELS = {
    "cognition_only": "Cognition only", "metadata_only": "MRI metadata only",
    "real_mri": "Real MRI", "subject_shuffled_mri": "Subject-shuffled MRI",
    "time_shuffled_mri": "Time-shuffled MRI", "mri_only": "MRI only",
}
COLORS = {
    "cognition_only": "#0072B2", "metadata_only": "#56B4E9", "real_mri": "#009E73",
    "subject_shuffled_mri": "#D55E00", "time_shuffled_mri": "#CC79A7", "mri_only": "#666666",
}
MARKERS = {"cognition_only": "o", "metadata_only": "s", "real_mri": "D",
           "subject_shuffled_mri": "^", "time_shuffled_mri": "v", "mri_only": "X"}


def style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 10,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "lines.linewidth": 1.5, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf")
    fig.savefig(FIGURES / f"{name}.png", dpi=450)
    plt.close(fig)


def cluster_interval(values: pd.Series, seed: int, repeats: int = 2000) -> tuple[float, float, float]:
    array = values.dropna().to_numpy(float)
    if not len(array):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = array[rng.integers(0, len(array), size=(repeats, len(array)))].mean(1)
    return float(array.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def grouped_effects(predictions: pd.DataFrame, group: pd.Series, group_name: str) -> pd.DataFrame:
    candidate_rows = predictions[predictions.arm.eq("real_mri")][["sample_id", "subject_id"]].copy()
    candidate_rows[group_name] = group.to_numpy()
    error = predictions.pivot_table(index=["sample_id", "subject_id"], columns="arm",
                                    values="composite_absolute_error").reset_index()
    frame = error.merge(candidate_rows, on=["sample_id", "subject_id"], validate="one_to_one")
    rows = []
    for comparator in ["cognition_only", "time_shuffled_mri"]:
        frame["difference"] = frame.real_mri - frame[comparator]
        by_subject = frame.groupby([group_name, "subject_id"], observed=True).difference.mean().reset_index()
        for index, (name, local) in enumerate(by_subject.groupby(group_name, observed=True)):
            mean, low, high = cluster_interval(local.difference, 20261029 + index + 100 * len(rows))
            rows.append({group_name: str(name), "comparator": comparator, "difference": mean,
                         "ci_low": low, "ci_high": high, "subjects": local.subject_id.nunique()})
    return pd.DataFrame(rows)


def main() -> None:
    style()
    aggregate = pd.read_csv(OUTPUT / "aggregate_metrics.csv")
    folds = pd.read_csv(OUTPUT / "metrics_by_fold.csv")
    horizon = pd.read_csv(OUTPUT / "metrics_by_horizon.csv")
    bootstrap = pd.read_csv(OUTPUT / "paired_subject_bootstrap.csv")
    predictions = pd.read_csv(OUTPUT / "crossfit_predictions.csv")

    # Figure 1: aggregate endpoint with outer-fold points.
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    for y, arm in enumerate(ORDER):
        local = folds[folds.arm.eq(arm)].sort_values("fold")
        ax.scatter(local.composite_mae, np.full(len(local), y), color=COLORS[arm], alpha=0.45,
                   s=22, marker=MARKERS[arm], edgecolor="none", label=None)
        mean = float(aggregate.set_index("arm").loc[arm, "composite_mae"])
        ax.scatter([mean], [y], color=COLORS[arm], s=55, marker=MARKERS[arm],
                   edgecolor="white", linewidth=0.7, zorder=3)
    ax.set_yticks(range(len(ORDER)), [LABELS[arm] for arm in ORDER]); ax.invert_yaxis()
    ax.set_xlabel("Future observed Memory composite MAE (lower is better)")
    ax.set_title("Real MRI improves the aggregate endpoint, but temporal ordering is not required", loc="left")
    ax.grid(axis="x", color="#dddddd", linewidth=0.5); save(fig, "figure_01_primary_clinical_endpoint")

    # Figure 2: paired participant effects.
    primary = bootstrap[bootstrap.metric.eq("composite_absolute_error")].copy()
    comparators = ["cognition_only", "metadata_only", "subject_shuffled_mri", "time_shuffled_mri"]
    primary = primary.set_index("comparator").loc[comparators].reset_index()
    fig, ax = plt.subplots(figsize=(6.4, 3.5)); y = np.arange(len(primary))
    color = [COLORS[arm] for arm in primary.comparator]
    ax.hlines(y, primary.ci_low, primary.ci_high, color=color, linewidth=2)
    ax.scatter(primary.difference, y, color=color, s=35, zorder=3)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y, [f"vs {LABELS[arm]}" for arm in primary.comparator]); ax.invert_yaxis()
    ax.set_xlabel("Real MRI minus comparator MAE (95% participant-bootstrap CI)")
    ax.set_title("MRI content effect is absent against within-subject time shuffling", loc="left")
    ax.grid(axis="x", color="#dddddd", linewidth=0.5); save(fig, "figure_02_paired_mri_effects")

    # Figure 3: horizon response for the five cognition-conditioned arms.
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for arm in ORDER[:-1]:
        local = horizon[horizon.arm.eq(arm)].sort_values("horizon_months")
        ax.plot(local.horizon_months, local.composite_mae, color=COLORS[arm], marker=MARKERS[arm],
                markersize=5, label=LABELS[arm])
    ax.set_xticks([6, 12, 24, 36]); ax.set_xlabel("Forecast horizon (months)")
    ax.set_ylabel("Future observed Memory composite MAE")
    ax.set_title("Real and time-shuffled MRI remain nearly indistinguishable across horizons", loc="left")
    ax.legend(frameon=False, ncol=2); ax.grid(color="#e5e5e5", linewidth=0.5)
    save(fig, "figure_03_horizon_response")

    # Figure 4: coverage-width calibration trade-off.
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.axhspan(0.90, 0.98, color="#E6F4EA", alpha=0.8, label="Prespecified coverage band")
    for arm in ORDER[:-1]:
        row = aggregate[aggregate.arm.eq(arm)].iloc[0]
        ax.scatter(row.interval_width, row.coverage_95, color=COLORS[arm], marker=MARKERS[arm],
                   s=50, label=LABELS[arm])
    ax.set_xlabel("Mean 95% prediction-interval width")
    ax.set_ylabel("Empirical 95% coverage")
    ax.set_title("All cognition-conditioned arms meet the clinical coverage target", loc="left")
    ax.set_ylim(0.90, 0.98); ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.grid(color="#e5e5e5", linewidth=0.5)
    save(fig, "figure_04_calibration_tradeoff")

    # Figure 5: recency-stratified paired effects.
    real = predictions[predictions.arm.eq("real_mri")].copy()
    recency = pd.cut(real.mri_recency_years, [-np.inf, 0.25, 0.5, 1.0, np.inf],
                     labels=["≤3 mo", "3–6 mo", "6–12 mo", ">12 mo"])
    effects = grouped_effects(predictions, recency, "recency")
    fig, ax = plt.subplots(figsize=(6.4, 3.7)); categories = ["≤3 mo", "3–6 mo", "6–12 mo", ">12 mo"]
    offsets = {"cognition_only": -0.10, "time_shuffled_mri": 0.10}
    for comparator in offsets:
        local = effects[effects.comparator.eq(comparator)].set_index("recency").reindex(categories)
        x = np.arange(len(categories)) + offsets[comparator]
        ax.errorbar(x, local.difference, yerr=[local.difference - local.ci_low, local.ci_high - local.difference],
                    fmt=MARKERS[comparator], color=COLORS[comparator], capsize=2.5,
                    label=f"Real MRI vs {LABELS[comparator]}")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(range(len(categories)), categories); ax.set_xlabel("MRI recency at prediction cutoff")
    ax.set_ylabel("Paired composite-MAE difference")
    ax.set_title("Recency does not reveal a temporal-ordering advantage", loc="left")
    ax.legend(frameon=False); ax.grid(axis="y", color="#e5e5e5", linewidth=0.5)
    save(fig, "figure_05_effect_by_mri_recency")

    # Figure 6: scan-count-stratified paired effects.
    count = pd.cut(real.mri_scan_count, [0, 1, 2, np.inf], labels=["1", "2", "≥3"])
    effects = grouped_effects(predictions, count, "scan_count")
    fig, ax = plt.subplots(figsize=(5.8, 3.7)); categories = ["1", "2", "≥3"]
    for comparator in offsets:
        local = effects[effects.comparator.eq(comparator)].set_index("scan_count").reindex(categories)
        x = np.arange(len(categories)) + offsets[comparator]
        ax.errorbar(x, local.difference, yerr=[local.difference - local.ci_low, local.ci_high - local.difference],
                    fmt=MARKERS[comparator], color=COLORS[comparator], capsize=2.5,
                    label=f"Real MRI vs {LABELS[comparator]}")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(range(len(categories)), categories); ax.set_xlabel("Pre-cutoff MRI scans (n)")
    ax.set_ylabel("Paired composite-MAE difference")
    ax.set_title("Additional scans do not establish longitudinal MRI utility", loc="left")
    ax.legend(frameon=False); ax.grid(axis="y", color="#e5e5e5", linewidth=0.5)
    save(fig, "figure_06_effect_by_scan_count")

    readme = """# Phase 6B figures

Each result is exported as an independent vector PDF and 450-DPI PNG; figures
are not screenshots or composite image montages.

1. `figure_01_primary_clinical_endpoint`: five outer-fold values and aggregate
   future observed Memory-composite MAE for all six arms.
2. `figure_02_paired_mri_effects`: participant-cluster bootstrap effects for
   real MRI minus every comparator; negative differences favor real MRI.
3. `figure_03_horizon_response`: 6/12/24/36-month errors for the five
   cognition-conditioned arms.
4. `figure_04_calibration_tradeoff`: empirical coverage against interval width;
   the shaded area is the prespecified 0.90--0.98 coverage band.
5. `figure_05_effect_by_mri_recency`: descriptive participant-bootstrap effects
   stratified by pre-cutoff MRI recency.
6. `figure_06_effect_by_scan_count`: descriptive effects stratified by the
   number of pre-cutoff MRI scans.

Recency and scan-count strata are mechanism diagnostics and were not used for
model or threshold selection. Error bars are 95% participant-bootstrap
intervals unless otherwise stated.
"""
    (FIGURES / "README.md").write_text(readme)


if __name__ == "__main__":
    main()
