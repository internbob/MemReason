"""Descriptive decomposition of frozen Phase 4F state/trend label errors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from ..data.common import ROOT


CONFIG = ROOT / "configs/phase5b_frozen_reasoning_error_decomposition.yaml"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def expected_state(value: float | None, threshold: float = 0.5) -> str:
    if value is None or not np.isfinite(value):
        return "unavailable"
    if value > threshold:
        return "elevated_impairment"
    if value < -threshold:
        return "lower_impairment"
    return "within_training_reference"


def expected_trend(start: float | None, end: float | None, timepoints: int,
                   threshold: float = 0.25) -> str:
    if timepoints < 2 or start is None or end is None or not np.isfinite(start) or not np.isfinite(end):
        return "insufficient_history"
    change = end - start
    if change > threshold:
        return "decline"
    if change < -threshold:
        return "improve"
    return "stable"


def margin_band(margin: float | None, unavailable: str = "unavailable_or_insufficient") -> str:
    if margin is None or not np.isfinite(margin):
        return unavailable
    if margin < 0.05:
        return "[0,0.05)"
    if margin < 0.10:
        return "[0.05,0.10)"
    if margin < 0.25:
        return "[0.10,0.25)"
    return ">=0.25"


def reconstruct_decisions(examples: list[dict[str, Any]], generations: list[dict[str, Any]],
                          state_threshold: float = 0.5, trend_threshold: float = 0.25,
                          near_boundary: float = 0.10) -> pd.DataFrame:
    generated = {row["sample_id"]: json.loads(row["generation"]) for row in generations}
    rows: list[dict[str, Any]] = []
    for example in examples:
        report = generated[example["sample_id"]]
        model_input = example["model_input"]
        reference = example["completion"]
        max_timepoints = max(int(stats["timepoints"]) for stats in model_input["domain_statistics"].values())
        history_stratum = "1" if max_timepoints <= 1 else "2-3" if max_timepoints <= 3 else "4+"
        missing_targets = sum(value is None for value in model_input["last_targets"].values())
        missingness_stratum = "0-1" if missing_targets <= 1 else "2+"
        common = {
            "sample_id": example["sample_id"], "subject_id": example["subject_id"],
            "horizon_months": int(model_input["horizon"]),
            "history_stratum": history_stratum, "max_domain_timepoints": max_timepoints,
            "missingness_stratum": missingness_stratum, "missing_last_targets": missing_targets,
        }
        for domain, stats in model_input["domain_statistics"].items():
            severity = stats["current_severity_z"]
            state_margin = None if severity is None else abs(abs(float(severity)) - state_threshold)
            state_expected = reference["current_state"][domain]["status"]
            recomputed_state = expected_state(severity, state_threshold)
            if state_expected != recomputed_state:
                raise ValueError(f"State reference mismatch: {example['sample_id']} {domain}")
            state_predicted = report["current_state"][domain]["status"]
            rows.append({
                **common, "domain": domain, "field": "current_state",
                "expected": state_expected, "predicted": state_predicted,
                "correct": int(state_expected == state_predicted),
                "mismatch": int(state_expected != state_predicted),
                "decision_value": severity, "start_severity_z": np.nan,
                "end_severity_z": np.nan, "timepoints": int(stats["timepoints"]),
                "items": int(stats["items"]), "threshold_margin": state_margin,
                "margin_band": margin_band(state_margin),
                "boundary_group": "unavailable" if state_margin is None else (
                    "near" if state_margin < near_boundary else "far"),
            })
            start, end, timepoints = stats["start_severity_z"], stats["end_severity_z"], int(stats["timepoints"])
            trend_expected = reference["trend"][domain]["label"]
            recomputed_trend = expected_trend(start, end, timepoints, trend_threshold)
            if trend_expected != recomputed_trend:
                raise ValueError(f"Trend reference mismatch: {example['sample_id']} {domain}")
            trend_predicted = report["trend"][domain]["label"]
            change = None if timepoints < 2 or start is None or end is None else float(end) - float(start)
            trend_margin = None if change is None else abs(abs(change) - trend_threshold)
            rows.append({
                **common, "domain": domain, "field": "trend",
                "expected": trend_expected, "predicted": trend_predicted,
                "correct": int(trend_expected == trend_predicted),
                "mismatch": int(trend_expected != trend_predicted),
                "decision_value": change, "start_severity_z": start,
                "end_severity_z": end, "timepoints": timepoints,
                "items": int(stats["items"]), "threshold_margin": trend_margin,
                "margin_band": margin_band(trend_margin),
                "boundary_group": "insufficient_history" if trend_margin is None else (
                    "near" if trend_margin < near_boundary else "far"),
            })
    return pd.DataFrame(rows)


def cluster_rate_interval(frame: pd.DataFrame, replicates: int, rng: np.random.Generator) -> tuple[float, float, float]:
    point = float(frame.mismatch.mean())
    clusters = frame.groupby("subject_id").mismatch.agg(["sum", "count"]).to_numpy(float)
    indices = rng.integers(0, len(clusters), size=(replicates, len(clusters)))
    selected = clusters[indices]
    draws = selected[:, :, 0].sum(1) / selected[:, :, 1].sum(1)
    return point, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def summarize(frame: pd.DataFrame, dimension: str, replicates: int,
              rng: np.random.Generator) -> list[dict[str, Any]]:
    rows = []
    keys = ["field"] if dimension == "overall" else ["field", dimension]
    for key, group in frame.groupby(keys, dropna=False, sort=False):
        values = key if isinstance(key, tuple) else (key,)
        field = values[0]
        label = "all" if dimension == "overall" else values[1]
        point, low, high = cluster_rate_interval(group, replicates, rng)
        rows.append({
            "dimension": dimension, "group": label, "field": field,
            "decisions": len(group), "subjects": group.subject_id.nunique(),
            "mismatches": int(group.mismatch.sum()), "mismatch_rate": point,
            "ci_low": low, "ci_high": high,
        })
    return rows


def near_far_contrast(frame: pd.DataFrame, replicates: int,
                      rng: np.random.Generator) -> pd.DataFrame:
    output = []
    for field, local in frame[frame.boundary_group.isin(["near", "far"])].groupby("field"):
        subjects = local.subject_id.unique()
        grouped = local.groupby(["subject_id", "boundary_group"]).mismatch.agg(["sum", "count"])
        arrays = {}
        for group_name in ("near", "far"):
            table = grouped.xs(group_name, level="boundary_group", drop_level=True) if group_name in grouped.index.get_level_values(1) else pd.DataFrame()
            arrays[group_name] = np.array([[float(table.loc[s, "sum"]), float(table.loc[s, "count"])]
                                           if s in table.index else [0.0, 0.0] for s in subjects])
        point = (arrays["near"][:, 0].sum() / arrays["near"][:, 1].sum()
                 - arrays["far"][:, 0].sum() / arrays["far"][:, 1].sum())
        indices = rng.integers(0, len(subjects), size=(replicates, len(subjects)))
        near_total = arrays["near"][indices].sum(axis=1)
        far_total = arrays["far"][indices].sum(axis=1)
        valid = (near_total[:, 1] > 0) & (far_total[:, 1] > 0)
        draws = near_total[valid, 0] / near_total[valid, 1] - far_total[valid, 0] / far_total[valid, 1]
        near = local[local.boundary_group.eq("near")]
        far = local[local.boundary_group.eq("far")]
        output.append({
            "field": field, "contrast": "near_minus_far_mismatch_rate",
            "near_decisions": len(near), "near_mismatches": int(near.mismatch.sum()),
            "near_rate": float(near.mismatch.mean()), "far_decisions": len(far),
            "far_mismatches": int(far.mismatch.sum()), "far_rate": float(far.mismatch.mean()),
            "difference": point, "ci_low": float(np.quantile(draws, .025)),
            "ci_high": float(np.quantile(draws, .975)), "subjects": len(subjects),
            "bootstrap_replicates": replicates, "valid_bootstrap_replicates": int(valid.sum()),
        })
    return pd.DataFrame(output)


def make_figure(summary: pd.DataFrame, confusion: pd.DataFrame, output: Path, dpi: int) -> None:
    colors = {"current_state": "#0072B2", "trend": "#D55E00"}
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8})
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6), constrained_layout=True)

    def grouped_bars(ax, dimension: str, title: str, order: list[Any], annotate_n: bool = False) -> None:
        local = summary[summary.dimension.eq(dimension)].copy()
        x = np.arange(len(order)); width = .36
        for offset, field in [(-width / 2, "current_state"), (width / 2, "trend")]:
            indexed = local[local.field.eq(field)].set_index("group").reindex(order)
            rates = indexed.mismatch_rate.to_numpy(float) * 100
            low = rates - indexed.ci_low.to_numpy(float) * 100
            high = indexed.ci_high.to_numpy(float) * 100 - rates
            bars = ax.bar(x + offset, rates, width, color=colors[field], label=field.replace("_", " "),
                          yerr=np.vstack([low, high]), capsize=2, linewidth=.6)
            if annotate_n:
                for bar, n in zip(bars, indexed.decisions.to_numpy(int)):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2.0, f"n={n}",
                            ha="center", va="bottom", fontsize=6, rotation=90)
        ax.set_xticks(x, [str(value) for value in order], rotation=25 if dimension == "domain" else 0)
        ax.set_ylabel("Mismatch rate (%)"); ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", alpha=.25); ax.spines[["top", "right"]].set_visible(False)

    domains = ["Memory", "Executive", "Attention", "Language", "Orientation"]
    grouped_bars(axes[0, 0], "domain", "A  Errors by cognitive domain", domains)
    grouped_bars(axes[0, 1], "horizon_months", "B  Errors by forecast horizon", [6, 12, 24, 36])
    axes[0, 0].legend(frameon=False, ncol=2, loc="upper right")

    band_order = ["[0,0.05)", "[0.05,0.10)", "[0.10,0.25)", ">=0.25"]
    grouped_bars(axes[1, 0], "margin_band", "C  Errors by distance to label boundary", band_order, True)
    axes[1, 0].set_xticklabels(["<.05", ".05–.10", ".10–.25", "≥.25"], rotation=20)

    trend = confusion[confusion.field.eq("trend")].copy()
    labels = ["decline", "improve", "stable", "insufficient_history"]
    matrix = trend.pivot_table(index="expected", columns="predicted", values="count", fill_value=0).reindex(
        index=labels, columns=labels, fill_value=0).to_numpy(float)
    row_sums = matrix.sum(1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0) * 100
    image = axes[1, 1].imshow(normalized, cmap="Blues", vmin=0, vmax=100)
    for i in range(len(labels)):
        for j in range(len(labels)):
            axes[1, 1].text(j, i, f"{int(matrix[i,j])}\n{normalized[i,j]:.1f}%", ha="center", va="center",
                            fontsize=7, color="white" if normalized[i, j] > 55 else "black")
    short = ["decline", "improve", "stable", "insuff."]
    axes[1, 1].set_xticks(range(4), short, rotation=25); axes[1, 1].set_yticks(range(4), short)
    axes[1, 1].set_xlabel("Predicted label"); axes[1, 1].set_ylabel("Expected label")
    axes[1, 1].set_title("D  Trend confusion (count and row %)", loc="left", fontweight="bold")
    fig.colorbar(image, ax=axes[1, 1], label="Row percentage", fraction=.046)
    fig.suptitle("Frozen Phase 4F structured-reasoning error decomposition", fontsize=12, fontweight="bold")
    fig.text(.01, .005, "Panel C excludes 1 unavailable state and 1,098 insufficient-history trend decisions; n denotes domain decisions.",
             fontsize=7, color="#444444")
    figure_dir = output / "figures"; figure_dir.mkdir(exist_ok=True)
    fig.savefig(figure_dir / "phase5b_error_decomposition.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "phase5b_error_decomposition.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_report(frame: pd.DataFrame, summary: pd.DataFrame, contrast: pd.DataFrame,
                 output: Path) -> None:
    overall = summary[summary.dimension.eq("overall")].set_index("field")
    by_domain = summary[summary.dimension.eq("domain")]
    by_horizon = summary[summary.dimension.eq("horizon_months")]
    by_expected = summary[summary.dimension.eq("expected")]
    state_top = by_domain[by_domain.field.eq("current_state")].sort_values("mismatch_rate", ascending=False).iloc[0]
    trend_top = by_domain[by_domain.field.eq("trend")].sort_values("mismatch_rate", ascending=False).iloc[0]
    horizon_top = by_horizon.groupby("group").agg(mismatches=("mismatches", "sum"), decisions=("decisions", "sum"))
    horizon_top["rate"] = horizon_top.mismatches / horizon_top.decisions
    horizon_top_row = horizon_top.sort_values("rate", ascending=False).iloc[0]
    horizon_top_name = horizon_top.sort_values("rate", ascending=False).index[0]
    c = contrast.set_index("field")
    confusion = frame[frame.mismatch.eq(1)].groupby(["field", "expected", "predicted"]).size().sort_values(ascending=False)
    dominant = confusion.index[0]; dominant_n = int(confusion.iloc[0])
    improve = by_expected[(by_expected.field.eq("trend")) & (by_expected.group.eq("improve"))].iloc[0]
    report = f"""# Phase 5B: Frozen Reasoning Error Decomposition

## Status

Completed as a post-test descriptive analysis of the unchanged Phase 4F grammar-model outputs. No training, decoding, prompt editing, threshold selection, or report regeneration was performed. The Phase 4F confirmation decision remains failed because temporal consistency was 0.98990 against the prespecified 0.99000 threshold.

## Cohort and denominators

- 495 subject-disjoint test reports, one per subject.
- 2,475 current-state decisions and 2,475 trend decisions across five cognitive domains.
- Current-state mismatches: **{int(overall.loc['current_state','mismatches'])}/2,475 ({100*overall.loc['current_state','mismatch_rate']:.2f}%; 95% subject-bootstrap CI {100*overall.loc['current_state','ci_low']:.2f}–{100*overall.loc['current_state','ci_high']:.2f}%)**.
- Trend mismatches: **{int(overall.loc['trend','mismatches'])}/2,475 ({100*overall.loc['trend','mismatch_rate']:.2f}%; 95% CI {100*overall.loc['trend','ci_low']:.2f}–{100*overall.loc['trend','ci_high']:.2f}%)**.

## Main decomposition

The highest current-state mismatch rate occurred in **{state_top['group']}** ({int(state_top['mismatches'])}/{int(state_top['decisions'])}, {100*state_top['mismatch_rate']:.2f}%). The highest trend mismatch rate occurred in **{trend_top['group']}** ({int(trend_top['mismatches'])}/{int(trend_top['decisions'])}, {100*trend_top['mismatch_rate']:.2f}%). Across both fields, the highest horizon-level rate was at **{horizon_top_name} months** ({int(horizon_top_row['mismatches'])}/{int(horizon_top_row['decisions'])}, {100*horizon_top_row['rate']:.2f}%). Expected `improve` trends had {int(improve['mismatches'])}/{int(improve['decisions'])} mismatches ({100*improve['mismatch_rate']:.2f}%), and the most frequent directed confusion was **{dominant[0]}: {dominant[1]} → {dominant[2]}** ({dominant_n} decisions).

For current state, the near-boundary (<0.10 z from ±0.5) mismatch rate was {int(c.loc['current_state','near_mismatches'])}/{int(c.loc['current_state','near_decisions'])} ({100*c.loc['current_state','near_rate']:.2f}%) versus {int(c.loc['current_state','far_mismatches'])}/{int(c.loc['current_state','far_decisions'])} ({100*c.loc['current_state','far_rate']:.2f}%) far from the boundary; the near-minus-far difference was {100*c.loc['current_state','difference']:.2f} percentage points (95% CI {100*c.loc['current_state','ci_low']:.2f} to {100*c.loc['current_state','ci_high']:.2f}). For trend, only {int(c.loc['trend','near_decisions'])} decisions were near the boundary: {int(c.loc['trend','near_mismatches'])}/{int(c.loc['trend','near_decisions'])} ({100*c.loc['trend','near_rate']:.2f}%) versus {int(c.loc['trend','far_mismatches'])}/{int(c.loc['trend','far_decisions'])} ({100*c.loc['trend','far_rate']:.2f}%) far from it, a difference of {100*c.loc['trend','difference']:.2f} points (95% CI {100*c.loc['trend','ci_low']:.2f} to {100*c.loc['trend','ci_high']:.2f}). This trend contrast is therefore imprecise despite its large point estimate. Insufficient-history trend decisions are retained as a separate structural category and are excluded from that contrast.

## Interpretation

These results localize the already observed Phase 4F label discrepancies; they do not revise the confirmatory outcome. All numeric severity values, timepoints, evidence citations, forecasts, and uncertainty fields remained preserved in Phase 4F. The error pattern therefore concerns categorical state/trend selection within otherwise valid structured reports. Because this decomposition was conducted after opening the test set, subgroup confidence intervals are descriptive and cannot justify test-set repair or a new claim of gate success.

The next admissible experiment is a prospectively frozen external-cohort validation of the unchanged **GRU-D cognitive forecasting + grammar-constrained structured reporting** system. Any revised label policy must be developed without these 495 subjects and evaluated on another untouched cohort.

## Artifacts

- `decision_level_decomposition.csv`: all 4,950 domain-field decisions and their frozen covariates.
- `mismatch_rate_summary.csv`: denominator-aware rates with subject-cluster bootstrap intervals.
- `confusion_matrix.csv`: full expected-to-predicted counts.
- `near_far_boundary_contrast.csv`: prespecified near-versus-far descriptive contrasts.
- `figures/phase5b_error_decomposition.pdf` and `.png`: publication-ready summary figure.
- `analysis_decision.json`: scope and next-action record.
- `verification.json`: independent source and metric verification.
"""
    (output / "report.md").write_text(report)


def run() -> Path:
    config = yaml.safe_load(CONFIG.read_text())
    source = ROOT / "outputs" / config["source_run_id"]
    output = ROOT / "outputs" / config["run_id"]
    output.mkdir(parents=True, exist_ok=True)
    examples_path = source / "test_reasoning_examples.jsonl"
    generations_path = source / f"test_generations_{config['source_arm']}.jsonl"
    decision_path = source / "final_decision.json"
    source_lock = {
        "analysis_status": config["analysis_status"], "source_arm": config["source_arm"],
        "training_performed": False, "generation_performed": False,
        "artifacts": {str(path): digest(path) for path in [examples_path, generations_path, decision_path, CONFIG]},
    }
    (output / "source_lock.json").write_text(json.dumps(source_lock, indent=2) + "\n")
    frame = reconstruct_decisions(
        read_jsonl(examples_path), read_jsonl(generations_path),
        float(config["thresholds"]["current_state_absolute_z"]),
        float(config["thresholds"]["trend_change_z"]),
        float(config["thresholds"]["near_boundary"]),
    )
    frame.to_csv(output / "decision_level_decomposition.csv", index=False)
    replicates = int(config["statistics"]["bootstrap_replicates"])
    rng = np.random.default_rng(int(config["statistics"]["bootstrap_seed"]))
    summary_rows: list[dict[str, Any]] = []
    for dimension in ["overall", "domain", "horizon_months", "expected", "history_stratum",
                      "missingness_stratum", "margin_band"]:
        summary_rows.extend(summarize(frame, dimension, replicates, rng))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "mismatch_rate_summary.csv", index=False)
    confusion = frame.groupby(["field", "expected", "predicted"], as_index=False).agg(
        count=("mismatch", "size"), subjects=("subject_id", "nunique"))
    confusion.to_csv(output / "confusion_matrix.csv", index=False)
    contrast = near_far_contrast(frame, replicates, rng)
    contrast.to_csv(output / "near_far_boundary_contrast.csv", index=False)
    mismatches = frame[frame.mismatch.eq(1)].copy()
    mismatches.to_csv(output / "mismatch_cases.csv", index=False)
    make_figure(summary, confusion, output, int(config["reporting"]["figure_dpi"]))
    build_report(frame, summary, contrast, output)
    phase4f = json.loads(decision_path.read_text())
    decision = {
        "run_id": config["run_id"], "analysis_only": True, "post_test_descriptive": True,
        "training_performed": False, "generation_performed": False,
        "reports": int(frame.sample_id.nunique()), "subjects": int(frame.subject_id.nunique()),
        "decisions_per_field": int((frame.field == "current_state").sum()),
        "current_state_mismatches": int(frame.query("field == 'current_state'").mismatch.sum()),
        "trend_mismatches": int(frame.query("field == 'trend'").mismatch.sum()),
        "phase4f_decision_changed": False,
        "phase4f_frozen_system_confirmed": bool(phase4f["phase4f_frozen_system_confirmed"]),
        "further_test_tuning_permitted": False,
        "next_experiment": "prospectively_frozen_external_cohort_validation",
    }
    (output / "analysis_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(run())


if __name__ == "__main__":
    main()
