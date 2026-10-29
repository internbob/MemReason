"""Phase 5G bridge-strength scan and residual-adaptive gate experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr

from ..data.common import DOMAINS, ROOT
from .phase5c_memory_measurement import (
    clean_matrix,
    cross_views,
    deterministic_targets,
    fit_scaling,
    stable_int,
    subject_fold,
    to_severity,
)
from .phase5d_anchored_mirt import concordance_correlation
from .phase5f_hierarchical_bridge import (
    HierarchicalBridgeAE,
    build_connection_graph,
    construct_membership,
    predict,
    residual_evidence_gate,
    shared_bridge_weights,
    train_model,
)


CONFIG = ROOT / "configs/phase5g_bridge_pareto_gate_dev.yaml"


def weight_tag(weight: float) -> str:
    text = f"{weight:g}".replace(".", "p")
    return f"w{text}"


def make_variants(config: dict[str, Any]) -> list[dict[str, Any]]:
    weights = [float(value) for value in config["scan"]["graph_weights"]]
    variants = [
        {"arm": f"ungated_{weight_tag(weight)}", "graph_weight": weight, "gated": False}
        for weight in weights
    ]
    for weight in weights:
        if weight == 0 and not bool(config["scan"].get("run_gated_zero_duplicate", False)):
            continue
        variants.append(
            {"arm": f"gated_{weight_tag(weight)}", "graph_weight": weight, "gated": True}
        )
    return variants


def state_arrays(
    model: HierarchicalBridgeAE,
    z: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
) -> dict[str, np.ndarray]:
    collected: dict[str, list[np.ndarray]] = {
        "shared_mean": [],
        "shared_logvar": [],
        "shared_available": [],
        "residual_mean": [],
        "residual_logvar": [],
        "residual_available": [],
    }
    with torch.no_grad():
        for start in range(0, len(z), 1024):
            values = torch.as_tensor(
                np.nan_to_num(z[start : start + 1024]), dtype=torch.float32, device=device
            )
            observed = torch.as_tensor(
                mask[start : start + 1024], dtype=torch.float32, device=device
            )
            state = model.state(values, observed)
            for key in collected:
                collected[key].append(state[key].cpu().numpy())
    return {key: np.concatenate(value) for key, value in collected.items()}


def gate_from_numpy(
    state_a: dict[str, np.ndarray],
    state_b: dict[str, np.ndarray],
    gate_config: dict[str, float],
) -> np.ndarray:
    keys = ["residual_mean", "residual_logvar", "residual_available"]
    tensors_a = {key: torch.as_tensor(state_a[key]) for key in keys}
    tensors_b = {key: torch.as_tensor(state_b[key]) for key in keys}
    return residual_evidence_gate(
        tensors_a,
        tensors_b,
        floor=float(gate_config["floor"]),
        temperature=float(gate_config["temperature"]),
    ).numpy()


def ccc_bootstrap_arm(
    states: pd.DataFrame,
    arm: str,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    data = states[states.arm.eq(arm)]
    sufficient = (
        data.assign(
            n=1.0,
            sx=data.view_a,
            sy=data.view_b,
            sxx=np.square(data.view_a),
            syy=np.square(data.view_b),
            sxy=data.view_a * data.view_b,
        )
        .groupby("subject_id")[["n", "sx", "sy", "sxx", "syy", "sxy"]]
        .sum()
    )
    values = sufficient.to_numpy(float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions)
    for repeat in range(repetitions):
        total = values[rng.integers(0, len(values), len(values))].sum(0)
        n, sx, sy, sxx, syy, sxy = total
        mx, my = sx / n, sy / n
        covariance = sxy / n - mx * my
        denominator = sxx / n - mx * mx + syy / n - my * my + (mx - my) ** 2
        estimates[repeat] = 2 * covariance / denominator if denominator > 0 else np.nan
    observed = concordance_correlation(data.view_a.to_numpy(), data.view_b.to_numpy())
    return {
        "arm": arm,
        "ccc": observed,
        "ccc_ci_low": float(np.nanquantile(estimates, 0.025)),
        "ccc_ci_high": float(np.nanquantile(estimates, 0.975)),
        "replicates": repetitions,
        "subjects": int(data.subject_id.nunique()),
        "pairs": len(data),
    }


def subject_bootstrap_comparisons(
    subject_metrics: pd.DataFrame,
    variants: list[dict[str, Any]],
    config: dict[str, Any],
) -> pd.DataFrame:
    repetitions = int(config["statistics"]["bootstrap_replicates"])
    rng = np.random.default_rng(int(config["statistics"]["bootstrap_seed"]))
    metadata = {variant["arm"]: variant for variant in variants}
    zero = next(v["arm"] for v in variants if v["graph_weight"] == 0 and not v["gated"])
    comparisons: list[tuple[str, str, str]] = []
    for variant in variants:
        arm = variant["arm"]
        if arm != zero:
            comparisons.append(("versus_zero", arm, zero))
        if variant["gated"]:
            baseline = next(
                v["arm"]
                for v in variants
                if not v["gated"] and v["graph_weight"] == variant["graph_weight"]
            )
            comparisons.append(("matched_gate", arm, baseline))
    rows = []
    for task in sorted(subject_metrics.task.unique()):
        task_frame = subject_metrics[subject_metrics.task.eq(task)]
        for comparison, candidate, baseline in comparisons:
            pair = task_frame[task_frame.arm.isin([candidate, baseline])].pivot(
                index="subject_id", columns="arm", values="mean_absolute_error"
            ).dropna()
            difference = (pair[candidate] - pair[baseline]).to_numpy()
            draws = np.empty(repetitions)
            for start in range(0, repetitions, 200):
                stop = min(start + 200, repetitions)
                indices = rng.integers(0, len(difference), size=(stop - start, len(difference)))
                draws[start:stop] = difference[indices].mean(1)
            rows.append(
                {
                    "comparison": comparison,
                    "task": task,
                    "candidate": candidate,
                    "baseline": baseline,
                    "graph_weight": metadata[candidate]["graph_weight"],
                    "difference": float(difference.mean()),
                    "ci_low": float(np.quantile(draws, 0.025)),
                    "ci_high": float(np.quantile(draws, 0.975)),
                    "subjects": len(difference),
                    "replicates": repetitions,
                }
            )
    return pd.DataFrame(rows)


def pareto_membership(frame: pd.DataFrame) -> np.ndarray:
    x = frame.heldout_instrument_nmae.to_numpy()
    y = frame.ccc.to_numpy()
    result = np.ones(len(frame), dtype=bool)
    for index in range(len(frame)):
        dominates = (x <= x[index]) & (y >= y[index]) & ((x < x[index]) | (y > y[index]))
        if dominates.any():
            result[index] = False
    return result


def run(
    config_path: Path = CONFIG,
    *,
    selected_fold: int | None = None,
    output_override: Path | None = None,
) -> Path:
    config = yaml.safe_load(config_path.read_text())
    contract_path = ROOT / config["item_contract"]
    items = yaml.safe_load(contract_path.read_text())["primary_state_items"]
    fields = np.asarray([item["field"] for item in items])
    item_domains = np.asarray([item["domain"] for item in items])
    item_instrument_names = np.asarray([item["instrument"] for item in items])
    instrument_names = sorted(set(item_instrument_names))
    instrument_lookup = {name: index for index, name in enumerate(instrument_names)}
    item_instrument_index = np.asarray([instrument_lookup[name] for name in item_instrument_names])
    domain_index = np.asarray([DOMAINS.index(item["domain"]) for item in items])
    construct_names, construct_index = construct_membership(items, config["memory_constructs"])

    frame = pd.read_csv(ROOT / "splits/train.csv", low_memory=False)
    raw = clean_matrix(frame, items)
    mask = np.isfinite(raw)
    keep = mask.sum(1) >= 2
    frame = frame.loc[keep].reset_index(drop=True)
    raw, mask = raw[keep], mask[keep]
    sample_ids = np.asarray(
        [f"{row.subject_id}:{row.visit_code}:{index}" for index, row in enumerate(frame.itertuples())]
    )
    folds, seed = int(config["folds"]), int(config["partition_seed"])
    assignment = np.asarray(
        [subject_fold(str(subject), seed, folds) for subject in frame.subject_id]
    )
    variants = make_variants(config)
    output = (ROOT / "outputs" / config["run_id"]
              if output_override is None else output_override.resolve())
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_root = output / "checkpoints"
    checkpoint_root.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    item_records: list[pd.DataFrame] = []
    subject_records: list[pd.DataFrame] = []
    state_records: list[pd.DataFrame] = []
    graph_records: list[pd.DataFrame] = []
    manifest_records: list[dict[str, Any]] = []

    fold_sequence = list(range(folds)) if selected_fold is None else [int(selected_fold)]
    if any(fold < 0 or fold >= folds for fold in fold_sequence):
        raise ValueError(f"Selected fold must lie in [0, {folds - 1}]")
    for fold in fold_sequence:
        evaluation = assignment == fold
        outer_fit = ~evaluation
        dev_subjects = {
            subject
            for subject in frame.loc[outer_fit, "subject_id"].astype(str).unique()
            if stable_int(f"inner:{seed}:{fold}:{subject}") / 2**64
            < float(config["inner_dev_fraction"])
        }
        development = outer_fit & frame.subject_id.astype(str).isin(dev_subjects).to_numpy()
        fit = outer_fit & ~development
        scaling = fit_scaling(raw, fit, items)
        z = to_severity(raw, scaling)
        graph, edge_matrix, graph_instruments = build_connection_graph(
            frame.loc[fit].reset_index(drop=True), mask[fit], items, config
        )
        if graph_instruments != instrument_names:
            raise RuntimeError("Instrument order changed while constructing graph")
        graph.insert(0, "fold", fold)
        graph_records.append(graph)
        dev_ids = sample_ids[development].tolist()
        evaluation_rows = np.flatnonzero(evaluation)
        local_ids = sample_ids[evaluation].tolist()
        task_masks = {
            task: deterministic_targets(
                mask[evaluation],
                item_instrument_names,
                local_ids,
                task,
                float(config["masking"]["item_fraction"]),
            )
            for task in ["heldout_item", "heldout_instrument"]
        }
        view_a, view_b = cross_views(mask[evaluation], item_instrument_names, local_ids)
        memory_dimension = 1 + DOMAINS.index("Memory")
        view_weights = shared_bridge_weights(
            view_a,
            view_b,
            item_instrument_index,
            domain_index,
            construct_index,
            edge_matrix,
        )[:, memory_dimension]

        for variant_index, variant in enumerate(variants):
            arm = variant["arm"]
            print(
                f"fold={fold} variant={variant_index + 1}/{len(variants)} arm={arm}",
                flush=True,
            )
            gate_config = config["residual_adaptive_gate"] if variant["gated"] else None
            model = train_model(
                "hierarchical_bridge",
                z[fit],
                mask[fit],
                z[development],
                mask[development],
                dev_ids,
                item_instrument_names,
                item_instrument_index,
                domain_index,
                construct_index,
                edge_matrix,
                config,
                seed + fold * 101,
                device,
                graph_weight_override=float(variant["graph_weight"]),
                residual_gate=gate_config,
            )
            destination = checkpoint_root / arm
            destination.mkdir(exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "fold": fold,
                    "fields": fields.tolist(),
                    "construct_names": construct_names,
                    "instrument_names": instrument_names,
                    "best_dev_loss": model.best_dev_loss,
                    "graph_weight": variant["graph_weight"],
                    "gated": variant["gated"],
                    "gate_config": gate_config,
                },
                destination / f"fold_{fold}.pt",
            )

            for task, (input_mask, target_mask) in task_masks.items():
                estimate = predict(model, z[evaluation], input_mask, device)
                row_index, item_index = np.where(target_mask)
                error = np.abs(estimate[row_index, item_index] - z[evaluation][row_index, item_index])
                decisions = pd.DataFrame(
                    {
                        "fold": fold,
                        "arm": arm,
                        "task": task,
                        "subject_id": frame.subject_id.iloc[evaluation_rows[row_index]].astype(str).to_numpy(),
                        "item": fields[item_index],
                        "domain": item_domains[item_index],
                        "absolute_error": error,
                    }
                )
                item_fold = decisions.groupby(
                    ["fold", "arm", "task", "item", "domain"], as_index=False
                ).absolute_error.agg(error_sum="sum", decisions="size")
                item_records.append(item_fold)
                subject = decisions.groupby(
                    ["fold", "arm", "task", "subject_id"], as_index=False
                ).absolute_error.agg(error_sum="sum", decisions="size")
                subject_records.append(subject)

            state_a = state_arrays(model, z[evaluation], view_a, device)
            state_b = state_arrays(model, z[evaluation], view_b, device)
            valid = (
                (view_weights > 0)
                & state_a["shared_available"][:, memory_dimension]
                & state_b["shared_available"][:, memory_dimension]
            )
            gate = (
                gate_from_numpy(state_a, state_b, config["residual_adaptive_gate"])
                if variant["gated"]
                else np.ones(len(view_weights), dtype=float)
            )
            valid_rows = np.flatnonzero(valid)
            state_records.append(
                pd.DataFrame(
                    {
                        "sample_id": sample_ids[evaluation_rows[valid_rows]],
                        "subject_id": frame.subject_id.iloc[evaluation_rows[valid_rows]].astype(str).to_numpy(),
                        "fold": fold,
                        "arm": arm,
                        "graph_weight": float(variant["graph_weight"]),
                        "gated": bool(variant["gated"]),
                        "edge_weight": view_weights[valid_rows],
                        "residual_gate": gate[valid_rows],
                        "effective_edge_weight": view_weights[valid_rows] * gate[valid_rows],
                        "view_a": state_a["shared_mean"][valid_rows, memory_dimension],
                        "view_b": state_b["shared_mean"][valid_rows, memory_dimension],
                        "view_a_sd": np.exp(0.5 * state_a["shared_logvar"][valid_rows, memory_dimension]),
                        "view_b_sd": np.exp(0.5 * state_b["shared_logvar"][valid_rows, memory_dimension]),
                    }
                )
            )
            loading = model.loading_summary()
            manifest_records.append(
                {
                    "fold": fold,
                    "arm": arm,
                    "graph_weight": variant["graph_weight"],
                    "gated": variant["gated"],
                    "best_dev_loss": model.best_dev_loss,
                    "residual_loading_fraction": loading["residual_fraction"],
                    "fit_subjects": int(frame.loc[fit, "subject_id"].nunique()),
                    "dev_subjects": int(frame.loc[development, "subject_id"].nunique()),
                    "evaluation_subjects": int(frame.loc[evaluation, "subject_id"].nunique()),
                }
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    item_fold = pd.concat(item_records, ignore_index=True)
    item_fold["mae"] = item_fold.error_sum / item_fold.decisions
    item_fold.to_csv(output / "item_fold_metrics.csv", index=False)
    item_metrics = item_fold.groupby(["arm", "task", "item", "domain"], as_index=False).agg(
        error_sum=("error_sum", "sum"), decisions=("decisions", "sum")
    )
    item_metrics["mae"] = item_metrics.error_sum / item_metrics.decisions
    item_metrics.to_csv(output / "item_metrics.csv", index=False)
    aggregate = item_metrics.groupby(["arm", "task"], as_index=False).agg(
        macro_item_mae=("mae", "mean"), items=("item", "nunique"), decisions=("decisions", "sum")
    )
    memory = item_metrics[item_metrics.domain.eq("Memory")].groupby(
        ["arm", "task"], as_index=False
    ).agg(memory_macro_item_mae=("mae", "mean"), memory_items=("item", "nunique"))
    aggregate = aggregate.merge(memory, on=["arm", "task"])
    aggregate.to_csv(output / "aggregate_metrics.csv", index=False)

    fold_metrics = item_fold.groupby(["fold", "arm", "task"], as_index=False).agg(
        macro_item_mae=("mae", "mean")
    )
    fold_metrics.to_csv(output / "fold_metrics.csv", index=False)
    subject_metrics = pd.concat(subject_records, ignore_index=True)
    subject_metrics = subject_metrics.groupby(
        ["fold", "arm", "task", "subject_id"], as_index=False
    ).agg(error_sum=("error_sum", "sum"), decisions=("decisions", "sum"))
    subject_metrics["mean_absolute_error"] = subject_metrics.error_sum / subject_metrics.decisions
    subject_metrics.to_csv(output / "subject_task_metrics.csv", index=False)

    states = pd.concat(state_records, ignore_index=True)
    states["absolute_difference"] = np.abs(states.view_a - states.view_b)
    states.to_csv(output / "cross_view_memory_states.csv", index=False)
    ccc_rows = [
        ccc_bootstrap_arm(
            states,
            variant["arm"],
            int(config["statistics"]["state_bootstrap_replicates"]),
            int(config["statistics"]["bootstrap_seed"]) + index,
        )
        for index, variant in enumerate(variants)
    ]
    ccc_frame = pd.DataFrame(ccc_rows)
    summaries = []
    for variant in variants:
        arm = variant["arm"]
        group = states[states.arm.eq(arm)]
        correlation = spearmanr(group.view_a, group.view_b)
        summaries.append(
            {
                "arm": arm,
                "graph_weight": variant["graph_weight"],
                "gated": variant["gated"],
                "spearman": float(correlation.statistic),
                "mean_absolute_difference": float(group.absolute_difference.mean()),
                "memory_state_std": float(pd.concat([group.view_a, group.view_b]).std()),
                "mean_posterior_sd": float(pd.concat([group.view_a_sd, group.view_b_sd]).mean()),
                "mean_edge_weight": float(group.edge_weight.mean()),
                "mean_residual_gate": float(group.residual_gate.mean()),
                "mean_effective_edge_weight": float(group.effective_edge_weight.mean()),
            }
        )
    summary = pd.DataFrame(summaries).merge(ccc_frame, on="arm")
    summary.to_csv(output / "cross_view_summary.csv", index=False)

    comparisons = subject_bootstrap_comparisons(subject_metrics, variants, config)
    comparisons.to_csv(output / "subject_bootstrap.csv", index=False)
    graph_frame = pd.concat(graph_records, ignore_index=True)
    graph_frame.to_csv(output / "connection_graph.csv", index=False)
    manifest_frame = pd.DataFrame(manifest_records)
    manifest_frame.to_csv(output / "fold_manifest.csv", index=False)

    instrument = aggregate[aggregate.task.eq("heldout_instrument")][
        ["arm", "macro_item_mae", "memory_macro_item_mae"]
    ].rename(columns={"macro_item_mae": "heldout_instrument_nmae"})
    item = aggregate[aggregate.task.eq("heldout_item")][["arm", "macro_item_mae"]].rename(
        columns={"macro_item_mae": "heldout_item_nmae"}
    )
    fold_range = fold_metrics[fold_metrics.task.eq("heldout_instrument")].groupby("arm").macro_item_mae.agg(
        heldout_instrument_fold_min="min", heldout_instrument_fold_max="max"
    ).reset_index()
    loading = manifest_frame.groupby("arm", as_index=False).residual_loading_fraction.mean()
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
        "models": len(variants) * len(fold_sequence),
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
        comparisons.comparison.eq("matched_gate")
        & comparisons.task.eq("heldout_instrument")
    ].sort_values("graph_weight")
    report_lines = [
        "# Phase 5G: Bridge-strength Pareto scan and residual-adaptive gate",
        "",
        "## Scope",
        "",
        f"Original-train-only development evaluated {len(variants)} variants across "
        f"{len(fold_sequence)} outer fold(s) ({len(variants) * len(fold_sequence)} neural fits). "
        f"Original validation and test subjects were not loaded.",
        "",
        "## Pareto result",
        "",
        f"The coefficient-zero reference obtained held-out-instrument macro NMAE "
        f"**{zero.heldout_instrument_nmae:.4f}** and Memory CCC **{zero.ccc:.4f}**. "
        f"Under the prespecified +{tolerance:.3f} prediction tolerance, the selected development "
        f"point was `{selected.arm}` with NMAE **{selected.heldout_instrument_nmae:.4f}** "
        f"and CCC **{selected.ccc:.4f}**, 95% participant-bootstrap CI "
        f"**[{selected.ccc_ci_low:.4f}, {selected.ccc_ci_high:.4f}]**.",
        "",
        "## Matched residual-gate effects",
        "",
    ]
    for row in matched.itertuples(index=False):
        report_lines.append(
            f"- λ={row.graph_weight:g}: gated-minus-ungated participant error "
            f"**{row.difference:+.5f}**, 95% CI **[{row.ci_low:+.5f}, {row.ci_high:+.5f}]**."
        )
    report_lines.extend(
        [
            "",
            "## Decision",
            "",
            "This is a development Pareto experiment, not a confirmation run. The Phase 5F gate "
            "is not retrospectively reopened and continuous-time dynamics remain closed.",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(report_lines))
    manifest = {
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256(
            (ROOT / "reports/experiments/phase5g_bridge_pareto_gate_protocol.md").read_bytes()
        ).hexdigest(),
        "train_rows_loaded": len(frame),
        "train_subjects_loaded": int(frame.subject_id.nunique()),
        "validation_or_test_loaded": False,
        "device": str(device),
        "variants": len(variants),
        "neural_checkpoints": len(variants) * len(fold_sequence),
        "selected_folds": fold_sequence,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    print(run(args.config, selected_fold=args.fold, output_override=args.output_dir))


if __name__ == "__main__":
    main()
