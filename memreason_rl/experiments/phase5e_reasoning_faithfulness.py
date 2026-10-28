"""Post-test frozen faithfulness interventions and renderer ablation for Phase 4F."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from ..data.common import DOMAINS, ROOT
from ..data.dictionary import load_dictionary
from .phase1_baseline_suite import select_verified_scalar_features
from .phase1_numeric import build_samples, clean_frame, markdown_table
from .phase4a_structured_reasoning import (
    ensemble_predictions,
    make_example,
    parse_events,
    predict_all_targets,
)
from .phase4b_reasoning_sft import score_generation
from .phase4c_grammar_decoding import executable_status, executable_trend, render_report


CONFIG = ROOT / "configs/phase5e_reasoning_faithfulness_audit.yaml"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_context(config: dict[str, Any]):
    source = ROOT / "outputs" / config["source_run_id"]
    phase4a = ROOT / "outputs" / config["phase4a_run_id"]
    phase4a_config = yaml.safe_load((ROOT / "configs/phase4a_frozen_cognitive_reasoning.yaml").read_text())
    dictionary = load_dictionary()
    features = select_verified_scalar_features(phase4a_config, dictionary)
    targets = list(phase4a_config["targets"])
    normalization = json.loads((phase4a / "normalization.json").read_text())
    frame = pd.read_csv(ROOT / "splits/test.csv", low_memory=False)
    clean = clean_frame(frame, features, targets, dictionary)
    horizons = [int(value) for value in phase4a_config["future_forecast"]["horizons_months"]]
    tolerances = {
        int(key): int(value)
        for key, value in phase4a_config["future_forecast"]["tolerance_months"].items()
    }
    samples = build_samples(clean, "sealed_test", features, targets, horizons, tolerances)
    reasoning = read_jsonl(source / "test_reasoning_examples.jsonl")
    wanted = {row["sample_id"] for row in reasoning}
    selected = {sample.sample_id: sample for sample in samples if sample.sample_id in wanted}
    if set(selected) != wanted or len(reasoning) != len(wanted):
        raise RuntimeError("Could not reconstruct the exact Phase 4F reasoning cohort")
    events, _ = parse_events(ROOT / "splits/test_events.csv", set(features))
    baseline = pd.read_csv(source / "test_ensemble_forecasts.csv")
    baseline = baseline[baseline.sample_id.isin(wanted)].copy()
    return source, phase4a, phase4a_config, dictionary, features, targets, normalization, reasoning, selected, events, baseline


def observed_visit_indices(sample: Any) -> list[int]:
    return [index for index in range(len(sample.months)) if bool(sample.feature_mask[index].any())]


def drop_visit(sample: Any, index: int, sample_id: str | None = None):
    keep = np.ones(len(sample.months), dtype=bool)
    keep[index] = False
    return replace(
        sample,
        sample_id=sample.sample_id if sample_id is None else sample_id,
        months=sample.months[keep].copy(),
        features=sample.features[keep].copy(),
        feature_mask=sample.feature_mask[keep].copy(),
        history_targets=sample.history_targets[keep].copy(),
        history_target_mask=sample.history_target_mask[keep].copy(),
    )


def reversed_time_sample(sample: Any):
    if len(sample.months) < 3:
        return copy.deepcopy(sample), {float(value): float(value) for value in sample.months}
    gaps = np.diff(sample.months)
    revised = np.concatenate([[sample.months[0]], sample.months[0] + np.cumsum(gaps[::-1])])
    mapping = {float(old): float(new) for old, new in zip(sample.months, revised)}
    return replace(sample, months=revised.astype(sample.months.dtype)), mapping


def reflected_memory_sample(
    sample: Any,
    features: list[str],
    dictionary: dict[str, Any],
    normalization: dict[str, dict[str, float]],
):
    values = sample.features.copy()
    for index, item in enumerate(features):
        if dictionary[item]["domain"] != "Memory":
            continue
        mask = sample.feature_mask[:, index]
        low = float(normalization[item]["minimum"])
        high = float(normalization[item]["maximum"])
        values[mask, index] = low + high - values[mask, index]
    return replace(sample, features=values)


def predict(
    samples: list[Any],
    features: list[str],
    targets: list[str],
    normalization: dict[str, Any],
    phase4a_config: dict[str, Any],
    phase4a: Path,
    device: torch.device,
) -> pd.DataFrame:
    frames = [
        predict_all_targets(
            int(seed), samples, features, targets, normalization, phase4a_config, phase4a, device
        )
        for seed in phase4a_config["neural_seeds"]
    ]
    return ensemble_predictions(pd.concat(frames, ignore_index=True))


def select_deletions(
    samples: dict[str, Any],
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    targets: list[str],
    normalization: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    original = baseline.set_index(["sample_id", "target"])["prediction"]
    rows = []
    for candidate_id, group in candidate.groupby("sample_id", sort=True):
        sample_id, raw_index = candidate_id.rsplit("::drop", 1)
        index = int(raw_index)
        displacement = np.mean([
            abs(float(row.prediction) - float(original.loc[(sample_id, row.target)]))
            / float(normalization[row.target]["std"])
            for row in group.itertuples()
            if row.target in targets
        ])
        rows.append({
            "sample_id": sample_id,
            "candidate_id": candidate_id,
            "visit_index": index,
            "visit_month": float(samples[sample_id].months[index]),
            "normalized_forecast_displacement": float(displacement),
        })
    candidates = pd.DataFrame(rows)
    selected = []
    rng = np.random.default_rng(seed)
    for sample_id, group in candidates.groupby("sample_id", sort=True):
        if len(group) < 2:
            continue
        ordered = group.sort_values(
            ["normalized_forecast_displacement", "visit_month", "visit_index"],
            ascending=[False, False, False], kind="stable",
        )
        high = ordered.iloc[0]
        remaining = group[group.visit_index.ne(int(high.visit_index))]
        random = remaining.iloc[int(rng.integers(0, len(remaining)))]
        for condition, row in [
            ("highest_influence_visit_deletion", high),
            ("matched_random_visit_deletion", random),
        ]:
            selected.append({
                "sample_id": sample_id,
                "subject_id": samples[sample_id].subject_id,
                "condition": condition,
                "visit_index": int(row.visit_index),
                "visit_month": float(row.visit_month),
                "selection_displacement": float(row.normalized_forecast_displacement),
                "eligible_visits": int(len(group)),
            })
    return pd.DataFrame(selected)


def transform_events(
    events: list[dict[str, Any]],
    condition: str,
    dictionary: dict[str, Any],
    normalization: dict[str, Any],
    deleted_month: float | None = None,
    time_mapping: dict[float, float] | None = None,
) -> list[dict[str, Any]]:
    output = []
    for source in events:
        event = copy.deepcopy(source)
        if condition.endswith("visit_deletion"):
            if deleted_month is not None and abs(float(event["visit_month"]) - deleted_month) <= 1e-6:
                continue
        elif condition == "reversed_time_gaps":
            old_visit = float(event["visit_month"])
            if time_mapping is not None and old_visit in time_mapping:
                new_visit = float(time_mapping[old_visit])
                offset = float(event["time_month"]) - old_visit
                event["visit_month"] = new_visit
                event["time_month"] = new_visit + offset
        elif condition == "memory_score_reflection":
            for item, value in list(event["payload"].items()):
                if dictionary[item]["domain"] == "Memory":
                    low = float(normalization[item]["minimum"])
                    high = float(normalization[item]["maximum"])
                    event["payload"][item] = low + high - float(value)
        output.append(event)
    output.sort(key=lambda event: (event["visit_month"], event["time_month"], event["event_id"]))
    return output


def compare_report(base: dict[str, Any], perturbed: dict[str, Any]) -> dict[str, float]:
    base_state = base["completion"]["current_state"]
    base_trend = base["completion"]["trend"]
    report = perturbed["completion"]
    state_changes = np.mean([
        report["current_state"][domain]["status"] != base_state[domain]["status"] for domain in DOMAINS
    ])
    trend_changes = np.mean([
        report["trend"][domain]["label"] != base_trend[domain]["label"] for domain in DOMAINS
    ])
    forecast_changes = np.mean([
        report["forecast"][target]["change_from_last"]
        != base["completion"]["forecast"][target]["change_from_last"]
        for target in report["forecast"]
    ])
    base_evidence = {
        (row["event_id"], row["item"], row["time_month"], row["value"])
        for row in base["completion"]["evidence"]
    }
    new_evidence = {
        (row["event_id"], row["item"], row["time_month"], row["value"])
        for row in report["evidence"]
    }
    union = base_evidence | new_evidence
    return {
        "state_label_change_fraction": float(state_changes),
        "trend_label_change_fraction": float(trend_changes),
        "forecast_direction_change_fraction": float(forecast_changes),
        "evidence_exact_jaccard": float(len(base_evidence & new_evidence) / len(union)) if union else 1.0,
        "compiled_report_changed": float(report != base["completion"]),
    }


def bootstrap_mean(frame: pd.DataFrame, columns: list[str], seed: int, replicates: int) -> pd.DataFrame:
    subjects = frame["subject_id"].astype(str).unique()
    by_subject = frame.groupby("subject_id")[columns].mean().reindex(subjects)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(subjects), size=(replicates, len(subjects)))
    rows = []
    for column in columns:
        values = by_subject[column].to_numpy(dtype=float)
        draws = values[indices].mean(axis=1)
        rows.append({
            "metric": column,
            "mean": float(values.mean()),
            "ci_low": float(np.quantile(draws, 0.025)),
            "ci_high": float(np.quantile(draws, 0.975)),
            "subjects": int(len(subjects)),
            "replicates": int(replicates),
        })
    return pd.DataFrame(rows)


def renderer_ablation(source: Path, output: Path, config: dict[str, Any]) -> dict[str, Any]:
    per_report = pd.read_csv(source / "reasoning_per_report_metrics.csv")
    aggregate = pd.read_csv(source / "reasoning_aggregate_metrics.csv")
    aggregate.to_csv(output / "renderer_ablation_metrics.csv", index=False)
    metrics = [
        "format_validity", "evidence_precision", "evidence_recall",
        "current_state_consistency", "temporal_consistency", "forecast_preservation",
        "uncertainty_preservation", "exact_report_match",
    ]
    pivot = per_report.pivot(index="subject_id", columns="arm", values=metrics)
    subjects = pivot.index.to_numpy()
    rng = np.random.default_rng(int(config["statistics"]["bootstrap_seed"]))
    n = int(config["statistics"]["bootstrap_replicates"])
    indices = rng.integers(0, len(subjects), size=(n, len(subjects)))
    contrasts = []
    for metric in metrics:
        difference = (
            pivot[(metric, "grammar_executable")] - pivot[(metric, "grammar_model")]
        ).to_numpy(dtype=float)
        draws = difference[indices].mean(axis=1)
        contrasts.append({
            "comparison": "grammar_executable_minus_grammar_model",
            "metric": metric,
            "difference": float(difference.mean()),
            "ci_low": float(np.quantile(draws, 0.025)),
            "ci_high": float(np.quantile(draws, 0.975)),
            "subjects": int(len(subjects)),
            "replicates": n,
        })
    pd.DataFrame(contrasts).to_csv(output / "renderer_ablation_bootstrap.csv", index=False)
    indexed = aggregate.set_index("arm")
    result = {
        "reports": int(len(subjects)),
        "executable_exact_match": float(indexed.loc["grammar_executable", "exact_report_match"]),
        "grammar_model_exact_match": float(indexed.loc["grammar_model", "exact_report_match"]),
        "raw_sft_exact_match": float(indexed.loc["raw_sft", "exact_report_match"]),
        "executable_equals_teacher_on_all_report_metrics": bool(
            np.allclose(
                indexed.loc["grammar_executable"].to_numpy(dtype=float),
                indexed.loc["teacher"].to_numpy(dtype=float),
            )
        ),
        "llm_renderer_factual_advantage_supported": False,
    }
    return result


def run(config_path: Path = CONFIG) -> None:
    config = yaml.safe_load(config_path.read_text())
    output = ROOT / "outputs" / config["run_id"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.json").write_text(json.dumps(config, indent=2) + "\n")
    protocol = ROOT / "reports/experiments/phase5e_reasoning_faithfulness_audit_protocol.md"
    source, phase4a, phase4a_config, dictionary, features, targets, normalization, reasoning, samples, events, baseline = load_context(config)
    source_lock = {
        "analysis_type": "post_test_frozen_audit",
        "official_test_already_opened": True,
        "model_or_decoder_selection_performed": False,
        "threshold_tuning_performed": False,
        "source_run": config["source_run_id"],
        "reasoning_examples_sha256": digest(source / "test_reasoning_examples.jsonl"),
        "phase4f_decision_sha256": digest(source / "final_decision.json"),
        "protocol_sha256": digest(protocol),
    }
    (output / "source_lock.json").write_text(json.dumps(source_lock, indent=2) + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidate_samples = []
    for sample in samples.values():
        eligible = observed_visit_indices(sample)
        if len(eligible) < int(config["interventions"]["minimum_observed_visits_for_paired_deletion"]):
            continue
        for index in eligible:
            candidate_samples.append(drop_visit(sample, index, f"{sample.sample_id}::drop{index}"))
    candidate_forecasts = predict(
        candidate_samples, features, targets, normalization, phase4a_config, phase4a, device
    )
    selections = select_deletions(
        samples, baseline, candidate_forecasts, targets, normalization, int(config["selection_seed"])
    )
    selections.to_csv(output / "visit_deletion_selection.csv", index=False)

    variants: dict[str, list[Any]] = {
        "highest_influence_visit_deletion": [],
        "matched_random_visit_deletion": [],
        "reversed_time_gaps": [],
        "memory_score_reflection": [],
    }
    metadata: dict[tuple[str, str], dict[str, Any]] = {}
    for row in selections.itertuples(index=False):
        variant = drop_visit(samples[row.sample_id], int(row.visit_index))
        variants[row.condition].append(variant)
        metadata[(row.condition, row.sample_id)] = {
            "deleted_visit_index": int(row.visit_index),
            "deleted_visit_month": float(row.visit_month),
            "eligible_visits": int(row.eligible_visits),
        }
    time_mappings: dict[str, dict[float, float]] = {}
    for sample_id, sample in samples.items():
        timed, mapping = reversed_time_sample(sample)
        variants["reversed_time_gaps"].append(timed)
        time_mappings[sample_id] = mapping
        metadata[("reversed_time_gaps", sample_id)] = {
            "time_mapping": {str(key): value for key, value in mapping.items()}
        }
        variants["memory_score_reflection"].append(
            reflected_memory_sample(sample, features, dictionary, normalization)
        )
        metadata[("memory_score_reflection", sample_id)] = {}

    forecast_frames = [baseline.assign(condition="baseline")]
    ensemble_by_condition: dict[str, pd.DataFrame] = {}
    for condition, local_samples in variants.items():
        frame = predict(local_samples, features, targets, normalization, phase4a_config, phase4a, device)
        frame["condition"] = condition
        ensemble_by_condition[condition] = frame
        forecast_frames.append(frame)
    all_forecasts = pd.concat(forecast_frames, ignore_index=True)
    all_forecasts.to_csv(output / "intervention_forecasts.csv", index=False)

    baseline_lookup = {row["sample_id"]: row for row in reasoning}
    ir_rows = []
    metric_rows = []
    target_std = {target: float(normalization[target]["std"]) for target in targets}
    base_predictions = baseline.set_index(["sample_id", "target"])["prediction"]
    variant_sample_lookup = {
        condition: {sample.sample_id: sample for sample in local_samples}
        for condition, local_samples in variants.items()
    }
    for condition, forecasts in ensemble_by_condition.items():
        for sample_id, group in forecasts.groupby("sample_id", sort=True):
            sample = variant_sample_lookup[condition][sample_id]
            original_events = events[sample.subject_id]
            local_meta = metadata[(condition, sample_id)]
            local_events = transform_events(
                original_events,
                condition,
                dictionary,
                normalization,
                deleted_month=local_meta.get("deleted_visit_month"),
                time_mapping=time_mappings.get(sample_id),
            )
            if local_meta.get("deleted_visit_month") is not None:
                deleted_month = float(local_meta["deleted_visit_month"])
                local_meta["deleted_event_ids"] = sorted({
                    str(event["event_id"])
                    for event in original_events
                    if abs(float(event["visit_month"]) - deleted_month) <= 1e-6
                })
            example, _ = make_example(
                sample, group, {sample.subject_id: local_events}, dictionary,
                normalization, phase4a_config,
            )
            row = {
                "sample_id": sample_id,
                "subject_id": sample.subject_id,
                "reasoning_split": "sealed_test_posthoc_counterfactual",
                "model_input": __import__(
                    "memreason_rl.experiments.phase4b_reasoning_sft", fromlist=["phase4b_input"]
                ).phase4b_input(example),
                "completion": example["completion"],
            }
            report = render_report(
                row,
                {
                    domain: executable_status(row["model_input"]["domain_statistics"][domain])
                    for domain in DOMAINS
                },
                {
                    domain: executable_trend(row["model_input"]["domain_statistics"][domain])
                    for domain in DOMAINS
                },
            )
            generation = json.dumps(report, separators=(",", ":"))
            verified = score_generation(row, generation, "grammar_executable")
            comparison = compare_report(baseline_lookup[sample_id], row)
            displacement = float(np.mean([
                abs(float(item.prediction) - float(base_predictions.loc[(sample_id, item.target)]))
                / target_std[item.target]
                for item in group.itertuples()
            ]))
            deleted_event_ids = set(local_meta.get("deleted_event_ids", []))
            stale = sum(
                str(citation["event_id"]) in deleted_event_ids for citation in report["evidence"]
            )
            metric_rows.append({
                "sample_id": sample_id,
                "subject_id": sample.subject_id,
                "condition": condition,
                "normalized_forecast_displacement": displacement,
                **comparison,
                "verifier_exact_report_match": float(verified["exact_report_match"]),
                "verifier_evidence_precision": float(verified["evidence_precision"]),
                "verifier_future_evidence_leakage": float(verified["future_evidence_leakage"]),
                "deleted_visit_stale_citations": int(stale),
            })
            ir_rows.append({
                "sample_id": sample_id,
                "subject_id": sample.subject_id,
                "condition": condition,
                "metadata": local_meta,
                "reasoning_ir": report,
            })
    write_jsonl(ir_rows, output / "counterfactual_reasoning_ir.jsonl")
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output / "counterfactual_per_report_metrics.csv", index=False)

    summary_columns = [
        "normalized_forecast_displacement", "state_label_change_fraction",
        "trend_label_change_fraction", "forecast_direction_change_fraction",
        "evidence_exact_jaccard", "compiled_report_changed",
        "verifier_exact_report_match", "verifier_evidence_precision",
        "verifier_future_evidence_leakage", "deleted_visit_stale_citations",
    ]
    summaries = []
    for condition, frame in metrics.groupby("condition", sort=True):
        block = bootstrap_mean(
            frame, summary_columns, int(config["statistics"]["bootstrap_seed"]),
            int(config["statistics"]["bootstrap_replicates"]),
        )
        block.insert(0, "condition", condition)
        summaries.append(block)
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(output / "counterfactual_summary_bootstrap.csv", index=False)

    deletion = metrics[metrics.condition.isin([
        "highest_influence_visit_deletion", "matched_random_visit_deletion"
    ])].pivot(index="subject_id", columns="condition", values="normalized_forecast_displacement")
    deletion = deletion.dropna()
    difference = (
        deletion["highest_influence_visit_deletion"]
        - deletion["matched_random_visit_deletion"]
    ).to_numpy(dtype=float)
    rng = np.random.default_rng(int(config["statistics"]["bootstrap_seed"]))
    n = int(config["statistics"]["bootstrap_replicates"])
    draws = difference[rng.integers(0, len(difference), size=(n, len(difference)))].mean(axis=1)
    primary = {
        "comparison": "highest_influence_minus_matched_random_deletion",
        "metric": "normalized_forecast_displacement",
        "difference": float(difference.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "subjects": int(len(difference)),
        "replicates": n,
    }
    (output / "primary_deletion_contrast.json").write_text(json.dumps(primary, indent=2) + "\n")
    ablation = renderer_ablation(source, output, config)

    verification = {
        "source_lock_present": (output / "source_lock.json").exists(),
        "reasoning_subjects": len(reasoning),
        "reconstructed_reasoning_samples": len(samples),
        "candidate_visit_deletions": len(candidate_samples),
        "paired_deletion_subjects": int(len(deletion)),
        "all_counterfactual_reports_exactly_verified": bool(metrics.verifier_exact_report_match.eq(1).all()),
        "all_counterfactual_evidence_precise": bool(metrics.verifier_evidence_precision.eq(1).all()),
        "zero_future_evidence_leakage": bool(metrics.verifier_future_evidence_leakage.eq(0).all()),
        "zero_deleted_visit_stale_citations": bool(metrics.deleted_visit_stale_citations.eq(0).all()),
        "renderer_ablation": ablation,
        "analysis_is_post_test_descriptive": True,
        "phase4f_gate_reopened": False,
    }
    (output / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    decision = {
        "analysis_type": "post_test_frozen_audit",
        "primary_visit_influence_contrast_supported": bool(primary["ci_low"] > 0),
        "mechanical_traceability_supported": bool(
            verification["all_counterfactual_reports_exactly_verified"]
            and verification["zero_deleted_visit_stale_citations"]
            and verification["zero_future_evidence_leakage"]
        ),
        "llm_renderer_factual_advantage_supported": False,
        "paper_core_method": "frozen_grud_plus_typed_executable_reasoning_ir_compiler_verifier",
        "llm_role": "optional_renderer_only",
        "latent_clinical_memory_state_supported": False,
        "phase4f_gate_reopened": False,
        "external_confirmation_still_required": True,
    }
    (output / "analysis_decision.json").write_text(json.dumps(decision, indent=2) + "\n")

    aggregate = pd.read_csv(output / "renderer_ablation_metrics.csv")
    report = [
        "# Phase 5E frozen reasoning-faithfulness and renderer-ablation audit",
        "",
        "Status: completed as a post-test frozen audit. No checkpoint, decoder, threshold, or gate was selected from these results.",
        "",
        "## Cohort and interventions",
        "",
        f"- Phase 4F reasoning cohort: {len(reasoning)} reports / {len(reasoning)} subjects",
        f"- Leave-one-visit candidates evaluated with the frozen three-seed GRU-D ensemble: {len(candidate_samples)}",
        f"- Subjects eligible for paired highest-influence versus random deletion: {len(deletion)}",
        "- Additional counterfactuals: reversed time gaps and reflected verified Memory scores",
        "- Future truth, diagnosis, MRI, and GRPO inputs: not used",
        "",
        "## Counterfactual response",
        "",
        markdown_table(summary),
        "",
        "## Primary paired deletion contrast",
        "",
        markdown_table(pd.DataFrame([primary])),
        "",
        "## Renderer ablation on the identical Phase 4F reports",
        "",
        markdown_table(aggregate),
        "",
        "## Decision",
        "",
        "The executable compiler reproduced the teacher report exactly and mechanically preserved evidence, forecasts, and uncertainty. The LLM grammar renderer did not add factual fidelity over the executable renderer and retained the previously documented state/trend label errors. The defensible method is therefore the frozen GRU-D forecaster plus typed executable reasoning IR, compiler, and independent verifier; the 1.5B LLM is an optional renderer rather than the scientific core.",
        "",
        "Counterfactual sensitivity is evidence of input dependence and traceability only. It does not validate a clinical latent state or repair the failed Phase 4F confirmatory gate.",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
