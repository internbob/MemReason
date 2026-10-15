"""Train-only frozen cognitive forecast and deterministic structured-reasoning baseline."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader

from ..data.common import DOMAINS, ROOT, save_csv, save_json, valid_value
from ..data.dictionary import load_dictionary
from ..models.reasoning_llm import parse_report
from .phase1_numeric import (
    NumericCollator,
    NumericForecastDataset,
    build_samples,
    clean_frame,
    fit_normalization,
    markdown_table,
)
from .phase1_baseline_suite import (
    build_model,
    checkpoint_path,
    select_verified_scalar_features,
    train_one,
)


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def assign_reasoning_splits(
    frame: pd.DataFrame, eligible_subjects: set[str], config: dict[str, Any]
) -> pd.DataFrame:
    severity = {"Unknown": 0, "CN": 1, "MCI": 2, "AD": 3}
    rows = []
    eligible = frame[frame.subject_id.astype(str).isin(eligible_subjects)]
    for subject, group in eligible.groupby("subject_id", sort=True):
        observed = group.dxsum_diagnosis_label.fillna("Unknown").astype(str)
        level = max([severity.get(value, 0) for value in observed], default=0)
        diagnosis = next(label for label, value in severity.items() if value == level)
        rows.append({"subject_id": str(subject), "stratum": diagnosis})
    registry = pd.DataFrame(rows)
    counts = registry.stratum.value_counts()
    registry.loc[registry.stratum.map(counts).lt(3), "stratum"] = registry.stratum.mode().iloc[0]
    settings = config["reasoning_split"]
    held_fraction = float(settings["dev_fraction"]) + float(settings["confirmation_fraction"])
    first = StratifiedShuffleSplit(n_splits=1, test_size=held_fraction, random_state=int(settings["seed"]))
    build_index, held_index = next(first.split(registry, registry.stratum))
    registry["reasoning_split"] = "build"
    held = registry.iloc[held_index].copy()
    confirmation_share = float(settings["confirmation_fraction"]) / held_fraction
    second = StratifiedShuffleSplit(n_splits=1, test_size=confirmation_share,
                                    random_state=int(settings["seed"]) + 1)
    dev_local, confirmation_local = next(second.split(held, held.stratum))
    registry.loc[held.iloc[dev_local].index, "reasoning_split"] = "dev"
    registry.loc[held.iloc[confirmation_local].index, "reasoning_split"] = "confirmation"
    if registry.subject_id.duplicated().any() or set(registry.reasoning_split) != {"build", "dev", "confirmation"}:
        raise RuntimeError("Invalid Phase 4A subject registry")
    return registry


def predict_all_targets(
    seed: int, samples: list, features: list[str], targets: list[str],
    normalization: dict[str, dict[str, float]], config: dict[str, Any],
    output: Path, device: torch.device,
) -> pd.DataFrame:
    record = torch.load(checkpoint_path(output, "GRU-D", seed), map_location="cpu", weights_only=True)
    if record["features"] != features or record["targets"] != targets:
        raise RuntimeError("Frozen GRU-D data contract mismatch")
    model = build_model(
        "GRU-D", len(features), len(targets), int(config["training"]["hidden_dim"]),
        int(config["training"]["layers"]), float(config["training"]["dropout"]),
    ).to(device)
    model.load_state_dict(record["model"])
    collator = NumericCollator(features, targets, normalization)
    loader = DataLoader(NumericForecastDataset(samples), batch_size=int(config["training"]["batch_size"]),
                        shuffle=False, collate_fn=collator, num_workers=0)
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            local_samples = batch.pop("samples")
            tensors = {key: value.to(device) for key, value in batch.items()}
            prediction = model(tensors).cpu().numpy() * collator.target_std[None] + collator.target_mean[None]
            for local, sample in enumerate(local_samples):
                for target_index, target in enumerate(targets):
                    rows.append({
                        "sample_id": sample.sample_id, "subject_id": sample.subject_id,
                        "reasoning_split": sample.split, "seed": seed, "target": target,
                        "horizon_months": sample.horizon_months, "cutoff_month": sample.cutoff_month,
                        "target_month": float(sample.target_months[target_index]) if sample.target_mask[target_index] else np.nan,
                        "truth": float(sample.targets[target_index]) if sample.target_mask[target_index] else np.nan,
                        "prediction": float(prediction[local, target_index]),
                    })
    return pd.DataFrame(rows)


def ensemble_predictions(seed_predictions: pd.DataFrame) -> pd.DataFrame:
    keys = ["sample_id", "subject_id", "reasoning_split", "target", "horizon_months", "cutoff_month"]
    ensemble = seed_predictions.groupby(keys, as_index=False).agg(
        prediction=("prediction", "mean"), ensemble_std=("prediction", "std"),
        truth=("truth", "first"), target_month=("target_month", "first"), seeds=("seed", "nunique"),
    )
    if not ensemble.seeds.eq(seed_predictions.seed.nunique()).all():
        raise RuntimeError("Incomplete forecast ensemble")
    ensemble["ensemble_std"] = ensemble.ensemble_std.fillna(0.0)
    return ensemble


def parse_events(path: Path, features: set[str]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    raw = pd.read_csv(path, low_memory=False)
    by_subject: dict[str, list[dict[str, Any]]] = {}
    store: dict[str, dict[str, Any]] = {}
    for row in raw.itertuples(index=False):
        try:
            payload = json.loads(row.payload_json) if isinstance(row.payload_json, str) else {}
        except json.JSONDecodeError:
            payload = {}
        payload = {key: float(value) for key, value in payload.items()
                   if key in features and value is not None and np.isfinite(float(value))}
        if not payload:
            continue
        event = {"event_id": str(row.event_id), "subject_id": str(row.subject_id),
                 "visit_month": float(row.visit_month), "time_month": float(row.time_month),
                 "instrument": str(row.instrument), "payload": payload}
        by_subject.setdefault(event["subject_id"], []).append(event)
        store[event["event_id"]] = event
    for events in by_subject.values():
        events.sort(key=lambda event: (event["visit_month"], event["time_month"], event["event_id"]))
    return by_subject, store


def severity(value: float, item: str, dictionary: dict[str, Any],
             normalization: dict[str, dict[str, float]]) -> float:
    z = (float(value) - float(normalization[item]["mean"])) / float(normalization[item]["std"])
    return float(-z if dictionary[item]["direction"] == "higher=better" else z)


def history_contract(
    subject: str, cutoff: float, events_by_subject: dict[str, list[dict[str, Any]]],
    dictionary: dict[str, Any], normalization: dict[str, dict[str, float]], config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], float]:
    history = [event for event in events_by_subject.get(subject, []) if event["visit_month"] <= cutoff + 1e-6]
    observations = []
    for event in history:
        for item, value in event["payload"].items():
            observations.append({
                "event_id": event["event_id"], "item": item, "time_month": event["time_month"],
                "visit_month": event["visit_month"], "value": value, "domain": dictionary[item]["domain"],
            })
    if not observations:
        raise RuntimeError(f"No verified evidence for {subject} at {cutoff}")
    observations.sort(key=lambda row: (row["visit_month"], row["time_month"], row["item"]))
    selected_items = []
    limit = int(config["reasoning"]["evidence_items_per_domain"])
    for domain in DOMAINS:
        candidates = [row for row in observations if row["domain"] == domain]
        latest = {}
        for row in candidates:
            latest[row["item"]] = row
        ranked = sorted(latest.values(), key=lambda row: (row["visit_month"], row["time_month"], row["item"]), reverse=True)
        selected_items.extend(row["item"] for row in ranked[:limit])
    selected = [row for row in observations if row["item"] in set(selected_items)]
    latest_by_item = {}
    for row in selected:
        latest_by_item[row["item"]] = row
    evidence = [
        {key: row[key] for key in ["event_id", "item", "time_month", "value", "domain"]}
        for row in sorted(latest_by_item.values(), key=lambda x: (x["domain"], x["item"]))
    ]
    state, trend = {}, {}
    state_threshold = float(config["reasoning"]["state_z_threshold"])
    trend_threshold = float(config["reasoning"]["trend_z_threshold"])
    for domain in DOMAINS:
        domain_rows = [row for row in selected if row["domain"] == domain]
        latest_rows = [row for row in latest_by_item.values() if row["domain"] == domain]
        if not latest_rows:
            state[domain] = {"status": "unavailable", "severity_z": None, "items": 0}
            trend[domain] = {"label": "insufficient_history", "start_severity_z": None,
                             "end_severity_z": None, "timepoints": 0}
            continue
        current = float(np.mean([severity(row["value"], row["item"], dictionary, normalization) for row in latest_rows]))
        status = "elevated_impairment" if current > state_threshold else (
            "lower_impairment" if current < -state_threshold else "within_training_reference"
        )
        state[domain] = {"status": status, "severity_z": round(current, 6), "items": len(latest_rows)}
        by_visit = []
        for visit, group in pd.DataFrame(domain_rows).groupby("visit_month", sort=True):
            by_visit.append((float(visit), float(np.mean([
                severity(row.value, row.item, dictionary, normalization) for row in group.itertuples(index=False)
            ]))))
        if len(by_visit) < 2:
            label = "insufficient_history"
        else:
            delta = by_visit[-1][1] - by_visit[0][1]
            label = "decline" if delta > trend_threshold else ("improve" if delta < -trend_threshold else "stable")
        trend[domain] = {"label": label, "start_severity_z": round(by_visit[0][1], 6),
                         "end_severity_z": round(by_visit[-1][1], 6), "timepoints": len(by_visit)}
    exact_cutoff = max(row["time_month"] for row in selected)
    return selected, state, trend, float(exact_cutoff)


def confidence_level(normalized_std: float, config: dict[str, Any]) -> str:
    if normalized_std <= float(config["reasoning"]["confidence_high_normalized_std"]):
        return "high"
    if normalized_std <= float(config["reasoning"]["confidence_moderate_normalized_std"]):
        return "moderate"
    return "low"


def make_example(
    sample: Any, forecasts: pd.DataFrame, events_by_subject: dict[str, list[dict[str, Any]]],
    dictionary: dict[str, Any], normalization: dict[str, dict[str, float]], config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    history, state, trend, exact_cutoff = history_contract(
        sample.subject_id, sample.cutoff_month, events_by_subject, dictionary, normalization, config
    )
    evidence_latest = {}
    for row in history:
        evidence_latest[row["item"]] = row
    evidence = [
        {key: row[key] for key in ["event_id", "item", "time_month", "value", "domain"]}
        for row in sorted(evidence_latest.values(), key=lambda x: (x["domain"], x["item"]))
    ]
    target_names = list(config["targets"])
    prediction_rows = forecasts.set_index("target")
    forecast, confidence, last_targets = {}, {}, {}
    directions = {"adas_total13": "higher=worse", "cdr_cdrsb": "higher=worse",
                  "faq_faqtotal": "higher=worse", "mmse_mmscore": "higher=better",
                  "moca_moca": "higher=better"}
    for index, target in enumerate(target_names):
        row = prediction_rows.loc[target]
        observed = sample.history_targets[:, index][sample.history_target_mask[:, index]]
        last = float(observed[-1]) if len(observed) else None
        mean, std = float(row.prediction), float(row.ensemble_std)
        if last is None:
            change = "unknown"
        else:
            delta = mean - last
            if abs(delta) <= 0.05 * float(normalization[target]["std"]):
                change = "stable"
            else:
                improves = delta > 0 if directions[target] == "higher=better" else delta < 0
                change = "improve" if improves else "decline"
        forecast[target] = {"mean": round(mean, 6), "horizon_months": int(sample.horizon_months),
                            "change_from_last": change}
        normalized_std = std / float(normalization[target]["std"])
        confidence[target] = {"ensemble_std": round(std, 6),
                              "normalized_std": round(normalized_std, 6),
                              "level": confidence_level(normalized_std, config)}
        last_targets[target] = last
    report = parse_report({"current_state": state, "evidence": evidence, "trend": trend,
                           "forecast": forecast, "confidence": confidence})
    prompt = {
        "task": "Produce the documented five-section structured longitudinal cognitive report.",
        "subject_id": sample.subject_id, "cutoff_visit_month": float(sample.cutoff_month),
        "evidence_cutoff_time_month": exact_cutoff, "forecast_horizon_months": int(sample.horizon_months),
        "verified_history": history, "last_observed_targets": last_targets,
        "constraints": {"use_only_verified_history": True, "do_not_infer_diagnosis": True,
                        "MRI_available_for_reasoning": False},
    }
    return {"sample_id": sample.sample_id, "subject_id": sample.subject_id,
            "reasoning_split": sample.split, "prompt": prompt, "completion": report}, report


def evaluate_example(
    example: dict[str, Any], reference: dict[str, Any], store: dict[str, dict[str, Any]],
    dictionary: dict[str, Any], expected_report: dict[str, Any],
) -> dict[str, Any]:
    try:
        report = parse_report(example["completion"])
        format_valid = True
    except (ValueError, TypeError, json.JSONDecodeError):
        report, format_valid = {}, False
    expected_evidence = {(row["event_id"], row["item"]) for row in expected_report["evidence"]}
    actual_evidence = report.get("evidence", []) if format_valid else []
    actual_keys = {(row.get("event_id"), row.get("item")) for row in actual_evidence}
    matches, numeric, times, domains, future = 0, 0, 0, 0, 0
    cutoff_visit = float(example["prompt"]["cutoff_visit_month"])
    cutoff_time = float(example["prompt"]["evidence_cutoff_time_month"])
    for citation in actual_evidence:
        event = store.get(citation.get("event_id"))
        item = citation.get("item")
        valid = bool(event and event["subject_id"] == example["subject_id"] and item in event["payload"]
                     and event["visit_month"] <= cutoff_visit + 1e-6 and event["time_month"] <= cutoff_time + 1e-6)
        matches += int(valid)
        if valid:
            numeric += int(np.isclose(float(citation["value"]), float(event["payload"][item]), atol=1e-6))
            times += int(np.isclose(float(citation["time_month"]), float(event["time_month"]), atol=1e-6))
            domains += int(citation.get("domain") == dictionary[item]["domain"])
        future += int(bool(event and event["visit_month"] > cutoff_visit + 1e-6))
    count = len(actual_evidence)
    precision = matches / count if count else 0.0
    recall = len(actual_keys & expected_evidence) / len(expected_evidence) if expected_evidence else 1.0
    return {
        "sample_id": example["sample_id"], "subject_id": example["subject_id"],
        "reasoning_split": example["reasoning_split"], "horizon_months": int(example["prompt"]["forecast_horizon_months"]),
        "format_valid": float(format_valid), "evidence_precision": precision, "evidence_recall": recall,
        "numeric_citation_accuracy": numeric / count if count else 0.0,
        "time_citation_accuracy": times / count if count else 0.0,
        "hallucinated_evidence_rate": 1.0 - precision, "domain_accuracy": domains / count if count else 0.0,
        "current_state_consistency": float(report.get("current_state") == expected_report["current_state"]),
        "temporal_consistency": float(report.get("trend") == expected_report["trend"]),
        "forecast_preservation": float(report.get("forecast") == expected_report["forecast"]),
        "uncertainty_preservation": float(report.get("confidence") == expected_report["confidence"]),
        "future_evidence_leakage": future / count if count else 0.0,
    }


def main(config_path: Path, smoke: bool = False) -> None:
    config = load_config(config_path)
    if smoke:
        config = copy.deepcopy(config)
        config["run_id"] += "_smoke"
        config["neural_seeds"] = config["neural_seeds"][:1]
        config["training"].update(max_epochs=2, patience=2, batch_size=64)
    output = ROOT / "outputs" / config["run_id"]
    output.mkdir(parents=True, exist_ok=True)
    save_json(config, output / "resolved_config.json")
    dictionary = load_dictionary()
    features = select_verified_scalar_features(config, dictionary)
    targets = list(config["targets"])
    source = pd.read_csv(ROOT / "splits/train.csv", low_memory=False)
    clean = clean_frame(source, features, targets, dictionary)
    horizons = [int(value) for value in config["future_forecast"]["horizons_months"]]
    tolerances = {int(key): int(value) for key, value in config["future_forecast"]["tolerance_months"].items()}
    all_samples = build_samples(clean, "all", features, targets, horizons, tolerances)
    eligible_subjects = {sample.subject_id for sample in all_samples}
    registry = assign_reasoning_splits(source, eligible_subjects, config)
    if smoke:
        registry = registry.groupby("reasoning_split", group_keys=False).head(40).reset_index(drop=True)
    save_csv(registry, output / "subject_registry.csv")
    frames, samples = {}, {}
    for split in ["build", "dev", "confirmation"]:
        subjects = set(registry.loc[registry.reasoning_split.eq(split), "subject_id"])
        frames[split] = clean[clean.subject_id.astype(str).isin(subjects)].copy()
        samples[split] = build_samples(frames[split], split, features, targets, horizons, tolerances)
    normalization = fit_normalization(frames["build"], features, targets)
    save_json(normalization, output / "normalization.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_predictions, parameter_counts = [], []
    for raw_seed in config["neural_seeds"]:
        seed = int(raw_seed)
        _, count = train_one("GRU-D", seed, samples["build"], samples["dev"], features, targets,
                             normalization, config, output, device)
        parameter_counts.append(count)
        for split in ["build", "dev", "confirmation"]:
            seed_predictions.append(predict_all_targets(seed, samples[split], features, targets,
                                                        normalization, config, output, device))
    seed_predictions = pd.concat(seed_predictions, ignore_index=True)
    ensemble = ensemble_predictions(seed_predictions)
    save_csv(seed_predictions, output / "seed_forecast_predictions.csv")
    save_csv(ensemble, output / "ensemble_forecast_predictions.csv")
    events_by_subject, event_store = parse_events(ROOT / "splits/train_events.csv", set(features))
    sample_lookup = {sample.sample_id: sample for values in samples.values() for sample in values}
    examples, evaluations = [], []
    jsonl = output / "structured_reasoning_examples.jsonl"
    with jsonl.open("w") as stream:
        for sample_id, group in ensemble.groupby("sample_id", sort=True):
            sample = sample_lookup[sample_id]
            example, expected = make_example(sample, group, events_by_subject, dictionary, normalization, config)
            stream.write(json.dumps(example, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
            examples.append(example)
            evaluations.append(evaluate_example(example, example["prompt"], event_store, dictionary, expected))
    evaluation = pd.DataFrame(evaluations)
    save_csv(evaluation, output / "reasoning_evaluation.csv")
    summary = evaluation.groupby("reasoning_split", as_index=False).agg(
        reports=("sample_id", "size"), subjects=("subject_id", "nunique"),
        format_validity=("format_valid", "mean"), evidence_precision=("evidence_precision", "mean"),
        evidence_recall=("evidence_recall", "mean"), numeric_citation_accuracy=("numeric_citation_accuracy", "mean"),
        time_citation_accuracy=("time_citation_accuracy", "mean"),
        hallucinated_evidence_rate=("hallucinated_evidence_rate", "mean"), domain_accuracy=("domain_accuracy", "mean"),
        current_state_consistency=("current_state_consistency", "mean"),
        temporal_consistency=("temporal_consistency", "mean"), forecast_preservation=("forecast_preservation", "mean"),
        uncertainty_preservation=("uncertainty_preservation", "mean"),
        future_evidence_leakage=("future_evidence_leakage", "mean"),
    )
    save_csv(summary, output / "reasoning_summary.csv")
    available = ensemble[ensemble.truth.notna()].copy()
    available["normalized_absolute_error"] = (
        (available.prediction - available.truth).abs()
        / available.target.map(lambda target: normalization[target]["std"])
    )
    available["normalized_uncertainty"] = (
        available.ensemble_std / available.target.map(lambda target: normalization[target]["std"])
    )
    metric_rows = []
    for (split, target, horizon), group in available.groupby(["reasoning_split", "target", "horizon_months"]):
        metric_rows.append({"reasoning_split": split, "target": target, "horizon_months": horizon,
                            "n": len(group), "subjects": group.subject_id.nunique(),
                            "mae": float((group.prediction - group.truth).abs().mean()),
                            "normalized_mae": float(group.normalized_absolute_error.mean()),
                            "mean_normalized_uncertainty": float(group.normalized_uncertainty.mean())})
    forecast_metrics = pd.DataFrame(metric_rows)
    save_csv(forecast_metrics, output / "forecast_metrics.csv")
    confirmation_available = available[available.reasoning_split.eq("confirmation")]
    correlation = spearmanr(
        confirmation_available.normalized_uncertainty,
        confirmation_available.normalized_absolute_error,
        nan_policy="omit",
    )
    correlation_value = float(correlation.statistic) if np.isfinite(correlation.statistic) else None
    correlation_pvalue = float(correlation.pvalue) if np.isfinite(correlation.pvalue) else None
    confirmation = summary.set_index("reasoning_split").loc["confirmation"]
    gate = config["gate"]
    criteria = {
        "format_validity": bool(confirmation.format_validity >= float(gate["minimum_format_validity"])),
        "evidence_precision": bool(confirmation.evidence_precision >= float(gate["minimum_evidence_precision"])),
        "evidence_recall": bool(confirmation.evidence_recall >= float(gate["minimum_evidence_recall"])),
        "numeric_citation_accuracy": bool(confirmation.numeric_citation_accuracy >= float(gate["minimum_numeric_citation_accuracy"])),
        "time_citation_accuracy": bool(confirmation.time_citation_accuracy >= float(gate["minimum_time_citation_accuracy"])),
        "hallucinated_evidence_rate": bool(confirmation.hallucinated_evidence_rate <= float(gate["maximum_hallucinated_evidence_rate"])),
        "domain_accuracy": bool(confirmation.domain_accuracy >= float(gate["minimum_domain_accuracy"])),
        "current_state_consistency": bool(confirmation.current_state_consistency >= float(gate["minimum_current_state_consistency"])),
        "temporal_consistency": bool(confirmation.temporal_consistency >= float(gate["minimum_temporal_consistency"])),
        "forecast_preservation": bool(confirmation.forecast_preservation >= float(gate["minimum_forecast_preservation"])),
        "uncertainty_preservation": bool(confirmation.uncertainty_preservation >= float(gate["minimum_uncertainty_preservation"])),
        "zero_future_evidence_leakage": bool(confirmation.future_evidence_leakage <= float(gate["maximum_future_evidence_leakage"])),
    }
    passed = bool(all(criteria.values())) and not smoke
    decision = {
        "run_id": config["run_id"], "engineering_smoke_only": smoke, "original_train_only": True,
        "validation_evaluated": False, "test_evaluated": False, "llm_trained": False, "grpo_performed": False,
        "criteria": criteria, "confirmation_reports": int(confirmation.reports),
        "confirmation_subjects": int(confirmation.subjects),
        "uncertainty_error_spearman": correlation_value,
        "uncertainty_error_pvalue": correlation_pvalue,
        "structured_reasoning_baseline_passed": passed,
        "phase4b_sft_ready": passed,
        "next_action": "train_cognitive_only_structured_reasoning_sft" if passed else "repair_structured_report_contract",
    }
    save_json(decision, output / "development_decision.json")
    preview = [example for example in examples if example["reasoning_split"] == "confirmation"][:3]
    save_json(preview, output / "confirmation_examples_preview.json")
    macro = forecast_metrics.groupby("reasoning_split", as_index=False).agg(
        normalized_mae=("normalized_mae", "mean"), mean_normalized_uncertainty=("mean_normalized_uncertainty", "mean")
    )
    report = f"""# Phase 4A frozen cognitive structured-reasoning baseline

Status: completed on a new subject-disjoint partition of original-train subjects. Original validation and test outcomes were not used.

## Data and frozen forecast

- Verified scalar cognitive inputs: {len(features)}
- Subject counts: {registry.groupby('reasoning_split').subject_id.nunique().to_dict()}
- Forecast-sample counts: { {key: len(value) for key, value in samples.items()} }
- GRU-D seeds/checkpoints: {config['neural_seeds']} / {len(config['neural_seeds'])}
- Trainable parameters per checkpoint: {sorted(set(parameter_counts))}
- LLM trained: no
- MRI/diagnosis used as reasoning input: no

## Frozen forecast description

{markdown_table(macro)}

Confirmation uncertainty-error Spearman: {correlation_value} (p={correlation_pvalue}). This is descriptive and is not part of the mechanical report gate.

## Structured-report verification

{markdown_table(summary)}

## Frozen decision

```json
{json.dumps(decision, indent=2)}
```

The JSONL completion is a deterministic, auditable teacher baseline rather than free chain-of-thought. Future truth is stored only in the forecast evaluation table and is absent from prompts and completions. A pass opens cognitive-only Phase 4B SFT development; it does not open multimodal reasoning or GRPO.
"""
    (output / "report.md").write_text(report)
    save_json({"features": features, "targets": targets,
               "subject_counts": registry.groupby("reasoning_split").subject_id.nunique().to_dict(),
               "sample_counts": {key: len(value) for key, value in samples.items()},
               "checkpoint_count": len(config["neural_seeds"]),
               "structured_examples": len(examples)}, output / "manifest.json")
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/phase4a_frozen_cognitive_reasoning.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    main(args.config, args.smoke)
