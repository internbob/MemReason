"""Phase 4C dev-only token-trie decoding and new-subject confirmation."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from ..data.common import DOMAINS, ROOT
from ..data.dictionary import load_dictionary
from .phase1_baseline_suite import select_verified_scalar_features
from .phase1_numeric import build_samples, clean_frame, markdown_table
from .phase4a_structured_reasoning import (
    ensemble_predictions, make_example, parse_events, predict_all_targets,
)
from .phase4b_reasoning_sft import (
    PromptCollator, Rows, load_base, load_rows, phase4b_input, score_generation,
    serialize_prompt, stable_subset, tokenizer_for,
)


CONFIG = ROOT / "configs/phase4c_grammar_constrained_decoding.yaml"


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG.read_text())


def output_dir(config: dict[str, Any]) -> Path:
    path = ROOT / "outputs" / config["run_id"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def prepare(config: dict[str, Any], tokenizer) -> None:
    output = output_dir(config)
    phase4a = ROOT / "outputs" / config["phase4a_run_id"]
    phase4b = ROOT / "outputs" / config["phase4b_run_id"]
    phase4a_config = yaml.safe_load((ROOT / "configs/phase4a_frozen_cognitive_reasoning.yaml").read_text())

    dev = stable_subset(
        load_rows(phase4b, "dev"),
        per_subject=int(config["selection"]["maximum_dev_examples_per_subject"]),
        seed=int(config["selection_seed"]),
    )
    write_jsonl(dev, output / "dev_examples.jsonl")

    dictionary = load_dictionary()
    features = select_verified_scalar_features(phase4a_config, dictionary)
    targets = list(phase4a_config["targets"])
    source = pd.read_csv(ROOT / "splits/val.csv", low_memory=False)
    clean = clean_frame(source, features, targets, dictionary)
    horizons = [int(value) for value in phase4a_config["future_forecast"]["horizons_months"]]
    tolerances = {int(key): int(value) for key, value in phase4a_config["future_forecast"]["tolerance_months"].items()}
    samples = build_samples(clean, "new_confirmation", features, targets, horizons, tolerances)
    normalization = json.loads((phase4a / "normalization.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_predictions = []
    for raw_seed in phase4a_config["neural_seeds"]:
        seed_predictions.append(predict_all_targets(
            int(raw_seed), samples, features, targets, normalization,
            phase4a_config, phase4a, device,
        ))
    seed_predictions = pd.concat(seed_predictions, ignore_index=True)
    ensemble = ensemble_predictions(seed_predictions)
    seed_predictions.to_csv(output / "new_confirmation_seed_forecasts.csv", index=False)
    ensemble.to_csv(output / "new_confirmation_ensemble_forecasts.csv", index=False)
    events_by_subject, _ = parse_events(ROOT / "splits/val_events.csv", set(features))
    sample_lookup = {sample.sample_id: sample for sample in samples}
    candidates = []
    for sample_id, group in ensemble.groupby("sample_id", sort=True):
        example, _ = make_example(
            sample_lookup[sample_id], group, events_by_subject, dictionary,
            normalization, phase4a_config,
        )
        candidates.append({
            "sample_id": example["sample_id"], "subject_id": example["subject_id"],
            "reasoning_split": "new_confirmation", "model_input": phase4b_input(example),
            "completion": example["completion"],
        })
    confirmation = stable_subset(
        candidates,
        per_subject=int(config["selection"]["maximum_confirmation_examples_per_subject"]),
        seed=int(config["selection_seed"]),
    )
    write_jsonl(confirmation, output / "new_confirmation_examples.jsonl")

    train_subjects = set(pd.read_csv(ROOT / "splits/train.csv", usecols=["subject_id"]).subject_id.astype(str))
    val_subjects = set(pd.read_csv(ROOT / "splits/val.csv", usecols=["subject_id"]).subject_id.astype(str))
    test_subjects = set(pd.read_csv(ROOT / "splits/test.csv", usecols=["subject_id"]).subject_id.astype(str))
    dev_subjects = {row["subject_id"] for row in dev}
    confirmation_subjects = {row["subject_id"] for row in confirmation}
    if not dev_subjects <= train_subjects or not confirmation_subjects <= val_subjects:
        raise RuntimeError("Phase 4C split provenance mismatch")
    if dev_subjects & confirmation_subjects or confirmation_subjects & test_subjects:
        raise RuntimeError("Phase 4C subject leakage")

    lengths = []
    for split, rows in [("dev", dev), ("new_confirmation", confirmation)]:
        for row in rows:
            prompt_tokens = len(tokenizer.encode(serialize_prompt(tokenizer, row["model_input"]), add_special_tokens=False))
            completion_tokens = len(tokenizer.encode(json.dumps(row["completion"], separators=(",", ":")) + tokenizer.eos_token,
                                                     add_special_tokens=False))
            lengths.append({"sample_id": row["sample_id"], "split": split,
                            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens})
    pd.DataFrame(lengths).to_csv(output / "token_lengths.csv", index=False)
    manifest = {
        "run_id": config["run_id"], "dev_examples": len(dev),
        "dev_subjects": len(dev_subjects), "new_confirmation_candidates": len(candidates),
        "new_confirmation_examples": len(confirmation),
        "new_confirmation_subjects": len(confirmation_subjects),
        "new_confirmation_horizons": sorted({int(row["model_input"]["horizon"]) for row in confirmation}),
        "maximum_total_tokens": int(pd.DataFrame(lengths).total_tokens.max()),
        "original_train_used_for_decoder_selection": True,
        "original_validation_used_only_after_decoder_freeze": True,
        "original_test_used": False, "previous_confirmation_reused": False,
        "future_truth_in_prompt": False, "diagnosis_input": False, "mri_input": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def load_split(output: Path, split: str) -> list[dict[str, Any]]:
    name = "dev_examples.jsonl" if split == "dev" else "new_confirmation_examples.jsonl"
    return read_jsonl(output / name)


def load_sft_model(config: dict[str, Any]):
    from peft import PeftModel
    phase4b = ROOT / "outputs" / config["phase4b_run_id"]
    phase4b_config = yaml.safe_load((ROOT / "configs/phase4b_cognitive_reasoning_sft.yaml").read_text())
    model = load_base(phase4b_config, training=False)
    model = PeftModel.from_pretrained(model, phase4b / "adapter", local_files_only=True)
    return model.to("cuda").eval()


def raw_decode(config: dict[str, Any], tokenizer, split: str) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Raw decoding requires CUDA")
    output = output_dir(config)
    rows = load_split(output, split)
    tokenizer.padding_side = "left"
    model = load_sft_model(config)
    loader = DataLoader(Rows(rows), batch_size=int(config["decoding"]["raw_batch_size"]), shuffle=False,
                        collate_fn=PromptCollator(tokenizer), num_workers=0)
    generations = []
    started = time.time()
    with torch.inference_mode():
        for batch in loader:
            local_rows = batch.pop("rows")
            tensors = {key: value.to("cuda") for key, value in batch.items()}
            width = tensors["input_ids"].shape[1]
            generated = model.generate(
                **tensors, max_new_tokens=int(config["decoding"]["raw_max_new_tokens"]),
                do_sample=False, temperature=None, top_p=None, use_cache=True,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
            )
            texts = tokenizer.batch_decode(generated[:, width:], skip_special_tokens=True)
            generations.extend({"sample_id": row["sample_id"], "subject_id": row["subject_id"],
                                "arm": "raw_sft", "generation": text}
                               for row, text in zip(local_rows, texts))
    write_jsonl(generations, output / f"{split}_generations_raw_sft.jsonl")
    (output / f"{split}_raw_state.json").write_text(json.dumps(
        {"examples": len(rows), "elapsed_minutes": (time.time() - started) / 60,
         "raw_without_repair": True}, indent=2) + "\n")


def latest_evidence(model_input: dict[str, Any]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in model_input["evidence_trajectories"]:
        old = latest.get(row["item"])
        key = (float(row["visit_month"]), float(row["time_month"]), row["event_id"])
        old_key = (-np.inf, -np.inf, "") if old is None else (
            float(old["visit_month"]), float(old["time_month"]), old["event_id"])
        if key > old_key:
            latest[row["item"]] = row
    return [{key: row[key] for key in ["event_id", "item", "time_month", "value", "domain"]}
            for row in sorted(latest.values(), key=lambda x: (x["domain"], x["item"]))]


def executable_status(statistics: dict[str, Any]) -> str:
    value = statistics["current_severity_z"]
    if value is None:
        return "unavailable"
    if float(value) > 0.5:
        return "elevated_impairment"
    if float(value) < -0.5:
        return "lower_impairment"
    return "within_training_reference"


def executable_trend(statistics: dict[str, Any]) -> str:
    if int(statistics["timepoints"]) < 2:
        return "insufficient_history"
    delta = float(statistics["end_severity_z"]) - float(statistics["start_severity_z"])
    if delta > 0.25:
        return "decline"
    if delta < -0.25:
        return "improve"
    return "stable"


def render_report(row: dict[str, Any], statuses: dict[str, str], trends: dict[str, str]) -> dict[str, Any]:
    model_input = row["model_input"]
    state, trend = {}, {}
    for domain in DOMAINS:
        stats = model_input["domain_statistics"][domain]
        state[domain] = {"status": statuses[domain], "severity_z": stats["current_severity_z"],
                         "items": stats["items"]}
        trend[domain] = {"label": trends[domain], "start_severity_z": stats["start_severity_z"],
                         "end_severity_z": stats["end_severity_z"], "timepoints": stats["timepoints"]}
    return {"current_state": state, "evidence": latest_evidence(model_input), "trend": trend,
            "forecast": model_input["frozen_forecast"], "confidence": model_input["frozen_confidence"]}


def token_trie_select(model, tokenizer, prefixes: list[str], choices: list[str], batch_size: int) -> list[str]:
    tokenizer.padding_side = "left"
    choice_sequences = [tokenizer.encode(choice + '"', add_special_tokens=False) for choice in choices]
    results = []
    for start in range(0, len(prefixes), batch_size):
        local = prefixes[start:start + batch_size]
        encoded = tokenizer(local, add_special_tokens=False, padding=True, return_tensors="pt").to("cuda")
        width = encoded.input_ids.shape[1]

        def allowed(_batch_id: int, input_ids: torch.Tensor) -> list[int]:
            suffix = input_ids[width:].tolist()
            candidates = [seq for seq in choice_sequences if seq[:len(suffix)] == suffix]
            if not candidates:
                return [tokenizer.eos_token_id]
            next_tokens = {seq[len(suffix)] for seq in candidates if len(seq) > len(suffix)}
            return sorted(next_tokens) if next_tokens else [tokenizer.eos_token_id]

        maximum = max(map(len, choice_sequences)) + 1
        with torch.inference_mode():
            generated = model.generate(
                **encoded, max_new_tokens=maximum, do_sample=False, temperature=None, top_p=None,
                prefix_allowed_tokens_fn=allowed, eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id, use_cache=True,
            )
        decoded = tokenizer.batch_decode(generated[:, width:], skip_special_tokens=True)
        for text in decoded:
            selected = text[:-1] if text.endswith('"') else text
            if selected not in choices:
                raise RuntimeError(f"Token trie emitted invalid choice: {text!r}")
            results.append(selected)
    return results


def grammar_decode(config: dict[str, Any], tokenizer, split: str) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Model-constrained decoding requires CUDA")
    output = output_dir(config)
    rows = load_split(output, split)
    model = load_sft_model(config)
    prompts = [serialize_prompt(tokenizer, row["model_input"]) for row in rows]
    parts = ['{"current_state":{' for _ in rows]
    status_values: list[dict[str, str]] = [dict() for _ in rows]
    trend_values: list[dict[str, str]] = [dict() for _ in rows]
    batch_size = int(config["decoding"]["constrained_batch_size"])
    for domain_index, domain in enumerate(DOMAINS):
        for index, row in enumerate(rows):
            if domain_index:
                parts[index] += ","
            parts[index] += json.dumps(domain) + ':{"status":"'
        prefixes = [prompt + part for prompt, part in zip(prompts, parts)]
        choices = token_trie_select(model, tokenizer, prefixes, list(config["decoding"]["status_choices"]), batch_size)
        for index, (row, choice) in enumerate(zip(rows, choices)):
            status_values[index][domain] = choice
            stats = row["model_input"]["domain_statistics"][domain]
            parts[index] += choice + '","severity_z":' + json.dumps(stats["current_severity_z"], separators=(",", ":"))
            parts[index] += ',"items":' + json.dumps(stats["items"]) + "}"
    for index, row in enumerate(rows):
        parts[index] += '},"evidence":' + json.dumps(latest_evidence(row["model_input"]), separators=(",", ":"))
        parts[index] += ',"trend":{'
    for domain_index, domain in enumerate(DOMAINS):
        for index in range(len(rows)):
            if domain_index:
                parts[index] += ","
            parts[index] += json.dumps(domain) + ':{"label":"'
        prefixes = [prompt + part for prompt, part in zip(prompts, parts)]
        choices = token_trie_select(model, tokenizer, prefixes, list(config["decoding"]["trend_choices"]), batch_size)
        for index, (row, choice) in enumerate(zip(rows, choices)):
            trend_values[index][domain] = choice
            stats = row["model_input"]["domain_statistics"][domain]
            parts[index] += choice + '","start_severity_z":' + json.dumps(stats["start_severity_z"])
            parts[index] += ',"end_severity_z":' + json.dumps(stats["end_severity_z"])
            parts[index] += ',"timepoints":' + json.dumps(stats["timepoints"]) + "}"
    model_generations, executable_generations = [], []
    for row, statuses, trends in zip(rows, status_values, trend_values):
        model_report = render_report(row, statuses, trends)
        executable_report = render_report(
            row,
            {domain: executable_status(row["model_input"]["domain_statistics"][domain]) for domain in DOMAINS},
            {domain: executable_trend(row["model_input"]["domain_statistics"][domain]) for domain in DOMAINS},
        )
        model_generations.append({"sample_id": row["sample_id"], "subject_id": row["subject_id"],
                                  "arm": "grammar_model", "generation": json.dumps(model_report, separators=(",", ":"))})
        executable_generations.append({"sample_id": row["sample_id"], "subject_id": row["subject_id"],
                                       "arm": "grammar_executable", "generation": json.dumps(executable_report, separators=(",", ":"))})
    write_jsonl(model_generations, output / f"{split}_generations_grammar_model.jsonl")
    write_jsonl(executable_generations, output / f"{split}_generations_grammar_executable.jsonl")


def score_file(rows: list[dict[str, Any]], path: Path, arm: str) -> pd.DataFrame:
    lookup = {row["sample_id"]: row for row in rows}
    generations = read_jsonl(path)
    if len(generations) != len(rows) or {x["sample_id"] for x in generations} != set(lookup):
        raise RuntimeError(f"Incomplete generations for {arm}")
    return pd.DataFrame([score_generation(lookup[x["sample_id"]], x["generation"], arm) for x in generations])


def gate_for(metrics: pd.Series, config: dict[str, Any]) -> dict[str, bool]:
    gate = config["gate"]
    return {
        "format_validity": metrics.format_validity >= gate["minimum_format_validity"],
        "evidence_precision": metrics.evidence_precision >= gate["minimum_evidence_precision"],
        "evidence_recall": metrics.evidence_recall >= gate["minimum_evidence_recall"],
        "numeric_citation_accuracy": metrics.numeric_citation_accuracy >= gate["minimum_numeric_citation_accuracy"],
        "time_citation_accuracy": metrics.time_citation_accuracy >= gate["minimum_time_citation_accuracy"],
        "domain_accuracy": metrics.domain_accuracy >= gate["minimum_domain_accuracy"],
        "current_state_consistency": metrics.current_state_consistency >= gate["minimum_current_state_consistency"],
        "temporal_consistency": metrics.temporal_consistency >= gate["minimum_temporal_consistency"],
        "forecast_preservation": metrics.forecast_preservation >= gate["minimum_forecast_preservation"],
        "uncertainty_preservation": metrics.uncertainty_preservation >= gate["minimum_uncertainty_preservation"],
        "hallucinated_evidence_rate": metrics.hallucinated_evidence_rate <= gate["maximum_hallucinated_evidence_rate"],
        "future_evidence_leakage": metrics.future_evidence_leakage <= gate["maximum_future_evidence_leakage"],
    }


def select_dev(config: dict[str, Any]) -> None:
    output = output_dir(config)
    rows = load_split(output, "dev")
    frames = []
    for arm in ["raw_sft", "grammar_model", "grammar_executable"]:
        frames.append(score_file(rows, output / f"dev_generations_{arm}.jsonl", arm))
    scored = pd.concat(frames, ignore_index=True)
    scored.to_csv(output / "dev_per_report_metrics.csv", index=False)
    metric_columns = [column for column in scored.columns if column not in {"sample_id", "subject_id", "arm"}]
    aggregate = scored.groupby("arm", as_index=False)[metric_columns].mean()
    aggregate.to_csv(output / "dev_aggregate_metrics.csv", index=False)
    indexed = aggregate.set_index("arm")
    gates = {arm: gate_for(indexed.loc[arm], config) for arm in indexed.index}
    selected = None
    for arm in config["decoding"]["arm_priority"]:
        if all(gates[arm].values()):
            selected = arm
            break
    decision = {
        "selection_uses_dev_only": True, "previous_confirmation_used": False,
        "arm_priority": config["decoding"]["arm_priority"],
        "criteria": {arm: {key: bool(value) for key, value in values.items()} for arm, values in gates.items()},
        "selected_arm": selected, "new_confirmation_opened": selected is not None,
    }
    (output / "dev_selection_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    if selected is None:
        raise RuntimeError("No Phase 4C decoder passed dev gate")


def confirm_selected(config: dict[str, Any], tokenizer) -> None:
    output = output_dir(config)
    decision = json.loads((output / "dev_selection_decision.json").read_text())
    arm = decision["selected_arm"]
    if arm == "grammar_model":
        grammar_decode(config, tokenizer, "new_confirmation")
    elif arm == "grammar_executable":
        rows = load_split(output, "new_confirmation")
        generations = []
        for row in rows:
            report = render_report(
                row,
                {domain: executable_status(row["model_input"]["domain_statistics"][domain]) for domain in DOMAINS},
                {domain: executable_trend(row["model_input"]["domain_statistics"][domain]) for domain in DOMAINS},
            )
            generations.append({"sample_id": row["sample_id"], "subject_id": row["subject_id"],
                                "arm": arm, "generation": json.dumps(report, separators=(",", ":"))})
        write_jsonl(generations, output / f"new_confirmation_generations_{arm}.jsonl")
    else:
        raise RuntimeError("Dev selection missing")


def subject_bootstrap(frame: pd.DataFrame, selected_arm: str, config: dict[str, Any]) -> pd.DataFrame:
    metrics = ["format_validity", "evidence_precision", "evidence_recall", "current_state_consistency",
               "temporal_consistency", "forecast_preservation", "uncertainty_preservation"]
    pivot = frame.pivot(index="subject_id", columns="arm", values=metrics)
    rng = np.random.default_rng(int(config["statistics"]["bootstrap_seed"]))
    indices = rng.integers(0, len(pivot), size=(int(config["statistics"]["bootstrap_replicates"]), len(pivot)))
    rows = []
    for metric in metrics:
        differences = (pivot[(metric, selected_arm)] - pivot[(metric, "raw_sft")]).to_numpy()
        draws = differences[indices].mean(axis=1)
        rows.append({"metric": metric, "selected_minus_raw": float(differences.mean()),
                     "ci_low": float(np.quantile(draws, .025)), "ci_high": float(np.quantile(draws, .975)),
                     "replicates": int(config["statistics"]["bootstrap_replicates"])})
    return pd.DataFrame(rows)


def summarize(config: dict[str, Any]) -> None:
    output = output_dir(config)
    decision = json.loads((output / "dev_selection_decision.json").read_text())
    selected = decision["selected_arm"]
    rows = load_split(output, "new_confirmation")
    frames = [
        score_file(rows, output / "new_confirmation_generations_raw_sft.jsonl", "raw_sft"),
        score_file(rows, output / f"new_confirmation_generations_{selected}.jsonl", selected),
    ]
    scored = pd.concat(frames, ignore_index=True)
    scored.to_csv(output / "confirmation_per_report_metrics.csv", index=False)
    metric_columns = [column for column in scored.columns if column not in {"sample_id", "subject_id", "arm"}]
    aggregate = scored.groupby("arm", as_index=False)[metric_columns].mean()
    aggregate.to_csv(output / "confirmation_aggregate_metrics.csv", index=False)
    bootstrap = subject_bootstrap(scored, selected, config)
    bootstrap.to_csv(output / "subject_cluster_bootstrap.csv", index=False)
    selected_metrics = aggregate.set_index("arm").loc[selected]
    criteria = gate_for(selected_metrics, config)
    passed = bool(all(criteria.values()))
    final = {
        "run_id": config["run_id"], "selected_arm": selected,
        "confirmation_examples": len(rows), "confirmation_subjects": len({row["subject_id"] for row in rows}),
        "criteria": {key: bool(value) for key, value in criteria.items()},
        "phase4c_structured_decoding_passed": passed,
        "stable_sft_policy_for_grpo": bool(passed and selected == "grammar_model"),
        "cognitive_grpo_ready": bool(passed and selected == "grammar_model"),
        "multimodal_reasoning_ready": False,
        "next_action": "freeze_constrained_cognitive_grpo_protocol" if passed and selected == "grammar_model"
                       else "retain_hybrid_executable_report_compiler",
    }
    (output / "development_decision.json").write_text(json.dumps(final, indent=2) + "\n")
    manifest = json.loads((output / "manifest.json").read_text())
    dev_metrics = pd.read_csv(output / "dev_aggregate_metrics.csv")
    report = f"""# Phase 4C dev-only grammar-constrained structured decoding

Status: completed with decoder selection on Phase 4A internal dev subjects and one-time confirmation on new original-validation subjects. The previous Phase 4B confirmation subjects were not reused.

## Cohorts and frozen components

- Dev: {manifest['dev_examples']} reports / {manifest['dev_subjects']} subjects
- New confirmation: {manifest['new_confirmation_examples']} reports / {manifest['new_confirmation_subjects']} subjects
- New confirmation source: original ADNI validation split
- Frozen components: Phase 4A GRU-D ensemble and normalization; Phase 4B DeepSeek-R1-Distill-Qwen-1.5B LoRA
- MRI, diagnosis, future truth, and original test used as reasoning inputs: no

## Dev-only decoder selection

{markdown_table(dev_metrics)}

Frozen selected arm: **{selected}**. The model-constrained arm had priority over the executable compiler whenever it satisfied every gate.

## New-subject confirmation

{markdown_table(aggregate)}

## Subject-cluster bootstrap: selected minus raw SFT

{markdown_table(bootstrap)}

## Decision

```json
{json.dumps(final, indent=2)}
```

`grammar_model` is token-trie-constrained model decoding. `grammar_executable` is a hybrid compiler using the documented thresholds and cannot by itself establish that the language-model policy is ready for GRPO.
"""
    (output / "report.md").write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=[
        "prepare", "dev-raw", "dev-grammar", "select-dev",
        "confirm-raw", "confirm-selected", "summarize",
    ])
    args = parser.parse_args()
    config = load_config()
    tokenizer = tokenizer_for(yaml.safe_load((ROOT / "configs/phase4b_cognitive_reasoning_sft.yaml").read_text()))
    if args.stage == "prepare": prepare(config, tokenizer)
    elif args.stage == "dev-raw": raw_decode(config, tokenizer, "dev")
    elif args.stage == "dev-grammar": grammar_decode(config, tokenizer, "dev")
    elif args.stage == "select-dev": select_dev(config)
    elif args.stage == "confirm-raw": raw_decode(config, tokenizer, "new_confirmation")
    elif args.stage == "confirm-selected": confirm_selected(config, tokenizer)
    else: summarize(config)


if __name__ == "__main__":
    main()
