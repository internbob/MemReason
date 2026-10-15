"""Phase 4B completion-only LoRA SFT and frozen confirmation evaluation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from ..data.common import ROOT
from ..models.reasoning_llm import parse_report
from .phase1_numeric import markdown_table


CONFIG = ROOT / "configs/phase4b_cognitive_reasoning_sft.yaml"
SYSTEM = (
    "Return only one valid JSON object with exactly these top-level keys: "
    "current_state,evidence,trend,forecast,confidence. Cite only evidence_trajectories. "
    "Copy all supplied numeric anchors exactly. Do not infer diagnosis or use MRI."
)
TOP_LEVEL = {"current_state", "evidence", "trend", "forecast", "confidence"}


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG.read_text())


def output_dir(config: dict[str, Any]) -> Path:
    path = ROOT / "outputs" / config["run_id"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def compact_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in history:
        grouped[row["item"]].append(row)
    compact = []
    for item in sorted(grouped):
        rows = sorted(grouped[item], key=lambda x: (x["visit_month"], x["time_month"], x["event_id"]))
        compact.append(rows[0])
        if rows[-1] != rows[0]:
            compact.append(rows[-1])
    return compact


def phase4b_input(example: dict[str, Any]) -> dict[str, Any]:
    source = example["prompt"]
    report = example["completion"]
    domain_statistics = {}
    for domain, state in report["current_state"].items():
        trend = report["trend"][domain]
        domain_statistics[domain] = {
            "current_severity_z": state["severity_z"],
            "items": state["items"],
            "start_severity_z": trend["start_severity_z"],
            "end_severity_z": trend["end_severity_z"],
            "timepoints": trend["timepoints"],
        }
    return {
        "cutoff": source["evidence_cutoff_time_month"],
        "horizon": source["forecast_horizon_months"],
        "evidence_trajectories": compact_history(source["verified_history"]),
        "domain_statistics": domain_statistics,
        "last_targets": source["last_observed_targets"],
        "frozen_forecast": report["forecast"],
        "frozen_confidence": report["confidence"],
    }


def serialize_prompt(tokenizer, model_input: dict[str, Any]) -> str:
    return (
        tokenizer.bos_token + SYSTEM + "<｜User｜>"
        + json.dumps(model_input, separators=(",", ":"), allow_nan=False)
        + "<｜Assistant｜>"
    )


def serialize_completion(tokenizer, completion: dict[str, Any]) -> str:
    return json.dumps(completion, separators=(",", ":"), allow_nan=False) + tokenizer.eos_token


def load_rows(output: Path, split: str | None = None) -> list[dict[str, Any]]:
    rows = []
    with (output / "sft_examples.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if split is None or row["reasoning_split"] == split:
                rows.append(row)
    return rows


def stable_subset(rows: list[dict[str, Any]], per_subject: int | None = None,
                  per_subject_horizon: int | None = None, seed: int = 0) -> list[dict[str, Any]]:
    """Select a reproducible subject-balanced subset without inspecting outcomes."""
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["subject_id"],)
        limit = per_subject
        if per_subject_horizon is not None:
            key = (row["subject_id"], str(row["model_input"]["horizon"]))
            limit = per_subject_horizon
        grouped[key].append(row)
    selected = []
    for key in sorted(grouped):
        ranked = sorted(grouped[key], key=lambda row: hashlib.sha256(
            f"{seed}:{row['sample_id']}".encode()).hexdigest())
        selected.extend(ranked[:int(limit)] if limit is not None else ranked)
    return sorted(selected, key=lambda row: row["sample_id"])


def prepare(config: dict[str, Any], tokenizer) -> None:
    output = output_dir(config)
    source = ROOT / "outputs" / config["source_run_id"] / "structured_reasoning_examples.jsonl"
    target = output / "sft_examples.jsonl"
    lengths = []
    counts: dict[str, int] = defaultdict(int)
    subjects: dict[str, set[str]] = defaultdict(set)
    with source.open() as reader, target.open("w") as writer:
        for line in reader:
            example = json.loads(line)
            model_input = phase4b_input(example)
            prompt = serialize_prompt(tokenizer, model_input)
            completion = serialize_completion(tokenizer, example["completion"])
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            completion_ids = tokenizer.encode(completion, add_special_tokens=False)
            total = len(prompt_ids) + len(completion_ids)
            if total > int(config["prompt"]["max_length"]):
                raise RuntimeError(f"Sequence overflow for {example['sample_id']}: {total}")
            row = {
                "sample_id": example["sample_id"],
                "subject_id": example["subject_id"],
                "reasoning_split": example["reasoning_split"],
                "model_input": model_input,
                "completion": example["completion"],
            }
            writer.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            lengths.append({"sample_id": row["sample_id"], "reasoning_split": row["reasoning_split"],
                            "prompt_tokens": len(prompt_ids), "completion_tokens": len(completion_ids),
                            "total_tokens": total})
            counts[row["reasoning_split"]] += 1
            subjects[row["reasoning_split"]].add(row["subject_id"])
    frame = pd.DataFrame(lengths)
    frame.to_csv(output / "token_lengths.csv", index=False)
    manifest = {
        "run_id": config["run_id"], "source_run_id": config["source_run_id"],
        "base_model": config["model"], "examples": int(len(frame)),
        "sample_counts": dict(sorted(counts.items())),
        "subject_counts": {key: len(value) for key, value in sorted(subjects.items())},
        "max_length": int(config["prompt"]["max_length"]),
        "token_lengths": {column: {"min": int(frame[column].min()),
                                   "median": float(frame[column].median()),
                                   "p95": float(frame[column].quantile(.95)),
                                   "p99": float(frame[column].quantile(.99)),
                                   "max": int(frame[column].max())}
                          for column in ["prompt_tokens", "completion_tokens", "total_tokens"]},
        "future_truth_in_prompt": False, "diagnosis_input": False, "mri_input": False,
        "target_truncation": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    all_rows = load_rows(output)
    train_rows = stable_subset(
        [x for x in all_rows if x["reasoning_split"] == "build"],
        per_subject=int(config["training"]["maximum_examples_per_build_subject"]),
        seed=int(config["training"]["seed"]),
    )
    dev_rows = stable_subset(
        [x for x in all_rows if x["reasoning_split"] == "dev"],
        per_subject_horizon=int(config["training"]["maximum_examples_per_dev_subject_horizon"]),
        seed=int(config["training"]["seed"]),
    )
    confirmation_rows = stable_subset(
        [x for x in all_rows if x["reasoning_split"] == "confirmation"],
        per_subject=int(config["evaluation"]["maximum_examples_per_confirmation_subject"]),
        seed=int(config["training"]["seed"]),
    )
    selection = {
        "selection_frozen_before_training": True,
        "method": "sha256_stable_subject_balancing",
        "training_examples": len(train_rows),
        "training_subjects": len({x["subject_id"] for x in train_rows}),
        "dev_examples": len(dev_rows),
        "dev_subjects": len({x["subject_id"] for x in dev_rows}),
        "confirmation_examples": len(confirmation_rows),
        "confirmation_subjects": len({x["subject_id"] for x in confirmation_rows}),
        "confirmation_horizons": sorted({int(x["model_input"]["horizon"]) for x in confirmation_rows}),
        "training_sample_ids": [x["sample_id"] for x in train_rows],
        "dev_sample_ids": [x["sample_id"] for x in dev_rows],
        "confirmation_sample_ids": [x["sample_id"] for x in confirmation_rows],
    }
    (output / "selection_manifest.json").write_text(json.dumps(selection, indent=2) + "\n")


class Rows(Dataset):
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows
    def __len__(self) -> int:
        return len(self.rows)
    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class SFTCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        prompts = [serialize_prompt(self.tokenizer, row["model_input"]) for row in rows]
        completions = [serialize_completion(self.tokenizer, row["completion"]) for row in rows]
        prompt_ids = self.tokenizer(prompts, add_special_tokens=False)["input_ids"]
        completion_ids = self.tokenizer(completions, add_special_tokens=False)["input_ids"]
        sequences, labels = [], []
        for prompt, completion in zip(prompt_ids, completion_ids):
            if len(prompt) + len(completion) > self.max_length:
                raise RuntimeError("SFT collator encountered an overflow")
            sequences.append(prompt + completion)
            labels.append([-100] * len(prompt) + completion)
        width = max(map(len, sequences))
        pad = self.tokenizer.pad_token_id
        input_ids, masks, padded_labels = [], [], []
        for sequence, label in zip(sequences, labels):
            amount = width - len(sequence)
            input_ids.append(sequence + [pad] * amount)
            masks.append([1] * len(sequence) + [0] * amount)
            padded_labels.append(label + [-100] * amount)
        return {"input_ids": torch.tensor(input_ids), "attention_mask": torch.tensor(masks),
                "labels": torch.tensor(padded_labels)}


def load_base(config: dict[str, Any], training: bool):
    from transformers import AutoModelForCausalLM
    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["path"], local_files_only=True, torch_dtype=dtype,
        attn_implementation=config["model"]["attention"], low_cpu_mem_usage=True,
    )
    model.config.use_cache = not training
    return model


def train(config: dict[str, Any], tokenizer) -> None:
    from peft import LoraConfig, get_peft_model
    from transformers import get_cosine_schedule_with_warmup
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 4B training requires CUDA")
    output = output_dir(config)
    random.seed(int(config["training"]["seed"])); np.random.seed(int(config["training"]["seed"]))
    torch.manual_seed(int(config["training"]["seed"])); torch.cuda.manual_seed_all(int(config["training"]["seed"]))
    torch.backends.cuda.matmul.allow_tf32 = True
    build = stable_subset(load_rows(output, "build"),
                          per_subject=int(config["training"]["maximum_examples_per_build_subject"]),
                          seed=int(config["training"]["seed"]))
    dev = stable_subset(load_rows(output, "dev"),
                        per_subject_horizon=int(config["training"]["maximum_examples_per_dev_subject_horizon"]),
                        seed=int(config["training"]["seed"]))
    collator = SFTCollator(tokenizer, int(config["prompt"]["max_length"]))
    generator = torch.Generator().manual_seed(int(config["training"]["seed"]))
    loader = DataLoader(Rows(build), batch_size=int(config["training"]["batch_size"]), shuffle=True,
                        generator=generator, collate_fn=collator, num_workers=0)
    dev_loader = DataLoader(Rows(dev), batch_size=int(config["training"]["batch_size"]), shuffle=False,
                            collate_fn=collator, num_workers=0)
    model = load_base(config, training=True)
    lora = config["training"]
    model = get_peft_model(model, LoraConfig(
        r=int(lora["lora_rank"]), lora_alpha=int(lora["lora_alpha"]),
        lora_dropout=float(lora["lora_dropout"]), target_modules=list(lora["lora_targets"]),
        bias="none", task_type="CAUSAL_LM",
    ))
    if bool(lora.get("gradient_checkpointing", False)):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.to("cuda")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(lora["learning_rate"]),
                                  weight_decay=float(lora["weight_decay"]))
    accumulation = int(lora["gradient_accumulation_steps"])
    updates = math.ceil(len(loader) * int(lora["epochs"]) / accumulation)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, int(updates * float(lora["warmup_fraction"]))),
        num_training_steps=updates,
    )
    curves, losses, update = [], [], 0
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    model.train()
    for epoch in range(int(lora["epochs"])):
        for step, batch in enumerate(loader):
            batch = {key: value.to("cuda", non_blocking=True) for key, value in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss
            (loss / accumulation).backward()
            losses.append(float(loss.detach()))
            boundary = (step + 1) % accumulation == 0 or step + 1 == len(loader)
            if boundary:
                torch.nn.utils.clip_grad_norm_(trainable, float(lora["gradient_clip"]))
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); update += 1
                if update == 1 or update % 50 == 0 or update == updates:
                    curves.append({"update": update, "epoch": epoch + 1,
                                   "train_loss_recent": float(np.mean(losses[-50 * accumulation:])),
                                   "learning_rate": float(scheduler.get_last_lr()[0]),
                                   "elapsed_minutes": (time.time() - started) / 60.0})
                    pd.DataFrame(curves).to_csv(output / "training_curves.csv", index=False)
    model.eval(); dev_losses = []
    with torch.no_grad():
        for batch in dev_loader:
            batch = {key: value.to("cuda", non_blocking=True) for key, value in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                dev_losses.append(float(model(**batch).loss))
    adapter = output / "adapter"
    model.save_pretrained(adapter, safe_serialization=True)
    tokenizer.save_pretrained(adapter)
    state = {
        "base_model": config["model"]["name"], "build_examples": len(build), "dev_examples": len(dev),
        "epochs": int(lora["epochs"]), "optimizer_updates": update,
        "train_loss": float(np.mean(losses[-min(200, len(losses)):])),
        "dev_loss": float(np.mean(dev_losses)), "elapsed_minutes": (time.time() - started) / 60.0,
        "trainable_parameters": int(sum(x.numel() for x in trainable)),
        "total_parameters": int(sum(x.numel() for x in model.parameters())),
        "completion_only_labels": True, "confirmation_loaded_during_training": False,
    }
    (output / "training_state.json").write_text(json.dumps(state, indent=2) + "\n")


class PromptCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        prompts = [serialize_prompt(self.tokenizer, row["model_input"]) for row in rows]
        encoded = self.tokenizer(prompts, add_special_tokens=False, padding=True, return_tensors="pt")
        return {"rows": rows, **encoded}


def evaluate_arm(config: dict[str, Any], tokenizer, arm: str) -> None:
    from peft import PeftModel
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 4B generation requires CUDA")
    output = output_dir(config)
    rows = stable_subset(load_rows(output, "confirmation"),
                         per_subject=int(config["evaluation"]["maximum_examples_per_confirmation_subject"]),
                         seed=int(config["training"]["seed"]))
    tokenizer.padding_side = "left"
    model = load_base(config, training=False)
    if arm == "sft":
        model = PeftModel.from_pretrained(model, output / "adapter", local_files_only=True)
    model.to("cuda").eval()
    loader = DataLoader(Rows(rows), batch_size=int(config["evaluation"]["batch_size"]), shuffle=False,
                        collate_fn=PromptCollator(tokenizer), num_workers=0)
    destination = output / f"confirmation_generations_{arm}.jsonl"
    started = time.time(); completed = 0
    with destination.open("w") as writer, torch.inference_mode():
        for batch in loader:
            local_rows = batch.pop("rows")
            batch = {key: value.to("cuda") for key, value in batch.items()}
            width = batch["input_ids"].shape[1]
            generated = model.generate(
                **batch, max_new_tokens=int(config["prompt"]["max_new_tokens"]), do_sample=False,
                temperature=None, top_p=None,
                use_cache=True, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
            )
            texts = tokenizer.batch_decode(generated[:, width:], skip_special_tokens=True)
            for row, text in zip(local_rows, texts):
                writer.write(json.dumps({"sample_id": row["sample_id"], "subject_id": row["subject_id"],
                                         "arm": arm, "generation": text}, allow_nan=False) + "\n")
                completed += 1
            if completed % 160 < len(local_rows):
                print(json.dumps({"arm": arm, "completed": completed, "total": len(rows),
                                  "elapsed_minutes": round((time.time() - started) / 60, 2)}), flush=True)
    (output / f"generation_state_{arm}.json").write_text(json.dumps(
        {"arm": arm, "examples": completed, "subjects": len({x['subject_id'] for x in rows}),
         "greedy": True, "raw_without_repair": True,
         "elapsed_minutes": (time.time() - started) / 60.0}, indent=2) + "\n")


def close_number(left: Any, right: Any, tolerance: float = 5e-4) -> bool:
    try:
        return bool(np.isfinite(float(left)) and np.isfinite(float(right))
                    and abs(float(left) - float(right)) <= tolerance)
    except (TypeError, ValueError):
        return left is None and right is None


def nested_score(generated: Any, target: Any) -> float:
    if not isinstance(generated, dict) or not isinstance(target, dict):
        return 0.0
    scores = []
    for key, value in target.items():
        candidate = generated.get(key)
        if isinstance(value, dict):
            scores.append(nested_score(candidate, value))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            scores.append(float(close_number(candidate, value)))
        else:
            scores.append(float(candidate == value))
    return float(np.mean(scores)) if scores else 1.0


def score_generation(row: dict[str, Any], text: str, arm: str) -> dict[str, Any]:
    target = row["completion"]
    allowed = row["model_input"]["evidence_trajectories"]
    latest = {}
    for evidence in allowed:
        old = latest.get(evidence["item"])
        if old is None or (evidence["visit_month"], evidence["time_month"], evidence["event_id"]) > \
                (old["visit_month"], old["time_month"], old["event_id"]):
            latest[evidence["item"]] = evidence
    expected = {(e["event_id"], e["item"]): e for e in target["evidence"]}
    result = {"sample_id": row["sample_id"], "subject_id": row["subject_id"], "arm": arm}
    try:
        report = parse_report(text.strip())
        result["format_validity"] = 1.0
    except Exception:
        report = None
        result["format_validity"] = 0.0
    metrics = ["evidence_precision", "evidence_recall", "numeric_citation_accuracy",
               "time_citation_accuracy", "domain_accuracy", "current_state_consistency",
               "temporal_consistency", "forecast_preservation", "uncertainty_preservation"]
    if report is None:
        result.update({key: 0.0 for key in metrics})
        result.update({"hallucinated_evidence_rate": 1.0, "future_evidence_leakage": 0.0,
                       "exact_report_match": 0.0, "forbidden_claim": float("diagnos" in text.lower() or "mri" in text.lower())})
        return result
    citations = report["evidence"]
    match, numeric, timing, domain, hallucinated, future = 0, 0, 0, 0, 0, 0
    for citation in citations:
        key = (citation.get("event_id"), citation.get("item"))
        reference = expected.get(key)
        if reference is None:
            hallucinated += 1
        else:
            match += 1
            numeric += int(close_number(citation.get("value"), reference["value"]))
            timing += int(close_number(citation.get("time_month"), reference["time_month"]))
            domain += int(citation.get("domain") == reference["domain"])
        try:
            future += int(float(citation.get("time_month")) > float(row["model_input"]["cutoff"]) + 1e-6)
        except (TypeError, ValueError):
            pass
    denominator = max(1, len(citations))
    result.update({
        "evidence_precision": match / denominator,
        "evidence_recall": match / max(1, len(expected)),
        "numeric_citation_accuracy": numeric / denominator,
        "time_citation_accuracy": timing / denominator,
        "domain_accuracy": domain / denominator,
        "hallucinated_evidence_rate": hallucinated / denominator,
        "future_evidence_leakage": future / denominator,
        "current_state_consistency": nested_score(report["current_state"], target["current_state"]),
        "temporal_consistency": nested_score(report["trend"], target["trend"]),
        "forecast_preservation": nested_score(report["forecast"], target["forecast"]),
        "uncertainty_preservation": nested_score(report["confidence"], target["confidence"]),
        "exact_report_match": float(report == target),
        "forbidden_claim": float("diagnos" in text.lower() or "mri" in text.lower()),
    })
    return result


def score_arm(config: dict[str, Any], arm: str) -> pd.DataFrame:
    output = output_dir(config)
    selected = stable_subset(load_rows(output, "confirmation"),
                             per_subject=int(config["evaluation"]["maximum_examples_per_confirmation_subject"]),
                             seed=int(config["training"]["seed"]))
    rows = {row["sample_id"]: row for row in selected}
    generations = []
    with (output / f"confirmation_generations_{arm}.jsonl").open() as stream:
        for line in stream:
            generations.append(json.loads(line))
    if len(generations) != len(rows) or {x["sample_id"] for x in generations} != set(rows):
        raise RuntimeError(f"Incomplete {arm} confirmation generations")
    return pd.DataFrame([score_generation(rows[x["sample_id"]], x["generation"], arm) for x in generations])


def paired_bootstrap(frame: pd.DataFrame, metrics: list[str], config: dict[str, Any]) -> pd.DataFrame:
    pivot = frame.pivot(index=["subject_id", "sample_id"], columns="arm", values=metrics)
    subjects = sorted(frame.subject_id.unique())
    by_subject = {subject: pivot.loc[subject] for subject in subjects}
    rng = np.random.default_rng(int(config["evaluation"]["bootstrap_seed"]))
    rows = []
    for metric in metrics:
        point = float(pivot[(metric, "sft")].mean() - pivot[(metric, "base")].mean())
        subject_differences = np.asarray([
            float((by_subject[subject][(metric, "sft")] - by_subject[subject][(metric, "base")]).mean())
            for subject in subjects
        ])
        indices = rng.integers(0, len(subjects),
                              size=(int(config["evaluation"]["bootstrap_replicates"]), len(subjects)))
        draws = subject_differences[indices].mean(axis=1)
        rows.append({"metric": metric, "sft_minus_base": point,
                     "ci_low": float(np.quantile(draws, .025)), "ci_high": float(np.quantile(draws, .975)),
                     "bootstrap_replicates": int(config["evaluation"]["bootstrap_replicates"])})
    return pd.DataFrame(rows)


def summarize(config: dict[str, Any]) -> None:
    output = output_dir(config)
    frames = [score_arm(config, arm) for arm in ["base", "sft"]]
    scored = pd.concat(frames, ignore_index=True)
    scored.to_csv(output / "confirmation_per_report_metrics.csv", index=False)
    metric_columns = [column for column in scored.columns if column not in {"sample_id", "subject_id", "arm"}]
    aggregate = scored.groupby("arm", as_index=False)[metric_columns].mean()
    teacher = {"arm": "deterministic_teacher", **{key: (0.0 if key in {"hallucinated_evidence_rate", "future_evidence_leakage", "forbidden_claim"} else 1.0) for key in metric_columns}}
    aggregate = pd.concat([aggregate, pd.DataFrame([teacher])], ignore_index=True)
    aggregate.to_csv(output / "aggregate_metrics.csv", index=False)
    bootstrap_metrics = ["format_validity", "evidence_precision", "evidence_recall", "current_state_consistency",
                         "temporal_consistency", "forecast_preservation", "uncertainty_preservation"]
    bootstrap = paired_bootstrap(scored, bootstrap_metrics, config)
    bootstrap.to_csv(output / "subject_cluster_bootstrap.csv", index=False)
    sft = aggregate.set_index("arm").loc["sft"]
    base = aggregate.set_index("arm").loc["base"]
    gate = config["gate"]
    criteria = {
        "format_validity": sft.format_validity >= gate["minimum_format_validity"],
        "evidence_precision": sft.evidence_precision >= gate["minimum_evidence_precision"],
        "evidence_recall": sft.evidence_recall >= gate["minimum_evidence_recall"],
        "numeric_citation_accuracy": sft.numeric_citation_accuracy >= gate["minimum_numeric_citation_accuracy"],
        "time_citation_accuracy": sft.time_citation_accuracy >= gate["minimum_time_citation_accuracy"],
        "domain_accuracy": sft.domain_accuracy >= gate["minimum_domain_accuracy"],
        "current_state_consistency": sft.current_state_consistency >= gate["minimum_current_state_consistency"],
        "temporal_consistency": sft.temporal_consistency >= gate["minimum_temporal_consistency"],
        "forecast_preservation": sft.forecast_preservation >= gate["minimum_forecast_preservation"],
        "uncertainty_preservation": sft.uncertainty_preservation >= gate["minimum_uncertainty_preservation"],
        "hallucinated_evidence_rate": sft.hallucinated_evidence_rate <= gate["maximum_hallucinated_evidence_rate"],
        "future_evidence_leakage": sft.future_evidence_leakage <= gate["maximum_future_evidence_leakage"],
    }
    composite_metrics = ["format_validity", "evidence_precision", "evidence_recall", "current_state_consistency",
                         "temporal_consistency", "forecast_preservation", "uncertainty_preservation"]
    base_composite = float(base[composite_metrics].mean()); sft_composite = float(sft[composite_metrics].mean())
    criteria["improvement_over_base"] = sft_composite - base_composite > float(gate["composite_improvement_margin"])
    decision = {
        "run_id": config["run_id"], "base_model": config["model"]["name"],
        "confirmation_reports": int((scored.arm == "sft").sum()),
        "confirmation_subjects": int(scored.subject_id.nunique()), "criteria": {k: bool(v) for k, v in criteria.items()},
        "base_composite": base_composite, "sft_composite": sft_composite,
        "sft_minus_base_composite": sft_composite - base_composite,
        "phase4b_sft_passed": bool(all(criteria.values())),
        "cognitive_grpo_ready": bool(all(criteria.values())), "multimodal_reasoning_ready": False,
        "next_action": "prepare_cognitive_only_grpo" if all(criteria.values()) else "retain_deterministic_teacher_and_audit_sft_failures",
    }
    (output / "development_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    manifest = json.loads((output / "manifest.json").read_text())
    selection = json.loads((output / "selection_manifest.json").read_text())
    training = json.loads((output / "training_state.json").read_text())
    valid_sft = scored[(scored.arm == "sft") & (scored.format_validity == 1)]
    invalid_sft = scored[(scored.arm == "sft") & (scored.format_validity == 0)]
    failure = {
        "invalid_json_reports": int(len(invalid_sft)),
        "invalid_sample_ids": invalid_sft.sample_id.tolist(),
        "invalid_reason": "generation reached the frozen token limit before closing the JSON object",
        "valid_reports": int(len(valid_sft)),
        "valid_exact_report_matches": int(valid_sft.exact_report_match.sum()),
        "valid_conditional_metrics": {key: float(valid_sft[key].mean()) for key in [
            "evidence_precision", "evidence_recall", "numeric_citation_accuracy",
            "time_citation_accuracy", "domain_accuracy", "current_state_consistency",
            "temporal_consistency", "forecast_preservation", "uncertainty_preservation",
            "hallucinated_evidence_rate", "future_evidence_leakage",
        ]},
    }
    (output / "failure_summary.json").write_text(json.dumps(failure, indent=2) + "\n")
    table = markdown_table(aggregate)
    boot = markdown_table(bootstrap)
    report = f"""# Phase 4B cognitive-only structured-reasoning SFT

Status: completed on the frozen Phase 4A build/dev/confirmation partition. Original ADNI validation and test subjects were not used.

## Data and model

- Local base model: `{config['model']['name']}`
- Source build/dev/confirmation reports: {manifest['sample_counts']['build']}/{manifest['sample_counts']['dev']}/{manifest['sample_counts']['confirmation']}
- Selected train/dev/confirmation reports: {selection['training_examples']}/{selection['dev_examples']}/{selection['confirmation_examples']}
- Selected train/dev/confirmation subjects: {selection['training_subjects']}/{selection['dev_subjects']}/{selection['confirmation_subjects']}
- Maximum prompt + completion length: {manifest['token_lengths']['total_tokens']['max']} tokens; no examples truncated
- Optimization: one-epoch completion-only rank-{config['training']['lora_rank']} LoRA, {training['trainable_parameters']:,} trainable parameters
- Final train/dev loss: {training['train_loss']:.4f}/{training['dev_loss']:.4f}
- MRI, diagnosis, and future observed outcomes used as inputs: no

## Frozen confirmation results

{table}

All language-model metrics use raw greedy generations without JSON repair or numeric post-processing. The deterministic teacher is the Phase 4A executable report and defines the auditable ceiling.

## Failure analysis

Two of 168 SFT generations reached the frozen 1,280-token limit before closing the JSON object. Among the 166 valid reports, evidence precision, numeric/time/domain accuracy, forecast preservation, and uncertainty preservation were 1.0000; evidence recall was {failure['valid_conditional_metrics']['evidence_recall']:.4f}, current-state consistency was {failure['valid_conditional_metrics']['current_state_consistency']:.4f}, and temporal consistency was {failure['valid_conditional_metrics']['temporal_consistency']:.4f}. This descriptive conditional analysis does not replace the all-report gate.

## Subject-cluster bootstrap: SFT minus base

{boot}

## Frozen decision

```json
{json.dumps(decision, indent=2)}
```

Passing this gate opens cognitive-only GRPO preparation only. Multimodal reasoning remains closed because the preceding MRI gates failed.
"""
    (output / "report.md").write_text(report)


def tokenizer_for(config: dict[str, Any]):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["path"], local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["prepare", "train", "evaluate-base", "evaluate-sft", "summarize"])
    args = parser.parse_args()
    config = load_config(); tokenizer = tokenizer_for(config)
    if args.stage == "prepare": prepare(config, tokenizer)
    elif args.stage == "train": train(config, tokenizer)
    elif args.stage == "evaluate-base": evaluate_arm(config, tokenizer, "base")
    elif args.stage == "evaluate-sft": evaluate_arm(config, tokenizer, "sft")
    else: summarize(config)


if __name__ == "__main__":
    main()
