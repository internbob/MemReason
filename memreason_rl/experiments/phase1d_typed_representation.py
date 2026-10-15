"""Validation-only Phase 1D typed question-conditioned representation study."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from ..data.common import ROOT, read_csv, save_csv, save_json
from ..data.dictionary import load_dictionary
from .phase1_numeric import clean_frame, fit_normalization, markdown_table
from .phase1b_semantic import seed_everything, select_verified_scalar_features
from .phase1c_joint_stage1 import (
    JointCollator,
    JointDataset,
    JointSample,
    bootstrap_difference,
    build_domain_assignment,
    build_joint_samples,
    equal_task_macro_score,
    flatten_tasks,
    load_config,
    metrics,
    task_text,
)


TASK_GROUPS = [
    "future_forecast",
    "cross_adas_cdr_to_moca",
    "cross_mmse_cdr_to_adas",
    "masked_item",
    "masked_instrument",
]
TYPE_IDS = {"continuous": 0, "binary": 1, "ordinal": 2}


def feature_metadata(
    features: list[str], dictionary: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    type_ids, bounds, directions, categories = [], [], [], []
    for feature in features:
        entry = dictionary[feature]
        value_type = entry["value_type"]
        if value_type not in TYPE_IDS:
            raise ValueError(f"Unsupported Phase 1D value type {value_type}: {feature}")
        type_ids.append(TYPE_IDS[value_type])
        allowed = entry.get("valid_values")
        if allowed:
            values = sorted(float(value) for value in allowed)
            low, high = values[0], values[-1]
        elif entry.get("range"):
            low, high = map(float, entry["range"])
            values = []
        elif value_type == "binary":
            low, high, values = 0.0, 1.0, [0.0, 1.0]
        else:
            raise ValueError(f"Missing verified bounds for {feature}")
        bounds.append([low, high])
        directions.append(-1.0 if entry["direction"] == "higher=better" else 1.0)
        categories.append(values if value_type in {"binary", "ordinal"} else [])
    max_categories = max(len(values) for values in categories)
    category_values = torch.zeros(len(features), max_categories)
    category_counts = torch.zeros(len(features), dtype=torch.long)
    for index, values in enumerate(categories):
        if values:
            category_values[index, : len(values)] = torch.tensor(values)
            category_counts[index] = len(values)
    return (
        torch.tensor(type_ids, dtype=torch.long),
        torch.tensor(bounds, dtype=torch.float32),
        torch.tensor(directions, dtype=torch.float32),
        category_values,
        category_counts,
    )


def severity_coordinate(
    raw_values: torch.Tensor,
    bounds: torch.Tensor,
    directions: torch.Tensor,
) -> torch.Tensor:
    low, high = bounds[:, 0], bounds[:, 1]
    coordinate = ((raw_values - low) / (high - low).clamp_min(1e-6)).clamp(0.0, 1.0)
    return torch.where(directions > 0, coordinate, 1.0 - coordinate)


class TypedCollator(JointCollator):
    def __init__(
        self,
        features: list[str],
        targets: list[str],
        normalization: dict[str, dict[str, float]],
        dictionary: dict[str, Any],
    ):
        response_vocab = {
            feature: sorted(
                {
                    round(float(value), 4)
                    for value in (
                        dictionary[feature].get("valid_values")
                        or dictionary[feature].get("range")
                        or [0.0]
                    )
                }
            )
            for feature in features
        }
        super().__init__(features, targets, normalization, response_vocab)
        (
            self.feature_types,
            self.bounds,
            self.directions,
            self.category_values,
            self.category_counts,
        ) = feature_metadata(features, dictionary)
        self.category_lists = []
        for feature in features:
            entry = dictionary[feature]
            if entry["value_type"] in {"binary", "ordinal"}:
                values = entry.get("valid_values") or [0.0, 1.0]
                self.category_lists.append(sorted(float(value) for value in values))
            else:
                self.category_lists.append([])

    def __call__(self, samples: list[JointSample]) -> dict[str, Any]:
        batch = super().__call__(samples)
        raw = (
            batch["values"] * torch.from_numpy(self.feature_std)[None, None]
            + torch.from_numpy(self.feature_mean)[None, None]
        )
        batch["severity"] = severity_coordinate(raw, self.bounds, self.directions) * batch[
            "mask"
        ].to(raw.dtype)
        category_target = torch.full(
            (len(samples), len(self.features)), -1, dtype=torch.long
        )
        for row, sample in enumerate(samples):
            for feature_index in np.flatnonzero(sample.reconstruction_mask):
                values = self.category_lists[feature_index]
                if values:
                    observed = float(sample.reconstruction_targets[feature_index])
                    category_target[row, feature_index] = int(
                        np.argmin(np.abs(np.asarray(values) - observed))
                    )
        batch["reconstruction_category"] = category_target
        batch["task_group_id"] = torch.tensor(
            [TASK_GROUPS.index(sample.task) for sample in samples], dtype=torch.long
        )
        return batch


def encode_task_descriptors(
    features: list[str],
    dictionary: dict[str, Any],
    config: dict[str, Any],
    output: Path,
    device: torch.device,
) -> torch.Tensor:
    texts = [task_text(feature, dictionary[feature]) for feature in features]
    signature = hashlib.sha256(json.dumps(texts).encode()).hexdigest()
    cache = output / "descriptor_cache" / f"task_{signature[:12]}.pt"
    if cache.exists():
        record = torch.load(cache, map_location="cpu", weights_only=True)
        if record["signature"] != signature:
            raise RuntimeError("Task descriptor cache mismatch")
        return record["embeddings"]
    from transformers import AutoModel, AutoTokenizer

    settings = config["text_encoder"]
    tokenizer = AutoTokenizer.from_pretrained(settings["path"], local_files_only=True)
    encoder = AutoModel.from_pretrained(settings["path"], local_files_only=True).to(device).eval()
    encoder.requires_grad_(False)
    batches = []
    for start in range(0, len(texts), settings["batch_size"]):
        tokens = tokenizer(
            texts[start : start + settings["batch_size"]],
            padding=True,
            truncation=True,
            max_length=settings["max_length"],
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            hidden = encoder(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            batches.append(F.normalize(pooled, dim=-1).cpu())
    embeddings = torch.cat(batches)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"signature": signature, "embeddings": embeddings}, cache)
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return embeddings


class TypedQuestionStage1(nn.Module):
    """Parameter-matched typed encoder with item-conditioned reconstruction."""

    def __init__(
        self,
        task_descriptors: torch.Tensor,
        domain_assignment: torch.Tensor,
        feature_types: torch.Tensor,
        category_values: torch.Tensor,
        category_counts: torch.Tensor,
        n_targets: int,
        arm: dict[str, Any],
        config: dict[str, Any],
    ):
        super().__init__()
        settings = config["model"]
        n_features = len(feature_types)
        item_dim = settings["item_dim"]
        hidden = settings["hidden_dim"]
        self.typed_value_active = bool(arm["typed_value"])
        self.task_semantics_active = bool(arm["task_semantics"])
        self.register_buffer("task_descriptors", task_descriptors.float())
        self.register_buffer("domain_assignment", domain_assignment.float())
        self.register_buffer("feature_types", feature_types.long())
        self.register_buffer("category_values", category_values.float())
        self.register_buffer("category_counts", category_counts)
        self.item_embedding = nn.Embedding(n_features, item_dim)
        self.type_embedding = nn.Embedding(len(TYPE_IDS), item_dim)
        self.semantic_projection = nn.Linear(
            settings["descriptor_input_dim"], item_dim, bias=False
        )
        self.value_encoders = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(1, item_dim), nn.GELU(), nn.Linear(item_dim, item_dim))
                for _ in TYPE_IDS
            ]
        )
        self.token_norm = nn.LayerNorm(item_dim)
        n_domains = domain_assignment.shape[1]
        self.domain_aggregation = nn.Sequential(
            nn.Linear(n_domains * item_dim, settings["domain_dim"]),
            nn.GELU(),
            nn.LayerNorm(settings["domain_dim"]),
        )
        self.decay_weight = nn.Parameter(torch.zeros(n_features))
        self.decay_bias = nn.Parameter(torch.zeros(n_features))
        first_input = n_features * 2 + settings["domain_dim"] + 2
        self.cells = nn.ModuleList(
            [nn.GRUCell(first_input, hidden)]
            + [nn.GRUCell(hidden, hidden) for _ in range(settings["layers"] - 1)]
        )
        self.dropout = nn.Dropout(settings["dropout"])
        self.horizon = nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.task_embedding = nn.Embedding(len(TASK_GROUPS), settings["task_dim"])
        self.total_head = nn.Linear(hidden * 2 + settings["task_dim"], n_targets)
        self.decoder_state = nn.Linear(hidden + settings["task_dim"], item_dim)
        self.decoder_query = nn.Linear(item_dim, item_dim)
        self.continuous_head = nn.Linear(item_dim, 1)
        self.category_head = nn.Linear(item_dim, category_values.shape[1])

    def _static_query(self) -> torch.Tensor:
        indices = torch.arange(len(self.feature_types), device=self.feature_types.device)
        query = self.item_embedding(indices) + self.type_embedding(self.feature_types)
        if self.task_semantics_active:
            query = query + self.semantic_projection(self.task_descriptors)
        return self.token_norm(query)

    def forward(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values, observed = batch["values"], batch["mask"]
        times, valid = batch["times"], batch["event_mask"]
        batch_size, steps, n_features = values.shape
        static = self._static_query()
        token = static.view(1, 1, n_features, -1).expand(batch_size, steps, -1, -1)
        if self.typed_value_active:
            typed = torch.zeros_like(token)
            for type_id, encoder in enumerate(self.value_encoders):
                encoded = encoder(batch["severity"].unsqueeze(-1))
                typed = typed + encoded * self.feature_types.eq(type_id).view(1, 1, -1, 1)
            token = self.token_norm(token + typed)
        token = token * observed.unsqueeze(-1).to(values.dtype)
        domain_sum = torch.einsum("btfe,fd->btde", token, self.domain_assignment)
        domain_count = torch.einsum(
            "btf,fd->btd", observed.to(values.dtype), self.domain_assignment
        ).clamp_min(1.0)
        domain_summary = self.domain_aggregation(
            (domain_sum / domain_count.unsqueeze(-1)).flatten(-2)
        )

        hidden = [values.new_zeros(batch_size, cell.hidden_size) for cell in self.cells]
        last_value = values.new_zeros(batch_size, n_features)
        last_observed_time = times[:, :1].expand(-1, n_features).clone()
        previous_time = times[:, 0]
        for step in range(steps):
            event_valid = valid[:, step]
            current_time = times[:, step]
            elapsed = (current_time[:, None] - last_observed_time).clamp_min(0)
            gamma = torch.exp(
                -F.relu(self.decay_weight[None] * elapsed + self.decay_bias[None])
            )
            imputed = torch.where(observed[:, step], values[:, step], gamma * last_value)
            gap = (current_time - previous_time).clamp_min(0)
            cell_input = torch.cat(
                [
                    imputed,
                    observed[:, step].to(values.dtype),
                    domain_summary[:, step],
                    current_time[:, None],
                    gap[:, None],
                ],
                dim=-1,
            )
            for layer, cell in enumerate(self.cells):
                proposed = cell(cell_input, hidden[layer])
                hidden[layer] = torch.where(event_valid[:, None], proposed, hidden[layer])
                cell_input = self.dropout(hidden[layer])
            last_value = torch.where(
                observed[:, step] & event_valid[:, None], values[:, step], last_value
            )
            last_observed_time = torch.where(
                observed[:, step] & event_valid[:, None],
                current_time[:, None],
                last_observed_time,
            )
            previous_time = torch.where(event_valid, current_time, previous_time)

        task = self.task_embedding(batch["task_group_id"])
        horizon = self.horizon(batch["horizon"][:, None])
        total = self.total_head(torch.cat([hidden[-1], horizon, task], dim=-1))
        state = self.decoder_state(torch.cat([hidden[-1], task], dim=-1))[:, None]
        interaction = F.gelu(state + self.decoder_query(static)[None])
        continuous = self.continuous_head(interaction).squeeze(-1)
        category = self.category_head(interaction)
        positions = torch.arange(category.shape[-1], device=category.device)
        invalid = positions.view(1, 1, -1) >= self.category_counts.view(1, -1, 1)
        category = category.masked_fill(invalid, -1e4)
        return total, continuous, category


def typed_loss(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    batch: dict[str, torch.Tensor],
    feature_types: torch.Tensor,
    reconstruction_weight: float,
) -> torch.Tensor:
    total, continuous, category = outputs
    losses = []
    for group_id in batch["task_group_id"].unique():
        rows = batch["task_group_id"].eq(group_id)
        if int(group_id) < 3:
            mask = batch["total_mask"] & rows[:, None]
            if mask.any():
                losses.append(F.huber_loss(total[mask], batch["total_targets"][mask]))
            continue
        reconstruction = batch["reconstruction_mask"] & rows[:, None]
        parts = []
        continuous_mask = reconstruction & feature_types.eq(TYPE_IDS["continuous"])[None]
        if continuous_mask.any():
            parts.append(
                F.huber_loss(
                    continuous[continuous_mask], batch["reconstruction_targets"][continuous_mask]
                )
            )
        category_mask = reconstruction & feature_types.ne(TYPE_IDS["continuous"])[None]
        if category_mask.any():
            parts.append(
                F.cross_entropy(
                    category[category_mask], batch["reconstruction_category"][category_mask]
                )
            )
        if parts:
            losses.append(torch.stack(parts).mean() * reconstruction_weight)
    return torch.stack(losses).mean() if losses else total.sum() * 0


def reconstruction_values(
    continuous: torch.Tensor,
    category: torch.Tensor,
    collator: TypedCollator,
) -> torch.Tensor:
    raw_continuous = (
        continuous * torch.from_numpy(collator.feature_std).to(continuous.device)[None]
        + torch.from_numpy(collator.feature_mean).to(continuous.device)[None]
    )
    expected = (category.softmax(-1) * collator.category_values.to(category.device)[None]).sum(-1)
    typed = collator.feature_types.to(category.device)
    return torch.where(typed.eq(TYPE_IDS["continuous"])[None], raw_continuous, expected)


def prediction_frame(
    samples: list[JointSample],
    total_prediction: np.ndarray,
    reconstruction_prediction: np.ndarray,
    model: str,
    seed: int,
    features: list[str],
    targets: list[str],
) -> pd.DataFrame:
    rows = []
    for sample, totals, reconstructed in zip(
        samples, total_prediction, reconstruction_prediction
    ):
        for index in np.flatnonzero(sample.total_target_mask):
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "subject_id": sample.subject_id,
                    "split": sample.split,
                    "task": sample.task,
                    "model": model,
                    "seed": seed,
                    "target": targets[index],
                    "horizon_months": sample.horizon_months,
                    "cutoff_month": sample.cutoff_month,
                    "target_month": float(sample.target_months[index]),
                    "truth": float(sample.total_targets[index]),
                    "prediction": float(totals[index]),
                }
            )
        for index in np.flatnonzero(sample.reconstruction_mask):
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "subject_id": sample.subject_id,
                    "split": sample.split,
                    "task": sample.task,
                    "model": model,
                    "seed": seed,
                    "target": features[index],
                    "horizon_months": 0,
                    "cutoff_month": sample.cutoff_month,
                    "target_month": sample.cutoff_month,
                    "truth": float(sample.reconstruction_targets[index]),
                    "prediction": float(reconstructed[index]),
                }
            )
    return pd.DataFrame(rows)


def evaluate(
    model: TypedQuestionStage1,
    samples: list[JointSample],
    collator: TypedCollator,
    batch_size: int,
    device: torch.device,
    arm: str,
    seed: int,
    features: list[str],
    targets: list[str],
) -> pd.DataFrame:
    loader = DataLoader(
        JointDataset(samples), batch_size=batch_size, shuffle=False, collate_fn=collator
    )
    model.eval()
    totals, reconstructions, ordered = [], [], []
    with torch.no_grad():
        for batch in loader:
            local = batch.pop("samples")
            batch = {key: value.to(device) for key, value in batch.items()}
            total, continuous, category = model(batch)
            totals.append(
                total.cpu().numpy() * collator.target_std + collator.target_mean
            )
            reconstructions.append(
                reconstruction_values(continuous, category, collator).cpu().numpy()
            )
            ordered.extend(local)
    return prediction_frame(
        ordered,
        np.concatenate(totals),
        np.concatenate(reconstructions),
        arm,
        seed,
        features,
        targets,
    )


def make_loader(
    tasks: dict[str, list[JointSample]],
    names: list[str],
    collator: TypedCollator,
    batch_size: int,
    seed: int,
) -> DataLoader:
    samples = [sample for name in names for sample in tasks[name]]
    counts = Counter(sample.task for sample in samples)
    weights = torch.tensor([1.0 / counts[sample.task] for sample in samples], dtype=torch.double)
    sampler = WeightedRandomSampler(
        weights,
        len(samples),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    return DataLoader(
        JointDataset(samples),
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=0,
    )


def checkpoint_path(output: Path, arm: str, seed: int) -> Path:
    return output / "checkpoints" / arm / str(seed) / "best.pt"


def train_one(
    arm_name: str,
    arm: dict[str, Any],
    seed: int,
    task_descriptors: torch.Tensor,
    domain_assignment: torch.Tensor,
    feature_types: torch.Tensor,
    category_values: torch.Tensor,
    category_counts: torch.Tensor,
    train_tasks: dict[str, list[JointSample]],
    val_samples: list[JointSample],
    collator: TypedCollator,
    features: list[str],
    targets: list[str],
    normalization: dict[str, dict[str, float]],
    config: dict[str, Any],
    output: Path,
    device: torch.device,
) -> tuple[pd.DataFrame, int]:
    seed_everything(seed)
    settings = config["training"]
    model = TypedQuestionStage1(
        task_descriptors,
        domain_assignment,
        feature_types,
        category_values,
        category_counts,
        len(targets),
        arm,
        config,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    path = checkpoint_path(output, arm_name, seed)
    if path.exists() and (path.parent / "curves.csv").exists():
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["features"] != features or checkpoint["targets"] != targets:
            raise RuntimeError(f"Checkpoint contract mismatch for {arm_name}/{seed}")
        model.load_state_dict(checkpoint["model"])
        return (
            evaluate(
                model,
                val_samples,
                collator,
                settings["batch_size"],
                device,
                arm_name,
                seed,
                features,
                targets,
            ),
            parameter_count,
        )

    if arm["schedule"] == "two_stage":
        phases = [
            {
                "name": "representation_pretrain",
                "tasks": TASK_GROUPS[1:],
                "epochs": settings["pretrain_max_epochs"],
                "patience": settings["pretrain_patience"],
                "reconstruction_weight": 1.0,
            },
            {
                "name": "forecast_finetune",
                "tasks": TASK_GROUPS,
                "epochs": settings["finetune_max_epochs"],
                "patience": settings["finetune_patience"],
                "reconstruction_weight": settings["finetune_reconstruction_weight"],
            },
        ]
    else:
        phases = [
            {
                "name": "joint",
                "tasks": TASK_GROUPS,
                "epochs": settings["joint_max_epochs"],
                "patience": settings["joint_patience"],
                "reconstruction_weight": 1.0,
            }
        ]

    curves, final_score = [], float("inf")
    for phase_index, phase in enumerate(phases):
        loader = make_loader(
            train_tasks,
            phase["tasks"],
            collator,
            settings["batch_size"],
            seed + phase_index,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=settings["learning_rate"],
            weight_decay=settings["weight_decay"],
        )
        best_score, best_state, patience = float("inf"), None, 0
        for epoch in range(phase["epochs"]):
            model.train()
            losses = []
            for batch in loader:
                batch.pop("samples")
                batch = {key: value.to(device) for key, value in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                outputs = model(batch)
                loss = typed_loss(
                    outputs,
                    batch,
                    model.feature_types,
                    phase["reconstruction_weight"],
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss for {arm_name}/{seed}")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"])
                optimizer.step()
                losses.append(float(loss.detach()))
            validation = evaluate(
                model,
                val_samples,
                collator,
                settings["batch_size"],
                device,
                arm_name,
                seed,
                features,
                targets,
            )
            score_frame = validation[validation.task.isin(phase["tasks"])]
            score = equal_task_macro_score(score_frame, normalization)
            record = {
                "phase": phase["name"],
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_equal_task_macro_normalized_mae": score,
            }
            curves.append(record)
            print(json.dumps({"arm": arm_name, "seed": seed, **record}), flush=True)
            if score < best_score - 1e-5:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
                patience = 0
            else:
                patience += 1
                if patience >= phase["patience"]:
                    break
        if best_state is None:
            raise RuntimeError(f"No checkpoint for {arm_name}/{seed}/{phase['name']}")
        model.load_state_dict(best_state)
        final_score = best_score
        if phase_index < len(phases) - 1:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": best_state, "phase": phase["name"]}, path.parent / "pretrain.pt")

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "arm": arm_name,
            "seed": seed,
            "features": features,
            "targets": targets,
            "parameter_count": parameter_count,
            "validation_score": final_score,
        },
        path,
    )
    save_csv(pd.DataFrame(curves), path.parent / "curves.csv")
    return (
        evaluate(
            model,
            val_samples,
            collator,
            settings["batch_size"],
            device,
            arm_name,
            seed,
            features,
            targets,
        ),
        parameter_count,
    )


def write_report(
    output: Path,
    macro: pd.DataFrame,
    bootstrap: pd.DataFrame,
    selection: dict[str, float],
    decision: dict[str, Any],
    parameter_counts: dict[str, int],
    sample_counts: dict[str, dict[str, int]],
) -> None:
    text = f"""# Phase 1D typed representation development results

Status: completed validation-only development study. The test split was not loaded.

## Contract

- Trainable parameter counts: {parameter_counts}
- Samples by split/task: {sample_counts}
- Equal-task validation scores: {selection}

## Validation macro normalized MAE

{markdown_table(macro)}

## Subject-cluster bootstrap against numeric-query joint baseline

Negative normalized-MAE differences favor the candidate.

{markdown_table(bootstrap)}

## Provisional decision

{json.dumps(decision, indent=2)}

## Interpretation

- This run is for architecture development and does not evaluate the test split.
- A provisional pass requires a new frozen three-seed confirmation run before any internal test re-evaluation.
- Failure should trigger another train/validation-only representation revision, not MRI Stage II.
"""
    (output / "report.md").write_text(text)


def run(config_path: Path, smoke: bool = False) -> None:
    config = load_config(config_path)
    if smoke:
        config = copy.deepcopy(config)
        config["arms"] = {
            "numeric_query_joint": config["arms"]["numeric_query_joint"],
            "typed_semantic_two_stage": config["arms"]["typed_semantic_two_stage"],
        }
        config["training"].update(
            joint_max_epochs=2,
            joint_patience=2,
            pretrain_max_epochs=1,
            pretrain_patience=1,
            finetune_max_epochs=1,
            finetune_patience=1,
            batch_size=64,
        )
        config["statistics"]["bootstrap_replicates"] = 50
    output = ROOT / "outputs" / (config["run_id"] + ("_smoke" if smoke else ""))
    output.mkdir(parents=True, exist_ok=True)
    resolved = output / "resolved_config.json"
    if resolved.exists() and json.loads(resolved.read_text()) != config:
        raise RuntimeError(f"Frozen config changed for {output}")
    if not resolved.exists():
        save_json(config, resolved)

    dictionary = load_dictionary()
    features = select_verified_scalar_features(config, dictionary)
    targets = list(config["targets"])
    frames = {
        split: clean_frame(read_csv(ROOT / f"splits/{split}.csv"), features, targets, dictionary)
        for split in ["train", "val"]
    }
    normalization = fit_normalization(frames["train"], features, targets)
    tasks = {
        split: build_joint_samples(frame, split, features, targets, dictionary, config)
        for split, frame in frames.items()
    }
    if smoke:
        for split_tasks in tasks.values():
            for task in split_tasks:
                split_tasks[task] = split_tasks[task][: min(96, len(split_tasks[task]))]
    val_samples = flatten_tasks(tasks["val"])
    collator = TypedCollator(features, targets, normalization, dictionary)
    domains, domain_assignment = build_domain_assignment(features, dictionary)
    feature_types, bounds, directions, category_values, category_counts = feature_metadata(
        features, dictionary
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_descriptors = encode_task_descriptors(features, dictionary, config, output, device)

    validation_frames, parameter_counts = [], {}
    for arm_name, arm in config["arms"].items():
        counts = []
        for seed in config["neural_seeds"]:
            frame, count = train_one(
                arm_name,
                arm,
                seed,
                task_descriptors,
                domain_assignment,
                feature_types,
                category_values,
                category_counts,
                tasks["train"],
                val_samples,
                collator,
                features,
                targets,
                normalization,
                config,
                output,
                device,
            )
            validation_frames.append(frame)
            counts.append(count)
        if len(set(counts)) != 1:
            raise RuntimeError(f"Parameter count changed across seeds: {arm_name}")
        parameter_counts[arm_name] = counts[0]
    if len(set(parameter_counts.values())) != 1:
        raise RuntimeError(f"Arms are not parameter matched: {parameter_counts}")

    validation = pd.concat(validation_frames, ignore_index=True)
    save_csv(validation, output / "validation_predictions.csv")
    validation_metrics = metrics(validation, normalization)
    save_csv(validation_metrics, output / "validation_metrics.csv")
    macro = (
        validation_metrics.groupby(["task", "model"], as_index=False).normalized_mae.mean()
    )
    selection = {
        arm: equal_task_macro_score(validation[validation.model.eq(arm)], normalization)
        for arm in config["arms"]
    }
    save_json(selection, output / "validation_selection.json")

    baseline_name = "numeric_query_joint"
    baseline = validation[validation.model.eq(baseline_name)]
    bootstrap_rows = []
    for arm in config["arms"]:
        if arm == baseline_name:
            continue
        for task in TASK_GROUPS:
            bootstrap_rows.append(
                bootstrap_difference(
                    validation[
                        validation.model.eq(arm) & validation.task.eq(task)
                    ],
                    baseline[baseline.task.eq(task)],
                    normalization,
                    config["statistics"]["bootstrap_replicates"],
                    config["seed"],
                )
            )
    bootstrap = pd.DataFrame(bootstrap_rows)
    save_csv(bootstrap, output / "paired_subject_bootstrap.csv")
    semantic_arms = [name for name, arm in config["arms"].items() if arm["task_semantics"]]
    selected = min(semantic_arms, key=selection.get)
    rows = bootstrap[bootstrap.candidate.eq(selected)].set_index("task")
    criteria = {
        "masked_item_improved": bool(rows.loc["masked_item", "ci_upper"] < 0),
        "masked_instrument_improved": bool(rows.loc["masked_instrument", "ci_upper"] < 0),
        "cross_scale_two_pairs_improved": bool(
            rows.loc[
                ["cross_adas_cdr_to_moca", "cross_mmse_cdr_to_adas"], "ci_upper"
            ].lt(0).all()
        ),
        "future_noninferior": bool(
            rows.loc["future_forecast", "ci_upper"]
            <= config["gate"]["future_noninferiority_margin_normalized_mae"]
        ),
    }
    decision = {
        "run_id": config["run_id"],
        "engineering_smoke_only": smoke,
        "selected_semantic_candidate": selected,
        "criteria": criteria,
        "provisional_validation_passed": bool(all(criteria.values())),
        "test_evaluated": False,
        "advance_to_mri_stage2": False,
        "next_action": (
            "freeze_three_seed_confirmation_protocol"
            if all(criteria.values())
            else "revise_on_train_validation_only"
        ),
    }
    save_json(decision, output / "development_decision.json")
    sample_counts = {
        split: {task: len(values) for task, values in split_tasks.items()}
        for split, split_tasks in tasks.items()
    }
    manifest = {
        "run_id": config["run_id"],
        "development_only": True,
        "engineering_smoke_only": smoke,
        "test_evaluated": False,
        "features": features,
        "targets": targets,
        "domains": domains,
        "parameter_counts": parameter_counts,
        "sample_counts": sample_counts,
        "typed_feature_counts": {
            name: int(feature_types.eq(type_id).sum()) for name, type_id in TYPE_IDS.items()
        },
        "direction_aligned": True,
        "item_conditioned_decoder": True,
        "typed_losses": True,
    }
    save_json(manifest, output / "manifest.json")
    write_report(
        output,
        macro,
        bootstrap,
        selection,
        decision,
        parameter_counts,
        sample_counts,
    )
    save_json(
        {
            "completed": True,
            "development_only": True,
            "engineering_smoke_only": smoke,
            "test_evaluated": False,
            "provisional_validation_passed": decision["provisional_validation_passed"],
            "report": str(output / "report.md"),
        },
        output / "status.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/phase1d_typed_representation_dev.yaml",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.config, args.smoke)


if __name__ == "__main__":
    main()
