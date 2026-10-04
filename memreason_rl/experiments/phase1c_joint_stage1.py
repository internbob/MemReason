"""Joint response-aware Stage I experiment with four learning objectives."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import yaml

from ..data.common import ROOT, read_csv, save_csv, save_json
from ..data.dictionary import load_dictionary
from .phase1_numeric import build_samples, clean_frame, fit_normalization, markdown_table
from .phase1b_semantic import build_cross_scale_samples, seed_everything, select_verified_scalar_features


TASK_IDS = {"future_forecast": 0, "cross_scale": 1, "masked_item": 2, "masked_instrument": 3}


@dataclass
class JointSample:
    sample_id: str
    subject_id: str
    split: str
    task: str
    cutoff_month: float
    horizon_months: int
    months: np.ndarray
    features: np.ndarray
    feature_mask: np.ndarray
    total_targets: np.ndarray
    total_target_mask: np.ndarray
    target_months: np.ndarray
    reconstruction_targets: np.ndarray
    reconstruction_mask: np.ndarray


class JointDataset(Dataset):
    def __init__(self, samples: list[JointSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> JointSample:
        return self.samples[index]


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def _stable_choice(size: int, text: str) -> int:
    digest = hashlib.sha256(text.encode()).hexdigest()
    return int(digest[:16], 16) % size


def _value_key(value: float) -> float:
    return round(float(value), 4)


def build_response_vocab(train: pd.DataFrame, features: list[str]) -> dict[str, list[float]]:
    vocab = {}
    for feature in features:
        values = pd.to_numeric(train[feature], errors="coerce").dropna().to_numpy(dtype=float)
        vocab[feature] = sorted({_value_key(value) for value in values})
        if not vocab[feature]:
            raise ValueError(f"No training response vocabulary for {feature}")
    return vocab


def task_text(feature: str, entry: dict[str, Any]) -> str:
    return (
        f"Instrument: {entry['instrument']}. Domain: {entry['domain']}. "
        f"Construct: {entry['construct']}. Task: {entry['task_description']}. "
        f"Score direction: {entry['direction']}. Source variable: {feature}."
    )


def _response_label(entry: dict[str, Any], value: float) -> str | None:
    for key, label in (entry.get("response_labels") or {}).items():
        try:
            if abs(float(key) - value) < 1e-4:
                return str(label)
        except ValueError:
            continue
    return None


def response_text(feature: str, entry: dict[str, Any], value: float) -> str:
    label = _response_label(entry, value)
    allowed = entry.get("valid_values")
    bounds = entry.get("range")
    if label:
        response = f"Observed response: {label}. Raw score: {value:g}."
    elif allowed:
        response = f"Observed response score: {value:g}. Allowed scores: {', '.join(map(str, allowed))}."
    elif bounds:
        response = f"Observed response score: {value:g}. Allowed range: {bounds[0]} to {bounds[1]}."
    else:
        response = f"Observed response score: {value:g}."
    return task_text(feature, entry) + " " + response


def encode_descriptor_bundle(
    features: list[str],
    response_vocab: dict[str, list[float]],
    dictionary: dict[str, Any],
    config: dict[str, Any],
    output: Path,
    device: torch.device,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    task_texts = [task_text(feature, dictionary[feature]) for feature in features]
    response_texts = [
        response_text(feature, dictionary[feature], value)
        for feature in features
        for value in response_vocab[feature]
    ]
    signature = hashlib.sha256(
        json.dumps(
            {"features": features, "vocab": response_vocab, "tasks": task_texts, "responses": response_texts},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    cache = output / "descriptor_cache" / f"response_bundle_{signature[:12]}.pt"
    dimension = config["model"]["descriptor_input_dim"]
    max_vocab = max(len(values) for values in response_vocab.values())
    if cache.exists():
        record = torch.load(cache, map_location="cpu", weights_only=True)
        if record["signature"] == signature:
            task_embeddings = record["task_embeddings"]
            response_embeddings = record["response_embeddings"]
        else:
            raise RuntimeError("Descriptor cache signature mismatch")
    else:
        from transformers import AutoModel, AutoTokenizer

        settings = config["text_encoder"]
        tokenizer = AutoTokenizer.from_pretrained(settings["path"], local_files_only=True)
        encoder = AutoModel.from_pretrained(settings["path"], local_files_only=True).to(device).eval()
        encoder.requires_grad_(False)

        def encode(texts: list[str]) -> torch.Tensor:
            batches = []
            for start in range(0, len(texts), settings["batch_size"]):
                encoded = tokenizer(
                    texts[start : start + settings["batch_size"]],
                    padding=True,
                    truncation=True,
                    max_length=settings["max_length"],
                    return_tensors="pt",
                ).to(device)
                with torch.no_grad():
                    hidden = encoder(**encoded).last_hidden_state
                    mask = encoded["attention_mask"].unsqueeze(-1)
                    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
                    batches.append(F.normalize(pooled, dim=-1).cpu())
            return torch.cat(batches)

        task_embeddings = encode(task_texts)
        flat_response = encode(response_texts)
        response_embeddings = torch.zeros(len(features), max_vocab, dimension)
        cursor = 0
        for feature_index, feature in enumerate(features):
            count = len(response_vocab[feature])
            response_embeddings[feature_index, :count] = flat_response[cursor : cursor + count]
            cursor += count
        if task_embeddings.shape != (len(features), dimension):
            raise ValueError(f"Unexpected task descriptor shape {task_embeddings.shape}")
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "signature": signature,
                "task_embeddings": task_embeddings,
                "response_embeddings": response_embeddings,
                "response_vocab": response_vocab,
            },
            cache,
        )
        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    zeros_task = torch.zeros_like(task_embeddings)
    zeros_response = torch.zeros_like(response_embeddings)
    task_repeated = task_embeddings[:, None, :].expand_as(response_embeddings).clone()
    return {
        "numeric_only": (zeros_task, zeros_response),
        "task_semantics": (task_embeddings, task_repeated),
        "response_aware": (task_embeddings, response_embeddings),
    }


def from_forecast(sample: Any, task: str) -> JointSample:
    n_features = sample.features.shape[1]
    return JointSample(
        sample_id=sample.sample_id,
        subject_id=sample.subject_id,
        split=sample.split,
        task=task,
        cutoff_month=sample.cutoff_month,
        horizon_months=sample.horizon_months,
        months=sample.months,
        features=sample.features,
        feature_mask=sample.feature_mask,
        total_targets=sample.targets,
        total_target_mask=sample.target_mask,
        target_months=sample.target_months,
        reconstruction_targets=np.full(n_features, np.nan, dtype=np.float32),
        reconstruction_mask=np.zeros(n_features, dtype=bool),
    )


def build_reconstruction_samples(
    frame: pd.DataFrame,
    split: str,
    features: list[str],
    targets: list[str],
    dictionary: dict[str, Any],
    seed: int,
) -> dict[str, list[JointSample]]:
    instruments = sorted({dictionary[feature]["instrument"] for feature in features})
    instrument_indices = {
        instrument: np.asarray(
            [index for index, feature in enumerate(features) if dictionary[feature]["instrument"] == instrument]
        )
        for instrument in instruments
    }
    output = {"masked_item": [], "masked_instrument": []}
    for row_index, row in enumerate(frame.itertuples(index=False)):
        values = np.asarray(
            [pd.to_numeric(getattr(row, feature), errors="coerce") for feature in features], dtype=np.float32
        )
        observed = np.isfinite(values)
        observed_indices = np.flatnonzero(observed)
        if len(observed_indices) < 2:
            continue
        identity = f"{seed}:{split}:{row.subject_id}:{float(row.visit_month):g}:{row_index}"

        hidden_item = observed_indices[_stable_choice(len(observed_indices), identity + ":item")]
        item_input_mask = observed.copy()
        item_input_mask[hidden_item] = False
        item_target_mask = np.zeros(len(features), dtype=bool)
        item_target_mask[hidden_item] = True
        output["masked_item"].append(
            JointSample(
                sample_id=f"masked_item:{row.subject_id}:m{float(row.visit_month):g}:r{row_index}",
                subject_id=row.subject_id,
                split=split,
                task="masked_item",
                cutoff_month=float(row.visit_month),
                horizon_months=0,
                months=np.asarray([float(row.visit_month)]),
                features=values[None],
                feature_mask=item_input_mask[None],
                total_targets=np.full(len(targets), np.nan, dtype=np.float32),
                total_target_mask=np.zeros(len(targets), dtype=bool),
                target_months=np.full(len(targets), np.nan, dtype=np.float32),
                reconstruction_targets=values.copy(),
                reconstruction_mask=item_target_mask,
            )
        )

        eligible = []
        for instrument, indices in instrument_indices.items():
            hidden = observed[indices].any()
            outside = observed[np.setdiff1d(np.arange(len(features)), indices)].any()
            if hidden and outside:
                eligible.append(instrument)
        if not eligible:
            continue
        instrument = eligible[_stable_choice(len(eligible), identity + ":instrument")]
        hidden_indices = instrument_indices[instrument]
        instrument_target_mask = np.zeros(len(features), dtype=bool)
        instrument_target_mask[hidden_indices] = observed[hidden_indices]
        instrument_input_mask = observed.copy()
        instrument_input_mask[hidden_indices] = False
        output["masked_instrument"].append(
            JointSample(
                sample_id=(
                    f"masked_instrument:{instrument}:{row.subject_id}:"
                    f"m{float(row.visit_month):g}:r{row_index}"
                ),
                subject_id=row.subject_id,
                split=split,
                task="masked_instrument",
                cutoff_month=float(row.visit_month),
                horizon_months=0,
                months=np.asarray([float(row.visit_month)]),
                features=values[None],
                feature_mask=instrument_input_mask[None],
                total_targets=np.full(len(targets), np.nan, dtype=np.float32),
                total_target_mask=np.zeros(len(targets), dtype=bool),
                target_months=np.full(len(targets), np.nan, dtype=np.float32),
                reconstruction_targets=values.copy(),
                reconstruction_mask=instrument_target_mask,
            )
        )
    return output


def build_joint_samples(
    frame: pd.DataFrame,
    split: str,
    features: list[str],
    targets: list[str],
    dictionary: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, list[JointSample]]:
    horizons = config["future_forecast"]["horizons_months"]
    tolerances = {int(key): int(value) for key, value in config["future_forecast"]["tolerance_months"].items()}
    future = [
        from_forecast(sample, "future_forecast")
        for sample in build_samples(frame, split, features, targets, horizons, tolerances)
    ]
    cross_raw = build_cross_scale_samples(frame, split, features, targets, config["cross_scale"])
    cross = {name: [] for name in [f"cross_{pair['name']}" for pair in config["cross_scale"]]}
    for sample in cross_raw:
        cross[sample.task].append(from_forecast(sample, sample.task))
    reconstruction = build_reconstruction_samples(
        frame,
        split,
        features,
        targets,
        dictionary,
        config["reconstruction"]["deterministic_mask_seed"],
    )
    return {"future_forecast": future, **cross, **reconstruction}


class JointCollator:
    def __init__(
        self,
        features: list[str],
        targets: list[str],
        normalization: dict[str, dict[str, float]],
        response_vocab: dict[str, list[float]],
    ):
        self.features = features
        self.targets = targets
        self.feature_mean = np.asarray([normalization[name]["mean"] for name in features], dtype=np.float32)
        self.feature_std = np.asarray([normalization[name]["std"] for name in features], dtype=np.float32)
        self.target_mean = np.asarray([normalization[name]["mean"] for name in targets], dtype=np.float32)
        self.target_std = np.asarray([normalization[name]["std"] for name in targets], dtype=np.float32)
        self.response_index = {
            feature: {_value_key(value): index for index, value in enumerate(response_vocab[feature])}
            for feature in features
        }

    def __call__(self, samples: list[JointSample]) -> dict[str, Any]:
        batch_size = len(samples)
        steps = max(len(sample.months) for sample in samples)
        n_features, n_targets = len(self.features), len(self.targets)
        values = np.zeros((batch_size, steps, n_features), dtype=np.float32)
        mask = np.zeros((batch_size, steps, n_features), dtype=bool)
        response_index = np.full((batch_size, steps, n_features), -1, dtype=np.int64)
        times = np.zeros((batch_size, steps), dtype=np.float32)
        event_mask = np.zeros((batch_size, steps), dtype=bool)
        total_targets = np.zeros((batch_size, n_targets), dtype=np.float32)
        total_mask = np.zeros((batch_size, n_targets), dtype=bool)
        reconstruction_targets = np.zeros((batch_size, n_features), dtype=np.float32)
        reconstruction_mask = np.zeros((batch_size, n_features), dtype=bool)
        for row, sample in enumerate(samples):
            length = len(sample.months)
            normalized = (sample.features - self.feature_mean) / self.feature_std
            values[row, :length] = np.where(sample.feature_mask, normalized, 0.0)
            mask[row, :length] = sample.feature_mask
            times[row, :length] = sample.months / 12.0
            event_mask[row, :length] = True
            for step in range(length):
                for feature_index in np.flatnonzero(sample.feature_mask[step]):
                    key = _value_key(sample.features[step, feature_index])
                    response_index[row, step, feature_index] = self.response_index[
                        self.features[feature_index]
                    ].get(key, -1)
            total_targets[row] = np.where(
                sample.total_target_mask,
                (sample.total_targets - self.target_mean) / self.target_std,
                0.0,
            )
            total_mask[row] = sample.total_target_mask
            reconstruction_targets[row] = np.where(
                sample.reconstruction_mask,
                (sample.reconstruction_targets - self.feature_mean) / self.feature_std,
                0.0,
            )
            reconstruction_mask[row] = sample.reconstruction_mask
        return {
            "values": torch.from_numpy(values),
            "mask": torch.from_numpy(mask),
            "response_index": torch.from_numpy(response_index),
            "times": torch.from_numpy(times),
            "event_mask": torch.from_numpy(event_mask),
            "horizon": torch.tensor([sample.horizon_months / 12 for sample in samples], dtype=torch.float32),
            "task_id": torch.tensor(
                [TASK_IDS["cross_scale"] if sample.task.startswith("cross_") else TASK_IDS[sample.task] for sample in samples],
                dtype=torch.long,
            ),
            "total_targets": torch.from_numpy(total_targets),
            "total_mask": torch.from_numpy(total_mask),
            "reconstruction_targets": torch.from_numpy(reconstruction_targets),
            "reconstruction_mask": torch.from_numpy(reconstruction_mask),
            "samples": samples,
        }


class ResponseAwareStage1(nn.Module):
    """Parameter-matched response-aware GRU-D with explicit domain aggregation."""

    def __init__(
        self,
        task_descriptors: torch.Tensor,
        response_descriptors: torch.Tensor,
        domain_assignment: torch.Tensor,
        n_targets: int,
        config: dict[str, Any],
    ):
        super().__init__()
        settings = config["model"]
        n_features, _, descriptor_input = response_descriptors.shape
        n_domains = domain_assignment.shape[1]
        descriptor_dim = settings["descriptor_dim"]
        hidden = settings["hidden_dim"]
        layers = settings["layers"]
        self.register_buffer("task_descriptors", task_descriptors.float())
        self.register_buffer("response_descriptors", response_descriptors.float())
        self.register_buffer("domain_assignment", domain_assignment.float())
        self.semantic_projection = nn.Linear(descriptor_input, descriptor_dim, bias=False)
        self.domain_aggregation = nn.Sequential(
            nn.Linear(n_domains * descriptor_dim, settings["domain_dim"]),
            nn.GELU(),
            nn.LayerNorm(settings["domain_dim"]),
        )
        self.decay_weight = nn.Parameter(torch.zeros(n_features))
        self.decay_bias = nn.Parameter(torch.zeros(n_features))
        first_input = n_features * 2 + settings["domain_dim"] + 2
        self.cells = nn.ModuleList(
            [nn.GRUCell(first_input, hidden)] + [nn.GRUCell(hidden, hidden) for _ in range(layers - 1)]
        )
        self.dropout = nn.Dropout(settings["dropout"])
        self.horizon = nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.task_embedding = nn.Embedding(len(TASK_IDS), settings["task_dim"])
        head_input = hidden * 2 + settings["task_dim"]
        self.total_head = nn.Linear(head_input, n_targets)
        self.reconstruction_head = nn.Linear(hidden + settings["task_dim"], n_features)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        values, observed = batch["values"], batch["mask"]
        indices, times, valid = batch["response_index"], batch["times"], batch["event_mask"]
        batch_size, steps, n_features = values.shape
        projected_task = self.semantic_projection(self.task_descriptors)
        projected_response = self.semantic_projection(self.response_descriptors)
        safe_indices = indices.clamp_min(0)
        feature_indices = torch.arange(n_features, device=values.device).view(1, 1, -1)
        semantic = projected_response[feature_indices, safe_indices]
        fallback = projected_task.view(1, 1, n_features, -1)
        semantic = torch.where((indices >= 0).unsqueeze(-1), semantic, fallback)
        semantic = semantic * observed.unsqueeze(-1).to(values.dtype)
        domain_sum = torch.einsum("btfe,fd->btde", semantic, self.domain_assignment)
        domain_count = torch.einsum(
            "btf,fd->btd", observed.to(values.dtype), self.domain_assignment
        ).clamp_min(1.0)
        domain_summary = self.domain_aggregation((domain_sum / domain_count.unsqueeze(-1)).flatten(-2))

        hidden = [values.new_zeros(batch_size, cell.hidden_size) for cell in self.cells]
        last_value = values.new_zeros(batch_size, n_features)
        last_observed_time = times[:, :1].expand(-1, n_features).clone()
        previous_time = times[:, 0]
        for step in range(steps):
            event_valid = valid[:, step]
            current_time = times[:, step]
            elapsed = (current_time[:, None] - last_observed_time).clamp_min(0)
            gamma = torch.exp(-F.relu(self.decay_weight[None] * elapsed + self.decay_bias[None]))
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
            last_value = torch.where(observed[:, step] & event_valid[:, None], values[:, step], last_value)
            last_observed_time = torch.where(
                observed[:, step] & event_valid[:, None], current_time[:, None], last_observed_time
            )
            previous_time = torch.where(event_valid, current_time, previous_time)

        task = self.task_embedding(batch["task_id"])
        horizon = self.horizon(batch["horizon"][:, None])
        total = self.total_head(torch.cat([hidden[-1], horizon, task], dim=-1))
        reconstruction = self.reconstruction_head(torch.cat([hidden[-1], task], dim=-1))
        return total, reconstruction


def task_balanced_loss(
    total_prediction: torch.Tensor,
    reconstruction_prediction: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    losses = []
    for task_id in batch["task_id"].unique():
        rows = batch["task_id"].eq(task_id)
        if task_id.item() in {TASK_IDS["future_forecast"], TASK_IDS["cross_scale"]}:
            mask = batch["total_mask"] & rows[:, None]
            if mask.any():
                losses.append(F.huber_loss(total_prediction[mask], batch["total_targets"][mask], delta=1.0))
        else:
            mask = batch["reconstruction_mask"] & rows[:, None]
            if mask.any():
                losses.append(
                    F.huber_loss(
                        reconstruction_prediction[mask], batch["reconstruction_targets"][mask], delta=1.0
                    )
                )
    if not losses:
        return total_prediction.sum() * 0
    return torch.stack(losses).mean()


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
    for sample, total_values, reconstruction_values in zip(
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
                    "prediction": float(total_values[index]),
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
                    "prediction": float(reconstruction_values[index]),
                }
            )
    return pd.DataFrame(rows)


def evaluate(
    model: nn.Module,
    samples: list[JointSample],
    collator: JointCollator,
    batch_size: int,
    device: torch.device,
    arm: str,
    seed: int,
    features: list[str],
    targets: list[str],
) -> pd.DataFrame:
    loader = DataLoader(JointDataset(samples), batch_size=batch_size, shuffle=False, collate_fn=collator)
    model.eval()
    total_predictions, reconstruction_predictions, ordered = [], [], []
    with torch.no_grad():
        for batch in loader:
            local = batch.pop("samples")
            batch = {key: value.to(device) for key, value in batch.items()}
            total, reconstruction = model(batch)
            total_predictions.append(total.cpu().numpy() * collator.target_std + collator.target_mean)
            reconstruction_predictions.append(
                reconstruction.cpu().numpy() * collator.feature_std + collator.feature_mean
            )
            ordered.extend(local)
    return prediction_frame(
        ordered,
        np.concatenate(total_predictions),
        np.concatenate(reconstruction_predictions),
        arm,
        seed,
        features,
        targets,
    )


def metrics(predictions: pd.DataFrame, normalization: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows = []
    keys = ["split", "task", "model", "seed", "target", "horizon_months"]
    for key, group in predictions.groupby(keys, dropna=False, sort=True):
        error = group.prediction - group.truth
        scale = normalization[key[4]]["std"]
        rows.append(
            {
                **dict(zip(keys, key)),
                "n": int(len(group)),
                "n_subjects": int(group.subject_id.nunique()),
                "mae": float(error.abs().mean()),
                "normalized_mae": float(error.abs().mean() / scale),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "pearson": float(group.prediction.corr(group.truth))
                if len(group) > 2 and group.prediction.std() > 0 and group.truth.std() > 0
                else None,
                "spearman": float(group.prediction.corr(group.truth, method="spearman"))
                if len(group) > 2 and group.prediction.nunique() > 1 and group.truth.nunique() > 1
                else None,
            }
        )
    return pd.DataFrame(rows)


def equal_task_macro_score(
    predictions: pd.DataFrame, normalization: dict[str, dict[str, float]]
) -> float:
    table = metrics(predictions, normalization)
    return float(table.groupby("task").normalized_mae.mean().mean())


def build_domain_assignment(
    features: list[str], dictionary: dict[str, Any]
) -> tuple[list[str], torch.Tensor]:
    domains = sorted({dictionary[feature]["domain"] for feature in features})
    assignment = torch.zeros(len(features), len(domains))
    for feature_index, feature in enumerate(features):
        assignment[feature_index, domains.index(dictionary[feature]["domain"])] = 1.0
    return domains, assignment


def flatten_tasks(tasks: dict[str, list[JointSample]]) -> list[JointSample]:
    return [sample for name in sorted(tasks) for sample in tasks[name]]


def checkpoint_path(output: Path, arm: str, seed: int) -> Path:
    return output / "checkpoints" / arm / str(seed) / "best.pt"


def train_one(
    arm: str,
    seed: int,
    descriptors: tuple[torch.Tensor, torch.Tensor],
    domain_assignment: torch.Tensor,
    train_tasks: dict[str, list[JointSample]],
    val_samples: list[JointSample],
    features: list[str],
    targets: list[str],
    normalization: dict[str, dict[str, float]],
    response_vocab: dict[str, list[float]],
    config: dict[str, Any],
    output: Path,
    device: torch.device,
) -> tuple[pd.DataFrame, int]:
    seed_everything(seed)
    settings = config["training"]
    train_samples = flatten_tasks(train_tasks)
    counts = Counter(sample.task for sample in train_samples)
    weights = torch.tensor([1.0 / counts[sample.task] for sample in train_samples], dtype=torch.double)
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(weights, len(train_samples), replacement=True, generator=generator)
    collator = JointCollator(features, targets, normalization, response_vocab)
    loader = DataLoader(
        JointDataset(train_samples),
        batch_size=settings["batch_size"],
        sampler=sampler,
        collate_fn=collator,
        num_workers=0,
    )
    model = ResponseAwareStage1(
        descriptors[0], descriptors[1], domain_assignment, len(targets), config
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    best_score, best_state, patience = float("inf"), None, 0
    curves = []
    for epoch in range(settings["max_epochs"]):
        model.train()
        losses = []
        for batch in loader:
            batch.pop("samples")
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            total, reconstruction = model(batch)
            loss = task_balanced_loss(total, reconstruction, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss for {arm}/{seed}")
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
            arm,
            seed,
            features,
            targets,
        )
        score = equal_task_macro_score(validation, normalization)
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_equal_task_macro_normalized_mae": score,
        }
        curves.append(record)
        print(json.dumps({"arm": arm, "seed": seed, **record}), flush=True)
        if score < best_score - 1e-5:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= settings["patience"]:
                break
    if best_state is None:
        raise RuntimeError(f"No checkpoint for {arm}/{seed}")
    path = checkpoint_path(output, arm, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": best_state,
            "arm": arm,
            "seed": seed,
            "features": features,
            "targets": targets,
            "parameter_count": parameter_count,
            "validation_score": best_score,
        },
        path,
    )
    save_csv(pd.DataFrame(curves), path.parent / "curves.csv")
    model.load_state_dict(best_state)
    return (
        evaluate(
            model,
            val_samples,
            collator,
            settings["batch_size"],
            device,
            arm,
            seed,
            features,
            targets,
        ),
        parameter_count,
    )


def evaluate_checkpoint(
    arm: str,
    seed: int,
    descriptors: tuple[torch.Tensor, torch.Tensor],
    domain_assignment: torch.Tensor,
    samples: list[JointSample],
    features: list[str],
    targets: list[str],
    normalization: dict[str, dict[str, float]],
    response_vocab: dict[str, list[float]],
    config: dict[str, Any],
    output: Path,
    device: torch.device,
) -> pd.DataFrame:
    checkpoint = torch.load(checkpoint_path(output, arm, seed), map_location="cpu", weights_only=True)
    if checkpoint["features"] != features or checkpoint["targets"] != targets:
        raise RuntimeError("Checkpoint data contract mismatch")
    model = ResponseAwareStage1(
        descriptors[0], descriptors[1], domain_assignment, len(targets), config
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    return evaluate(
        model,
        samples,
        JointCollator(features, targets, normalization, response_vocab),
        config["training"]["batch_size"],
        device,
        arm,
        seed,
        features,
        targets,
    )


def ensembles(predictions: pd.DataFrame, arms: list[str]) -> pd.DataFrame:
    keys = [
        "sample_id", "subject_id", "split", "task", "target", "horizon_months",
        "cutoff_month", "target_month", "truth",
    ]
    frames = []
    for arm in arms:
        frame = predictions[predictions.model.eq(arm)].groupby(keys, as_index=False).prediction.mean()
        frame["model"] = arm + " ensemble"
        frame["seed"] = pd.NA
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def bootstrap_difference(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    normalization: dict[str, dict[str, float]],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    keys = ["sample_id", "subject_id", "task", "target", "horizon_months", "truth"]
    paired = candidate[keys + ["prediction"]].merge(
        baseline[keys + ["prediction"]],
        on=keys,
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    ).dropna()
    scale = paired.target.map(lambda target: normalization[target]["std"]).to_numpy()
    paired["difference"] = (
        np.abs(paired.prediction_candidate - paired.truth)
        - np.abs(paired.prediction_baseline - paired.truth)
    ) / scale
    subject = paired.groupby("subject_id").difference.agg(["sum", "count"])
    sums, counts = subject["sum"].to_numpy(), subject["count"].to_numpy()
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(subject), size=(replicates, len(subject)))
    draws = sums[indices].sum(1) / counts[indices].sum(1)
    return {
        "task": str(paired.task.iloc[0]),
        "candidate": str(candidate.model.iloc[0]),
        "baseline": str(baseline.model.iloc[0]),
        "n_paired_targets": int(len(paired)),
        "n_subjects": int(len(subject)),
        "normalized_mae_difference": float(paired.difference.mean()),
        "ci_lower": float(np.quantile(draws, 0.025)),
        "ci_upper": float(np.quantile(draws, 0.975)),
        "bootstrap_probability_candidate_better": float(np.mean(draws < 0)),
        "replicates": replicates,
    }


def reconstruction_type_metrics(
    predictions: pd.DataFrame, dictionary: dict[str, Any]
) -> pd.DataFrame:
    reconstruction = predictions[predictions.task.isin(["masked_item", "masked_instrument"])].copy()
    reconstruction["value_type"] = reconstruction.target.map(lambda target: dictionary[target]["value_type"])
    rows = []
    for (task, model, value_type), group in reconstruction.groupby(["task", "model", "value_type"]):
        record: dict[str, Any] = {
            "task": task,
            "model": model,
            "value_type": value_type,
            "n": int(len(group)),
            "mae": float(np.abs(group.prediction - group.truth).mean()),
        }
        if value_type == "binary":
            labels = group.truth.astype(int)
            record["accuracy"] = float(((group.prediction >= 0.5).astype(int) == labels).mean())
            record["auroc"] = float(roc_auc_score(labels, group.prediction)) if labels.nunique() == 2 else None
        if value_type == "ordinal":
            rounded = []
            for target, prediction in zip(group.target, group.prediction):
                allowed = np.asarray(dictionary[target]["valid_values"], dtype=float)
                rounded.append(float(allowed[np.argmin(np.abs(allowed - prediction))]))
            record["weighted_kappa"] = float(cohen_kappa_score(group.truth, rounded, weights="quadratic"))
        rows.append(record)
    return pd.DataFrame(rows)


def write_report(
    output: Path,
    config: dict[str, Any],
    features: list[str],
    domains: list[str],
    sample_counts: dict[str, dict[str, int]],
    selection: dict[str, Any],
    test_metrics: pd.DataFrame,
    reconstruction_metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    parameter_counts: dict[str, int],
    gate: dict[str, Any],
) -> None:
    macro = (
        test_metrics[test_metrics.model.str.endswith("ensemble")]
        .groupby(["task", "model"], as_index=False)
        .normalized_mae.mean()
    )
    text = f"""# Joint response-aware Stage I results

Status: completed under the frozen Phase 1C protocol.

## Fairness and data

- Verified scalar inputs: {len(features)}
- Cognitive domains: {domains}
- Arms: {config['arms']}
- Trainable parameter counts: {parameter_counts}
- Samples by task/split: {sample_counts}
- Validation scores: {selection}
- Test samples were loaded only after all nine validation-trained checkpoints completed.

## Test macro normalized MAE

{markdown_table(macro)}

## Reconstruction metrics by value type

{markdown_table(reconstruction_metrics)}

## Paired subject bootstrap

Negative normalized-MAE differences favor the candidate.

{markdown_table(bootstrap)}

## Frozen Gate 1 decision

{json.dumps(gate, indent=2)}

## Interpretation limits

- Bio_ClinicalBERT is frozen; this experiment tests fixed response-text descriptors inside a trainable cognitive encoder.
- The internal ADNI test split has appeared in earlier project experiments and is not an external confirmation cohort.
- Cross-scale outcomes are concurrent. Future forecasting is temporally prospective under nominal visit months.
- MRI, diagnosis, SFT and RL are absent from model inputs.
- These test results cannot be used to modify this run ID.
"""
    (output / "report.md").write_text(text)


def run(config_path: Path, smoke: bool = False) -> None:
    config = load_config(config_path)
    if smoke:
        config = copy.deepcopy(config)
        config["neural_seeds"] = [config["neural_seeds"][0]]
        config["training"].update(max_epochs=2, patience=2, batch_size=64)
    output = ROOT / "outputs" / (config["run_id"] + ("_smoke" if smoke else ""))
    output.mkdir(parents=True, exist_ok=True)
    dictionary = load_dictionary()
    features = select_verified_scalar_features(config, dictionary)
    targets = list(config["targets"])
    train_val = {
        split: clean_frame(read_csv(ROOT / f"splits/{split}.csv"), features, targets, dictionary)
        for split in ["train", "val"]
    }
    normalization = fit_normalization(train_val["train"], features, targets)
    response_vocab = build_response_vocab(train_val["train"], features)
    tasks = {
        split: build_joint_samples(frame, split, features, targets, dictionary, config)
        for split, frame in train_val.items()
    }
    if smoke:
        for split_tasks in tasks.values():
            for name in split_tasks:
                split_tasks[name] = split_tasks[name][: min(96, len(split_tasks[name]))]
    val_samples = flatten_tasks(tasks["val"])
    device = torch.device("cuda" if torch.cuda.is_available() and not smoke else "cpu")
    descriptor_bundle = encode_descriptor_bundle(
        features, response_vocab, dictionary, config, output, device
    )
    domains, domain_assignment = build_domain_assignment(features, dictionary)
    validation_frames, parameter_counts = [], {}
    for arm in config["arms"]:
        arm_counts = []
        for seed in config["neural_seeds"]:
            frame, count = train_one(
                arm,
                seed,
                descriptor_bundle[arm],
                domain_assignment,
                tasks["train"],
                val_samples,
                features,
                targets,
                normalization,
                response_vocab,
                config,
                output,
                device,
            )
            validation_frames.append(frame)
            arm_counts.append(count)
        if len(set(arm_counts)) != 1:
            raise RuntimeError(f"Parameter count changed across seeds for {arm}")
        parameter_counts[arm] = arm_counts[0]
    if len(set(parameter_counts.values())) != 1:
        raise RuntimeError(f"Unfair trainable parameter counts: {parameter_counts}")

    validation = pd.concat(validation_frames, ignore_index=True)
    validation = pd.concat([validation, ensembles(validation, config["arms"])], ignore_index=True)
    save_csv(validation, output / "validation_predictions.csv")
    validation_metrics = metrics(validation, normalization)
    save_csv(validation_metrics, output / "validation_metrics.csv")
    selection = {
        arm: {
            str(int(seed)): equal_task_macro_score(frame, normalization)
            for seed, frame in validation[validation.model.eq(arm)].groupby("seed")
        }
        for arm in config["arms"]
    }
    save_json(selection, output / "validation_selection.json")
    sample_counts = {
        split: {task: len(values) for task, values in split_tasks.items()}
        for split, split_tasks in tasks.items()
    }
    manifest = {
        "run_id": config["run_id"],
        "engineering_smoke_only": smoke,
        "features": features,
        "targets": targets,
        "domains": domains,
        "arms": config["arms"],
        "parameter_counts": parameter_counts,
        "sample_counts_before_test": sample_counts,
        "response_vocab_source": "train_only",
        "response_vocab_sizes": {feature: len(values) for feature, values in response_vocab.items()},
        "dictionary_verified": all(dictionary[feature]["verified_from_dictionary"] for feature in features),
        "diagnosis_input": False,
        "test_evaluated": False,
        "deterministic_algorithms": True,
        "cuda_attention_backend": "math_sdp",
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }
    save_json(manifest, output / "manifest.json")
    save_json(config, output / "resolved_config.json")
    save_json(response_vocab, output / "response_vocab_train.json")
    if smoke:
        save_json(
            {"completed": True, "engineering_smoke_only": True, "test_evaluated": False},
            output / "status.json",
        )
        return

    test_frame = clean_frame(read_csv(ROOT / "splits/test.csv"), features, targets, dictionary)
    test_tasks = build_joint_samples(test_frame, "test", features, targets, dictionary, config)
    test_samples = flatten_tasks(test_tasks)
    test_frames = []
    for arm in config["arms"]:
        for seed in config["neural_seeds"]:
            test_frames.append(
                evaluate_checkpoint(
                    arm,
                    seed,
                    descriptor_bundle[arm],
                    domain_assignment,
                    test_samples,
                    features,
                    targets,
                    normalization,
                    response_vocab,
                    config,
                    output,
                    device,
                )
            )
    test = pd.concat(test_frames, ignore_index=True)
    test = pd.concat([test, ensembles(test, config["arms"])], ignore_index=True)
    save_csv(test, output / "test_predictions.csv")
    test_metrics = metrics(test, normalization)
    save_csv(test_metrics, output / "test_metrics.csv")
    reconstruction_metrics = reconstruction_type_metrics(
        test[test.model.str.endswith("ensemble")], dictionary
    )
    save_csv(reconstruction_metrics, output / "reconstruction_type_metrics.csv")

    bootstrap_rows = []
    for task in sorted(test.task.unique()):
        task_frame = test[test.task.eq(task)]
        baseline = task_frame[task_frame.model.eq("numeric_only ensemble")]
        for arm in ["task_semantics", "response_aware"]:
            bootstrap_rows.append(
                bootstrap_difference(
                    task_frame[task_frame.model.eq(arm + " ensemble")],
                    baseline,
                    normalization,
                    config["statistics"]["bootstrap_replicates"],
                    config["seed"],
                )
            )
    bootstrap = pd.DataFrame(bootstrap_rows)
    save_csv(bootstrap, output / "paired_subject_bootstrap.csv")
    response_rows = bootstrap[bootstrap.candidate.eq("response_aware ensemble")].set_index("task")
    leakage = json.loads((ROOT / "reports/data_audit/protocol_leakage.json").read_text())
    criteria = {
        "masked_item_improved": bool(response_rows.loc["masked_item", "ci_upper"] < 0),
        "masked_instrument_improved": bool(response_rows.loc["masked_instrument", "ci_upper"] < 0),
        "cross_scale_two_pairs_improved": bool(
            response_rows.loc[
                ["cross_adas_cdr_to_moca", "cross_mmse_cdr_to_adas"], "ci_upper"
            ].lt(0).all()
        ),
        "no_diagnosis_leakage": bool(
            leakage["validation_balanced_accuracy"]
            < config["gate"]["missingness_only_balanced_accuracy_max"]
        ),
        "future_noninferior": bool(
            response_rows.loc["future_forecast", "ci_upper"]
            <= config["gate"]["future_noninferiority_margin_normalized_mae"]
        ),
    }
    gate = {
        "run_id": config["run_id"],
        "criteria": criteria,
        "stage1_gate_passed": bool(
            criteria["masked_item_improved"]
            and criteria["masked_instrument_improved"]
            and criteria["cross_scale_two_pairs_improved"]
            and criteria["no_diagnosis_leakage"]
        ),
        "future_noninferiority_supportive": criteria["future_noninferior"],
        "missingness_only_validation_balanced_accuracy": leakage["validation_balanced_accuracy"],
        "advance_to_mri_stage2": False,
        "note": "advance_to_mri_stage2 remains false unless all frozen Stage I semantic criteria pass.",
    }
    gate["advance_to_mri_stage2"] = gate["stage1_gate_passed"]
    save_json(gate, output / "gate_decision.json")
    sample_counts["test"] = {task: len(values) for task, values in test_tasks.items()}
    manifest.update(sample_counts=sample_counts, test_evaluated=True)
    save_json(manifest, output / "manifest.json")
    write_report(
        output,
        config,
        features,
        domains,
        sample_counts,
        selection,
        test_metrics,
        reconstruction_metrics,
        bootstrap,
        parameter_counts,
        gate,
    )
    save_json(
        {
            "run_id": config["run_id"],
            "completed": True,
            "engineering_smoke_only": False,
            "test_evaluated": True,
            "stage1_gate_passed": gate["stage1_gate_passed"],
            "report": str(output / "report.md"),
        },
        output / "status.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/phase1c_joint_stage1.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.config, args.smoke)


if __name__ == "__main__":
    main()
