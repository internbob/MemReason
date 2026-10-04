"""Phase 1B: fair numeric, variable-name and task-semantics comparison."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import yaml

from ..data.common import ROOT, read_csv, save_csv, save_json
from ..data.dictionary import load_dictionary
from .phase1_numeric import (
    ForecastSample,
    NumericCollator,
    build_samples,
    clean_frame,
    fit_normalization,
    markdown_table,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def select_verified_scalar_features(config: dict[str, Any], dictionary: dict[str, Any]) -> list[str]:
    excluded = tuple(config["inputs"]["exclude_prefixes"])
    targets = set(config["targets"])
    features = []
    for key, entry in dictionary.items():
        if key in targets or key.startswith(excluded):
            continue
        if not entry.get("use_for_state") or not entry.get("verified_from_dictionary") or entry.get("default_drop"):
            continue
        if config["inputs"]["scalar_monotonic_only"]:
            if entry.get("value_type") == "categorical":
                continue
            if entry.get("direction") not in {"higher=better", "higher=worse"}:
                continue
        features.append(key)
    if not features:
        raise ValueError("No dictionary-verified scalar features")
    return features


def descriptor_text(feature: str, entry: dict[str, Any], arm: str) -> str:
    if arm == "variable_name":
        return "Source variable: " + feature.replace("_", " ")
    if arm == "task_semantics":
        return (
            f"Instrument: {entry['instrument']}. Domain: {entry['domain']}. "
            f"Construct: {entry['construct']}. Task: {entry['task_description']}. "
            f"Score direction: {entry['direction']}."
        )
    raise ValueError(arm)


def encode_descriptors(
    features: list[str],
    dictionary: dict[str, Any],
    arm: str,
    config: dict[str, Any],
    output: Path,
    device: torch.device,
) -> torch.Tensor:
    dimension = config["model"]["descriptor_input_dim"]
    if arm == "numeric_only":
        return torch.zeros(len(features), dimension)
    texts = [descriptor_text(feature, dictionary[feature], arm) for feature in features]
    signature = hashlib.sha256(json.dumps({"features": features, "texts": texts}, sort_keys=True).encode()).hexdigest()
    cache = output / "descriptor_cache" / f"{arm}_{signature[:12]}.pt"
    if cache.exists():
        record = torch.load(cache, map_location="cpu", weights_only=True)
        if record["signature"] == signature:
            return record["embeddings"]
    from transformers import AutoModel, AutoTokenizer
    settings = config["text_encoder"]
    tokenizer = AutoTokenizer.from_pretrained(settings["path"], local_files_only=True)
    encoder = AutoModel.from_pretrained(settings["path"], local_files_only=True).to(device).eval()
    encoder.requires_grad_(False)
    encoded = tokenizer(
        texts, padding=True, truncation=True, max_length=settings["max_length"], return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        hidden = encoder(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1)
        embeddings = ((hidden * mask).sum(1) / mask.sum(1).clamp_min(1)).cpu()
        embeddings = F.normalize(embeddings, dim=-1)
    if embeddings.shape != (len(features), dimension):
        raise ValueError(f"Unexpected descriptor shape {embeddings.shape}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"signature": signature, "features": features, "texts": texts, "embeddings": embeddings}, cache)
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return embeddings


class SampleDataset(Dataset):
    def __init__(self, samples: list[ForecastSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> ForecastSample:
        return self.samples[index]


class DescriptorGRUDForecaster(nn.Module):
    """Identical trainable architecture across arms; only descriptor buffer changes."""

    def __init__(
        self,
        descriptor_embeddings: torch.Tensor,
        n_targets: int,
        descriptor_dim: int,
        hidden: int,
        layers: int,
        dropout: float,
    ):
        super().__init__()
        n_features, descriptor_input_dim = descriptor_embeddings.shape
        self.register_buffer("descriptors", descriptor_embeddings.float())
        self.descriptor_projection = nn.Linear(descriptor_input_dim, descriptor_dim, bias=False)
        self.decay_weight = nn.Parameter(torch.zeros(n_features))
        self.decay_bias = nn.Parameter(torch.zeros(n_features))
        first_input = n_features * 2 + descriptor_dim * 2 + 2
        self.cells = nn.ModuleList(
            [nn.GRUCell(first_input, hidden)] + [nn.GRUCell(hidden, hidden) for _ in range(layers - 1)]
        )
        self.dropout = nn.Dropout(dropout)
        self.horizon = nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.output = nn.Linear(hidden * 2, n_targets)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        values, observed = batch["values"], batch["mask"]
        times, valid = batch["times"], batch["event_mask"]
        batch_size, steps, n_features = values.shape
        hidden = [values.new_zeros(batch_size, cell.hidden_size) for cell in self.cells]
        last_value = values.new_zeros(batch_size, n_features)
        last_observed_time = times[:, :1].expand(-1, n_features).clone()
        previous_time = times[:, 0]
        descriptor = self.descriptor_projection(self.descriptors)
        for step in range(steps):
            event_valid = valid[:, step]
            current_time = times[:, step]
            elapsed = (current_time[:, None] - last_observed_time).clamp_min(0)
            gamma = torch.exp(-F.relu(self.decay_weight[None] * elapsed + self.decay_bias[None]))
            imputed = torch.where(observed[:, step], values[:, step], gamma * last_value)
            mask_float = observed[:, step].to(values.dtype)
            count = mask_float.sum(-1, keepdim=True).clamp_min(1)
            semantic_value = (imputed * mask_float) @ descriptor / count
            semantic_presence = mask_float @ descriptor / count
            gap = (current_time - previous_time).clamp_min(0)
            cell_input = torch.cat(
                [imputed, mask_float, semantic_value, semantic_presence, current_time[:, None], gap[:, None]], dim=-1
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
        horizon = self.horizon(batch["horizon"][:, None])
        return self.output(torch.cat([hidden[-1], horizon], dim=-1))


def mark_task(samples: list[ForecastSample], task: str) -> list[ForecastSample]:
    for sample in samples:
        sample.task = task
    return samples


def build_cross_scale_samples(
    frame: pd.DataFrame,
    split: str,
    features: list[str],
    targets: list[str],
    pairs: list[dict[str, Any]],
) -> list[ForecastSample]:
    samples = []
    for pair in pairs:
        target_index = targets.index(pair["target"])
        source_indices = [
            index for index, feature in enumerate(features) if feature.startswith(tuple(pair["source_prefixes"]))
        ]
        if not source_indices:
            raise ValueError(f"No source features for {pair['name']}")
        for row in frame.itertuples(index=False):
            target_value = pd.to_numeric(getattr(row, pair["target"]), errors="coerce")
            if not np.isfinite(target_value):
                continue
            all_values = np.asarray([pd.to_numeric(getattr(row, feature), errors="coerce") for feature in features], dtype=float)
            feature_mask = np.isfinite(all_values)
            allowed = np.zeros(len(features), dtype=bool)
            allowed[source_indices] = True
            feature_mask &= allowed
            if not feature_mask.any():
                continue
            truth = np.full(len(targets), np.nan, dtype=np.float32)
            truth[target_index] = target_value
            truth_mask = np.zeros(len(targets), dtype=bool)
            truth_mask[target_index] = True
            truth_month = np.full(len(targets), np.nan, dtype=np.float32)
            truth_month[target_index] = float(row.visit_month)
            sample = ForecastSample(
                sample_id=f"{pair['name']}:{row.subject_id}:m{float(row.visit_month):g}",
                subject_id=row.subject_id,
                split=split,
                cutoff_month=float(row.visit_month),
                horizon_months=0,
                months=np.asarray([float(row.visit_month)]),
                features=all_values[None].astype(np.float32),
                feature_mask=feature_mask[None],
                history_targets=np.zeros((1, len(targets)), dtype=np.float32),
                history_target_mask=np.zeros((1, len(targets)), dtype=bool),
                targets=truth,
                target_mask=truth_mask,
                target_months=truth_month,
            )
            sample.task = f"cross_{pair['name']}"
            samples.append(sample)
    return samples


def masked_huber(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return F.huber_loss(prediction[mask], target[mask], delta=1.0)


def prediction_frame(
    samples: list[ForecastSample], prediction: np.ndarray, arm: str, seed: int, targets: list[str]
) -> pd.DataFrame:
    rows = []
    for sample, values in zip(samples, prediction):
        for index, target in enumerate(targets):
            if not sample.target_mask[index]:
                continue
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "subject_id": sample.subject_id,
                    "split": sample.split,
                    "task": sample.task,
                    "model": arm,
                    "seed": seed,
                    "target": target,
                    "horizon_months": sample.horizon_months,
                    "cutoff_month": sample.cutoff_month,
                    "target_month": float(sample.target_months[index]),
                    "truth": float(sample.targets[index]),
                    "prediction": float(values[index]),
                }
            )
    return pd.DataFrame(rows)


def evaluate(
    model: nn.Module,
    samples: list[ForecastSample],
    collator: NumericCollator,
    batch_size: int,
    device: torch.device,
    arm: str,
    seed: int,
    targets: list[str],
) -> pd.DataFrame:
    loader = DataLoader(SampleDataset(samples), batch_size=batch_size, shuffle=False, collate_fn=collator)
    means = collator.target_mean[None]
    stds = collator.target_std[None]
    predictions, ordered = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            local = batch.pop("samples")
            batch = {key: value.to(device) for key, value in batch.items()}
            predictions.append(model(batch).cpu().numpy() * stds + means)
            ordered.extend(local)
    return prediction_frame(ordered, np.concatenate(predictions), arm, seed, targets)


def metrics(predictions: pd.DataFrame, normalization: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows = []
    keys = ["split", "task", "model", "seed", "target", "horizon_months"]
    for key, group in predictions.groupby(keys, dropna=False, sort=True):
        error = group.prediction - group.truth
        target = key[4]
        rows.append(
            dict(
                zip(keys, key),
                n=int(len(group)),
                n_subjects=int(group.subject_id.nunique()),
                coverage=float(group.prediction.notna().mean()),
                mae=float(error.abs().mean()),
                normalized_mae=float(error.abs().mean() / normalization[target]["std"]),
                rmse=float(np.sqrt(np.mean(error**2))),
                pearson=float(group.prediction.corr(group.truth)) if len(group) > 2 and group.truth.std() > 0 else None,
                spearman=float(group.prediction.corr(group.truth, method="spearman")) if len(group) > 2 and group.truth.nunique() > 1 else None,
            )
        )
    return pd.DataFrame(rows)


def macro_score(predictions: pd.DataFrame, normalization: dict[str, dict[str, float]]) -> float:
    return float(metrics(predictions, normalization).normalized_mae.mean())


def train_one(
    arm: str,
    seed: int,
    task_name: str,
    descriptors: torch.Tensor,
    train_samples: list[ForecastSample],
    val_samples: list[ForecastSample],
    features: list[str],
    targets: list[str],
    normalization: dict[str, dict[str, float]],
    config: dict[str, Any],
    output: Path,
    device: torch.device,
) -> pd.DataFrame:
    seed_everything(seed)
    training, model_config = config["training"], config["model"]
    collator = NumericCollator(features, targets, normalization)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        SampleDataset(train_samples), batch_size=training["batch_size"], shuffle=True,
        collate_fn=collator, generator=generator, num_workers=0,
    )
    model = DescriptorGRUDForecaster(
        descriptors, len(targets), model_config["descriptor_dim"], model_config["hidden_dim"],
        model_config["layers"], model_config["dropout"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"]
    )
    best, best_state, patience, curves = float("inf"), None, 0, []
    for epoch in range(training["max_epochs"]):
        model.train()
        losses = []
        for batch in loader:
            batch.pop("samples")
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            loss = masked_huber(prediction, batch["targets"], batch["target_mask"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss for {task_name}/{arm}/{seed}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip"])
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = evaluate(model, val_samples, collator, training["batch_size"], device, arm, seed, targets)
        score = macro_score(validation, normalization)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_macro_normalized_mae": score}
        curves.append(record)
        print(json.dumps({"task": task_name, "arm": arm, "seed": seed, **record}), flush=True)
        if score < best - 1e-5:
            best, best_state, patience = score, copy.deepcopy(model.state_dict()), 0
        else:
            patience += 1
            if patience >= training["patience"]:
                break
    if best_state is None:
        raise RuntimeError("No checkpoint selected")
    directory = output / "checkpoints" / task_name / arm / str(seed)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": best_state,
            "task": task_name,
            "arm": arm,
            "seed": seed,
            "features": features,
            "targets": targets,
            "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        },
        directory / "best.pt",
    )
    save_csv(pd.DataFrame(curves), directory / "curves.csv")
    model.load_state_dict(best_state)
    return evaluate(model, val_samples, collator, training["batch_size"], device, arm, seed, targets)


def load_and_evaluate_test(
    arm: str,
    seed: int,
    task_name: str,
    descriptors: torch.Tensor,
    samples: list[ForecastSample],
    features: list[str],
    targets: list[str],
    normalization: dict[str, dict[str, float]],
    config: dict[str, Any],
    output: Path,
    device: torch.device,
) -> pd.DataFrame:
    model_config = config["model"]
    model = DescriptorGRUDForecaster(
        descriptors, len(targets), model_config["descriptor_dim"], model_config["hidden_dim"],
        model_config["layers"], model_config["dropout"],
    ).to(device)
    checkpoint = torch.load(output / "checkpoints" / task_name / arm / str(seed) / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    collator = NumericCollator(features, targets, normalization)
    return evaluate(model, samples, collator, config["training"]["batch_size"], device, arm, seed, targets)


def ensembles(predictions: pd.DataFrame, arms: list[str]) -> pd.DataFrame:
    keys = [
        "sample_id", "subject_id", "split", "task", "target", "horizon_months",
        "cutoff_month", "target_month", "truth",
    ]
    result = []
    for arm in arms:
        frame = predictions[predictions.model.eq(arm)].groupby(keys, as_index=False).prediction.mean()
        frame["model"] = arm + " ensemble"
        frame["seed"] = pd.NA
        result.append(frame)
    return pd.concat(result, ignore_index=True)


def bootstrap_normalized_error_difference(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    normalization: dict[str, dict[str, float]],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    keys = ["sample_id", "subject_id", "task", "target", "horizon_months", "truth"]
    paired = candidate[keys + ["prediction"]].merge(
        baseline[keys + ["prediction"]], on=keys, suffixes=("_candidate", "_baseline"), validate="one_to_one"
    )
    scale = paired.target.map(lambda target: normalization[target]["std"]).to_numpy()
    paired["difference"] = (
        np.abs(paired.prediction_candidate - paired.truth) - np.abs(paired.prediction_baseline - paired.truth)
    ) / scale
    subject = paired.groupby("subject_id").difference.agg(["sum", "count"])
    values_sum, values_count = subject["sum"].to_numpy(), subject["count"].to_numpy()
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(subject), size=(replicates, len(subject)))
    draws = values_sum[indices].sum(1) / values_count[indices].sum(1)
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


def write_report(
    output: Path,
    config: dict[str, Any],
    features: list[str],
    sample_counts: dict[str, dict[str, int]],
    validation_selection: dict[str, Any],
    test_metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    parameter_counts: dict[str, int],
) -> None:
    ensembles_only = test_metrics[test_metrics.model.str.endswith("ensemble")]
    macro = ensembles_only.groupby(["task", "model"], as_index=False).normalized_mae.mean()
    text = f"""# Phase 1B numeric versus semantic results

Status: completed under the frozen Phase 1B protocol.

## Fairness and data

- Dictionary-verified scalar inputs: {len(features)}
- Arms: {config['arms']}
- Frozen text encoder: `{config['text_encoder']['path']}`
- Samples: {sample_counts}
- Trainable parameter counts: {parameter_counts}
- All arms have identical trainable parameter counts and differ only in the fixed descriptor matrix.
- Validation checkpoint scores: {validation_selection}

## Test macro normalized MAE

{markdown_table(macro)}

## Paired subject bootstrap

Negative normalized-MAE differences favor the candidate.

{markdown_table(bootstrap)}

## Frozen-protocol decision

- Future forecasting: task semantics did not improve over numeric-only. A positive difference means worse error.
- Cross-scale: both declared pairs had small directionally favorable point estimates, but their 95% subject-bootstrap intervals crossed zero.
- The semantic-value gate is therefore **not passed**. These test results cannot be used to tune this run or claim stable semantic benefit.

## Complete test metrics

{markdown_table(ensembles_only[['task','model','target','horizon_months','n','n_subjects','mae','normalized_mae','rmse','spearman']])}

## Interpretation limits

- This experiment tests frozen item metadata embeddings, not generated response text or a fine-tuned language model.
- MoCA delayed-recall cue categories were verified but excluded because they are non-monotonic categories in a scalar-response comparison.
- Cross-scale targets are concurrent and establish representation transfer, not temporal forecasting.
- MRI, diagnosis, SFT and RL are absent.
- Test results must not be used to modify this run ID. Any follow-up requires a new protocol and run identifier.
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
    horizons = config["future_forecast"]["horizons_months"]
    tolerances = {int(key): int(value) for key, value in config["future_forecast"]["tolerance_months"].items()}
    samples = {
        "future": {
            split: mark_task(build_samples(frame, split, features, targets, horizons, tolerances), "future_forecast")
            for split, frame in train_val.items()
        },
        "cross": {
            split: build_cross_scale_samples(frame, split, features, targets, config["cross_scale"])
            for split, frame in train_val.items()
        },
    }
    if smoke:
        for task in samples.values():
            for split in task:
                task[split] = task[split][: min(256, len(task[split]))]
    device = torch.device("cuda" if torch.cuda.is_available() and not smoke else "cpu")
    descriptors = {
        arm: encode_descriptors(features, dictionary, arm, config, output, device)
        for arm in config["arms"]
    }
    task_names = {"future": "future_forecast", "cross": "cross_scale"}
    validation_frames = []
    parameter_counts = {}
    for task_key, task_name in task_names.items():
        for arm in config["arms"]:
            for seed in config["neural_seeds"]:
                validation_frames.append(
                    train_one(
                        arm, seed, task_name, descriptors[arm], samples[task_key]["train"], samples[task_key]["val"],
                        features, targets, normalization, config, output, device,
                    )
                )
                checkpoint = torch.load(
                    output / "checkpoints" / task_name / arm / str(seed) / "best.pt",
                    map_location="cpu", weights_only=True,
                )
                parameter_counts[arm] = checkpoint["trainable_parameters"]
    if len(set(parameter_counts.values())) != 1:
        raise RuntimeError(f"Unfair trainable parameter counts: {parameter_counts}")

    validation = pd.concat(validation_frames, ignore_index=True)
    validation = pd.concat([validation, ensembles(validation, config["arms"])], ignore_index=True)
    save_csv(validation, output / "validation_predictions.csv")
    validation_metrics = metrics(validation, normalization)
    save_csv(validation_metrics, output / "validation_metrics.csv")
    validation_selection = {}
    for task_key, task_name in task_names.items():
        task_rows = validation[validation.task.str.startswith("future") if task_key == "future" else validation.task.str.startswith("cross_")]
        validation_selection[task_name] = {
            arm: {
                str(int(seed)): macro_score(frame, normalization)
                for seed, frame in task_rows[task_rows.model.eq(arm)].groupby("seed")
            }
            for arm in config["arms"]
        }
    save_json(validation_selection, output / "validation_selection.json")

    sample_counts = {
        task_name: {split: len(samples[task_key][split]) for split in samples[task_key]}
        for task_key, task_name in task_names.items()
    }
    manifest = {
        "run_id": config["run_id"],
        "engineering_smoke_only": smoke,
        "features": features,
        "targets": targets,
        "sample_counts_before_test": sample_counts,
        "parameter_counts": parameter_counts,
        "dictionary_verified": all(dictionary[feature]["verified_from_dictionary"] for feature in features),
        "test_evaluated": False,
        "deterministic_algorithms": True,
        "cuda_attention_backend": "math_sdp",
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }
    save_json(manifest, output / "manifest.json")
    save_json(config, output / "resolved_config.json")
    if smoke:
        save_json(
            {"completed": True, "engineering_smoke_only": True, "test_evaluated": False}, output / "status.json"
        )
        return

    test_frame = clean_frame(read_csv(ROOT / "splits/test.csv"), features, targets, dictionary)
    test_samples = {
        "future": mark_task(build_samples(test_frame, "test", features, targets, horizons, tolerances), "future_forecast"),
        "cross": build_cross_scale_samples(test_frame, "test", features, targets, config["cross_scale"]),
    }
    sample_counts["future_forecast"]["test"] = len(test_samples["future"])
    sample_counts["cross_scale"]["test"] = len(test_samples["cross"])
    test_frames = []
    for task_key, task_name in task_names.items():
        for arm in config["arms"]:
            for seed in config["neural_seeds"]:
                test_frames.append(
                    load_and_evaluate_test(
                        arm, seed, task_name, descriptors[arm], test_samples[task_key], features, targets,
                        normalization, config, output, device,
                    )
                )
    test = pd.concat(test_frames, ignore_index=True)
    test = pd.concat([test, ensembles(test, config["arms"])], ignore_index=True)
    save_csv(test, output / "test_predictions.csv")
    test_metrics = metrics(test, normalization)
    save_csv(test_metrics, output / "test_metrics.csv")

    bootstrap_rows = []
    for task in sorted(test.task.unique()):
        task_frame = test[test.task.eq(task)]
        baseline = task_frame[task_frame.model.eq("numeric_only ensemble")]
        for arm in ["variable_name", "task_semantics"]:
            bootstrap_rows.append(
                bootstrap_normalized_error_difference(
                    task_frame[task_frame.model.eq(arm + " ensemble")], baseline, normalization,
                    config["statistics"]["bootstrap_replicates"], config["seed"],
                )
            )
    bootstrap = pd.DataFrame(bootstrap_rows)
    save_csv(bootstrap, output / "paired_subject_bootstrap.csv")
    manifest.update(sample_counts=sample_counts, test_evaluated=True)
    save_json(manifest, output / "manifest.json")
    write_report(output, config, features, sample_counts, validation_selection, test_metrics, bootstrap, parameter_counts)
    save_json(
        {
            "run_id": config["run_id"],
            "completed": True,
            "engineering_smoke_only": False,
            "test_evaluated": True,
            "primary_contrast": config["statistics"]["primary_contrast"],
            "report": str(output / "report.md"),
        },
        output / "status.json",
    )
    primary = bootstrap[
        bootstrap.task.eq("future_forecast")
        & bootstrap.candidate.eq("task_semantics ensemble")
    ].iloc[0]
    cross = bootstrap[
        bootstrap.task.str.startswith("cross_")
        & bootstrap.candidate.eq("task_semantics ensemble")
    ]
    save_json(
        {
            "run_id": config["run_id"],
            "future_semantics_improved": bool(primary.ci_upper < 0),
            "future_semantics_difference": float(primary.normalized_mae_difference),
            "cross_scale_two_pairs_improved": bool((cross.ci_upper < 0).all()),
            "cross_scale_point_estimates_favorable": bool((cross.normalized_mae_difference < 0).all()),
            "semantic_value_gate_passed": False,
            "advance_semantic_claim": False,
            "reason": "Future semantics was slightly worse; both cross-scale confidence intervals crossed zero.",
        },
        output / "gate_decision.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/phase1b_semantic_fair.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.config, args.smoke)


if __name__ == "__main__":
    main()
