"""Validation-only multimodal cognitive-item to MRI-spatial grounding."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import rankdata
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from ..data.common import ROOT, save_csv, save_json
from ..data.dictionary import load_dictionary
from .phase1_numeric import markdown_table, seed_everything
from .phase1b_semantic import select_verified_scalar_features
from .phase3b_mri_representation import build_concurrent_frame, make_shuffle_indices
from .phase3c_spatial_grounding import coordinate_grid


ITEM_DATES = {
    "ADAS": "adas_visit_date",
    "CDR": "cdr_visit_date",
    "MMSE": "mmse_visit_date",
    "MOCA": "moca_visit_date",
}


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16) % (2**32)


def prepare_frame(config: dict[str, Any], features: list[str], dictionary: dict[str, Any]) -> pd.DataFrame:
    frame = build_concurrent_frame(config)
    pieces = []
    for split in ["train", "val"]:
        raw = pd.read_csv(ROOT / f"splits/{split}.csv", low_memory=False)
        raw = raw[["subject_id", "visit_month"] + features]
        current = frame[frame.split.eq(split)].merge(
            raw,
            left_on=["subject_id", "cutoff_month"],
            right_on=["subject_id", "visit_month"],
            how="left",
            validate="one_to_one",
        )
        pieces.append(current.drop(columns="visit_month", errors="ignore"))
    frame = pd.concat(pieces, ignore_index=True).sort_values(
        ["split", "subject_id", "cutoff_month"]
    ).reset_index(drop=True)
    source = ROOT / "outputs" / config["backbone"]["source_run"] / "sample_manifest.csv"
    reference = pd.read_csv(source)
    if frame.scan_id.astype(str).tolist() != reference.scan_id.astype(str).tolist():
        raise RuntimeError("Phase 3D MRI anchors do not match the frozen Phase 3C token order")
    scan_date = pd.to_datetime(frame.scan_date, errors="coerce")
    for feature in features:
        instrument = str(dictionary[feature]["instrument"]).upper()
        date_column = ITEM_DATES[instrument]
        item_date = pd.to_datetime(frame[date_column], errors="coerce")
        eligible = item_date.notna() & scan_date.notna() & (
            (item_date - scan_date).dt.days.abs() <= int(config["date_window_days"])
        )
        frame.loc[~eligible, feature] = np.nan
        frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
    return frame


def fit_arrays(frame: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = frame[features].to_numpy(dtype=np.float32)
    observed = np.isfinite(raw)
    train = frame.split.eq("train").to_numpy()
    mean = np.nanmean(np.where(observed[train], raw[train], np.nan), axis=0).astype(np.float32)
    std = np.nanstd(np.where(observed[train], raw[train], np.nan), axis=0).astype(np.float32)
    mean[~np.isfinite(mean)] = 0
    std[~np.isfinite(std) | (std < 1e-6)] = 1
    values = np.where(observed, (raw - mean) / std, 0).astype(np.float32)
    age = pd.to_numeric(frame.age, errors="coerce").to_numpy(dtype=np.float32)
    age_mean, age_std = float(age[train].mean()), float(age[train].std())
    if age_std < 1e-6:
        age_std = 1.0
    demo = np.stack([(age - age_mean) / age_std, frame.sex_numeric.to_numpy(dtype=np.float32)], 1)
    return values, observed, mean, std, demo.astype(np.float32)


def load_descriptors(config: dict[str, Any], features: list[str]) -> tuple[torch.Tensor, Path]:
    source = ROOT / "outputs" / config["items"]["source_descriptor_run"] / "descriptor_cache"
    candidates = sorted(source.glob("task_semantics_*.pt"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one frozen task-semantics descriptor cache, found {candidates}")
    payload = torch.load(candidates[0], map_location="cpu", weights_only=True)
    if list(payload["features"]) != features:
        raise RuntimeError("Frozen descriptor feature order differs from verified item order")
    return payload["embeddings"].float(), candidates[0]


def build_samples(
    frame: pd.DataFrame,
    observed: np.ndarray,
    features: list[str],
    dictionary: dict[str, Any],
    config: dict[str, Any],
) -> pd.DataFrame:
    instrument_indices: dict[str, np.ndarray] = {}
    for instrument in ITEM_DATES:
        instrument_indices[instrument] = np.asarray(
            [i for i, feature in enumerate(features) if str(dictionary[feature]["instrument"]).upper() == instrument],
            dtype=np.int64,
        )
    records = []
    n_items = int(config["items"]["masked_items_per_visit"])
    n_instruments = int(config["items"]["masked_instruments_per_visit"])
    for row in range(len(frame)):
        present = np.flatnonzero(observed[row])
        if len(present) < 2:
            continue
        rng = np.random.default_rng(stable_seed(str(frame.anchor_key.iloc[row])))
        selected = rng.choice(present, size=min(n_items, len(present)), replace=False)
        for position, item in enumerate(selected):
            records.append(
                {"sample_id": f"{frame.anchor_key.iloc[row]}|item|{position}|{features[item]}", "row": row,
                 "task": "masked_item", "instrument": str(dictionary[features[item]]["instrument"]).upper(),
                 "targets": str(int(item))}
            )
        available = [
            instrument for instrument, indices in instrument_indices.items()
            if int(observed[row, indices].sum()) >= 2
        ]
        if available:
            order = rng.permutation(len(available))[: min(n_instruments, len(available))]
            for position in order:
                instrument = available[int(position)]
                targets = instrument_indices[instrument][observed[row, instrument_indices[instrument]]]
                records.append(
                    {"sample_id": f"{frame.anchor_key.iloc[row]}|instrument|{instrument}", "row": row,
                     "task": "masked_instrument", "instrument": instrument,
                     "targets": ",".join(map(str, targets.tolist()))}
                )
    samples = pd.DataFrame(records)
    samples["split"] = samples.row.map(frame.split)
    samples["subject_id"] = samples.row.map(frame.subject_id)
    samples["scan_id"] = samples.row.map(frame.scan_id)
    return samples


class GroundingDataset(Dataset):
    def __init__(
        self,
        samples: pd.DataFrame,
        values: np.ndarray,
        observed: np.ndarray,
        demo: np.ndarray,
        feature_path: Path,
        source_rows: np.ndarray | None = None,
    ):
        self.samples = samples.reset_index(drop=True)
        self.values = values
        self.observed = observed
        self.demo = demo
        self.feature_path = feature_path
        self.source_rows = source_rows
        self._features = None

    @property
    def features(self):
        if self._features is None:
            self._features = np.load(self.feature_path, mmap_mode="r")
        return self._features

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples.iloc[index]
        row = int(sample.row)
        source = int(self.source_rows[row]) if self.source_rows is not None else row
        output_mask = np.zeros(self.values.shape[1], dtype=bool)
        output_mask[np.fromstring(sample.targets, dtype=np.int64, sep=",")] = True
        input_mask = self.observed[row] & ~output_mask
        input_value = np.where(input_mask, self.values[row], 0).astype(np.float32)
        return {
            "sample_index": index,
            "row": row,
            "source": source,
            "value": torch.from_numpy(input_value),
            "target": torch.from_numpy(self.values[row]),
            "input_mask": torch.from_numpy(input_mask),
            "output_mask": torch.from_numpy(output_mask),
            "demo": torch.from_numpy(self.demo[row]),
            "tokens": torch.from_numpy(np.asarray(self.features[source], dtype=np.float32)),
        }


class MultimodalItemGrounder(nn.Module):
    """Matched cognitive, global-MRI, and item-conditioned spatial models."""

    def __init__(
        self,
        arm: str,
        descriptors: torch.Tensor,
        mri_dim: int,
        hidden: int,
        layers: int,
        heads: int,
        dropout: float,
        coordinates: np.ndarray,
    ):
        super().__init__()
        self.arm = arm
        self.register_buffer("descriptors", descriptors)
        self.register_buffer("coordinates", torch.from_numpy(coordinates.astype(np.float32)))
        self.semantic = nn.Sequential(nn.Linear(descriptors.shape[1], hidden, bias=False), nn.LayerNorm(hidden))
        self.response = nn.Sequential(nn.Linear(3, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.demo = nn.Sequential(nn.Linear(2, hidden), nn.GELU(), nn.LayerNorm(hidden))
        encoder_layer = nn.TransformerEncoderLayer(
            hidden, heads, hidden * 2, dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.cognitive_encoder = nn.TransformerEncoder(encoder_layer, layers, enable_nested_tensor=False)
        self.mri_token = nn.Sequential(nn.Linear(mri_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.position = nn.Linear(3, hidden, bias=False)
        self.cross_attention = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.fusion = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.decoder = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(
        self,
        value: torch.Tensor,
        input_mask: torch.Tensor,
        output_mask: torch.Tensor,
        demo: torch.Tensor,
        tokens: torch.Tensor,
        spatial_permutation: torch.Tensor | None = None,
        query_permutation: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        semantic = self.semantic(self.descriptors)
        if query_permutation is not None:
            semantic = semantic[query_permutation]
        query = semantic[None] + self.response(
            torch.stack([value, input_mask.float(), output_mask.float()], dim=-1)
        ) + self.demo(demo)[:, None]
        cognitive = self.cognitive_encoder(query)
        if spatial_permutation is not None:
            tokens = tokens[:, spatial_permutation]
        mri = self.mri_token(tokens)
        keys = mri + self.position(self.coordinates)[None]
        weights = None
        if self.arm == "cognition_only":
            context = torch.zeros_like(cognitive)
        elif self.arm == "global_mri":
            context = mri.mean(1)[:, None].expand_as(cognitive)
        elif self.arm == "item_spatial_attention":
            context, weights = self.cross_attention(
                cognitive, keys, mri, need_weights=need_weights, average_attn_weights=True
            )
        else:
            raise ValueError(self.arm)
        fused = self.fusion(torch.cat([cognitive, context], dim=-1)) + cognitive
        return self.decoder(fused).squeeze(-1), weights


def macro_huber(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    losses = []
    raw = F.smooth_l1_loss(prediction, target, reduction="none")
    for item in range(mask.shape[1]):
        if mask[:, item].any():
            losses.append(raw[:, item][mask[:, item]].mean())
    if not losses:
        raise RuntimeError("Batch contains no reconstruction targets")
    return torch.stack(losses).mean()


def evaluate(
    model: MultimodalItemGrounder,
    dataset: GroundingDataset,
    batch_size: int,
    device: torch.device,
    frame: pd.DataFrame,
    features: list[str],
    mean: np.ndarray,
    std: np.ndarray,
    arm: str,
    seed: int,
    pairing: str,
    spatial_permutation: np.ndarray | None = None,
    query_permutation: np.ndarray | None = None,
    return_attention: bool = False,
) -> tuple[pd.DataFrame, np.ndarray | None, list[str]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    spatial = None if spatial_permutation is None else torch.from_numpy(spatial_permutation).to(device)
    query = None if query_permutation is None else torch.from_numpy(query_permutation).to(device)
    rows, attention, attention_keys = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            prediction, weights = model(
                batch["value"].to(device), batch["input_mask"].to(device),
                batch["output_mask"].to(device), batch["demo"].to(device),
                batch["tokens"].to(device), spatial, query, return_attention,
            )
            prediction = prediction.cpu().numpy()
            output_mask = batch["output_mask"].numpy()
            sample_indices = batch["sample_index"].numpy()
            source_rows = batch["source"].numpy()
            for local, sample_index in enumerate(sample_indices):
                sample = dataset.samples.iloc[int(sample_index)]
                row_index = int(sample.row)
                for item in np.flatnonzero(output_mask[local]):
                    truth_normalized = float(dataset.values[row_index, item])
                    rows.append(
                        {
                            "sample_id": sample.sample_id,
                            "scan_id": sample.scan_id,
                            "source_scan_id": frame.scan_id.iloc[int(source_rows[local])],
                            "subject_id": sample.subject_id,
                            "split": sample.split,
                            "model": arm,
                            "seed": seed,
                            "pairing": pairing,
                            "task": sample.task,
                            "instrument": sample.instrument,
                            "target": features[item],
                            "truth": truth_normalized * float(std[item]) + float(mean[item]),
                            "prediction": float(prediction[local, item]) * float(std[item]) + float(mean[item]),
                            "normalized_truth": truth_normalized,
                            "normalized_prediction": float(prediction[local, item]),
                        }
                    )
                    if weights is not None:
                        attention.append(weights[local, item].cpu().numpy())
                        attention_keys.append(f"{sample.sample_id}|{features[item]}")
    return pd.DataFrame(rows), (np.stack(attention) if attention else None), attention_keys


def validation_score(predictions: pd.DataFrame) -> float:
    task_scores = []
    for _, task in predictions.groupby("task"):
        target_mae = task.assign(
            error=(task.normalized_prediction - task.normalized_truth).abs()
        ).groupby("target").error.mean()
        task_scores.append(float(target_mae.mean()))
    return float(np.mean(task_scores))


def train_one(
    arm: str,
    seed: int,
    feature_path: Path,
    samples: pd.DataFrame,
    frame: pd.DataFrame,
    values: np.ndarray,
    observed: np.ndarray,
    demo: np.ndarray,
    descriptors: torch.Tensor,
    features: list[str],
    mean: np.ndarray,
    std: np.ndarray,
    val_shuffle: np.ndarray,
    config: dict[str, Any],
    output: Path,
    device: torch.device,
) -> tuple[pd.DataFrame, np.ndarray | None, list[str], int]:
    seed_everything(seed)
    train_samples = samples[samples.split.eq("train")].reset_index(drop=True)
    val_samples = samples[samples.split.eq("val")].reset_index(drop=True)
    train_data = GroundingDataset(train_samples, values, observed, demo, feature_path)
    val_data = GroundingDataset(val_samples, values, observed, demo, feature_path)
    shuffled_data = GroundingDataset(val_samples, values, observed, demo, feature_path, val_shuffle)
    settings = config["training"]
    loader = DataLoader(
        train_data, batch_size=int(settings["batch_size"]), shuffle=True,
        generator=torch.Generator().manual_seed(seed), num_workers=0,
    )
    model = MultimodalItemGrounder(
        arm, descriptors, int(config["backbone"]["feature_channels"]),
        int(settings["hidden_dim"]), int(settings["self_attention_layers"]),
        int(settings["attention_heads"]), float(settings["dropout"]),
        coordinate_grid(tuple(config["backbone"]["feature_shape"])),
    ).to(device)
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    directory = output / "checkpoints" / arm / str(seed)
    checkpoint_path = directory / "best.pt"
    if checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if payload["arm"] != arm or int(payload["seed"]) != seed or payload["features"] != features:
            raise RuntimeError(f"Incompatible resume checkpoint: {checkpoint_path}")
        model.load_state_dict(payload["model"])
        print(json.dumps({"arm": arm, "seed": seed, "resumed_checkpoint": str(checkpoint_path)}), flush=True)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"])
        )
        best_score, best_state, patience, curves = float("inf"), None, 0, []
        for epoch in range(int(settings["max_epochs"])):
            model.train()
            losses = []
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                prediction, _ = model(
                    batch["value"].to(device), batch["input_mask"].to(device),
                    batch["output_mask"].to(device), batch["demo"].to(device), batch["tokens"].to(device)
                )
                loss = macro_huber(prediction, batch["target"].to(device), batch["output_mask"].to(device))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss for {arm}/{seed}")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), float(settings["gradient_clip"]))
                optimizer.step()
                losses.append(float(loss.detach()))
            val_prediction, _, _ = evaluate(
                model, val_data, int(settings["batch_size"]), device, frame, features, mean, std,
                arm, seed, "observed"
            )
            score = validation_score(val_prediction)
            curve = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_macro_normalized_mae": score}
            curves.append(curve)
            print(json.dumps({"arm": arm, "seed": seed, **curve}), flush=True)
            if score < best_score - 1e-5:
                best_score, best_state, patience = score, copy.deepcopy(model.state_dict()), 0
            else:
                patience += 1
                if patience >= int(settings["patience"]):
                    break
        if best_state is None:
            raise RuntimeError("No checkpoint selected")
        model.load_state_dict(best_state)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model": best_state, "arm": arm, "seed": seed, "trainable_parameters": count,
             "best_score": best_score, "features": features}, checkpoint_path
        )
        save_csv(pd.DataFrame(curves), directory / "curves.csv")
    observed_prediction, weights, keys = evaluate(
        model, val_data, int(settings["batch_size"]), device, frame, features, mean, std,
        arm, seed, "observed", return_attention=arm == "item_spatial_attention"
    )
    predictions = [observed_prediction]
    if arm in {"global_mri", "item_spatial_attention"}:
        shuffled_prediction, _, _ = evaluate(
            model, shuffled_data, int(settings["batch_size"]), device, frame, features, mean, std,
            arm, seed, "shuffled_subject"
        )
        predictions.append(shuffled_prediction)
    if arm == "item_spatial_attention":
        spatial = np.roll(np.arange(int(np.prod(config["backbone"]["feature_shape"]))), 1).astype(np.int64)
        query = np.roll(np.arange(len(features)), 1).astype(np.int64)
        spatial_prediction, _, _ = evaluate(
            model, val_data, int(settings["batch_size"]), device, frame, features, mean, std,
            arm, seed, "spatial_permutation", spatial_permutation=spatial
        )
        query_prediction, _, _ = evaluate(
            model, val_data, int(settings["batch_size"]), device, frame, features, mean, std,
            arm, seed, "query_permutation", query_permutation=query
        )
        predictions.extend([spatial_prediction, query_prediction])
    return pd.concat(predictions, ignore_index=True), weights, keys, count


def ensemble(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "sample_id", "scan_id", "source_scan_id", "subject_id", "split", "model", "pairing",
        "task", "instrument", "target", "truth", "normalized_truth",
    ]
    result = frame.groupby(keys, as_index=False)[["prediction", "normalized_prediction"]].mean()
    result["seed"] = pd.NA
    return result


def metric_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(["model", "pairing", "task", "target"], sort=True):
        normalized_error = group.normalized_prediction - group.normalized_truth
        raw_error = group.prediction - group.truth
        rows.append(
            {"model": keys[0], "pairing": keys[1], "task": keys[2], "target": keys[3],
             "n": len(group), "subjects": group.subject_id.nunique(),
             "mae": float(raw_error.abs().mean()), "normalized_mae": float(normalized_error.abs().mean()),
             "rmse": float(np.sqrt(np.mean(raw_error**2))),
             "spearman": float(group.prediction.corr(group.truth, method="spearman"))}
        )
    return pd.DataFrame(rows)


def bootstrap_difference(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    replicates: int,
    seed: int,
    task: str | None = None,
) -> dict[str, Any]:
    if task is not None:
        candidate, baseline = candidate[candidate.task.eq(task)], baseline[baseline.task.eq(task)]
    keys = ["sample_id", "scan_id", "subject_id", "task", "target", "normalized_truth"]
    paired = candidate[keys + ["normalized_prediction"]].merge(
        baseline[keys + ["normalized_prediction"]], on=keys,
        suffixes=("_candidate", "_baseline"), validate="one_to_one"
    )
    if paired.empty:
        raise RuntimeError("Bootstrap comparison has no paired predictions")
    paired["difference"] = (
        (paired.normalized_prediction_candidate - paired.normalized_truth).abs()
        - (paired.normalized_prediction_baseline - paired.normalized_truth).abs()
    )
    subjects = paired.groupby("subject_id").difference.mean().to_numpy()
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(subjects), size=(replicates, len(subjects)))
    distribution = subjects[indices].mean(1)
    return {
        "candidate": str(candidate.model.iloc[0]), "candidate_pairing": str(candidate.pairing.iloc[0]),
        "baseline": str(baseline.model.iloc[0]), "baseline_pairing": str(baseline.pairing.iloc[0]),
        "task": task or "overall", "subjects": len(subjects), "targets": len(paired),
        "difference": float(subjects.mean()), "ci_lower": float(np.quantile(distribution, 0.025)),
        "ci_upper": float(np.quantile(distribution, 0.975)),
    }


def attention_stability(
    weights: dict[int, np.ndarray], top_fraction: float
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    token_count = next(iter(weights.values())).shape[1]
    top_k = max(1, int(np.ceil(token_count * top_fraction)))
    for left, right in combinations(sorted(weights), 2):
        for sample in range(len(weights[left])):
            a, b = weights[left][sample], weights[right][sample]
            correlation = float(np.corrcoef(rankdata(a), rankdata(b))[0, 1]) if a.std() and b.std() else 0.0
            top_a = set(np.argpartition(a, -top_k)[-top_k:])
            top_b = set(np.argpartition(b, -top_k)[-top_k:])
            rows.append({"seed_left": left, "seed_right": right, "attention_key_index": sample,
                         "spearman": correlation, "top_token_overlap": len(top_a & top_b) / top_k})
    table = pd.DataFrame(rows)
    stacked = np.stack([weights[seed] for seed in sorted(weights)])
    entropy = -(stacked * np.log(stacked.clip(1e-12))).sum(-1) / np.log(token_count)
    summary = {"median_spearman": float(table.spearman.median()),
               "median_top_token_overlap": float(table.top_token_overlap.median()),
               "mean_normalized_entropy": float(entropy.mean())}
    return table, summary


def main(config_path: Path, smoke: bool = False) -> None:
    config = load_config(config_path)
    if smoke:
        config = copy.deepcopy(config)
        config["run_id"] += "_smoke"
        config["seeds"] = config["seeds"][:1]
        config["training"]["max_epochs"] = 2
        config["training"]["patience"] = 2
        config["statistics"]["bootstrap_replicates"] = 100
    output = ROOT / "outputs" / config["run_id"]
    output.mkdir(parents=True, exist_ok=True)
    item_config = load_config(ROOT / "configs" / config["items"]["source_config"])
    dictionary = load_dictionary()
    features = select_verified_scalar_features(item_config, dictionary)
    descriptors, descriptor_source = load_descriptors(config, features)
    torch.save({"features": features, "embeddings": descriptors, "source": str(descriptor_source)}, output / "frozen_item_descriptors.pt")
    frame = prepare_frame(config, features, dictionary)
    values, observed, mean, std, demo = fit_arrays(frame, features)
    samples = build_samples(frame, observed, features, dictionary, config)
    if smoke:
        samples = pd.concat(
            [samples[samples.split.eq("train")].head(768), samples[samples.split.eq("val")].head(256)],
            ignore_index=True,
        )
    save_csv(samples, output / "sample_manifest.csv")
    save_csv(frame[["anchor_key", "scan_id", "subject_id", "split", "cutoff_month", "scan_date"]], output / "anchor_manifest.csv")
    save_json({"features": features, "mean": dict(zip(features, map(float, mean))),
               "std": dict(zip(features, map(float, std)))}, output / "normalization.json")
    source_run = ROOT / "outputs" / config["backbone"]["source_run"]
    feature_paths = {int(seed): source_run / f"spatial_features_seed{seed}.npy" for seed in config["seeds"]}
    expected = (len(frame), int(np.prod(config["backbone"]["feature_shape"])), int(config["backbone"]["feature_channels"]))
    for path in feature_paths.values():
        if tuple(np.load(path, mmap_mode="r").shape) != expected:
            raise RuntimeError(f"Unexpected frozen spatial-token shape at {path}")
    val_rows = np.flatnonzero(frame.split.eq("val").to_numpy())
    local_shuffle = make_shuffle_indices(frame.iloc[val_rows].reset_index(drop=True), int(config["statistics"]["bootstrap_seed"]))
    val_shuffle = np.arange(len(frame))
    val_shuffle[val_rows] = val_rows[local_shuffle]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_predictions, weights, attention_keys, parameter_counts = [], {}, None, {}
    for arm in config["arms"]:
        for seed in config["seeds"]:
            prediction, attention, keys, count = train_one(
                arm, int(seed), feature_paths[int(seed)], samples, frame, values, observed, demo,
                descriptors, features, mean, std, val_shuffle, config, output, device
            )
            all_predictions.append(prediction)
            parameter_counts[arm] = count
            if attention is not None:
                weights[int(seed)] = attention
                if attention_keys is None:
                    attention_keys = keys
                elif attention_keys != keys:
                    raise RuntimeError("Attention target order differs across seeds")
    seed_predictions = pd.concat(all_predictions, ignore_index=True)
    ensembles = ensemble(seed_predictions)
    save_csv(pd.concat([seed_predictions, ensembles], ignore_index=True), output / "validation_predictions.csv")
    metrics = metric_table(ensembles)
    save_csv(metrics, output / "validation_metrics.csv")
    candidate = ensembles[ensembles.model.eq("item_spatial_attention") & ensembles.pairing.eq("observed")]
    baselines = [
        ensembles[ensembles.model.eq("cognition_only") & ensembles.pairing.eq("observed")],
        ensembles[ensembles.model.eq("global_mri") & ensembles.pairing.eq("observed")],
        ensembles[ensembles.model.eq("item_spatial_attention") & ensembles.pairing.eq("shuffled_subject")],
        ensembles[ensembles.model.eq("item_spatial_attention") & ensembles.pairing.eq("spatial_permutation")],
        ensembles[ensembles.model.eq("item_spatial_attention") & ensembles.pairing.eq("query_permutation")],
    ]
    comparisons = []
    for baseline in baselines:
        for task in [None, "masked_item", "masked_instrument"]:
            comparisons.append(bootstrap_difference(
                candidate, baseline, int(config["statistics"]["bootstrap_replicates"]),
                int(config["statistics"]["bootstrap_seed"]) + len(comparisons), task
            ))
    bootstrap = pd.DataFrame(comparisons)
    save_csv(bootstrap, output / "bootstrap_comparisons.csv")
    if len(weights) >= 2:
        stability, stability_summary = attention_stability(weights, float(config["statistics"]["attention_top_fraction"]))
    else:
        stability = pd.DataFrame()
        stability_summary = {"median_spearman": None, "median_top_token_overlap": None, "mean_normalized_entropy": None}
    save_csv(stability, output / "attention_stability.csv")
    np.savez_compressed(output / "attention_weights.npz", seeds=np.asarray(sorted(weights)),
                        keys=np.asarray(attention_keys or []),
                        weights=np.stack([weights[seed] for seed in sorted(weights)]) if weights else np.empty(0))
    indexed = bootstrap.set_index(["baseline", "baseline_pairing", "task"])
    cognition_tasks = [indexed.loc[("cognition_only", "observed", task), "ci_upper"] < 0 for task in ["masked_item", "masked_instrument"]]
    control_keys = [("item_spatial_attention", pairing, "overall") for pairing in ["shuffled_subject", "spatial_permutation", "query_permutation"]]
    criteria = {
        "both_tasks_better_than_cognition": bool(all(cognition_tasks)),
        "overall_better_than_global_mri": bool(indexed.loc[("global_mri", "observed", "overall"), "ci_upper"] < 0),
        "better_than_all_negative_controls": bool(all(indexed.loc[key, "ci_upper"] < 0 for key in control_keys)),
        "attention_spearman_stable": bool(stability_summary["median_spearman"] is not None and stability_summary["median_spearman"] >= float(config["gate"]["minimum_attention_spearman"])),
        "top_token_overlap_stable": bool(stability_summary["median_top_token_overlap"] is not None and stability_summary["median_top_token_overlap"] >= float(config["gate"]["minimum_top_token_overlap"])),
        "attention_noncollapsed": bool(stability_summary["mean_normalized_entropy"] is not None and stability_summary["mean_normalized_entropy"] < float(config["gate"]["maximum_normalized_attention_entropy"])),
    }
    passed = bool(all(criteria.values())) and not smoke
    decision = {"run_id": config["run_id"], "engineering_smoke_only": smoke, "development_only": True,
                "test_evaluated": False, "criteria": criteria, "attention_stability": stability_summary,
                "multimodal_item_grounding_passed": passed,
                "next_action": "freeze_multimodal_concurrent_encoder_for_downstream_reasoning" if passed else "retain_phase3c_total_score_grounding_only"}
    save_json(decision, output / "development_decision.json")
    summary = metrics.groupby(["model", "pairing", "task"], as_index=False).normalized_mae.mean().sort_values("normalized_mae")
    report = f"""# Phase 3D multimodal concurrent item grounding\n\nStatus: completed validation-only experiment. Test was not loaded.\n\n## Contract\n\n- MRI anchors: {int(frame.split.eq('train').sum())} train / {int(frame.split.eq('val').sum())} validation\n- Reconstruction samples: {int(samples.split.eq('train').sum())} train / {int(samples.split.eq('val').sum())} validation\n- Verified cognitive items: {len(features)}\n- Frozen MRI tokens: {config['backbone']['feature_shape']} × {config['backbone']['feature_channels']} channels\n- Seeds: {config['seeds']}\n- Trainable parameters: {parameter_counts}\n\n## Validation task summary\n\n{markdown_table(summary)}\n\n## Paired subject-cluster bootstrap\n\nNegative differences favor observed item-conditioned spatial attention.\n\n{markdown_table(bootstrap)}\n\n## Attention stability and concentration\n\n```json\n{json.dumps(stability_summary, indent=2)}\n```\n\n## Decision\n\n```json\n{json.dumps(decision, indent=2)}\n```\n\nThe attention maps are predictive grounding weights over the aligned SFCN grid. They are not causal anatomical evidence and no atlas-region label is assigned.\n"""
    (output / "report.md").write_text(report)
    save_json({"run_id": config["run_id"], "device": str(device), "test_evaluated": False,
               "anchors": frame.split.value_counts().to_dict(), "samples": samples.split.value_counts().to_dict(),
               "subjects": frame.groupby("split").subject_id.nunique().to_dict(),
               "parameter_counts": parameter_counts, "descriptor_source": str(descriptor_source),
               "spatial_feature_sources": [str(path) for path in feature_paths.values()]}, output / "run_manifest.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/phase3d_multimodal_item_grounding_dev.yaml")
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    main(arguments.config, arguments.smoke)
