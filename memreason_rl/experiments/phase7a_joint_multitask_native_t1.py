"""End-to-end native-T1 CN/MCI/AD and Memory multitask experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from ..data.common import ROOT
from .phase1_numeric import markdown_table, seed_everything
from .phase3b_mri_representation import SFCNEncoder
from .phase3g_mri_diagnosis import (
    arm_metrics,
    bootstrap_contrast,
    bootstrap_metric_intervals,
    shuffled_indices,
    subject_weights,
)
from .phase3i_native_resolution_adcn import _cache_native_volume
from .phase6a_latent_state_dynamics import reconstruct_state_records
from .phase6d0_native_embedding_extract import scan_frame


CONFIG = ROOT / "configs/phase7a_joint_multitask_native_t1.yaml"
PROTOCOL = ROOT / "reports/experiments/phase7a_joint_multitask_native_t1_protocol.md"


def load_config(path: Path = CONFIG) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def output_dir(config: dict[str, Any]) -> Path:
    path = ROOT / "outputs" / config["run_id"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(value: Any, path: Path) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def normalize_diagnostic_probabilities(frame: pd.DataFrame) -> pd.DataFrame:
    """Renormalize serialized probabilities after mixed-precision rounding."""
    columns = ["prob_CN", "prob_MCI", "prob_AD"]
    available = frame[columns].notna().all(axis=1)
    values = np.clip(frame.loc[available, columns].to_numpy(float), 0.0, 1.0)
    totals = values.sum(axis=1, keepdims=True)
    if (totals <= 0).any():
        raise RuntimeError("Diagnostic probabilities have a nonpositive row sum")
    frame.loc[available, columns] = values / totals
    return frame


def native_cache_path(config: dict[str, Any], output: Path) -> Path:
    settings = config["native_volume"]
    cache_root = Path(
        os.environ.get("MEMREASON_PHASE7A_CACHE_ROOT", settings.get("cache_root", output))
    ).expanduser()
    return cache_root / settings["cache_filename"]


def _nearest_future(
    months: np.ndarray,
    values: np.ndarray,
    cutoff: float,
    horizon: int,
    tolerance: float,
) -> tuple[float, float]:
    eligible = np.flatnonzero((months > cutoff) & np.isfinite(values))
    if not len(eligible):
        return math.nan, math.nan
    desired = cutoff + horizon
    index = int(eligible[np.argmin(np.abs(months[eligible] - desired))])
    if abs(float(months[index]) - desired) > tolerance:
        return math.nan, math.nan
    return float(values[index]), float(months[index])


def memory_history(months: np.ndarray, values: np.ndarray, cutoff: float) -> tuple[np.ndarray, float]:
    """Return cutoff-safe Memory summaries and the latest month consulted."""
    valid = (months <= cutoff + 1e-8) & np.isfinite(values)
    observed_months = months[valid]
    observed_values = values[valid]
    if not len(observed_values):
        return np.asarray([math.nan, math.nan, math.nan, 0.0, math.nan], np.float32), math.nan
    years = observed_months / 12.0
    slope = (
        float(np.polyfit(years, observed_values, 1)[0])
        if len(np.unique(years)) >= 2 else 0.0
    )
    features = np.asarray([
        observed_values[-1], observed_values.mean(), slope,
        np.log1p(len(observed_values)), max((cutoff - observed_months[-1]) / 12.0, 0.0),
    ], np.float32)
    return features, float(observed_months[-1])


def build_cohort(config: dict[str, Any], output: Path) -> pd.DataFrame:
    d0_config = load_config(ROOT / config["phase6d0_config"])
    scans = scan_frame(d0_config)
    scans["subject_id"] = scans.subject_id.astype(str)
    scans["visit_month"] = pd.to_numeric(scans.visit_month, errors="raise")
    contract = yaml.safe_load((ROOT / config["item_contract"]).read_text())
    state_config = {
        "phase5g_states": config["phase5g_states"],
        "phase5g_selected_arm": config["phase5g_selected_arm"],
    }
    records = reconstruct_state_records(state_config, contract["primary_state_items"])
    records["subject_id"] = records.subject_id.astype(str)
    records["visit_month"] = pd.to_numeric(records.visit_month, errors="raise")
    records = (
        records.sort_values(["subject_id", "visit_month", "sample_id"], kind="stable")
        .drop_duplicates(["subject_id", "visit_month"], keep="last")
    )
    current = records[[
        "subject_id", "visit_month", "sample_id", "diagnosis", "memory_composite", "fold"
    ]].rename(columns={"memory_composite": "current_memory"})
    frame = scans.merge(current, on=["subject_id", "visit_month"], how="inner", validate="many_to_one")
    frame["diagnosis"] = frame.diagnosis.astype(str).str.upper()
    frame = frame[
        frame.diagnosis.isin(config["classes"])
        & np.isfinite(pd.to_numeric(frame.current_memory, errors="coerce"))
    ].copy()
    label_map = {name: index for index, name in enumerate(config["classes"])}
    frame["label"] = frame.diagnosis.map(label_map).astype(int)
    frame["fold"] = frame.fold.astype(int)
    horizons = [int(value) for value in config["forecast"]["horizons_months"]]
    tolerances = {int(key): float(value) for key, value in config["forecast"]["tolerance_months"].items()}
    history_rows: dict[int, np.ndarray] = {}
    history_latest: dict[int, float] = {}
    targets: dict[int, dict[int, tuple[float, float]]] = {}
    record_groups = {subject: group.sort_values("visit_month") for subject, group in records.groupby("subject_id")}
    for index, row in frame.iterrows():
        group = record_groups[str(row.subject_id)]
        months = group.visit_month.to_numpy(float)
        values = group.memory_composite.to_numpy(float)
        history_rows[index], history_latest[index] = memory_history(months, values, float(row.visit_month))
        targets[index] = {
            horizon: _nearest_future(
                months, values, float(row.visit_month), horizon, tolerances[horizon]
            ) for horizon in horizons
        }
    history = np.stack([history_rows[index] for index in frame.index])
    for column, values in zip(config["forecast"]["history_features"], history.T):
        frame[f"history_{column}"] = values
    frame["history_latest_month"] = [history_latest[index] for index in frame.index]
    for horizon in horizons:
        frame[f"future_memory_{horizon}"] = [targets[index][horizon][0] for index in frame.index]
        frame[f"future_month_{horizon}"] = [targets[index][horizon][1] for index in frame.index]
    frame = frame.sort_values(["subject_id", "visit_month", "scan_id"], kind="stable").reset_index(drop=True)
    frame["volume_index"] = np.arange(len(frame), dtype=int)
    if frame.scan_id.duplicated().any():
        raise RuntimeError("Phase 7A cohort is not unique by scan")
    if not frame.fold.between(0, int(config["folds"]) - 1).all():
        raise RuntimeError("Invalid outer fold in Phase 7A cohort")
    if (frame.history_latest_month > frame.visit_month + 1e-8).any():
        raise RuntimeError("Future cognition entered a history feature")
    for horizon in horizons:
        available = frame[f"future_memory_{horizon}"].notna()
        if not (frame.loc[available, f"future_month_{horizon}"] > frame.loc[available, "visit_month"]).all():
            raise RuntimeError(f"Non-future target found at horizon {horizon}")
    official_train = set(pd.read_csv(ROOT / "splits/train.csv", usecols=["subject_id"]).subject_id.astype(str))
    if not set(frame.subject_id) <= official_train:
        raise RuntimeError("Non-training participant entered Phase 7A")
    frame.to_csv(output / "cohort_manifest.csv", index=False)
    audit = {
        "scans": len(frame), "subjects": int(frame.subject_id.nunique()),
        "class_visits": frame.diagnosis.value_counts().to_dict(),
        "class_subjects": frame.groupby("diagnosis").subject_id.nunique().to_dict(),
        "future_targets": {str(h): int(frame[f"future_memory_{h}"].notna().sum()) for h in horizons},
        "history_future_leakage_rows": int((frame.history_latest_month > frame.visit_month + 1e-8).sum()),
        "all_subjects_original_train": True, "validation_or_test_loaded": False,
    }
    save_json(audit, output / "cohort_audit.json")
    return frame


def prepare_native_cache(frame: pd.DataFrame, config: dict[str, Any], output: Path) -> Path:
    settings = config["native_volume"]
    shape = tuple(int(value) for value in settings["expected_shape"])
    voxel_size = tuple(float(value) for value in settings["expected_voxel_size_mm"])
    cache_path = native_cache_path(config, output)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        cache = np.load(cache_path, mmap_mode="r")
        if cache.shape == (len(frame),) + shape and cache.dtype == np.float16:
            return cache_path
        raise RuntimeError(f"Existing native cache has the wrong shape or dtype: {cache.shape}/{cache.dtype}")
    temporary = cache_path.with_name(cache_path.stem + ".tmp.npy")
    temporary.unlink(missing_ok=True)
    cache = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float16, shape=(len(frame),) + shape
    )
    cache.flush(); del cache
    tasks = [
        (int(row.volume_index), str(row.image_path), str(row.mask_path), str(temporary), shape, voxel_size)
        for row in frame.itertuples(index=False)
    ]
    failures: list[str] = []
    observed_shapes: dict[str, int] = {}
    observed_zooms: dict[str, int] = {}
    with ProcessPoolExecutor(max_workers=int(settings["extraction_workers"])) as pool:
        futures = [pool.submit(_cache_native_volume, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), 1):
            _, actual_shape, zooms, error = future.result()
            if error is not None:
                failures.append(error)
            else:
                shape_key = "x".join(map(str, actual_shape))
                zoom_key = "x".join(map(str, zooms))
                observed_shapes[shape_key] = observed_shapes.get(shape_key, 0) + 1
                observed_zooms[zoom_key] = observed_zooms.get(zoom_key, 0) + 1
            if completed % 50 == 0 or completed == len(tasks):
                print(json.dumps({"cached": completed, "total": len(tasks)}), flush=True)
    if failures:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Native cache failures ({len(failures)}): {failures[:10]}")
    temporary.replace(cache_path)
    save_json({
        "scans": len(frame), "shape": list(shape), "voxel_size_mm": list(voxel_size),
        "observed_shapes": observed_shapes, "observed_voxel_sizes_mm": observed_zooms,
        "dtype": "float16", "spatial_interpolation": False, "crop_or_pad": False,
        "cache_path": str(cache_path),
    }, output / "native_cache_audit.json")
    return cache_path


class SubjectBalancedSampler(Sampler[int]):
    """Draw one randomly selected scan per participant on every epoch."""

    def __init__(self, frame: pd.DataFrame, indices: np.ndarray, seed: int):
        self.groups = [
            group.index.to_numpy(dtype=int)
            for _, group in frame.iloc[indices].groupby("subject_id", sort=True)
        ]
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self.epoch)
        selected = np.asarray([rng.choice(group) for group in self.groups], dtype=int)
        rng.shuffle(selected)
        self.epoch += 1
        return iter(selected.tolist())

    def __len__(self) -> int:
        return len(self.groups)


class JointVolumeDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        indices: np.ndarray,
        cache_path: Path,
        history_mean: np.ndarray,
        history_std: np.ndarray,
        target_mean: float,
        target_std: float,
        horizons: list[int],
        weights: np.ndarray,
        volume_source: np.ndarray | None = None,
    ):
        self.frame = frame
        self.indices = np.asarray(indices, dtype=int)
        self.cache_path = cache_path
        self.history_mean = np.asarray(history_mean, np.float32)
        self.history_std = np.asarray(history_std, np.float32)
        self.target_mean = float(target_mean)
        self.target_std = float(target_std)
        self.horizons = horizons
        self.weights = np.asarray(weights, np.float32)
        self.volume_source = np.arange(len(frame), dtype=int) if volume_source is None else np.asarray(volume_source, dtype=int)
        self._cache = None

    @property
    def cache(self):
        if self._cache is None:
            self._cache = np.load(self.cache_path, mmap_mode="r")
        return self._cache

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        row_index = int(self.indices[item])
        source_index = int(self.volume_source[row_index])
        volume_index = int(self.frame.iloc[source_index].volume_index)
        row = self.frame.iloc[row_index]
        history_columns = [f"history_{name}" for name in ["last", "mean", "slope_per_year", "log_count", "recency_years"]]
        history = row[history_columns].to_numpy(dtype=np.float32)
        history = np.nan_to_num((history - self.history_mean) / self.history_std)
        future = np.asarray([row[f"future_memory_{h}"] for h in self.horizons], np.float32)
        future_mask = np.isfinite(future)
        future = np.nan_to_num((future - self.target_mean) / self.target_std)
        current = (float(row.current_memory) - self.target_mean) / self.target_std
        volume = np.asarray(self.cache[volume_index], dtype=np.float32)[None]
        return {
            "row": torch.tensor(row_index, dtype=torch.long),
            "volume": torch.from_numpy(volume),
            "history": torch.from_numpy(history),
            "label": torch.tensor(int(row.label), dtype=torch.long),
            "current": torch.tensor(current, dtype=torch.float32),
            "future": torch.from_numpy(future),
            "future_mask": torch.from_numpy(future_mask),
            "weight": torch.tensor(float(self.weights[row_index]), dtype=torch.float32),
        }


class JointSFCNMemoryModel(nn.Module):
    def __init__(self, horizons: int, dropout: float, log_variance_bounds: tuple[float, float]):
        super().__init__()
        self.encoder = SFCNEncoder(dropout)
        self.diagnosis_head = nn.Linear(128, 3)
        self.current_head = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(), nn.LayerNorm(64), nn.Dropout(dropout), nn.Linear(64, 2)
        )
        self.history_encoder = nn.Sequential(
            nn.Linear(5, 32), nn.GELU(), nn.LayerNorm(32), nn.Dropout(dropout)
        )
        self.future_head = nn.Sequential(
            nn.Linear(160, 128), nn.GELU(), nn.LayerNorm(128), nn.Dropout(dropout),
            nn.Linear(128, horizons * 2),
        )
        self.horizons = int(horizons)
        self.log_variance_bounds = tuple(float(value) for value in log_variance_bounds)

    def forward_heads(self, embedding: torch.Tensor, history: torch.Tensor) -> dict[str, torch.Tensor]:
        current = self.current_head(embedding)
        future = self.future_head(torch.cat([embedding, self.history_encoder(history)], dim=1))
        future = future.view(len(embedding), self.horizons, 2)
        low, high = self.log_variance_bounds
        return {
            "embedding": embedding,
            "diagnosis_logits": self.diagnosis_head(embedding),
            "current_mean": current[:, 0],
            "current_logvar": current[:, 1].clamp(low, high),
            "future_mean": future[:, :, 0],
            "future_logvar": future[:, :, 1].clamp(low, high),
        }

    def forward(self, volume: torch.Tensor, history: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.forward_heads(self.encoder(volume), history)


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


def gaussian_nll(mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 0.5 * (math.log(2 * math.pi) + logvar + (target - mean).square() * torch.exp(-logvar))


def multitask_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    arm: str,
    class_weight: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    settings = config["training"]
    sample_weight = batch["weight"]
    ce_values = F.cross_entropy(
        outputs["diagnosis_logits"].float(), batch["label"], weight=class_weight,
        reduction="none", label_smoothing=float(settings["label_smoothing"]),
    )
    diagnosis = weighted_mean(ce_values, sample_weight)
    current_values = gaussian_nll(outputs["current_mean"], outputs["current_logvar"], batch["current"])
    current = weighted_mean(current_values, sample_weight)
    future_values = gaussian_nll(outputs["future_mean"], outputs["future_logvar"], batch["future"])
    mask = batch["future_mask"].float()
    per_sample = (future_values * mask).sum(1) / mask.sum(1).clamp_min(1.0)
    available = mask.sum(1) > 0
    future = (
        weighted_mean(per_sample[available], sample_weight[available])
        if available.any() else outputs["future_mean"].sum() * 0
    )
    if arm == "classification_only":
        total = float(settings["diagnosis_weight"]) * diagnosis
    elif arm == "regression_only":
        total = (float(settings["current_memory_weight"]) * current
                 + float(settings["future_memory_weight"]) * future)
    elif arm == "joint_multitask":
        total = (float(settings["diagnosis_weight"]) * diagnosis
                 + float(settings["current_memory_weight"]) * current
                 + float(settings["future_memory_weight"]) * future)
    else:
        raise ValueError(arm)
    return total, {"diagnosis": diagnosis, "current": current, "future": future}


def partitions(frame: pd.DataFrame, fold: int, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    evaluation_subjects = set(frame.loc[frame.fold.eq(fold), "subject_id"])
    subject_label = (
        frame.loc[frame.fold.ne(fold)].groupby("subject_id").label
        .agg(lambda values: int(values.value_counts().sort_index().idxmax()))
    )
    subjects = subject_label.index.to_numpy()
    split = StratifiedShuffleSplit(
        n_splits=1, test_size=float(config["inner_dev_fraction"]),
        random_state=int(config["partition_seed"]) + fold,
    )
    fit_local, dev_local = next(split.split(subjects, subject_label.to_numpy()))
    fit_subjects, dev_subjects = set(subjects[fit_local]), set(subjects[dev_local])
    fit = np.flatnonzero(frame.subject_id.isin(fit_subjects))
    development = np.flatnonzero(frame.subject_id.isin(dev_subjects))
    evaluation = np.flatnonzero(frame.subject_id.isin(evaluation_subjects))
    sets = [set(frame.iloc[index].subject_id) for index in [fit, development, evaluation]]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise RuntimeError(f"Subject leakage in fold {fold}")
    return fit, development, evaluation


def fit_normalizers(
    frame: pd.DataFrame, fit: np.ndarray, config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
    history_columns = [f"history_{name}" for name in config["forecast"]["history_features"]]
    history = frame.loc[fit, history_columns].to_numpy(np.float32)
    history_mean = np.nanmean(history, axis=0)
    history_std = np.nanstd(history, axis=0)
    history_mean = np.where(np.isfinite(history_mean), history_mean, 0.0)
    history_std = np.where(np.isfinite(history_std) & (history_std > 1e-6), history_std, 1.0)
    target = frame.loc[fit, "current_memory"].to_numpy(float)
    target_mean, target_std = float(np.mean(target)), float(np.std(target))
    if target_std <= 1e-6:
        raise RuntimeError("Degenerate Memory normalization")
    base = subject_weights(frame.iloc[fit])
    labels = frame.iloc[fit].label.to_numpy(int)
    mass = np.asarray([base[labels == label].sum() for label in range(3)])
    class_weight = mass.sum() / (3.0 * mass)
    return history_mean, history_std, target_mean, target_std, class_weight.astype(np.float32)


def load_warm_start(model: JointSFCNMemoryModel, fold: int, config: dict[str, Any]) -> str:
    path = ROOT / config["warm_start_checkpoints"] / f"fold{fold}" / f"seed{int(config['warm_start_seed'])}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload["model"]
    encoder_state = {key.removeprefix("encoder."): value for key, value in state.items() if key.startswith("encoder.")}
    model.encoder.load_state_dict(encoder_state, strict=True)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _amp_dtype(config: dict[str, Any]) -> torch.dtype:
    return torch.bfloat16 if config["training"].get("amp_dtype") == "bfloat16" else torch.float16


@torch.no_grad()
def evaluate_model(
    model: JointSFCNMemoryModel,
    dataset: JointVolumeDataset,
    arm: str,
    class_weight: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[float, pd.DataFrame]:
    settings = config["training"]
    loader = DataLoader(
        dataset, batch_size=int(settings["batch_size"]), shuffle=False,
        num_workers=int(settings["num_workers"]), pin_memory=device.type == "cuda",
        persistent_workers=False,
    )
    model.eval(); losses = []; blocks = []
    amp = bool(settings["amp"]) and device.type == "cuda"
    for batch in loader:
        cpu_rows = batch["row"].numpy()
        moved = {key: value.to(device, non_blocking=True) for key, value in batch.items() if key != "row"}
        with torch.amp.autocast(device_type=device.type, enabled=amp, dtype=_amp_dtype(config)):
            outputs = model(moved["volume"], moved["history"])
            loss, _ = multitask_loss(outputs, moved, arm, class_weight, config)
        losses.append((float(loss), len(cpu_rows)))
        probability = torch.softmax(outputs["diagnosis_logits"].float(), dim=1).cpu().numpy()
        current_mean = outputs["current_mean"].float().cpu().numpy() * dataset.target_std + dataset.target_mean
        current_var = np.exp(outputs["current_logvar"].float().cpu().numpy()) * dataset.target_std ** 2
        future_mean = outputs["future_mean"].float().cpu().numpy() * dataset.target_std + dataset.target_mean
        future_var = np.exp(outputs["future_logvar"].float().cpu().numpy()) * dataset.target_std ** 2
        block = pd.DataFrame({"row_index": cpu_rows})
        for index, name in enumerate(["CN", "MCI", "AD"]):
            block[f"prob_{name}"] = probability[:, index]
        block["prediction_current"] = current_mean
        block["variance_current"] = current_var
        for index, horizon in enumerate(dataset.horizons):
            block[f"prediction_{horizon}"] = future_mean[:, index]
            block[f"variance_{horizon}"] = future_var[:, index]
        blocks.append(block)
    mean_loss = sum(value * count for value, count in losses) / sum(count for _, count in losses)
    return mean_loss, pd.concat(blocks, ignore_index=True)


def attach_truth(
    predictions: pd.DataFrame, frame: pd.DataFrame, arm: str, fold: int, config: dict[str, Any]
) -> pd.DataFrame:
    truth_columns = ["scan_id", "subject_id", "visit_month", "fold", "label", "diagnosis", "current_memory"]
    for horizon in config["forecast"]["horizons_months"]:
        truth_columns.extend([f"future_memory_{horizon}", f"future_month_{horizon}"])
    truth = frame[truth_columns].copy(); truth["row_index"] = frame.index
    result = predictions.merge(truth, on="row_index", how="left", validate="one_to_one", suffixes=("", "_truth"))
    result["fold"] = fold; result["arm"] = arm
    if arm == "classification_only":
        for column in [name for name in result if name.startswith("prediction_") or name.startswith("variance_")]:
            result[column] = np.nan
    if arm == "regression_only":
        result[["prob_CN", "prob_MCI", "prob_AD"]] = np.nan
    return result


def train_arm(
    frame: pd.DataFrame,
    cache_path: Path,
    fold: int,
    arm: str,
    fit: np.ndarray,
    development: np.ndarray,
    evaluation: np.ndarray,
    normalizers: tuple[np.ndarray, np.ndarray, float, float, np.ndarray],
    config: dict[str, Any],
    output: Path,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    prediction_path = output / "predictions" / f"fold{fold}_{arm}.csv"
    curve_path = output / "curves" / f"fold{fold}_{arm}.csv"
    checkpoint = output / "checkpoints" / f"fold{fold}" / f"{arm}.pt"
    if prediction_path.exists() and curve_path.exists() and checkpoint.exists():
        return pd.read_csv(prediction_path), pd.read_csv(curve_path), "reused"
    settings = config["training"]
    seed = int(settings["seed"]) + fold * 100
    seed_everything(seed)
    history_mean, history_std, target_mean, target_std, class_values = normalizers
    base_weights = subject_weights(frame)
    train_weights = np.ones(len(frame), np.float32)
    horizons = [int(value) for value in config["forecast"]["horizons_months"]]
    fit_dataset = JointVolumeDataset(
        frame, np.arange(len(frame)), cache_path, history_mean, history_std,
        target_mean, target_std, horizons, train_weights,
    )
    sampler = SubjectBalancedSampler(frame, fit, seed)
    loader = DataLoader(
        fit_dataset, batch_size=int(settings["batch_size"]), sampler=sampler,
        num_workers=int(settings["num_workers"]), pin_memory=device.type == "cuda",
        persistent_workers=False,
    )
    dev_dataset = JointVolumeDataset(
        frame, development, cache_path, history_mean, history_std,
        target_mean, target_std, horizons, base_weights,
    )
    eval_dataset = JointVolumeDataset(
        frame, evaluation, cache_path, history_mean, history_std,
        target_mean, target_std, horizons, base_weights,
    )
    model = JointSFCNMemoryModel(
        len(horizons), float(settings["dropout"]), tuple(settings["log_variance_bounds"])
    ).to(device)
    warm_hash = load_warm_start(model, fold, config)
    head_parameters = [parameter for name, parameter in model.named_parameters() if not name.startswith("encoder.")]
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": float(settings["encoder_learning_rate"])},
        {"params": head_parameters, "lr": float(settings["head_learning_rate"])},
    ], weight_decay=float(settings["weight_decay"]), betas=(0.9, 0.95), eps=1e-10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(settings["max_epochs"]))
    class_weight = torch.as_tensor(class_values, device=device)
    amp = bool(settings["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(
        device.type, enabled=amp and _amp_dtype(config) == torch.float16
    )
    accumulation = int(settings["gradient_accumulation_steps"])
    best_loss, best_state, stale = math.inf, None, 0
    curves: list[dict[str, Any]] = []
    for epoch in range(int(settings["max_epochs"])):
        model.train(); optimizer.zero_grad(set_to_none=True); train_losses = []
        for batch_index, batch in enumerate(loader):
            moved = {key: value.to(device, non_blocking=True) for key, value in batch.items() if key != "row"}
            with torch.amp.autocast(device_type=device.type, enabled=amp, dtype=_amp_dtype(config)):
                outputs = model(moved["volume"], moved["history"])
                loss, _ = multitask_loss(outputs, moved, arm, class_weight, config)
            scaler.scale(loss / accumulation).backward()
            should_step = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader)
            if should_step:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), float(settings["gradient_clip"]))
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            train_losses.append(float(loss.detach()))
        scheduler.step()
        dev_loss, _ = evaluate_model(model, dev_dataset, arm, class_weight, config, device)
        record = {
            "fold": fold, "arm": arm, "epoch": epoch,
            "train_loss": float(np.mean(train_losses)), "dev_loss": dev_loss,
            "encoder_lr": optimizer.param_groups[0]["lr"], "head_lr": optimizer.param_groups[1]["lr"],
        }
        curves.append(record); print(json.dumps(record), flush=True)
        if np.isfinite(dev_loss) and dev_loss < best_loss - 1e-5:
            best_loss = dev_loss; best_state = copy.deepcopy(model.state_dict()); stale = 0
        else:
            stale += 1
            if stale >= int(settings["patience"]):
                break
    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for fold {fold} arm {arm}")
    model.load_state_dict(best_state)
    _, raw = evaluate_model(model, eval_dataset, arm, class_weight, config, device)
    predictions = attach_truth(raw, frame, arm, fold, config)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": best_state, "fold": fold, "arm": arm, "seed": seed,
        "best_dev_loss": best_loss, "warm_start_sha256": warm_hash,
        "history_mean": history_mean, "history_std": history_std,
        "target_mean": target_mean, "target_std": target_std,
        "class_weight": class_values,
    }, checkpoint)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(prediction_path, index=False)
    curve = pd.DataFrame(curves); curve.to_csv(curve_path, index=False)
    return predictions, curve, warm_hash


def run_folds(config: dict[str, Any], output: Path, folds: list[int]) -> None:
    frame = pd.read_csv(output / "cohort_manifest.csv", low_memory=False)
    cache_path = native_cache_path(config, output)
    if not cache_path.exists():
        raise FileNotFoundError("Run --mode prepare before training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for fold in folds:
        fit, development, evaluation = partitions(frame, fold, config)
        normalizers = fit_normalizers(frame, fit, config)
        fit_subjects = set(frame.iloc[fit].subject_id.astype(str))
        dev_subjects = set(frame.iloc[development].subject_id.astype(str))
        eval_subjects = set(frame.iloc[evaluation].subject_id.astype(str))
        fold_record = {
            "fold": fold, "fit_subjects": len(fit_subjects),
            "development_subjects": len(dev_subjects), "evaluation_subjects": len(eval_subjects),
            "fit_evaluation_overlap": len(fit_subjects & eval_subjects),
            "development_evaluation_overlap": len(dev_subjects & eval_subjects),
            "fit_development_overlap": len(fit_subjects & dev_subjects),
            "fit_scans": len(fit), "development_scans": len(development), "evaluation_scans": len(evaluation),
        }
        save_json(fold_record, output / f"fold_manifest_{fold}.json")
        for arm in config["arms"]:
            predictions, _, _ = train_arm(
                frame, cache_path, fold, arm, fit, development, evaluation,
                normalizers, config, output, device,
            )
            if arm == "joint_multitask":
                shuffle_path = output / "predictions" / f"fold{fold}_joint_subject_shuffled_mri.csv"
                if not shuffle_path.exists():
                    history_mean, history_std, target_mean, target_std, class_values = normalizers
                    base_weights = subject_weights(frame)
                    source = np.arange(len(frame), dtype=int)
                    source[evaluation] = shuffled_indices(frame, evaluation, int(config["training"]["seed"]) + 9000 + fold)
                    dataset = JointVolumeDataset(
                        frame, evaluation, cache_path, history_mean, history_std,
                        target_mean, target_std,
                        [int(value) for value in config["forecast"]["horizons_months"]],
                        base_weights, volume_source=source,
                    )
                    checkpoint = torch.load(
                        output / "checkpoints" / f"fold{fold}" / "joint_multitask.pt",
                        map_location=device, weights_only=False,
                    )
                    model = JointSFCNMemoryModel(
                        len(config["forecast"]["horizons_months"]),
                        float(config["training"]["dropout"]),
                        tuple(config["training"]["log_variance_bounds"]),
                    ).to(device)
                    model.load_state_dict(checkpoint["model"])
                    _, raw = evaluate_model(
                        model, dataset, "joint_multitask",
                        torch.as_tensor(class_values, device=device), config, device,
                    )
                    shuffled = attach_truth(raw, frame, "joint_subject_shuffled_mri", fold, config)
                    shuffle_path.parent.mkdir(parents=True, exist_ok=True)
                    shuffled.to_csv(shuffle_path, index=False)
        print(json.dumps({"completed_fold": fold, "device": str(device)}), flush=True)


def regression_metrics(predictions: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    horizons = [int(value) for value in config["forecast"]["horizons_months"]]
    for arm, group in predictions.groupby("arm", sort=True):
        if group.prediction_current.notna().any():
            weights = subject_weights(group)
            error = np.abs(group.current_memory.to_numpy(float) - group.prediction_current.to_numpy(float))
            variance = group.variance_current.to_numpy(float).clip(1e-8)
            residual = group.current_memory.to_numpy(float) - group.prediction_current.to_numpy(float)
            overall_rows.append({
                "arm": arm, "endpoint": "current", "observations": len(group),
                "subjects": group.subject_id.nunique(), "mae": float(np.average(error, weights=weights)),
                "nll": float(np.average(0.5 * (np.log(2 * np.pi * variance) + residual ** 2 / variance), weights=weights)),
                "coverage_95": float(np.average(np.abs(residual) <= 1.96 * np.sqrt(variance), weights=weights)),
            })
        long_blocks = []
        for horizon in horizons:
            local = group[group[f"future_memory_{horizon}"].notna() & group[f"prediction_{horizon}"].notna()].copy()
            if local.empty:
                continue
            weights = subject_weights(local)
            truth = local[f"future_memory_{horizon}"].to_numpy(float)
            prediction = local[f"prediction_{horizon}"].to_numpy(float)
            variance = local[f"variance_{horizon}"].to_numpy(float).clip(1e-8)
            error = truth - prediction
            row = {
                "arm": arm, "horizon_months": horizon, "observations": len(local),
                "subjects": local.subject_id.nunique(),
                "mae": float(np.average(np.abs(error), weights=weights)),
                "nll": float(np.average(0.5 * (np.log(2 * np.pi * variance) + error ** 2 / variance), weights=weights)),
                "coverage_95": float(np.average(np.abs(error) <= 1.96 * np.sqrt(variance), weights=weights)),
            }
            horizon_rows.append(row)
            long_blocks.append(pd.DataFrame({
                "subject_id": local.subject_id.astype(str), "error": np.abs(error),
                "nll": 0.5 * (np.log(2 * np.pi * variance) + error ** 2 / variance),
                "covered": np.abs(error) <= 1.96 * np.sqrt(variance),
            }))
        if long_blocks:
            long = pd.concat(long_blocks, ignore_index=True)
            weights = subject_weights(long)
            overall_rows.append({
                "arm": arm, "endpoint": "future", "observations": len(long),
                "subjects": long.subject_id.nunique(),
                "mae": float(np.average(long.error, weights=weights)),
                "nll": float(np.average(long.nll, weights=weights)),
                "coverage_95": float(np.average(long.covered, weights=weights)),
            })
    return pd.DataFrame(overall_rows), pd.DataFrame(horizon_rows)


def future_subject_errors(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for record in frame.itertuples(index=False):
        values = []
        for horizon in config["forecast"]["horizons_months"]:
            truth = getattr(record, f"future_memory_{horizon}")
            prediction = getattr(record, f"prediction_{horizon}")
            if np.isfinite(truth) and np.isfinite(prediction):
                values.append(abs(float(truth) - float(prediction)))
        if values:
            rows.append({"subject_id": str(record.subject_id), "fold": int(record.fold), "error": float(np.mean(values))})
    return pd.DataFrame(rows).groupby(["subject_id", "fold"], as_index=False).error.mean()


def regression_bootstrap(predictions: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    joint = future_subject_errors(predictions[predictions.arm.eq("joint_multitask")], config)
    baseline = future_subject_errors(predictions[predictions.arm.eq("regression_only")], config)
    paired = joint.merge(baseline, on=["subject_id", "fold"], suffixes=("_joint", "_baseline"), validate="one_to_one")
    paired["difference"] = paired.error_joint - paired.error_baseline
    rng = np.random.default_rng(int(config["statistics"]["bootstrap_seed"]))
    values = paired.difference.to_numpy(float)
    draws = np.asarray([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(int(config["statistics"]["bootstrap_replicates"]))
    ])
    fold_differences = paired.groupby("fold").difference.mean()
    joint_horizon = regression_metrics(
        predictions[predictions.arm.isin(["joint_multitask", "regression_only"])], config
    )[1].pivot(index="horizon_months", columns="arm", values="mae")
    horizon_difference = joint_horizon.joint_multitask - joint_horizon.regression_only
    return {
        "comparison": "joint_multitask_minus_regression_only",
        "participant_difference": float(values.mean()),
        "ci_low": float(np.quantile(draws, 0.025)), "ci_high": float(np.quantile(draws, 0.975)),
        "subjects": len(paired), "replicates": int(config["statistics"]["bootstrap_replicates"]),
        "folds_better": int((fold_differences < 0).sum()),
        "horizons_noninferior": int((horizon_difference <= 0).sum()),
        "fold_differences": {str(key): float(value) for key, value in fold_differences.items()},
        "horizon_differences": {str(key): float(value) for key, value in horizon_difference.items()},
    }


def summarize(config: dict[str, Any], output: Path) -> None:
    paths = []
    for fold in range(int(config["folds"])):
        for arm in [*config["arms"], "joint_subject_shuffled_mri"]:
            path = output / "predictions" / f"fold{fold}_{arm}.csv"
            if arm == "joint_subject_shuffled_mri" and not path.exists():
                raise FileNotFoundError(path)
            if arm != "joint_subject_shuffled_mri" or path.exists():
                paths.append(path)
    predictions = pd.concat([pd.read_csv(path, low_memory=False) for path in paths], ignore_index=True)
    predictions = normalize_diagnostic_probabilities(predictions)
    predictions.to_csv(output / "crossfit_predictions.csv", index=False)
    classification = predictions[predictions.arm.isin([
        "classification_only", "joint_multitask", "joint_subject_shuffled_mri"
    ])].copy()
    diagnostic = arm_metrics(classification)
    diagnostic.to_csv(output / "diagnostic_metrics.csv", index=False)
    bootstrap_metric_intervals(classification, config).to_csv(
        output / "diagnostic_metric_intervals.csv", index=False
    )
    diagnostic_contrast = pd.DataFrame(bootstrap_contrast(
        classification, "joint_multitask", "classification_only", config
    ))
    diagnostic_contrast.to_csv(output / "diagnostic_bootstrap.csv", index=False)
    per_class = []
    for row in diagnostic.itertuples(index=False):
        for label in config["classes"]:
            per_class.append({
                "arm": row.arm, "class": label,
                "sensitivity": getattr(row, f"sensitivity_{label}"),
                "specificity": getattr(row, f"specificity_{label}"),
                "f1": getattr(row, f"f1_{label}"),
            })
    pd.DataFrame(per_class).to_csv(output / "diagnostic_per_class_metrics.csv", index=False)
    regression_predictions = predictions[predictions.arm.isin([
        "regression_only", "joint_multitask", "joint_subject_shuffled_mri"
    ])].copy()
    regression, horizons = regression_metrics(regression_predictions, config)
    regression.to_csv(output / "regression_metrics.csv", index=False)
    horizons.to_csv(output / "regression_metrics_by_horizon.csv", index=False)
    regression_comparison = regression_bootstrap(regression_predictions, config)
    save_json(regression_comparison, output / "regression_bootstrap.json")
    auc = diagnostic_contrast[diagnostic_contrast.metric.eq("macro_auc")].iloc[0]
    joint_future = regression[(regression.arm == "joint_multitask") & (regression.endpoint == "future")].iloc[0]
    gate = config["gate"]
    criteria = {
        "classification_auc_point_nonnegative": bool(
            (not gate["require_joint_auc_point_nonnegative"]) or auc.difference >= 0
        ),
        "classification_auc_noninferior": bool(
            auc.ci_low > float(gate["classification_auc_noninferiority_margin"])
        ),
        "future_mae_ci_below_zero": bool(
            (not gate["require_joint_future_mae_ci_below_zero"]) or regression_comparison["ci_high"] < 0
        ),
        "future_folds_better": bool(
            regression_comparison["folds_better"] >= int(gate["minimum_folds_future_better"])
        ),
        "future_horizons_noninferior": bool(
            regression_comparison["horizons_noninferior"] >= int(gate["minimum_horizons_future_noninferior"])
        ),
        "future_coverage_calibrated": bool(
            float(gate["coverage_95_bounds"][0]) <= joint_future.coverage_95 <= float(gate["coverage_95_bounds"][1])
        ),
    }
    decision = {
        "run_id": config["run_id"], "original_train_only": True,
        "validation_or_test_loaded": False, "criteria": criteria,
        "joint_multitask_gate_passed": bool(all(criteria.values())),
        "classification_contrast": {
            "macro_auc_difference": float(auc.difference), "ci_low": float(auc.ci_low), "ci_high": float(auc.ci_high)
        },
        "regression_contrast": regression_comparison,
        "confirmatory_claim_open": False,
    }
    save_json(decision, output / "development_decision.json")
    manifest = pd.read_csv(output / "cohort_manifest.csv", low_memory=False)
    diag_table = diagnostic[[
        "arm", "accuracy", "macro_auc", "macro_sensitivity", "macro_specificity", "macro_f1"
    ]]
    reg_table = regression[["arm", "endpoint", "mae", "nll", "coverage_95"]]
    lines = [
        "# Phase 7A: native-T1 CN/MCI/AD and Memory multitask learning", "",
        "Original-train-only five-fold subject cross-fitting; official validation and test were not loaded.", "",
        "## Cohort", "",
        f"- Scans: **{len(manifest):,}**", f"- Participants: **{manifest.subject_id.nunique():,}**",
        f"- CN/MCI/AD visits: **{manifest.diagnosis.value_counts().to_dict()}**", "",
        "## Subject-balanced diagnosis", "",
        markdown_table(diag_table), "",
        "## Subject-balanced Memory regression", "",
        markdown_table(reg_table), "",
        "## Prespecified contrasts", "",
        f"- Joint minus classification-only macro AUC: **{auc.difference:+.6f}** "
        f"(95% CI **[{auc.ci_low:+.6f}, {auc.ci_high:+.6f}]**).",
        f"- Joint minus regression-only future Memory MAE: **{regression_comparison['participant_difference']:+.6f}** "
        f"(95% CI **[{regression_comparison['ci_low']:+.6f}, {regression_comparison['ci_high']:+.6f}]**); "
        f"folds better **{regression_comparison['folds_better']}/5**, horizons noninferior "
        f"**{regression_comparison['horizons_noninferior']}/4**.", "",
        f"Joint multitask development gate passed: **`{str(all(criteria.values())).lower()}`**.", "",
        "This is internal original-train development evidence and does not open a confirmatory clinical claim.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    save_json({
        "config_sha256": hashlib.sha256((ROOT / "configs/phase7a_joint_multitask_native_t1.yaml").read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "cohort_sha256": hashlib.sha256((output / "cohort_manifest.csv").read_bytes()).hexdigest(),
        "checkpoints": len(list((output / "checkpoints").glob("fold*/*.pt"))),
        "folds": int(config["folds"]), "arms": config["arms"],
    }, output / "manifest.json")
    print(output)


def prepare(config: dict[str, Any], output: Path) -> None:
    frame = build_cohort(config, output)
    prepare_native_cache(frame, config, output)
    save_json(config, output / "resolved_config.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--mode", choices=["prepare", "train", "summarize", "all"], default="all")
    parser.add_argument("--folds", nargs="*", type=int)
    args = parser.parse_args(); config = load_config(args.config); output = output_dir(config)
    folds = args.folds if args.folds else list(range(int(config["folds"])))
    if args.mode in {"prepare", "all"}:
        prepare(config, output)
    if args.mode in {"train", "all"}:
        run_folds(config, output, folds)
    if args.mode in {"summarize", "all"}:
        summarize(config, output)


if __name__ == "__main__":
    main()
