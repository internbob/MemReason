"""Validation-only MRI representation and future-change experiment."""

from __future__ import annotations

import argparse
import copy
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import Ridge
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from ..data.common import ROOT, save_csv, save_json
from .phase1_numeric import markdown_table, seed_everything


TARGET_DATES = {
    "adas_total13": "adas_visit_date",
    "cdr_cdrsb": "cdr_visit_date",
    "faq_faqtotal": "faq_visit_date",
    "mmse_mmscore": "mmse_visit_date",
    "moca_moca": "moca_visit_date",
}


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def build_concurrent_frame(config: dict[str, Any]) -> pd.DataFrame:
    phase3a = ROOT / "outputs/phase3a_global_mri_anchor_dev_20261013"
    anchors = pd.read_csv(phase3a / "sample_manifest.csv").drop_duplicates(
        ["split", "anchor_key"]
    )
    metadata = pd.read_csv(ROOT / "processed/mri_metadata.csv", low_memory=False)
    metadata["image_id"] = pd.to_numeric(metadata.image_id, errors="coerce")
    ida = pd.read_csv(ROOT / "idaSearch_10_12_2026 (2).csv", low_memory=False).rename(
        columns={"Image ID": "image_id", "Age": "age", "Sex": "sex"}
    )
    ida["image_id"] = pd.to_numeric(ida.image_id, errors="coerce")
    ida = ida[["image_id", "age", "sex"]].dropna(subset=["image_id"]).drop_duplicates("image_id")
    anchors = anchors.merge(
        metadata[["scan_id", "image_id"]].drop_duplicates("scan_id"),
        on="scan_id",
        how="left",
        validate="one_to_one",
    ).merge(ida, on="image_id", how="left", validate="many_to_one")
    targets = list(config["targets"])
    frames = []
    for split in ["train", "val"]:
        source = pd.read_csv(ROOT / f"splits/{split}.csv", low_memory=False)
        columns = ["subject_id", "visit_month"] + targets + list(TARGET_DATES.values())
        current = anchors[anchors.split.eq(split)].merge(
            source[columns],
            left_on=["subject_id", "cutoff_month"],
            right_on=["subject_id", "visit_month"],
            how="left",
            validate="one_to_one",
        )
        frames.append(current)
    frame = pd.concat(frames, ignore_index=True)
    frame["age"] = pd.to_numeric(frame.age, errors="coerce")
    frame["sex_numeric"] = frame.sex.astype(str).str.upper().map({"M": 0.0, "F": 1.0})
    scan_date = pd.to_datetime(frame.scan_date, errors="coerce")
    for target, date_column in TARGET_DATES.items():
        target_date = pd.to_datetime(frame[date_column], errors="coerce")
        eligible = (
            frame[target].notna()
            & target_date.notna()
            & scan_date.notna()
            & ((target_date - scan_date).dt.days.abs() <= int(config["date_window_days"]))
        )
        frame.loc[~eligible, target] = np.nan
    frame = frame[frame.age.notna() & frame.sex_numeric.notna()]
    frame = frame[frame[targets].notna().any(axis=1)].copy()
    if set(frame.split) != {"train", "val"}:
        raise RuntimeError("Concurrent MRI frame must contain train and validation only")
    if frame.scan_id.duplicated().any():
        raise RuntimeError("Concurrent MRI frame is not unique by scan")
    return frame.sort_values(["split", "subject_id", "cutoff_month"]).reset_index(drop=True)


def load_global_features(frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, int]]:
    archive = np.load(
        ROOT / "outputs/phase3a_global_mri_anchor_dev_20261013/mri_features.npz",
        allow_pickle=False,
    )
    scan_ids = archive["scan_ids"].astype(str)
    features = archive["features"].astype(np.float32)
    lookup = {scan_id: index for index, scan_id in enumerate(scan_ids)}
    matrix = np.stack([features[lookup[str(scan_id)]] for scan_id in frame.scan_id])
    train = frame.split.eq("train").to_numpy()
    mean, std = matrix[train].mean(0), matrix[train].std(0)
    std[std < 1e-6] = 1.0
    return ((matrix - mean) / std).astype(np.float32), {
        str(scan_id): index for index, scan_id in enumerate(frame.scan_id)
    }


def _extract_volume(task: tuple[int, str, str, tuple[int, int, int]]):
    index, image_path, mask_path, shape = task
    torch.set_num_threads(1)
    try:
        image = nib.as_closest_canonical(nib.load(image_path))
        mask_image = nib.as_closest_canonical(nib.load(mask_path))
        array = np.asarray(image.dataobj, dtype=np.float32)
        mask = np.asarray(mask_image.dataobj, dtype=np.float32) > 0.5
        if array.shape != mask.shape or not mask.any() or not np.isfinite(array).all():
            raise ValueError("invalid_image_or_mask")
        foreground = array[mask]
        std = float(foreground.std())
        if std < 1e-6:
            raise ValueError("constant_foreground")
        normalized = np.zeros_like(array, dtype=np.float32)
        normalized[mask] = np.clip((foreground - float(foreground.mean())) / std, -5, 5)
        tensor = torch.from_numpy(normalized)[None, None]
        resized = F.interpolate(tensor, size=shape, mode="trilinear", align_corners=False)
        return index, resized[0, 0].numpy().astype(np.float16), None
    except Exception as error:
        return index, None, f"{type(error).__name__}:{error}"


def prepare_volume_cache(
    frame: pd.DataFrame, config: dict[str, Any], output: Path
) -> tuple[Path, pd.DataFrame]:
    cache = output / "volumes_48x56x48.npy"
    index_path = output / "volume_index.csv"
    shape = tuple(int(value) for value in config["volume"]["output_shape"])
    index = frame[["scan_id", "subject_id", "split", "image_path", "mask_path"]].copy()
    index["volume_index"] = np.arange(len(index))
    if cache.exists() and index_path.exists():
        existing = pd.read_csv(index_path)
        if existing.scan_id.astype(str).tolist() == index.scan_id.astype(str).tolist():
            array = np.load(cache, mmap_mode="r")
            if tuple(array.shape) == (len(index),) + shape:
                return cache, existing
    temporary = output / "volumes_48x56x48.tmp.npy"
    volume = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float16, shape=(len(index),) + shape
    )
    tasks = [
        (int(row.volume_index), str(row.image_path), str(row.mask_path), shape)
        for row in index.itertuples(index=False)
    ]
    failures = {}
    with ProcessPoolExecutor(max_workers=int(config["volume"]["extraction_workers"])) as pool:
        futures = [pool.submit(_extract_volume, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), 1):
            volume_index, values, error = future.result()
            if error is None:
                volume[volume_index] = values
            else:
                failures[volume_index] = error
            if completed % 250 == 0 or completed == len(futures):
                print(json.dumps({"volume_completed": completed, "total": len(futures)}), flush=True)
    volume.flush()
    del volume
    if failures:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Volume extraction failures: {list(failures.items())[:10]}")
    temporary.replace(cache)
    save_csv(index, index_path)
    return cache, index


def make_shuffle_indices(frame: pd.DataFrame, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    subjects = frame.subject_id.astype(str).to_numpy()
    output = np.arange(len(frame))
    for index, subject in enumerate(subjects):
        candidates = np.flatnonzero(subjects != subject)
        output[index] = candidates[int(rng.integers(0, len(candidates)))]
    if np.any(subjects[output] == subjects):
        raise RuntimeError("Subject-shuffle control retained subject identity")
    return output


class ConcurrentDataset(Dataset):
    def __init__(
        self,
        row_indices: np.ndarray,
        frame: pd.DataFrame,
        demos: np.ndarray,
        global_features: np.ndarray,
        targets: np.ndarray,
        masks: np.ndarray,
        representation: str,
        volume_path: Path,
        source_indices: np.ndarray | None = None,
    ):
        self.row_indices = np.asarray(row_indices)
        self.frame = frame
        self.demos = demos
        self.global_features = global_features
        self.targets = targets
        self.masks = masks
        self.representation = representation
        self.volume_path = volume_path
        self.source_indices = source_indices
        self._volumes = None

    def __len__(self) -> int:
        return len(self.row_indices)

    @property
    def volumes(self):
        if self._volumes is None:
            self._volumes = np.load(self.volume_path, mmap_mode="r")
        return self._volumes

    def __getitem__(self, item: int) -> dict[str, Any]:
        row = int(self.row_indices[item])
        source = int(self.source_indices[row]) if self.source_indices is not None else row
        result = {
            "row": row,
            "demo": torch.from_numpy(self.demos[row]),
            "target": torch.from_numpy(self.targets[row]),
            "mask": torch.from_numpy(self.masks[row]),
        }
        if self.representation == "global_mri":
            result["image"] = torch.from_numpy(self.global_features[source])
        elif self.representation == "sfcn_3d":
            result["image"] = torch.from_numpy(
                np.asarray(self.volumes[source], dtype=np.float32)[None]
            )
        else:
            result["image"] = torch.zeros(1, dtype=torch.float32)
        return result


class SFCNEncoder(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        blocks = []
        channels = [1, 8, 16, 32, 64, 128]
        for source, target in zip(channels[:-1], channels[1:]):
            blocks.extend(
                [
                    nn.Conv3d(source, target, 3, stride=2, padding=1, bias=False),
                    nn.GroupNorm(min(8, target), target),
                    nn.GELU(),
                ]
            )
        self.network = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.pool(self.network(image)).flatten(1))


class ConcurrentRegressor(nn.Module):
    def __init__(self, representation: str, global_dim: int, n_targets: int, dropout: float):
        super().__init__()
        self.representation = representation
        self.demo = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.LayerNorm(32))
        if representation == "demographics_only":
            self.encoder = None
            representation_dim = 0
        elif representation == "global_mri":
            self.encoder = nn.Sequential(
                nn.Linear(global_dim, 256),
                nn.GELU(),
                nn.LayerNorm(256),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.GELU(),
                nn.LayerNorm(128),
            )
            representation_dim = 128
        elif representation == "sfcn_3d":
            self.encoder = SFCNEncoder(dropout)
            representation_dim = 128
        else:
            raise ValueError(representation)
        self.head = nn.Sequential(
            nn.Linear(32 + representation_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.Linear(128, n_targets),
        )

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        if self.encoder is None:
            return image.new_zeros((len(image), 0))
        return self.encoder(image)

    def forward(self, demo: torch.Tensor, image: torch.Tensor):
        representation = self.encode(image)
        prediction = self.head(torch.cat([self.demo(demo), representation], dim=1))
        return prediction, representation


def macro_huber(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    losses = []
    for index in range(prediction.shape[1]):
        available = mask[:, index]
        if available.any():
            losses.append(F.huber_loss(prediction[available, index], target[available, index]))
    return torch.stack(losses).mean()


def evaluate_concurrent(
    model: ConcurrentRegressor,
    dataset: ConcurrentDataset,
    batch_size: int,
    device: torch.device,
    target_names: list[str],
    target_mean: np.ndarray,
    target_std: np.ndarray,
    model_name: str,
    seed: int,
    pairing: str,
) -> pd.DataFrame:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            prediction, _ = model(batch["demo"].to(device), batch["image"].to(device))
            prediction = prediction.cpu().numpy() * target_std[None] + target_mean[None]
            for local, row_index in enumerate(batch["row"].numpy()):
                source = dataset.source_indices[row_index] if dataset.source_indices is not None else row_index
                record = dataset.frame.iloc[row_index]
                for target_index, target in enumerate(target_names):
                    if dataset.masks[row_index, target_index]:
                        rows.append(
                            {
                                "scan_id": record.scan_id,
                                "source_scan_id": dataset.frame.iloc[int(source)].scan_id,
                                "subject_id": record.subject_id,
                                "split": record.split,
                                "model": model_name,
                                "seed": seed,
                                "pairing": pairing,
                                "target": target,
                                "truth": float(dataset.targets[row_index, target_index] * target_std[target_index] + target_mean[target_index]),
                                "prediction": float(prediction[local, target_index]),
                            }
                        )
    return pd.DataFrame(rows)


def normalized_macro(predictions: pd.DataFrame, scales: dict[str, float]) -> float:
    values = []
    for target, group in predictions.groupby("target"):
        values.append(float(np.abs(group.prediction - group.truth).mean() / scales[target]))
    return float(np.mean(values))


def train_concurrent(
    representation: str,
    seed: int,
    frame: pd.DataFrame,
    demos: np.ndarray,
    global_features: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    volume_path: Path,
    val_shuffle: np.ndarray,
    config: dict[str, Any],
    output: Path,
    device: torch.device,
) -> tuple[pd.DataFrame, int]:
    seed_everything(seed)
    target_names = list(config["targets"])
    settings = config["training"]
    train_indices = np.flatnonzero(frame.split.eq("train").to_numpy())
    val_indices = np.flatnonzero(frame.split.eq("val").to_numpy())
    train_dataset = ConcurrentDataset(
        train_indices, frame, demos, global_features, targets, masks, representation, volume_path
    )
    val_dataset = ConcurrentDataset(
        val_indices, frame, demos, global_features, targets, masks, representation, volume_path
    )
    shuffled_dataset = ConcurrentDataset(
        val_indices,
        frame,
        demos,
        global_features,
        targets,
        masks,
        representation,
        volume_path,
        val_shuffle,
    )
    batch_size = int(
        settings["batch_size_volume"]
        if representation == "sfcn_3d"
        else settings["batch_size_tabular"]
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    model = ConcurrentRegressor(
        representation,
        global_features.shape[1],
        len(target_names),
        float(settings["dropout"]),
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    scales = dict(zip(target_names, target_std))
    best_score, best_state, patience = float("inf"), None, 0
    curves = []
    for epoch in range(int(settings["max_epochs"])):
        model.train()
        losses = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = model(batch["demo"].to(device), batch["image"].to(device))
            loss = macro_huber(
                prediction,
                batch["target"].to(device),
                batch["mask"].to(device),
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss for {representation}/{seed}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(settings["gradient_clip"]))
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = evaluate_concurrent(
            model,
            val_dataset,
            batch_size,
            device,
            target_names,
            target_mean,
            target_std,
            representation,
            seed,
            "observed",
        )
        score = normalized_macro(validation, scales)
        curve = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_macro_normalized_mae": score}
        curves.append(curve)
        print(json.dumps({"model": representation, "seed": seed, **curve}), flush=True)
        if score < best_score - 1e-5:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= int(settings["patience"]):
                break
    if best_state is None:
        raise RuntimeError("No checkpoint selected")
    model.load_state_dict(best_state)
    directory = output / "checkpoints" / representation / str(seed)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": best_state,
            "representation": representation,
            "seed": seed,
            "trainable_parameters": parameter_count,
            "validation_macro_normalized_mae": best_score,
        },
        directory / "best.pt",
    )
    save_csv(pd.DataFrame(curves), directory / "curves.csv")
    observed = evaluate_concurrent(
        model, val_dataset, batch_size, device, target_names, target_mean, target_std,
        representation, seed, "observed"
    )
    outputs = [observed]
    if representation != "demographics_only":
        outputs.append(
            evaluate_concurrent(
                model, shuffled_dataset, batch_size, device, target_names, target_mean,
                target_std, representation, seed, "shuffled_subject"
            )
        )
    return pd.concat(outputs, ignore_index=True), parameter_count


def ensemble(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [
        key
        for key in [
            "sample_id",
            "scan_id",
            "source_scan_id",
            "subject_id",
            "split",
            "model",
            "pairing",
            "target",
            "horizon_months",
            "current",
            "truth",
            "future_value",
        ]
        if key in frame.columns
    ]
    output = frame.groupby(keys, as_index=False).prediction.mean()
    output["seed"] = pd.NA
    return output


def metrics(frame: pd.DataFrame, scales: dict[str, float]) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(["model", "pairing", "target"], sort=True):
        error = group.prediction - group.truth
        rows.append(
            {
                "model": keys[0],
                "pairing": keys[1],
                "target": keys[2],
                "n": len(group),
                "subjects": group.subject_id.nunique(),
                "mae": float(error.abs().mean()),
                "normalized_mae": float(error.abs().mean() / scales[keys[2]]),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "spearman": float(group.prediction.corr(group.truth, method="spearman")),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_difference(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    scales: dict[str, float],
    replicates: int,
    seed: int,
    target: str | None = None,
    horizon: int | None = None,
) -> dict[str, Any]:
    if target is not None:
        candidate = candidate[candidate.target.eq(target)]
        baseline = baseline[baseline.target.eq(target)]
    if horizon is not None:
        candidate = candidate[candidate.horizon_months.eq(horizon)]
        baseline = baseline[baseline.horizon_months.eq(horizon)]
    keys = [key for key in ["sample_id", "scan_id", "subject_id", "target", "horizon_months", "truth"] if key in candidate.columns]
    paired = candidate[keys + ["prediction"]].merge(
        baseline[keys + ["prediction"]], on=keys, suffixes=("_candidate", "_baseline"), validate="one_to_one"
    )
    scale = paired.target.map(scales).to_numpy()
    paired["difference"] = (
        np.abs(paired.prediction_candidate - paired.truth)
        - np.abs(paired.prediction_baseline - paired.truth)
    ) / scale
    grouped = paired.groupby("subject_id").difference.agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(grouped), size=(replicates, len(grouped)))
    draws = grouped["sum"].to_numpy()[indices].sum(1) / grouped["count"].to_numpy()[indices].sum(1)
    return {
        "target": target or "overall",
        "horizon_months": horizon,
        "candidate": str(candidate.model.iloc[0]),
        "candidate_pairing": str(candidate.pairing.iloc[0]),
        "baseline": str(baseline.model.iloc[0]),
        "baseline_pairing": str(baseline.pairing.iloc[0]),
        "n": len(paired),
        "subjects": len(grouped),
        "normalized_mae_difference": float(paired.difference.mean()),
        "ci_lower": float(np.quantile(draws, 0.025)),
        "ci_upper": float(np.quantile(draws, 0.975)),
        "probability_candidate_better": float(np.mean(draws < 0)),
        "replicates": replicates,
    }


def extract_sfcn_embeddings(
    frame: pd.DataFrame,
    demos: np.ndarray,
    global_features: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    volume_path: Path,
    seed: int,
    config: dict[str, Any],
    output: Path,
    device: torch.device,
) -> np.ndarray:
    model = ConcurrentRegressor(
        "sfcn_3d", global_features.shape[1], len(config["targets"]), float(config["training"]["dropout"])
    ).to(device)
    checkpoint = torch.load(output / "checkpoints/sfcn_3d" / str(seed) / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    dataset = ConcurrentDataset(
        np.arange(len(frame)), frame, demos, global_features, targets, masks, "sfcn_3d", volume_path
    )
    loader = DataLoader(dataset, batch_size=int(config["training"]["batch_size_volume"]), shuffle=False, num_workers=0)
    embeddings = np.zeros((len(frame), 128), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            representation = model.encode(batch["image"].to(device)).cpu().numpy()
            embeddings[batch["row"].numpy()] = representation
    return embeddings


def build_delta_frame(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    targets = list(config["targets"])
    horizons = [int(value) for value in config["future_change"]["horizons_months"]]
    tolerances = {int(key): int(value) for key, value in config["future_change"]["tolerance_months"].items()}
    rows = []
    for split in ["train", "val"]:
        source = pd.read_csv(ROOT / f"splits/{split}.csv", low_memory=False)
        anchors = frame[frame.split.eq(split)]
        groups = {str(subject): group.sort_values("visit_month") for subject, group in source.groupby("subject_id")}
        for anchor in anchors.itertuples(index=False):
            subject_data = groups[str(anchor.subject_id)]
            scan_date = pd.to_datetime(anchor.scan_date)
            for target in targets:
                current = getattr(anchor, target)
                if not np.isfinite(current):
                    continue
                values = pd.to_numeric(subject_data[target], errors="coerce")
                months = pd.to_numeric(subject_data.visit_month, errors="coerce")
                dates = pd.to_datetime(subject_data[TARGET_DATES[target]], errors="coerce")
                for horizon in horizons:
                    center = float(anchor.cutoff_month) + horizon
                    eligible = (
                        values.notna()
                        & months.gt(float(anchor.cutoff_month))
                        & months.sub(center).abs().le(tolerances[horizon])
                        & dates.gt(scan_date)
                    )
                    candidates = subject_data.loc[eligible].copy()
                    if candidates.empty:
                        continue
                    candidates["distance"] = (candidates.visit_month - center).abs()
                    selected = candidates.sort_values(["distance", "visit_month"]).iloc[0]
                    future = float(selected[target])
                    rows.append(
                        {
                            "sample_id": f"{anchor.anchor_key}:h{horizon}:{target}",
                            "scan_id": anchor.scan_id,
                            "subject_id": anchor.subject_id,
                            "split": split,
                            "target": target,
                            "horizon_months": horizon,
                            "target_month": float(selected.visit_month),
                            "target_date": pd.to_datetime(
                                selected[TARGET_DATES[target]]
                            ).date().isoformat(),
                            "current": float(current),
                            "truth": future - float(current),
                            "future_value": future,
                        }
                    )
    return pd.DataFrame(rows)


def fit_delta_predictions(
    delta: pd.DataFrame,
    frame: pd.DataFrame,
    demos: np.ndarray,
    representation: np.ndarray | None,
    representation_name: str,
    scales: dict[str, float],
    alpha: float,
    val_shuffle: np.ndarray | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    scan_to_row = {str(scan_id): index for index, scan_id in enumerate(frame.scan_id)}
    outputs = []
    for (target, horizon), group in delta.groupby(["target", "horizon_months"]):
        train = group[group.split.eq("train")]
        val = group[group.split.eq("val")]
        if len(train) < 20 or val.empty:
            continue
        def design(part: pd.DataFrame, shuffled: bool = False):
            indices = np.asarray([scan_to_row[str(value)] for value in part.scan_id])
            source = val_shuffle[indices] if shuffled and val_shuffle is not None else indices
            current = (part.current.to_numpy() / scales[target])[:, None]
            components = [current, demos[indices]]
            if representation is not None:
                components.append(representation[source])
            return np.concatenate(components, axis=1)
        x_train = design(train)
        x_val = design(val)
        mean, std = x_train.mean(0), x_train.std(0)
        std[std < 1e-6] = 1.0
        model = Ridge(alpha=alpha)
        model.fit((x_train - mean) / std, train.truth.to_numpy() / scales[target])
        prediction = model.predict((x_val - mean) / std) * scales[target]
        for record, value in zip(val.itertuples(index=False), prediction):
            outputs.append({**record._asdict(), "model": representation_name, "seed": seed, "pairing": "observed", "prediction": float(value)})
        if representation is not None and val_shuffle is not None:
            shuffled_x = design(val, shuffled=True)
            shuffled_prediction = model.predict((shuffled_x - mean) / std) * scales[target]
            for record, value in zip(val.itertuples(index=False), shuffled_prediction):
                outputs.append({**record._asdict(), "model": representation_name, "seed": seed, "pairing": "shuffled_subject", "prediction": float(value)})
    return pd.DataFrame(outputs)


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
    frame = build_concurrent_frame(config)
    if smoke:
        frame = pd.concat(
            [frame[frame.split.eq("train")].head(256), frame[frame.split.eq("val")].head(96)],
            ignore_index=True,
        )
    target_names = list(config["targets"])
    train_mask = frame.split.eq("train").to_numpy()
    age_mean, age_std = frame.loc[train_mask, "age"].mean(), frame.loc[train_mask, "age"].std()
    demos = np.stack(
        [(frame.age.to_numpy() - age_mean) / age_std, frame.sex_numeric.to_numpy()], axis=1
    ).astype(np.float32)
    raw_targets = frame[target_names].to_numpy(dtype=np.float32)
    masks = np.isfinite(raw_targets)
    target_mean = np.nanmean(raw_targets[train_mask], axis=0).astype(np.float32)
    target_std = np.nanstd(raw_targets[train_mask], axis=0).astype(np.float32)
    targets = np.where(masks, (raw_targets - target_mean) / target_std, 0).astype(np.float32)
    scales = dict(zip(target_names, target_std))
    global_features, _ = load_global_features(frame)
    volume_path, volume_index = prepare_volume_cache(frame, config, output)
    if volume_index.scan_id.astype(str).tolist() != frame.scan_id.astype(str).tolist():
        raise RuntimeError("Volume cache order differs from experiment frame")
    val_rows = np.flatnonzero(~train_mask)
    local_shuffle = make_shuffle_indices(frame.iloc[val_rows].reset_index(drop=True), int(config["statistics"]["bootstrap_seed"]))
    val_shuffle = np.arange(len(frame))
    val_shuffle[val_rows] = val_rows[local_shuffle]
    save_csv(
        frame[["anchor_key", "scan_id", "subject_id", "split", "cutoff_month", "scan_date", "age", "sex"] + target_names],
        output / "concurrent_manifest.csv",
    )
    save_json(
        {
            "age_mean": float(age_mean),
            "age_std": float(age_std),
            "target_mean": dict(zip(target_names, target_mean.astype(float))),
            "target_std": dict(zip(target_names, target_std.astype(float))),
        },
        output / "normalization.json",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions, parameter_counts = [], {}
    for representation in config["representations"]:
        for seed in config["seeds"]:
            result, count = train_concurrent(
                representation,
                int(seed),
                frame,
                demos,
                global_features,
                targets,
                masks,
                target_mean,
                target_std,
                volume_path,
                val_shuffle,
                config,
                output,
                device,
            )
            predictions.append(result)
            parameter_counts[representation] = count
    seed_predictions = pd.concat(predictions, ignore_index=True)
    ensemble_predictions = ensemble(seed_predictions)
    save_csv(pd.concat([seed_predictions, ensemble_predictions], ignore_index=True), output / "concurrent_predictions.csv")
    concurrent_metrics = metrics(ensemble_predictions, scales)
    save_csv(concurrent_metrics, output / "concurrent_metrics.csv")
    replicates = int(config["statistics"]["bootstrap_replicates"])
    bootstrap_seed = int(config["statistics"]["bootstrap_seed"])
    comparisons = []
    demo = ensemble_predictions[
        ensemble_predictions.model.eq("demographics_only") & ensemble_predictions.pairing.eq("observed")
    ]
    for representation in ["global_mri", "sfcn_3d"]:
        observed = ensemble_predictions[
            ensemble_predictions.model.eq(representation) & ensemble_predictions.pairing.eq("observed")
        ]
        shuffled = ensemble_predictions[
            ensemble_predictions.model.eq(representation) & ensemble_predictions.pairing.eq("shuffled_subject")
        ]
        for baseline in [demo, shuffled]:
            for target in [None] + target_names:
                comparisons.append(
                    bootstrap_difference(
                        observed, baseline, scales, replicates, bootstrap_seed + len(comparisons), target=target
                    )
                )
    concurrent_bootstrap = pd.DataFrame(comparisons)
    save_csv(concurrent_bootstrap, output / "concurrent_bootstrap.csv")

    sfcn_embeddings = {}
    for seed in config["seeds"]:
        sfcn_embeddings[int(seed)] = extract_sfcn_embeddings(
            frame, demos, global_features, targets, masks, volume_path, int(seed), config, output, device
        )
    delta = build_delta_frame(frame, config)
    save_csv(delta, output / "future_change_manifest.csv")
    alpha = float(config["future_change"]["ridge_alpha"])
    delta_predictions = [
        fit_delta_predictions(delta, frame, demos, None, "delta_control", scales, alpha),
        fit_delta_predictions(delta, frame, demos, global_features, "delta_global_mri", scales, alpha, val_shuffle),
    ]
    for seed, embedding in sfcn_embeddings.items():
        delta_predictions.append(
            fit_delta_predictions(
                delta, frame, demos, embedding, "delta_sfcn_3d", scales, alpha, val_shuffle, seed
            )
        )
    delta_seed = pd.concat(delta_predictions, ignore_index=True)
    delta_ensemble = ensemble(delta_seed)
    save_csv(pd.concat([delta_seed, delta_ensemble], ignore_index=True), output / "future_change_predictions.csv")
    delta_metrics = metrics(delta_ensemble, scales)
    save_csv(delta_metrics, output / "future_change_metrics.csv")
    delta_comparisons = []
    control = delta_ensemble[
        delta_ensemble.model.eq("delta_control") & delta_ensemble.pairing.eq("observed")
    ]
    for representation in ["delta_global_mri", "delta_sfcn_3d"]:
        observed = delta_ensemble[
            delta_ensemble.model.eq(representation) & delta_ensemble.pairing.eq("observed")
        ]
        shuffled = delta_ensemble[
            delta_ensemble.model.eq(representation) & delta_ensemble.pairing.eq("shuffled_subject")
        ]
        for baseline in [control, shuffled]:
            for horizon in [None] + list(config["future_change"]["horizons_months"]):
                delta_comparisons.append(
                    bootstrap_difference(
                        observed, baseline, scales, replicates, bootstrap_seed + 100 + len(delta_comparisons), horizon=horizon
                    )
                )
    delta_bootstrap = pd.DataFrame(delta_comparisons)
    save_csv(delta_bootstrap, output / "future_change_bootstrap.csv")

    representation_decisions = {}
    for representation in ["global_mri", "sfcn_3d"]:
        versus_demo = concurrent_bootstrap[
            concurrent_bootstrap.candidate.eq(representation)
            & concurrent_bootstrap.baseline.eq("demographics_only")
        ]
        versus_shuffle = concurrent_bootstrap[
            concurrent_bootstrap.candidate.eq(representation)
            & concurrent_bootstrap.baseline.eq(representation)
            & concurrent_bootstrap.baseline_pairing.eq("shuffled_subject")
        ]
        significant_endpoints = int(
            ((versus_demo.target != "overall") & (versus_demo.ci_upper < 0)).sum()
        )
        representation_decisions[representation] = {
            "overall_better_than_demographics": bool(versus_demo.loc[versus_demo.target.eq("overall"), "ci_upper"].iloc[0] < 0),
            "overall_better_than_subject_shuffle": bool(versus_shuffle.loc[versus_shuffle.target.eq("overall"), "ci_upper"].iloc[0] < 0),
            "significant_endpoints_vs_demographics": significant_endpoints,
            "concurrent_signal_passed": bool(
                versus_demo.loc[versus_demo.target.eq("overall"), "ci_upper"].iloc[0] < 0
                and versus_shuffle.loc[versus_shuffle.target.eq("overall"), "ci_upper"].iloc[0] < 0
                and significant_endpoints >= int(config["gate"]["minimum_significant_endpoints"])
            ),
        }
    future_decisions = {}
    for representation in ["delta_global_mri", "delta_sfcn_3d"]:
        comparison = delta_bootstrap[
            delta_bootstrap.candidate.eq(representation)
            & delta_bootstrap.baseline.eq("delta_control")
        ]
        overall = bool(comparison.loc[comparison.horizon_months.isna(), "ci_upper"].iloc[0] < 0)
        horizon = bool((comparison.loc[comparison.horizon_months.notna(), "ci_upper"] < 0).any())
        future_decisions[representation] = {
            "overall_better_than_control": overall,
            "supported_12_or_24_month_improvement": horizon,
            "future_change_passed": overall and horizon,
        }
    concurrent_grounding_ready = any(
        representation_decisions[name]["concurrent_signal_passed"]
        for name in ["global_mri", "sfcn_3d"]
    )
    longitudinal_fusion_ready = any(
        representation_decisions[name]["concurrent_signal_passed"]
        and future_decisions["delta_" + name]["future_change_passed"]
        for name in ["global_mri", "sfcn_3d"]
    )
    decision = {
        "run_id": config["run_id"],
        "engineering_smoke_only": smoke,
        "development_only": True,
        "test_evaluated": False,
        "concurrent_representation_decisions": representation_decisions,
        "future_change_decisions": future_decisions,
        "phase3b_concurrent_grounding_ready": bool(concurrent_grounding_ready and not smoke),
        "phase3b_longitudinal_fusion_ready": bool(longitudinal_fusion_ready and not smoke),
        "next_action": (
            "freeze_longitudinal_spatial_grounding_protocol"
            if longitudinal_fusion_ready and not smoke
            else (
                "freeze_concurrent_spatial_grounding_protocol"
                if concurrent_grounding_ready and not smoke
                else "retain_mri_as_exploratory_and_prioritize_non_mri_robustness"
            )
        ),
    }
    save_json(decision, output / "development_decision.json")
    concurrent_summary = (
        concurrent_metrics.groupby(["model", "pairing"], as_index=False)
        .agg(normalized_mae=("normalized_mae", "mean"), mae=("mae", "mean"))
        .sort_values("normalized_mae")
    )
    delta_summary = (
        delta_metrics.groupby(["model", "pairing"], as_index=False)
        .agg(normalized_mae=("normalized_mae", "mean"), mae=("mae", "mean"))
        .sort_values("normalized_mae")
    )
    report = f"""# Phase 3B MRI representation validation

Status: completed validation-only experiment. Test was not loaded.

## Cohort

- Concurrent train/validation MRI anchors: {int(train_mask.sum())} / {int((~train_mask).sum())}
- Train/validation subjects: {frame.loc[train_mask, 'subject_id'].nunique()} / {frame.loc[~train_mask, 'subject_id'].nunique()}
- Concurrent validation targets: {int(masks[~train_mask].sum())}
- Future-change train/validation targets: {int((delta.split == 'train').sum())} / {int((delta.split == 'val').sum())}
- SFCN input: {config['volume']['output_shape']}; seeds: {config['seeds']}
- Parameter counts: {parameter_counts}

## Concurrent cognition

{markdown_table(concurrent_summary)}

### Paired subject-cluster bootstrap

Negative differences favor the correctly paired candidate.

{markdown_table(concurrent_bootstrap)}

## Future cognitive change

{markdown_table(delta_summary)}

### Paired subject-cluster bootstrap

{markdown_table(delta_bootstrap)}

## Decision

```json
{json.dumps(decision, indent=2)}
```

The experiment tests information value of the supplied aligned MRI. It does not establish regional anatomical grounding or clinical MRI QC.
"""
    (output / "report.md").write_text(report)
    save_json(
        {
            "run_id": config["run_id"],
            "device": str(device),
            "concurrent_counts": {"train": int(train_mask.sum()), "val": int((~train_mask).sum())},
            "subjects": {
                "train": int(frame.loc[train_mask, "subject_id"].nunique()),
                "val": int(frame.loc[~train_mask, "subject_id"].nunique()),
            },
            "parameter_counts": parameter_counts,
            "test_evaluated": False,
        },
        output / "run_manifest.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/phase3b_mri_representation_dev.yaml"
    )
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    main(arguments.config, arguments.smoke)
