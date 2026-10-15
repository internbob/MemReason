"""Validation-only global MRI-anchor experiment with shuffled negative controls."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import yaml

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from ..data.common import ROOT, read_csv, save_csv, save_json
from ..data.dictionary import load_dictionary
from .phase1_numeric import (
    ForecastSample,
    NumericCollator,
    build_samples,
    clean_frame,
    compute_metrics,
    fit_normalization,
    markdown_table,
    masked_huber,
    prediction_rows,
    seed_everything,
)
from .phase1b_semantic import select_verified_scalar_features


TARGET_DATE_COLUMNS = {
    "adas_total13": "adas_visit_date",
    "cdr_cdrsb": "cdr_visit_date",
    "faq_faqtotal": "faq_visit_date",
    "mmse_mmscore": "mmse_visit_date",
    "moca_moca": "moca_visit_date",
}


@dataclass
class AnchorSample:
    base: ForecastSample
    anchor_key: str
    scan_id: str
    source_scan_id: str
    mri_features: np.ndarray
    mri_available: float


class AnchorDataset(Dataset):
    def __init__(self, samples: list[AnchorSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> AnchorSample:
        return self.samples[index]


class AnchorCollator:
    def __init__(self, numeric: NumericCollator):
        self.numeric = numeric

    def __call__(self, samples: list[AnchorSample]) -> dict[str, Any]:
        batch = self.numeric([sample.base for sample in samples])
        batch["mri_features"] = torch.from_numpy(
            np.stack([sample.mri_features for sample in samples]).astype(np.float32)
        )
        batch["mri_available"] = torch.tensor(
            [sample.mri_available for sample in samples], dtype=torch.float32
        )[:, None]
        batch["anchor_samples"] = samples
        return batch


class MRIAnchorForecaster(nn.Module):
    """Parameter-matched GRU-D with an optional global MRI channel."""

    def __init__(
        self,
        n_features: int,
        n_mri_features: int,
        n_targets: int,
        hidden: int,
        layers: int,
        dropout: float,
    ):
        super().__init__()
        self.decay_weight = nn.Parameter(torch.zeros(n_features))
        self.decay_bias = nn.Parameter(torch.zeros(n_features))
        self.cells = nn.ModuleList(
            [nn.GRUCell(n_features * 2 + 2, hidden)]
            + [nn.GRUCell(hidden, hidden) for _ in range(layers - 1)]
        )
        self.dropout = nn.Dropout(dropout)
        self.horizon = nn.Sequential(
            nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.mri = nn.Sequential(
            nn.Linear(n_mri_features, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.mri_gate = nn.Parameter(torch.tensor(-2.0))
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 3 + 1, hidden * 2),
            nn.GELU(),
            nn.LayerNorm(hidden * 2),
        )
        self.output = nn.Linear(hidden * 2, n_targets)

    def cognitive_summary(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x, observed = batch["values"], batch["mask"]
        times, valid = batch["times"], batch["event_mask"]
        batch_size, steps, features = x.shape
        hidden = [x.new_zeros(batch_size, cell.hidden_size) for cell in self.cells]
        last_value = x.new_zeros(batch_size, features)
        last_observed_time = times[:, :1].expand(-1, features).clone()
        previous_time = times[:, 0]
        for step in range(steps):
            event_valid = valid[:, step]
            current_time = times[:, step]
            elapsed = (current_time[:, None] - last_observed_time).clamp_min(0)
            gamma = torch.exp(
                -F.relu(self.decay_weight[None] * elapsed + self.decay_bias[None])
            )
            imputed = torch.where(observed[:, step], x[:, step], gamma * last_value)
            gap = (current_time - previous_time).clamp_min(0)
            cell_input = torch.cat(
                [
                    imputed,
                    observed[:, step].to(x.dtype),
                    current_time[:, None],
                    gap[:, None],
                ],
                dim=-1,
            )
            for layer, cell in enumerate(self.cells):
                proposed = cell(cell_input, hidden[layer])
                hidden[layer] = torch.where(
                    event_valid[:, None], proposed, hidden[layer]
                )
                cell_input = self.dropout(hidden[layer])
            last_value = torch.where(
                observed[:, step] & event_valid[:, None], x[:, step], last_value
            )
            last_observed_time = torch.where(
                observed[:, step] & event_valid[:, None],
                current_time[:, None],
                last_observed_time,
            )
            previous_time = torch.where(event_valid, current_time, previous_time)
        return hidden[-1]

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        cognition = self.cognitive_summary(batch)
        horizon = self.horizon(batch["horizon"][:, None])
        available = batch["mri_available"]
        structural = self.mri(batch["mri_features"])
        structural = structural * available * torch.sigmoid(self.mri_gate)
        fused = self.fusion(torch.cat([cognition, horizon, structural, available], dim=-1))
        return self.output(fused)


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def build_anchor_samples(
    frame: pd.DataFrame,
    split: str,
    features: list[str],
    targets: list[str],
    config: dict[str, Any],
) -> tuple[list[AnchorSample], pd.DataFrame]:
    forecast = config["future_forecast"]
    tolerances = {int(key): int(value) for key, value in forecast["tolerance_months"].items()}
    base_samples = build_samples(
        frame,
        split,
        features,
        targets,
        forecast["horizons_months"],
        tolerances,
    )
    eligible = frame[frame.date_window_verified.fillna(False)].copy()
    eligible = eligible[
        eligible.scan_id.notna()
        & eligible.image_path.notna()
        & eligible.mask_path.notna()
        & eligible.scan_date.notna()
    ]
    if eligible.duplicated(["subject_id", "visit_month"]).any():
        raise RuntimeError("MRI cutoff anchor is not unique by subject and visit month")
    anchors = {
        (str(row.subject_id), float(row.visit_month)): row
        for row in eligible.itertuples(index=False)
    }
    visit_rows = {
        (str(row.subject_id), float(row.visit_month)): row
        for row in frame.itertuples(index=False)
    }
    retained: list[AnchorSample] = []
    manifest: list[dict[str, Any]] = []
    for base in base_samples:
        anchor = anchors.get((str(base.subject_id), float(base.cutoff_month)))
        if anchor is None:
            continue
        scan_date = pd.to_datetime(anchor.scan_date, errors="coerce")
        if pd.isna(scan_date):
            continue
        target_mask = base.target_mask.copy()
        for index, target in enumerate(targets):
            if not target_mask[index]:
                continue
            target_row = visit_rows.get(
                (str(base.subject_id), float(base.target_months[index]))
            )
            date_column = TARGET_DATE_COLUMNS[target]
            target_date = pd.to_datetime(
                getattr(target_row, date_column, None) if target_row is not None else None,
                errors="coerce",
            )
            if pd.isna(target_date) or target_date <= scan_date:
                target_mask[index] = False
        if not target_mask.any():
            continue
        base = replace(base, target_mask=target_mask)
        anchor_key = f"{base.subject_id}:m{base.cutoff_month:g}"
        retained.append(
            AnchorSample(
                base=base,
                anchor_key=anchor_key,
                scan_id=str(anchor.scan_id),
                source_scan_id=str(anchor.scan_id),
                mri_features=np.empty(0, dtype=np.float32),
                mri_available=1.0,
            )
        )
        manifest.append(
            {
                "anchor_key": anchor_key,
                "sample_id": base.sample_id,
                "subject_id": base.subject_id,
                "split": split,
                "cutoff_month": base.cutoff_month,
                "horizon_months": base.horizon_months,
                "scan_id": str(anchor.scan_id),
                "image_path": str(anchor.image_path),
                "mask_path": str(anchor.mask_path),
                "scan_date": scan_date.date().isoformat(),
                "date_difference_days": float(anchor.date_difference_days),
                "n_targets": int(target_mask.sum()),
            }
        )
    if not retained:
        raise RuntimeError(f"No real-date MRI forecasting samples for {split}")
    return retained, pd.DataFrame(manifest)


def _extract_one(task: tuple[dict[str, Any], tuple[int, int, int], dict[str, float]]):
    record, grid, qc = task
    torch.set_num_threads(1)
    output = dict(record)
    try:
        image = nib.as_closest_canonical(nib.load(record["image_path"]))
        mask_image = nib.as_closest_canonical(nib.load(record["mask_path"]))
        if len(image.shape) != 3 or len(mask_image.shape) != 3:
            raise ValueError("image_or_mask_not_3d")
        if image.shape != mask_image.shape:
            raise ValueError("image_mask_shape_mismatch")
        if not np.allclose(image.affine, mask_image.affine, atol=1e-3):
            raise ValueError("image_mask_affine_mismatch")
        array = np.asarray(image.dataobj, dtype=np.float32)
        mask = np.asarray(mask_image.dataobj, dtype=np.float32) > 0.5
        if not np.isfinite(array).all():
            raise ValueError("nonfinite_image")
        voxels = int(mask.sum())
        fraction = float(mask.mean())
        if voxels < int(qc["min_mask_voxels"]):
            raise ValueError("mask_too_small")
        if not float(qc["min_mask_fraction"]) <= fraction <= float(qc["max_mask_fraction"]):
            raise ValueError("mask_fraction_out_of_bounds")
        foreground = array[mask]
        foreground_std = float(foreground.std())
        if not np.isfinite(foreground_std) or foreground_std < float(qc["min_foreground_std"]):
            raise ValueError("empty_or_constant_foreground")
        normalized = np.zeros_like(array, dtype=np.float32)
        normalized[mask] = (foreground - float(foreground.mean())) / foreground_std
        coords = np.argwhere(mask).astype(np.float32)
        denominators = np.maximum(np.asarray(mask.shape, dtype=np.float32) - 1, 1)
        unit = coords / denominators[None]
        centroid = unit.mean(0)
        spread = unit.std(0)
        extent = unit.max(0) - unit.min(0)
        asymmetry = []
        for axis in range(3):
            half = mask.shape[axis] // 2
            left = np.take(mask, np.arange(half), axis=axis).mean()
            right = np.take(mask, np.arange(mask.shape[axis] - half, mask.shape[axis]), axis=axis).mean()
            asymmetry.append(float(left - right))
        voxel_volume_ml = abs(float(np.linalg.det(image.affine[:3, :3]))) / 1000.0
        scalars = np.asarray(
            [math.log1p(voxels * voxel_volume_ml), fraction]
            + centroid.tolist()
            + spread.tolist()
            + extent.tolist()
            + asymmetry,
            dtype=np.float32,
        )
        mask_tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
        value_tensor = torch.from_numpy(normalized)[None, None]
        occupancy = F.adaptive_avg_pool3d(mask_tensor, grid)
        pooled_sum = F.adaptive_avg_pool3d(value_tensor * mask_tensor, grid)
        pooled_intensity = torch.where(
            occupancy > 1e-6, pooled_sum / occupancy.clamp_min(1e-6), torch.zeros_like(pooled_sum)
        )
        vector = np.concatenate(
            [
                scalars,
                occupancy.flatten().numpy().astype(np.float32),
                pooled_intensity.flatten().numpy().astype(np.float32),
            ]
        )
        if not np.isfinite(vector).all():
            raise ValueError("nonfinite_features")
        output.update(
            project_qc_pass=True,
            qc_failure=None,
            image_shape="x".join(str(value) for value in image.shape),
            spacing="x".join(f"{value:.4g}" for value in image.header.get_zooms()[:3]),
            mask_voxels=voxels,
            mask_fraction=fraction,
            brain_volume_ml=voxels * voxel_volume_ml,
            feature_dim=len(vector),
        )
        return output, vector
    except Exception as error:
        output.update(
            project_qc_pass=False,
            qc_failure=f"{type(error).__name__}:{error}",
            image_shape=None,
            spacing=None,
            mask_voxels=None,
            mask_fraction=None,
            brain_volume_ml=None,
            feature_dim=None,
        )
        return output, None


def extract_features(
    manifest: pd.DataFrame,
    config: dict[str, Any],
    output: Path,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    feature_path = output / "mri_features.npz"
    qc_path = output / "project_qc.csv"
    unique = manifest.drop_duplicates("scan_id")[
        ["scan_id", "subject_id", "split", "image_path", "mask_path", "scan_date"]
    ].sort_values("scan_id")
    required_ids = unique.scan_id.astype(str).tolist()
    if feature_path.exists() and qc_path.exists():
        archive = np.load(feature_path, allow_pickle=False)
        cached_ids = archive["scan_ids"].astype(str).tolist()
        if cached_ids == required_ids:
            matrix = archive["features"].astype(np.float32)
            return dict(zip(cached_ids, matrix)), read_csv(qc_path)
    records = unique.to_dict("records")
    grid = tuple(int(value) for value in config["mri"]["pooled_grid"])
    qc = config["mri"]["technical_qc"]
    results: dict[str, np.ndarray] = {}
    audit = []
    tasks = [(record, grid, qc) for record in records]
    workers = int(config["mri"]["extraction_workers"])
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_extract_one, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            row, vector = future.result()
            audit.append(row)
            if vector is not None:
                results[str(row["scan_id"])] = vector
            if index % 250 == 0 or index == len(futures):
                print(json.dumps({"mri_features_completed": index, "total": len(futures)}), flush=True)
    audit_frame = pd.DataFrame(audit).sort_values("scan_id").reset_index(drop=True)
    save_csv(audit_frame, qc_path)
    passed_ids = [scan_id for scan_id in required_ids if scan_id in results]
    if not passed_ids:
        raise RuntimeError("No MRI passed project technical QC")
    np.savez_compressed(
        feature_path,
        scan_ids=np.asarray(passed_ids),
        features=np.stack([results[scan_id] for scan_id in passed_ids]).astype(np.float32),
    )
    return results, audit_frame


def fit_mri_normalization(
    samples: list[AnchorSample], features: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    scan_ids = sorted({sample.scan_id for sample in samples})
    matrix = np.stack([features[scan_id] for scan_id in scan_ids])
    mean = matrix.mean(0).astype(np.float32)
    std = matrix.std(0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def shuffle_maps(
    samples: list[AnchorSample], seed: int
) -> tuple[dict[str, str], dict[str, str], dict[str, float]]:
    anchors = {}
    for sample in samples:
        anchors.setdefault(
            sample.anchor_key,
            {"subject_id": sample.base.subject_id, "scan_id": sample.scan_id},
        )
    keys = sorted(anchors)
    rng = np.random.default_rng(seed)
    subject_map = {}
    for key in keys:
        candidates = [other for other in keys if anchors[other]["subject_id"] != anchors[key]["subject_id"]]
        if not candidates:
            raise RuntimeError("Cannot build shuffled-subject control")
        donor = candidates[int(rng.integers(0, len(candidates)))]
        subject_map[key] = anchors[donor]["scan_id"]
    scans_by_subject: dict[str, list[str]] = {}
    for value in anchors.values():
        scans_by_subject.setdefault(value["subject_id"], []).append(value["scan_id"])
    scans_by_subject = {
        subject: sorted(set(scan_ids)) for subject, scan_ids in scans_by_subject.items()
    }
    time_map = {}
    for key in keys:
        current = anchors[key]["scan_id"]
        scans = scans_by_subject[anchors[key]["subject_id"]]
        if len(scans) == 1:
            time_map[key] = current
        else:
            time_map[key] = scans[(scans.index(current) + 1) % len(scans)]
    changed_subject = np.mean([subject_map[key] != anchors[key]["scan_id"] for key in keys])
    changed_time = np.mean([time_map[key] != anchors[key]["scan_id"] for key in keys])
    return subject_map, time_map, {
        "unique_anchors": len(keys),
        "shuffled_subject_changed_fraction": float(changed_subject),
        "shuffled_time_changed_fraction": float(changed_time),
    }


def construct_arm(
    samples: list[AnchorSample],
    arm: str,
    features: dict[str, np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
    subject_map: dict[str, str],
    time_map: dict[str, str],
) -> list[AnchorSample]:
    output = []
    for sample in samples:
        if arm == "cognition_only":
            source = sample.scan_id
            vector = np.zeros_like(mean)
            available = 0.0
        elif arm == "observed_mri":
            source = sample.scan_id
            vector = (features[source] - mean) / std
            available = 1.0
        elif arm == "shuffled_subject_mri":
            source = subject_map[sample.anchor_key]
            vector = (features[source] - mean) / std
            available = 1.0
        elif arm == "shuffled_time_mri":
            source = time_map[sample.anchor_key]
            vector = (features[source] - mean) / std
            available = 1.0
        else:
            raise ValueError(f"Unknown arm: {arm}")
        output.append(
            replace(
                sample,
                source_scan_id=source,
                mri_features=vector.astype(np.float32),
                mri_available=available,
            )
        )
    return output


def evaluate(
    model: nn.Module,
    samples: list[AnchorSample],
    collator: AnchorCollator,
    batch_size: int,
    device: torch.device,
    targets: list[str],
    arm: str,
    seed: int,
) -> pd.DataFrame:
    loader = DataLoader(
        AnchorDataset(samples), batch_size=batch_size, shuffle=False, collate_fn=collator
    )
    model.eval()
    predictions, ordered = [], []
    mean = collator.numeric.target_mean[None]
    std = collator.numeric.target_std[None]
    with torch.no_grad():
        for batch in loader:
            anchors = batch.pop("anchor_samples")
            batch.pop("samples")
            batch = {key: value.to(device) for key, value in batch.items()}
            values = model(batch).cpu().numpy() * std + mean
            predictions.append(values)
            ordered.extend(sample.base for sample in anchors)
    return pd.DataFrame(
        prediction_rows(
            ordered,
            np.concatenate(predictions),
            arm,
            seed,
            targets,
        )
    )


def train_one(
    arm: str,
    seed: int,
    train_samples: list[AnchorSample],
    val_samples: list[AnchorSample],
    features: list[str],
    targets: list[str],
    normalization: dict[str, dict[str, float]],
    config: dict[str, Any],
    output: Path,
    device: torch.device,
) -> tuple[pd.DataFrame, int]:
    seed_everything(seed)
    settings = config["training"]
    collator = AnchorCollator(NumericCollator(features, targets, normalization))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        AnchorDataset(train_samples),
        batch_size=int(settings["batch_size"]),
        shuffle=True,
        collate_fn=collator,
        generator=generator,
        num_workers=0,
    )
    model = MRIAnchorForecaster(
        len(features),
        len(train_samples[0].mri_features),
        len(targets),
        int(settings["hidden_dim"]),
        int(settings["layers"]),
        float(settings["dropout"]),
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    best_score, best_state, patience = float("inf"), None, 0
    curves = []
    for epoch in range(int(settings["max_epochs"])):
        model.train()
        losses = []
        for batch in loader:
            batch.pop("anchor_samples")
            batch.pop("samples")
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            loss = masked_huber(prediction, batch["targets"], batch["target_mask"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss for {arm}/{seed}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(settings["gradient_clip"]))
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = evaluate(
            model,
            val_samples,
            collator,
            int(settings["batch_size"]),
            device,
            targets,
            arm,
            seed,
        )
        score = float(compute_metrics(validation, normalization).normalized_mae.mean())
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_macro_normalized_mae": score,
        }
        curves.append(record)
        print(json.dumps({"arm": arm, "seed": seed, **record}), flush=True)
        if score < best_score - 1e-5:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= int(settings["patience"]):
                break
    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {arm}/{seed}")
    model.load_state_dict(best_state)
    directory = output / "checkpoints" / arm / str(seed)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": best_state,
            "arm": arm,
            "seed": seed,
            "features": features,
            "targets": targets,
            "mri_feature_dim": len(train_samples[0].mri_features),
            "trainable_parameters": parameter_count,
            "validation_macro_normalized_mae": best_score,
        },
        directory / "best.pt",
    )
    save_csv(pd.DataFrame(curves), directory / "curves.csv")
    validation = evaluate(
        model,
        val_samples,
        collator,
        int(settings["batch_size"]),
        device,
        targets,
        arm,
        seed,
    )
    return validation, parameter_count


def ensemble_predictions(frame: pd.DataFrame, arm: str) -> pd.DataFrame:
    keys = [
        "sample_id",
        "subject_id",
        "split",
        "target",
        "horizon_months",
        "cutoff_month",
        "target_month",
        "truth",
    ]
    output = frame[frame.model.eq(arm)].groupby(keys, as_index=False).prediction.mean()
    output["model"] = arm + "_ensemble"
    output["seed"] = pd.NA
    return output


def bootstrap_difference(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    normalization: dict[str, dict[str, float]],
    replicates: int,
    seed: int,
    horizon: int | None,
) -> dict[str, Any]:
    if horizon is not None:
        candidate = candidate[candidate.horizon_months.eq(horizon)]
        baseline = baseline[baseline.horizon_months.eq(horizon)]
    keys = ["sample_id", "subject_id", "target", "horizon_months", "truth"]
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
    grouped = paired.groupby("subject_id").difference.agg(["sum", "count"])
    sums, counts = grouped["sum"].to_numpy(), grouped["count"].to_numpy()
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(grouped), size=(replicates, len(grouped)))
    draws = sums[indices].sum(1) / counts[indices].sum(1)
    return {
        "scope": "overall" if horizon is None else f"horizon_{horizon}",
        "horizon_months": horizon,
        "candidate": str(candidate.model.iloc[0]),
        "baseline": str(baseline.model.iloc[0]),
        "n_paired_targets": int(len(paired)),
        "n_subjects": int(len(grouped)),
        "normalized_mae_difference": float(paired.difference.mean()),
        "ci_lower": float(np.quantile(draws, 0.025)),
        "ci_upper": float(np.quantile(draws, 0.975)),
        "bootstrap_probability_candidate_better": float(np.mean(draws < 0)),
        "replicates": replicates,
    }


def main(config_path: Path, smoke: bool = False) -> None:
    config = load_config(config_path)
    run_id = config["run_id"] + ("_smoke" if smoke else "")
    output = ROOT / "outputs" / run_id
    output.mkdir(parents=True, exist_ok=True)
    dictionary = load_dictionary()
    baseline_config = load_config(ROOT / "configs/phase1_baseline_suite_formal.yaml")
    features = select_verified_scalar_features(baseline_config, dictionary)
    targets = list(config["targets"])
    train = clean_frame(read_csv(ROOT / "splits/train.csv"), features, targets, dictionary)
    validation = clean_frame(read_csv(ROOT / "splits/val.csv"), features, targets, dictionary)
    normalization = fit_normalization(train, features, targets)
    train_samples, train_manifest = build_anchor_samples(
        train, "train", features, targets, config
    )
    val_samples, val_manifest = build_anchor_samples(
        validation, "val", features, targets, config
    )
    if smoke:
        train_samples = train_samples[:512]
        val_samples = val_samples[:128]
        train_ids = {sample.base.sample_id for sample in train_samples}
        val_ids = {sample.base.sample_id for sample in val_samples}
        train_manifest = train_manifest[train_manifest.sample_id.isin(train_ids)]
        val_manifest = val_manifest[val_manifest.sample_id.isin(val_ids)]
        config = copy.deepcopy(config)
        config["neural_seeds"] = config["neural_seeds"][:1]
        config["training"]["max_epochs"] = 2
        config["training"]["patience"] = 2
        config["statistics"]["bootstrap_replicates"] = 100
    manifest = pd.concat([train_manifest, val_manifest], ignore_index=True)
    if set(manifest.split) != {"train", "val"}:
        raise RuntimeError("MRI development manifest must contain train and validation only")
    save_csv(manifest, output / "sample_manifest.csv")
    raw_features, qc = extract_features(manifest, config, output)
    pass_ids = set(qc.loc[qc.project_qc_pass, "scan_id"].astype(str))
    train_samples = [sample for sample in train_samples if sample.scan_id in pass_ids]
    val_samples = [sample for sample in val_samples if sample.scan_id in pass_ids]
    if not train_samples or not val_samples:
        raise RuntimeError("Project QC removed every sample from a split")
    mean, std = fit_mri_normalization(train_samples, raw_features)
    np.savez_compressed(output / "mri_normalization.npz", mean=mean, std=std)
    train_subject_map, train_time_map, train_shuffle = shuffle_maps(
        train_samples, int(config["mri"]["shuffle_seed"])
    )
    val_subject_map, val_time_map, val_shuffle = shuffle_maps(
        val_samples, int(config["mri"]["shuffle_seed"]) + 1
    )
    arms = config["arms"]
    arm_samples = {}
    for arm in arms:
        arm_samples[(arm, "train")] = construct_arm(
            train_samples, arm, raw_features, mean, std, train_subject_map, train_time_map
        )
        arm_samples[(arm, "val")] = construct_arm(
            val_samples, arm, raw_features, mean, std, val_subject_map, val_time_map
        )
    lengths = {
        split: {len(arm_samples[(arm, split)]) for arm in arms}
        for split in ["train", "val"]
    }
    if any(len(values) != 1 for values in lengths.values()):
        raise RuntimeError("Comparison arms do not use identical sample counts")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions, parameter_counts = [], {}
    for arm in arms:
        for seed in config["neural_seeds"]:
            frame, count = train_one(
                arm,
                int(seed),
                arm_samples[(arm, "train")],
                arm_samples[(arm, "val")],
                features,
                targets,
                normalization,
                config,
                output,
                device,
            )
            predictions.append(frame)
            parameter_counts[arm] = count
    seed_predictions = pd.concat(predictions, ignore_index=True)
    ensembles = [ensemble_predictions(seed_predictions, arm) for arm in arms]
    ensemble = pd.concat(ensembles, ignore_index=True)
    all_predictions = pd.concat([seed_predictions, ensemble], ignore_index=True)
    save_csv(all_predictions, output / "validation_predictions.csv")
    metrics = compute_metrics(all_predictions, normalization)
    save_csv(metrics, output / "validation_metrics.csv")
    by_arm = {arm: ensemble[ensemble.model.eq(arm + "_ensemble")] for arm in arms}
    comparisons = []
    replicate_count = int(config["statistics"]["bootstrap_replicates"])
    bootstrap_seed = int(config["statistics"]["bootstrap_seed"])
    for baseline in ["cognition_only", "shuffled_subject_mri", "shuffled_time_mri"]:
        for horizon in [None, 12, 24]:
            comparisons.append(
                bootstrap_difference(
                    by_arm["observed_mri"],
                    by_arm[baseline],
                    normalization,
                    replicate_count,
                    bootstrap_seed + len(comparisons),
                    horizon,
                )
            )
    bootstrap = pd.DataFrame(comparisons)
    save_csv(bootstrap, output / "bootstrap_comparisons.csv")
    qc_rate = float(qc.project_qc_pass.mean())
    overall = bootstrap[bootstrap.scope.eq("overall")].set_index("baseline")
    cognition_name = "cognition_only_ensemble"
    subject_name = "shuffled_subject_mri_ensemble"
    time_name = "shuffled_time_mri_ensemble"
    horizon_rows = bootstrap[
        bootstrap.baseline.eq(cognition_name) & bootstrap.horizon_months.isin([12, 24])
    ]
    criteria = {
        "technical_qc_pass_rate": qc_rate >= float(config["gate"]["minimum_technical_qc_pass_rate"]),
        "observed_better_than_cognition": float(overall.loc[cognition_name, "ci_upper"]) < 0,
        "observed_better_than_shuffled_subject": float(overall.loc[subject_name, "ci_upper"]) < 0,
        "observed_better_than_shuffled_time": float(overall.loc[time_name, "ci_upper"]) < 0,
        "supported_12_or_24_month_improvement": bool((horizon_rows.ci_upper < 0).any()),
    }
    gate_passed = bool(all(criteria.values())) and not smoke
    decision = {
        "run_id": run_id,
        "engineering_smoke_only": smoke,
        "development_only": True,
        "test_evaluated": False,
        "candidate": "observed_mri_ensemble",
        "baseline": cognition_name,
        "criteria": criteria,
        "global_mri_anchor_gate_passed": gate_passed,
        "advance_to_spatial_grounding": gate_passed,
        "clinical_qc_claim": False,
        "next_action": "freeze_spatial_grounding_protocol" if gate_passed else "retain_cognition_primary_and_stop_current_mri_anchor",
    }
    save_json(decision, output / "development_decision.json")
    summary = (
        metrics[metrics.model.str.endswith("_ensemble")]
        .groupby("model", as_index=False)
        .agg(
            normalized_mae=("normalized_mae", "mean"),
            mae=("mae", "mean"),
            coverage=("coverage", "mean"),
        )
        .sort_values("normalized_mae")
    )
    report = f"""# Phase 3A global MRI-anchor development experiment

Status: completed validation-only development experiment. Test was not loaded.

## Contract

- Cognitive inputs: {len(features)} verified scalar items
- Outcomes: {targets}
- Train/validation samples after temporal checks and project QC: {len(train_samples)} / {len(val_samples)}
- Unique MRI processed: {qc.scan_id.nunique()}
- Project technical-QC pass rate: {qc_rate:.4f}
- Train/validation shuffled-time changed fractions: {train_shuffle['shuffled_time_changed_fraction']:.4f} / {val_shuffle['shuffled_time_changed_fraction']:.4f}
- Seeds: {config['neural_seeds']}
- Trainable parameters: {parameter_counts}

## Validation ensemble summary

{markdown_table(summary)}

## Subject-cluster bootstrap

Negative normalized-MAE differences favor observed MRI.

{markdown_table(bootstrap)}

## Development decision

```json
{json.dumps(decision, indent=2)}
```

This experiment evaluates global information value in the supplied preprocessed MRI. It does not establish clinical QC or regional anatomical grounding.
"""
    (output / "report.md").write_text(report)
    save_json(
        {
            "run_id": run_id,
            "config": str(config_path),
            "device": str(device),
            "features": features,
            "targets": targets,
            "sample_counts": {"train": len(train_samples), "val": len(val_samples)},
            "unique_mri": int(qc.scan_id.nunique()),
            "project_qc_pass_rate": qc_rate,
            "shuffle_diagnostics": {"train": train_shuffle, "val": val_shuffle},
            "parameter_counts": parameter_counts,
            "test_evaluated": False,
        },
        output / "run_manifest.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/phase3a_global_mri_anchor_dev.yaml",
    )
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    main(arguments.config, arguments.smoke)
