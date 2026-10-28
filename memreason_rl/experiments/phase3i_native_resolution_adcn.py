"""Native-resolution, train-only AD-vs-CN MRI sanity benchmark."""

from __future__ import annotations

import argparse
import copy
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from ..data.common import ROOT
from .phase1_numeric import seed_everything
from .phase3b_mri_representation import SFCNEncoder


CONFIG = ROOT / "configs/phase3i_native_resolution_adcn.yaml"


def load_config(path: Path = CONFIG) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def output_dir(config: dict[str, Any]) -> Path:
    path = ROOT / "outputs" / config["run_id"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_cohort(config: dict[str, Any]) -> pd.DataFrame:
    diagnostic = pd.read_csv(
        ROOT / "outputs" / config["source_diagnostic_run_id"] / "diagnostic_manifest.csv",
        low_memory=False,
    )
    frame = diagnostic[
        diagnostic.dxsum_diagnosis_label.isin(config["classes"])
    ].copy()
    frame["scan_date"] = pd.to_datetime(frame.scan_date, errors="raise")
    frame = (
        frame.sort_values(["subject_id", "scan_date", "scan_id"])
        .drop_duplicates("subject_id", keep="first")
        .reset_index(drop=True)
    )
    paths = pd.read_csv(
        ROOT / "outputs" / config["source_mri_run_id"] / "volume_index.csv",
        usecols=["scan_id", "image_path", "mask_path"],
    )
    frame = frame.drop(columns=["volume_index"], errors="ignore").merge(
        paths, on="scan_id", how="inner", validate="one_to_one"
    )
    label = {str(name): index for index, name in enumerate(config["classes"])}
    frame["label"] = frame.dxsum_diagnosis_label.map(label).astype(int)
    frame["native_volume_index"] = np.arange(len(frame), dtype=int)
    if frame.subject_id.duplicated().any():
        raise RuntimeError("Native-resolution cohort is not unique by subject")
    official_train = set(
        pd.read_csv(ROOT / "splits/train.csv", usecols=["subject_id"])
        .subject_id.astype(str)
    )
    if not set(frame.subject_id.astype(str)) <= official_train:
        raise RuntimeError("Non-training subject entered Phase 3I")
    if set(frame.dxsum_diagnosis_label) != set(config["classes"]):
        raise RuntimeError("Both frozen endpoint classes are required")
    return frame


def _cache_native_volume(task: tuple[int, str, str, str, tuple[int, int, int], tuple[float, float, float]]):
    index, image_path, mask_path, cache_path, shape, voxel_size = task
    try:
        image = nib.as_closest_canonical(nib.load(image_path))
        mask_image = nib.as_closest_canonical(nib.load(mask_path))
        actual_shape = tuple(int(value) for value in image.shape)
        mask_shape = tuple(int(value) for value in mask_image.shape)
        zooms = tuple(round(float(value), 5) for value in image.header.get_zooms()[:3])
        expected_zooms = tuple(round(float(value), 5) for value in voxel_size)
        if actual_shape != shape or mask_shape != shape:
            raise ValueError(f"shape image={actual_shape} mask={mask_shape} expected={shape}")
        if zooms != expected_zooms:
            raise ValueError(f"voxel_size={zooms} expected={expected_zooms}")
        array = np.asarray(image.dataobj, dtype=np.float32)
        mask = np.asarray(mask_image.dataobj, dtype=np.float32) > 0.5
        if not np.isfinite(array).all() or int(mask.sum()) < 100:
            raise ValueError("nonfinite_or_empty")
        foreground = array[mask]
        std = float(foreground.std())
        if std < 1e-6:
            raise ValueError("constant_foreground")
        normalized = np.zeros(shape, dtype=np.float32)
        normalized[mask] = np.clip(
            (foreground - float(foreground.mean())) / std, -5.0, 5.0
        )
        cache = np.load(cache_path, mmap_mode="r+")
        cache[index] = normalized.astype(np.float16)
        cache.flush()
        return index, actual_shape, zooms, None
    except Exception as error:
        return index, None, None, f"{type(error).__name__}:{error}"


def prepare_native_cache(
    frame: pd.DataFrame, config: dict[str, Any], output: Path
) -> tuple[Path, pd.DataFrame]:
    settings = config["native_volume"]
    shape = tuple(int(value) for value in settings["expected_shape"])
    voxel_size = tuple(float(value) for value in settings["expected_voxel_size_mm"])
    cache_path = output / "volumes_native_193x229x193_float16.npy"
    index_path = output / "native_volume_index.csv"
    index = frame[
        ["scan_id", "subject_id", "dxsum_diagnosis_label", "image_path", "mask_path", "native_volume_index"]
    ].copy()
    if cache_path.exists() and index_path.exists():
        old = pd.read_csv(index_path)
        cache = np.load(cache_path, mmap_mode="r")
        if (
            old.scan_id.astype(str).tolist() == index.scan_id.astype(str).tolist()
            and tuple(cache.shape) == (len(index),) + shape
        ):
            return cache_path, old
    temporary = output / "volumes_native_193x229x193_float16.tmp.npy"
    temporary.unlink(missing_ok=True)
    cache = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float16, shape=(len(index),) + shape
    )
    cache.flush()
    del cache
    tasks = [
        (
            int(row.native_volume_index), str(row.image_path), str(row.mask_path),
            str(temporary), shape, voxel_size,
        )
        for row in index.itertuples(index=False)
    ]
    failures = []
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
    index.to_csv(index_path, index=False)
    (output / "native_grid_audit.json").write_text(json.dumps({
        "n_scans": len(index),
        "observed_shapes": observed_shapes,
        "observed_voxel_sizes_mm": observed_zooms,
        "spatial_interpolation": False,
        "crop_or_pad": False,
        "storage_dtype": "float16",
    }, indent=2) + "\n")
    return cache_path, index


def make_folds(frame: pd.DataFrame, config: dict[str, Any], output: Path) -> pd.DataFrame:
    splitter = StratifiedKFold(
        n_splits=int(config["outer_folds"]), shuffle=True,
        random_state=int(config["fold_seed"]),
    )
    rows = []
    for fold, (_, test_indices) in enumerate(splitter.split(frame.subject_id, frame.label)):
        for index in test_indices:
            rows.append({
                "subject_id": str(frame.iloc[index].subject_id),
                "fold": fold,
                "label": int(frame.iloc[index].label),
            })
    folds = pd.DataFrame(rows)
    if folds.subject_id.duplicated().any() or len(folds) != len(frame):
        raise RuntimeError("Invalid Phase 3I subject folds")
    folds.to_csv(output / "subject_folds.csv", index=False)
    return folds


class NativeVolumeDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, indices: np.ndarray, cache_path: Path):
        self.frame = frame
        self.indices = np.asarray(indices, dtype=int)
        self.cache_path = cache_path
        self._cache = None

    @property
    def cache(self):
        if self._cache is None:
            self._cache = np.load(self.cache_path, mmap_mode="r")
        return self._cache

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        row = int(self.indices[item])
        volume_index = int(self.frame.iloc[row].native_volume_index)
        volume = np.asarray(self.cache[volume_index], dtype=np.float32)[None]
        return {
            "row": torch.tensor(row, dtype=torch.long),
            "volume": torch.from_numpy(volume),
            "label": torch.tensor(int(self.frame.iloc[row].label), dtype=torch.long),
        }


class NativeSFCNClassifier(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.encoder = SFCNEncoder(dropout)
        self.head = nn.Linear(128, 2)

    def forward(self, volume: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(volume)
        return self.head(embedding), embedding


def evaluate_network(
    model: nn.Module, dataset: NativeVolumeDataset, batch_size: int,
    workers: int, device: torch.device, amp: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=device.type == "cuda", persistent_workers=workers > 0,
    )
    model.eval()
    rows, probabilities, embeddings = [], [], []
    with torch.no_grad():
        for batch in loader:
            volume = batch["volume"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                logits, embedding = model(volume)
            rows.append(batch["row"].numpy())
            probabilities.append(torch.softmax(logits.float(), 1).cpu().numpy())
            embeddings.append(embedding.float().cpu().numpy())
    return np.concatenate(rows), np.concatenate(probabilities), np.concatenate(embeddings)


def fit_tabular(x: np.ndarray, y: np.ndarray, config: dict[str, Any]) -> Pipeline:
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=float(config["logistic"]["C"]),
            max_iter=int(config["logistic"]["max_iter"]),
            class_weight="balanced",
            solver="lbfgs",
        )),
    ])
    model.fit(x, y)
    return model


def select_temperature(
    model: Pipeline, x: np.ndarray, y: np.ndarray, grid: list[float]
) -> float:
    logits = model.decision_function(x)
    values = [(float(log_loss(y, expit(logits / float(t)), labels=[0, 1])), float(t)) for t in grid]
    return min(values)[1]


def predict_tabular(model: Pipeline, x: np.ndarray, temperature: float) -> np.ndarray:
    return expit(model.decision_function(x) / float(temperature))


def train_native_sfcn(
    fold: int, seed: int, frame: pd.DataFrame, fit_indices: np.ndarray,
    tune_indices: np.ndarray, cache_path: Path, config: dict[str, Any],
    output: Path, device: torch.device,
) -> tuple[np.ndarray, pd.DataFrame]:
    seed_everything(seed)
    settings = config["sfcn_training"]
    batch_size = int(settings["batch_size"])
    workers = int(settings["num_workers"])
    amp = bool(settings["amp"])
    fit = NativeVolumeDataset(frame, fit_indices, cache_path)
    tune = NativeVolumeDataset(frame, tune_indices, cache_path)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        fit, batch_size=batch_size, shuffle=True, generator=generator,
        num_workers=workers, pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    counts = np.bincount(frame.iloc[fit_indices].label.to_numpy(), minlength=2)
    class_weight = torch.tensor(counts.sum() / (2.0 * counts), dtype=torch.float32, device=device)
    model = NativeSFCNClassifier(float(settings["dropout"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp and device.type == "cuda")
    best_auc, best_state, patience = -np.inf, None, 0
    curves = []
    for epoch in range(int(settings["max_epochs"])):
        model.train()
        losses = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            volume = batch["volume"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                logits, _ = model(volume)
                loss = F.cross_entropy(logits.float(), label, weight=class_weight)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(settings["gradient_clip"]))
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        rows, probability, _ = evaluate_network(
            model, tune, batch_size, workers, device, amp
        )
        tune_auc = float(roc_auc_score(frame.iloc[rows].label, probability[:, 1]))
        record = {
            "fold": fold, "seed": seed, "epoch": epoch,
            "train_loss": float(np.mean(losses)), "tune_auc": tune_auc,
        }
        curves.append(record)
        print(json.dumps(record), flush=True)
        if tune_auc > best_auc + 1e-5:
            best_auc = tune_auc
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= int(settings["patience"]):
                break
    if best_state is None:
        raise RuntimeError("No native-resolution checkpoint selected")
    model.load_state_dict(best_state)
    checkpoint = output / "checkpoints" / f"fold{fold}" / f"seed{seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": best_state, "fold": fold, "seed": seed,
        "tune_auc": best_auc, "input_shape": config["native_volume"]["expected_shape"],
        "spatial_interpolation": False,
    }, checkpoint)
    all_rows = np.arange(len(frame))
    rows, _, embedding = evaluate_network(
        model, NativeVolumeDataset(frame, all_rows, cache_path),
        batch_size, workers, device, amp,
    )
    matrix = np.empty((len(frame), embedding.shape[1]), dtype=np.float32)
    matrix[rows] = embedding
    return matrix, pd.DataFrame(curves)


def metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = (probability >= threshold).astype(int)
    true_positive = int(((prediction == 1) & (y == 1)).sum())
    true_negative = int(((prediction == 0) & (y == 0)).sum())
    positive = int((y == 1).sum())
    negative = int((y == 0).sum())
    return {
        "ACC": float(accuracy_score(y, prediction)),
        "AUC": float(roc_auc_score(y, probability)),
        "SEN": true_positive / positive,
        "SEC": true_negative / negative,
        "F1": float(f1_score(y, prediction)),
    }


def run_crossfit(
    frame: pd.DataFrame, cache_path: Path, config: dict[str, Any], output: Path
) -> pd.DataFrame:
    folds = make_folds(frame, config, output)
    fold_lookup = folds.set_index("subject_id").fold
    frame["fold"] = frame.subject_id.astype(str).map(fold_lookup).astype(int)
    frame.drop(columns=["image_path", "mask_path"]).to_csv(output / "cohort_manifest.csv", index=False)
    demo = frame[["age", "sex_numeric"]].to_numpy(dtype=np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions, curves = [], []
    for fold in range(int(config["outer_folds"])):
        test_indices = np.flatnonzero(frame.fold.eq(fold))
        outer_indices = np.flatnonzero(frame.fold.ne(fold))
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=float(config["inner_tune_fraction"]),
            random_state=int(config["fold_seed"]) + fold,
        )
        fit_local, tune_local = next(splitter.split(outer_indices, frame.iloc[outer_indices].label))
        fit_indices = outer_indices[fit_local]
        tune_indices = outer_indices[tune_local]
        if set(frame.iloc[fit_indices].subject_id) & set(frame.iloc[test_indices].subject_id):
            raise RuntimeError("Outer-fold subject leakage")
        demo_model = fit_tabular(demo[fit_indices], frame.iloc[fit_indices].label.to_numpy(), config)
        demo_temperature = select_temperature(
            demo_model, demo[tune_indices], frame.iloc[tune_indices].label.to_numpy(),
            list(config["temperature_grid"]),
        )
        demo_probability = predict_tabular(demo_model, demo[test_indices], demo_temperature)
        for row, probability in zip(test_indices, demo_probability):
            predictions.append({
                "row_index": row, "subject_id": frame.iloc[row].subject_id,
                "scan_id": frame.iloc[row].scan_id, "label": int(frame.iloc[row].label),
                "arm": "demographics", "fold": fold, "seed": -1,
                "probability_AD": float(probability), "temperature": demo_temperature,
            })
        for seed in config["neural_seeds"]:
            embedding, curve = train_native_sfcn(
                fold, int(seed), frame, fit_indices, tune_indices, cache_path,
                config, output, device,
            )
            curves.append(curve)
            for arm, matrix in {
                "native_fullres_mri": embedding,
                "demographics_native_fullres_mri": np.concatenate([demo, embedding], axis=1),
            }.items():
                model = fit_tabular(
                    matrix[fit_indices], frame.iloc[fit_indices].label.to_numpy(), config
                )
                temperature = select_temperature(
                    model, matrix[tune_indices], frame.iloc[tune_indices].label.to_numpy(),
                    list(config["temperature_grid"]),
                )
                probability = predict_tabular(model, matrix[test_indices], temperature)
                for row, value in zip(test_indices, probability):
                    predictions.append({
                        "row_index": row, "subject_id": frame.iloc[row].subject_id,
                        "scan_id": frame.iloc[row].scan_id, "label": int(frame.iloc[row].label),
                        "arm": arm, "fold": fold, "seed": int(seed),
                        "probability_AD": float(value), "temperature": temperature,
                    })
        print(json.dumps({"completed_fold": fold, "test_subjects": len(test_indices)}), flush=True)
    pd.concat(curves, ignore_index=True).to_csv(output / "sfcn_training_curves.csv", index=False)
    seed_predictions = pd.DataFrame(predictions)
    seed_predictions.to_csv(output / "crossfit_seed_predictions.csv", index=False)
    ensemble = seed_predictions.groupby(
        ["row_index", "subject_id", "scan_id", "label", "arm", "fold"], as_index=False
    ).probability_AD.mean()
    ensemble.to_csv(output / "crossfit_predictions.csv", index=False)
    return ensemble


def summarize(
    frame: pd.DataFrame, predictions: pd.DataFrame,
    config: dict[str, Any], output: Path,
) -> None:
    threshold = float(config["classification_threshold"])
    metric_rows = []
    for arm, group in predictions.groupby("arm"):
        result = metrics(group.label.to_numpy(), group.probability_AD.to_numpy(), threshold)
        metric_rows.extend({"arm": arm, "metric": name, "value": value} for name, value in result.items())
    pd.DataFrame(metric_rows).to_csv(output / "classification_metrics.csv", index=False)

    rng = np.random.default_rng(int(config["statistics"]["bootstrap_seed"]))
    replicates = int(config["statistics"]["bootstrap_replicates"])
    bootstrap_rows = []
    arm_groups = {arm: group.sort_values("subject_id") for arm, group in predictions.groupby("arm")}
    reference_subjects = next(iter(arm_groups.values())).subject_id.astype(str).to_numpy()
    for arm, group in arm_groups.items():
        if not np.array_equal(group.subject_id.astype(str).to_numpy(), reference_subjects):
            raise RuntimeError("Arms do not contain identical subject order")
    samples = rng.integers(0, len(reference_subjects), size=(replicates, len(reference_subjects)))
    for arm, group in arm_groups.items():
        y = group.label.to_numpy()
        probability = group.probability_AD.to_numpy()
        point = metrics(y, probability, threshold)
        values = {name: [] for name in point}
        for sample in samples:
            if np.unique(y[sample]).size < 2:
                continue
            current = metrics(y[sample], probability[sample], threshold)
            for name, value in current.items():
                values[name].append(value)
        for name, point_value in point.items():
            bootstrap_rows.append({
                "arm": arm, "metric": name, "value": point_value,
                "ci_low": float(np.quantile(values[name], 0.025)),
                "ci_high": float(np.quantile(values[name], 0.975)),
            })
    intervals = pd.DataFrame(bootstrap_rows)
    intervals.to_csv(output / "classification_metric_intervals.csv", index=False)

    primary_arm = "demographics_native_fullres_mri"
    primary_auc = float(intervals[(intervals.arm == primary_arm) & (intervals.metric == "AUC")].value.iloc[0])
    passed = primary_auc >= float(config["stopping"]["minimum_native_mri_auc"])
    decision = {
        "native_resolution_mri_auc": primary_auc,
        "minimum_native_mri_auc": float(config["stopping"]["minimum_native_mri_auc"]),
        "native_resolution_sanity_gate_passed": bool(passed),
        "expand_to_three_seed_multiclass": bool(passed and config["stopping"]["expand_to_three_seed_multiclass_if_passed"]),
        "official_validation_loaded": False,
        "official_test_loaded": False,
    }
    (output / "development_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    verification = {
        "passed": True,
        "n_subjects": int(frame.subject_id.nunique()),
        "one_scan_per_subject": bool(len(frame) == frame.subject_id.nunique()),
        "classes": frame.dxsum_diagnosis_label.value_counts().to_dict(),
        "input_shape": config["native_volume"]["expected_shape"],
        "voxel_size_mm": config["native_volume"]["expected_voxel_size_mm"],
        "spatial_interpolation": False,
        "crop_or_pad": False,
        "all_subjects_original_train": True,
        "official_validation_loaded": False,
        "official_test_loaded": False,
    }
    (output / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")

    pivot = intervals.pivot(index="arm", columns="metric", values="value").reset_index()
    table = ["| Arm | ACC | AUC | SEN | SEC | F1 |", "|---|---:|---:|---:|---:|---:|"]
    for row in pivot.itertuples(index=False):
        table.append(
            f"| {row.arm} | {row.ACC:.4f} | {row.AUC:.4f} | {row.SEN:.4f} | {row.SEC:.4f} | {row.F1:.4f} |"
        )
    report = f"""# Phase 3I native-resolution AD-vs-CN MRI benchmark

This train-only, subject-wise five-fold experiment used one earliest eligible
MRI per subject. Every image retained the complete `193 x 229 x 193`, 1-mm
voxel grid. No interpolation, cropping, padding, or slice selection was used.
Float16 was used only for cache storage.

## Cohort

- Subjects/scans: {len(frame)}
- CN: {int((frame.dxsum_diagnosis_label == 'CN').sum())}
- AD: {int((frame.dxsum_diagnosis_label == 'AD').sum())}
- Official validation/test loaded: no/no

## Cross-fitted results

{chr(10).join(table)}

The prespecified native-MRI sanity threshold was AUC >=
{float(config['stopping']['minimum_native_mri_auc']):.2f}. The observed primary
MRI-plus-demographics AUC was {primary_auc:.4f}; gate passed: `{str(passed).lower()}`.
See `classification_metric_intervals.csv` for 2,000 subject-bootstrap confidence
intervals.
"""
    (output / "report.md").write_text(report)


def run(config_path: Path = CONFIG, prepare_only: bool = False) -> None:
    config = load_config(config_path)
    output = output_dir(config)
    (output / "resolved_config.json").write_text(json.dumps(config, indent=2) + "\n")
    frame = build_cohort(config)
    cache_path, _ = prepare_native_cache(frame, config, output)
    if prepare_only:
        return
    predictions = run_crossfit(frame, cache_path, config, output)
    summarize(frame, predictions, config, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    run(args.config, prepare_only=args.prepare_only)


if __name__ == "__main__":
    main()
