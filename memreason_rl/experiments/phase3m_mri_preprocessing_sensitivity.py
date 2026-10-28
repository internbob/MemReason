"""Matched MRI preprocessing sensitivity on the frozen Phase 3I AD/CN cohort."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from ..data.common import ROOT
from .phase1_numeric import seed_everything
from .phase3b_mri_representation import SFCNEncoder


CONFIG = ROOT / "configs/phase3m_mri_preprocessing_sensitivity.yaml"


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


class IndexedVolumeDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, rows: np.ndarray, cache_path: Path):
        self.frame = frame
        self.rows = np.asarray(rows, dtype=int)
        self.cache_path = cache_path
        self._cache = None

    @property
    def cache(self):
        if self._cache is None:
            self._cache = np.load(self.cache_path, mmap_mode="r")
        return self._cache

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        row = int(self.rows[item])
        index = int(self.frame.iloc[row].volume_index)
        volume = np.asarray(self.cache[index], dtype=np.float32)[None]
        return {
            "row": torch.tensor(row, dtype=torch.long),
            "volume": torch.from_numpy(volume),
            "label": torch.tensor(int(self.frame.iloc[row].label), dtype=torch.long),
        }


class SFCNClassifier(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.encoder = SFCNEncoder(dropout)
        self.head = nn.Linear(128, 2)

    def forward(self, volume: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(volume)
        return self.head(embedding), embedding


def evaluate(model: nn.Module, dataset: Dataset, batch_size: int, workers: int,
             device: torch.device, amp: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=device.type == "cuda", persistent_workers=False)
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
        ("classifier", LogisticRegression(C=float(config["logistic"]["C"]),
                                          max_iter=int(config["logistic"]["max_iter"]),
                                          class_weight="balanced", solver="lbfgs")),
    ])
    model.fit(x, y)
    return model


def select_temperature(model: Pipeline, x: np.ndarray, y: np.ndarray,
                       grid: list[float]) -> float:
    logits = model.decision_function(x)
    return min((float(log_loss(y, expit(logits / float(t)), labels=[0, 1])), float(t))
               for t in grid)[1]


def prepare_frame(config: dict[str, Any], representation: str) -> pd.DataFrame:
    source = ROOT / "outputs" / config["source_fullres_run_id"]
    frame = pd.read_csv(source / "cohort_manifest.csv", low_memory=False)
    frozen_folds = pd.read_csv(source / "subject_folds.csv", dtype={"subject_id": str})
    frame["subject_id"] = frame.subject_id.astype(str)
    frozen_folds["subject_id"] = frozen_folds.subject_id.astype(str)
    fold_map = frozen_folds.set_index("subject_id").fold
    expected = frame.subject_id.map(fold_map)
    if expected.isna().any() or not np.array_equal(expected.astype(int), frame.fold.astype(int)):
        raise RuntimeError("Phase 3I folds do not match the frozen cohort")
    index = pd.read_csv(resolve(config["representations"][representation]["cache_index"]),
                        usecols=["scan_id", "volume_index"])
    if index.scan_id.duplicated().any():
        raise RuntimeError("MRI cache index contains duplicate scan IDs")
    frame = frame.drop(columns=["native_volume_index"], errors="ignore").merge(
        index, on="scan_id", how="left", validate="one_to_one")
    if frame.volume_index.isna().any() or len(frame) != 896:
        raise RuntimeError("Representation does not cover all 896 frozen scans")
    frame["volume_index"] = frame.volume_index.astype(int)
    return frame.reset_index(drop=True)


def train_fold(frame: pd.DataFrame, fit_rows: np.ndarray, tune_rows: np.ndarray,
               cache_path: Path, config: dict[str, Any], representation: str,
               fold: int, output: Path, device: torch.device) -> tuple[np.ndarray, pd.DataFrame, dict]:
    settings = config["sfcn_training"]
    seed = int(config["neural_seed"])
    seed_everything(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    fit_data = IndexedVolumeDataset(frame, fit_rows, cache_path)
    tune_data = IndexedVolumeDataset(frame, tune_rows, cache_path)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(fit_data, batch_size=int(settings["batch_size"]), shuffle=True,
                        generator=generator, num_workers=int(settings["num_workers"]),
                        pin_memory=device.type == "cuda", persistent_workers=False)
    counts = np.bincount(frame.iloc[fit_rows].label.to_numpy(), minlength=2)
    class_weight = torch.tensor(counts.sum() / (2.0 * counts), dtype=torch.float32, device=device)
    model = SFCNClassifier(float(settings["dropout"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["learning_rate"]),
                                  weight_decay=float(settings["weight_decay"]))
    amp = bool(settings["amp"])
    scaler = torch.amp.GradScaler(device.type, enabled=amp and device.type == "cuda")
    best_auc, best_state, patience = -np.inf, None, 0
    curve = []
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
        rows, probability, _ = evaluate(model, tune_data, int(settings["batch_size"]),
                                        int(settings["num_workers"]), device, amp)
        tune_auc = float(roc_auc_score(frame.iloc[rows].label, probability[:, 1]))
        record = {"representation": representation, "fold": fold, "seed": seed,
                  "epoch": epoch, "train_loss": float(np.mean(losses)), "tune_auc": tune_auc}
        curve.append(record)
        print(json.dumps(record), flush=True)
        if tune_auc > best_auc + 1e-5:
            best_auc, best_state, patience = tune_auc, copy.deepcopy(model.state_dict()), 0
        else:
            patience += 1
            if patience >= int(settings["patience"]):
                break
    if best_state is None:
        raise RuntimeError("No SFCN checkpoint selected")
    model.load_state_dict(best_state)
    checkpoint = output / "checkpoints" / representation / f"fold{fold}_seed{seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "fold": fold, "seed": seed, "tune_auc": best_auc,
                "representation": representation,
                "input_shape": config["representations"][representation]["shape"]}, checkpoint)
    rows, _, embedding = evaluate(model, IndexedVolumeDataset(frame, np.arange(len(frame)), cache_path),
                                  int(settings["batch_size"]), int(settings["num_workers"]), device, amp)
    matrix = np.empty((len(frame), embedding.shape[1]), dtype=np.float32)
    matrix[rows] = embedding
    elapsed = time.perf_counter() - started
    resource = {"representation": representation, "fold": fold, "elapsed_seconds": elapsed,
                "epochs": len(curve), "best_tune_auc": best_auc,
                "peak_cuda_memory_mb": (torch.cuda.max_memory_allocated(device) / 2**20
                                        if device.type == "cuda" else 0.0)}
    return matrix, pd.DataFrame(curve), resource


def run_branch(config_path: Path, representation: str, cache_override: Path | None) -> None:
    config = load_config(config_path)
    if representation not in config["representations"]:
        raise ValueError(f"Unknown representation: {representation}")
    output = ROOT / "outputs" / config["run_id"]
    output.mkdir(parents=True, exist_ok=True)
    frame = prepare_frame(config, representation)
    cache_path = cache_override or resolve(config["representations"][representation]["cache_path"])
    cache = np.load(cache_path, mmap_mode="r")
    expected_shape = tuple(config["representations"][representation]["shape"])
    if tuple(cache.shape[1:]) != expected_shape or frame.volume_index.max() >= len(cache):
        raise RuntimeError(f"Invalid cache {cache.shape}; expected (*, {expected_shape})")
    del cache
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    demo = frame[["age", "sex_numeric"]].to_numpy(dtype=np.float32)
    predictions, curves, resources = [], [], []
    for fold in range(int(config["outer_folds"])):
        test_rows = np.flatnonzero(frame.fold.eq(fold))
        outer_rows = np.flatnonzero(frame.fold.ne(fold))
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=float(config["inner_tune_fraction"]),
                                          random_state=int(config["fold_seed"]) + fold)
        fit_local, tune_local = next(splitter.split(outer_rows, frame.iloc[outer_rows].label))
        fit_rows, tune_rows = outer_rows[fit_local], outer_rows[tune_local]
        embedding, curve, resource = train_fold(frame, fit_rows, tune_rows, cache_path, config,
                                                representation, fold, output, device)
        curves.append(curve)
        resources.append(resource)
        matrices = {f"{representation}_mri": embedding,
                    f"demographics_{representation}_mri": np.concatenate([demo, embedding], axis=1)}
        for arm, matrix in matrices.items():
            model = fit_tabular(matrix[fit_rows], frame.iloc[fit_rows].label.to_numpy(), config)
            temperature = select_temperature(model, matrix[tune_rows],
                                             frame.iloc[tune_rows].label.to_numpy(),
                                             list(config["temperature_grid"]))
            probability = expit(model.decision_function(matrix[test_rows]) / temperature)
            for row, value in zip(test_rows, probability):
                predictions.append({"row_index": int(row), "subject_id": frame.iloc[row].subject_id,
                                    "scan_id": frame.iloc[row].scan_id, "label": int(frame.iloc[row].label),
                                    "arm": arm, "fold": fold, "seed": int(config["neural_seed"]),
                                    "probability_AD": float(value), "temperature": temperature})
        print(json.dumps({"representation": representation, "completed_fold": fold,
                          "test_subjects": len(test_rows)}), flush=True)
    pd.concat(curves, ignore_index=True).to_csv(output / f"{representation}_training_curves.csv", index=False)
    pd.DataFrame(resources).to_csv(output / f"{representation}_resource_usage.csv", index=False)
    pd.DataFrame(predictions).to_csv(output / f"{representation}_crossfit_predictions.csv", index=False)
    branch = {"representation": representation, "n_subjects": len(frame),
              "input_shape": list(expected_shape), "cache_path_used": str(cache_path),
              "frozen_phase3i_folds": True, "official_validation_loaded": False,
              "official_test_loaded": False}
    (output / f"{representation}_verification.json").write_text(json.dumps(branch, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--representation", choices=["center_crop", "lowres"], required=True)
    parser.add_argument("--cache-override", type=Path)
    args = parser.parse_args()
    run_branch(args.config, args.representation, args.cache_override)


if __name__ == "__main__":
    main()
