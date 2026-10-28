"""Train-only subject-cross-fitted MRI diagnostic incremental-value experiment."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.special import softmax
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from ..data.common import ROOT
from .phase1_numeric import markdown_table, seed_everything
from .phase3b_mri_representation import SFCNEncoder


CONFIG = ROOT / "configs/phase3g_mri_diagnosis_crossfit.yaml"


def load_config(path: Path = CONFIG) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def output_dir(config: dict[str, Any]) -> Path:
    path = ROOT / "outputs" / config["run_id"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_frame(config: dict[str, Any]) -> tuple[pd.DataFrame, Path]:
    source = ROOT / "outputs" / config["source_mri_run_id"]
    manifest = pd.read_csv(source / "concurrent_manifest.csv", low_memory=False)
    manifest = manifest[manifest.split.eq("train")].copy()
    clinical = pd.read_csv(ROOT / "splits/train.csv", low_memory=False)
    diagnosis_columns = [
        "subject_id", "visit_month", "dxsum_diagnosis_label", "dxsum_visit_date", "dxsum_phase"
    ]
    frame = manifest.merge(
        clinical[diagnosis_columns],
        left_on=["subject_id", "cutoff_month"],
        right_on=["subject_id", "visit_month"],
        how="left",
        validate="one_to_one",
    )
    frame["scan_date"] = pd.to_datetime(frame.scan_date, errors="coerce")
    frame["dxsum_visit_date"] = pd.to_datetime(frame.dxsum_visit_date, errors="coerce")
    frame["diagnosis_date_difference_days"] = (
        frame.dxsum_visit_date - frame.scan_date
    ).dt.days.abs()
    frame = frame[
        frame.dxsum_diagnosis_label.isin(config["classes"])
        & frame.diagnosis_date_difference_days.le(int(config["date_window_days"]))
    ].copy()
    label = {name: index for index, name in enumerate(config["classes"])}
    frame["label"] = frame.dxsum_diagnosis_label.map(label).astype(int)
    frame["sex_numeric"] = frame.sex.astype(str).str.upper().map({"M": 0.0, "F": 1.0})
    frame["age"] = pd.to_numeric(frame.age, errors="coerce")
    frame = frame[frame.age.notna() & frame.sex_numeric.notna()].copy()
    cache_settings = config.get("volume_cache")
    if cache_settings:
        cache_index = pd.read_csv(ROOT / cache_settings["index"])[
            ["scan_id", "volume_index"]
        ]
        frame = frame.drop(columns=["volume_index"], errors="ignore").merge(
            cache_index, on="scan_id", how="inner", validate="one_to_one"
        )
        durable_path = ROOT / cache_settings["path"]
        runtime_path = Path(cache_settings.get("runtime_path", durable_path))
        volume_path = runtime_path if runtime_path.exists() else durable_path
    else:
        volume_index = pd.read_csv(source / "volume_index.csv")[["scan_id", "volume_index"]]
        frame = frame.merge(volume_index, on="scan_id", how="inner", validate="one_to_one")
        volume_path = source / "volumes_48x56x48.npy"
    if frame.scan_id.duplicated().any():
        raise RuntimeError("Diagnostic frame is not unique by scan")
    official_train = set(pd.read_csv(ROOT / "splits/train.csv", usecols=["subject_id"]).subject_id.astype(str))
    if not set(frame.subject_id.astype(str)) <= official_train:
        raise RuntimeError("Non-train subject entered Phase 3G")
    return frame.sort_values(["subject_id", "cutoff_month"]).reset_index(drop=True), volume_path


def global_features(frame: pd.DataFrame) -> np.ndarray:
    archive = np.load(
        ROOT / "outputs/phase3a_global_mri_anchor_dev_20261013/mri_features.npz",
        allow_pickle=False,
    )
    scan_ids = archive["scan_ids"].astype(str)
    features = archive["features"].astype(np.float32)
    lookup = {str(scan): index for index, scan in enumerate(scan_ids)}
    return np.stack([features[lookup[str(scan)]] for scan in frame.scan_id]).astype(np.float32)


def cognitive_history_features(frame: pd.DataFrame, targets: list[str]) -> tuple[np.ndarray, list[str]]:
    clinical = pd.read_csv(ROOT / "splits/train.csv", low_memory=False)
    clinical["visit_month"] = pd.to_numeric(clinical.visit_month, errors="coerce")
    for target in targets:
        clinical[target] = pd.to_numeric(clinical[target], errors="coerce")
    by_subject = {}
    for subject, group in clinical.groupby("subject_id", sort=False):
        group = group.sort_values("visit_month")
        by_subject[str(subject)] = (
            group.visit_month.to_numpy(dtype=float),
            group[targets].to_numpy(dtype=float),
        )
    names = []
    for target in targets:
        names.extend([
            f"{target}_last", f"{target}_mean", f"{target}_slope",
            f"{target}_log_count", f"{target}_recency", f"{target}_available",
        ])
    rows = []
    missing_fraction = []
    for record in frame.itertuples(index=False):
        months, values = by_subject[str(record.subject_id)]
        history_mask = months <= float(record.cutoff_month) + 1e-6
        features = []
        available_targets = 0
        for target_index, target in enumerate(targets):
            observed_mask = history_mask & np.isfinite(values[:, target_index])
            observed_months = months[observed_mask]
            observed_values = values[observed_mask, target_index]
            count = len(observed_values)
            if count:
                available_targets += 1
                slope = float(np.polyfit(observed_months, observed_values, 1)[0]) \
                    if len(np.unique(observed_months)) >= 2 else 0.0
                features.extend([
                    float(observed_values[-1]), float(observed_values.mean()), slope,
                    float(np.log1p(count)), float(record.cutoff_month - observed_months[-1]), 1.0,
                ])
            else:
                features.extend([np.nan, np.nan, np.nan, 0.0, np.nan, 0.0])
        rows.append(features)
        missing_fraction.append(1.0 - available_targets / len(targets))
    frame["cognitive_missing_fraction"] = missing_fraction
    return np.asarray(rows, dtype=np.float32), names


def subject_weights(frame: pd.DataFrame) -> np.ndarray:
    count = frame.groupby("subject_id")["subject_id"].transform("size").to_numpy(dtype=float)
    weights = 1.0 / count
    return weights / weights.mean()


def make_folds(frame: pd.DataFrame, config: dict[str, Any], output: Path) -> pd.DataFrame:
    if config.get("reuse_subject_folds"):
        folds = pd.read_csv(ROOT / config["reuse_subject_folds"])
        if set(folds.subject_id.astype(str)) != set(frame.subject_id.astype(str)):
            raise RuntimeError("Reused diagnostic folds do not match the current cohort")
        folds.to_csv(output / "subject_folds.csv", index=False)
        return folds
    subject_label = frame.groupby("subject_id").label.agg(
        lambda values: int(values.value_counts().sort_index().idxmax())
    )
    subjects = subject_label.index.to_numpy()
    labels = subject_label.to_numpy()
    splitter = StratifiedKFold(
        n_splits=int(config["outer_folds"]), shuffle=True, random_state=int(config["fold_seed"])
    )
    rows = []
    for fold, (_, test_index) in enumerate(splitter.split(subjects, labels)):
        for index in test_index:
            rows.append({"subject_id": subjects[index], "fold": fold, "stratification_label": labels[index]})
    folds = pd.DataFrame(rows)
    if folds.subject_id.duplicated().any() or folds.subject_id.nunique() != frame.subject_id.nunique():
        raise RuntimeError("Invalid subject folds")
    folds.to_csv(output / "subject_folds.csv", index=False)
    return folds


class VolumeDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, indices: np.ndarray, volume_path: Path, weights: np.ndarray):
        self.frame = frame
        self.indices = np.asarray(indices, dtype=int)
        self.volume_path = volume_path
        self.weights = weights
        self._volume = None

    @property
    def volumes(self):
        if self._volume is None:
            self._volume = np.load(self.volume_path, mmap_mode="r")
        return self._volume

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int):
        row = int(self.indices[item])
        volume_index = int(self.frame.iloc[row].volume_index)
        volume = np.asarray(self.volumes[volume_index], dtype=np.float32)[None]
        return {
            "row": row,
            "volume": torch.from_numpy(volume),
            "label": torch.tensor(int(self.frame.iloc[row].label), dtype=torch.long),
            "weight": torch.tensor(float(self.weights[row]), dtype=torch.float32),
        }


class SFCNClassifier(nn.Module):
    def __init__(self, classes: int, dropout: float):
        super().__init__()
        self.encoder = SFCNEncoder(dropout)
        self.head = nn.Linear(128, classes)

    def forward(self, volume: torch.Tensor):
        embedding = self.encoder(volume)
        return self.head(embedding), embedding


def weighted_macro_auc(labels: np.ndarray, probabilities: np.ndarray, weights: np.ndarray) -> float:
    return float(roc_auc_score(
        labels, probabilities, multi_class="ovr", average="macro", sample_weight=weights
    ))


def evaluate_sfcn(
    model: nn.Module, dataset: VolumeDataset, batch_size: int, device: torch.device,
    workers: int = 0, amp: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=device.type == "cuda", persistent_workers=False,
    )
    model.eval()
    rows, probabilities, embeddings = [], [], []
    with torch.no_grad():
        for batch in loader:
            volume = batch["volume"].to(device, non_blocking=True)
            with torch.amp.autocast(
                device_type=device.type, enabled=amp and device.type == "cuda"
            ):
                logits, embedding = model(volume)
            rows.extend(batch["row"].numpy().tolist())
            probabilities.append(torch.softmax(logits.float(), 1).cpu().numpy())
            embeddings.append(embedding.float().cpu().numpy())
    return np.asarray(rows), np.concatenate(probabilities), np.concatenate(embeddings)


def train_sfcn(
    fold: int,
    seed: int,
    frame: pd.DataFrame,
    fit_indices: np.ndarray,
    tune_indices: np.ndarray,
    volume_path: Path,
    config: dict[str, Any],
    output: Path,
    device: torch.device,
) -> tuple[np.ndarray, pd.DataFrame]:
    seed_everything(seed)
    settings = config["sfcn_training"]
    base_weights = subject_weights(frame)
    fit_labels = frame.iloc[fit_indices].label.to_numpy()
    class_subject_mass = np.asarray([
        base_weights[fit_indices][fit_labels == label].sum() for label in range(len(config["classes"]))
    ])
    class_factor = class_subject_mass.sum() / (len(class_subject_mass) * class_subject_mass)
    training_weights = base_weights.copy()
    training_weights[fit_indices] *= class_factor[fit_labels]
    fit_dataset = VolumeDataset(frame, fit_indices, volume_path, training_weights)
    tune_dataset = VolumeDataset(frame, tune_indices, volume_path, base_weights)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        fit_dataset, batch_size=int(settings["batch_size"]), shuffle=True,
        generator=generator, num_workers=int(settings["num_workers"]),
        pin_memory=device.type == "cuda",
        # Do not keep the large training prefetch queue alive while validation
        # creates its own loader.
        persistent_workers=False,
    )
    model = SFCNClassifier(len(config["classes"]), float(settings["dropout"])).to(device)
    checkpoint = output / "checkpoints" / f"fold{fold}" / f"seed{seed}.pt"
    if bool(config.get("resume_completed_checkpoints", False)) and checkpoint.exists():
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        if int(payload["fold"]) != int(fold) or int(payload["seed"]) != int(seed):
            raise RuntimeError(f"Checkpoint identity mismatch: {checkpoint}")
        model.load_state_dict(payload["model"])
        all_indices = np.arange(len(frame))
        rows, _, embedding = evaluate_sfcn(
            model, VolumeDataset(frame, all_indices, volume_path, base_weights),
            int(settings["batch_size"]), device,
            int(settings.get("export_num_workers", settings["num_workers"])),
            bool(settings.get("amp", False)),
        )
        matrix = np.empty((len(frame), embedding.shape[1]), dtype=np.float32)
        matrix[rows] = embedding
        curve = pd.DataFrame([{
            "fold": fold, "seed": seed, "epoch": -1, "train_loss": np.nan,
            "tune_macro_auc": float(payload["tune_macro_auc"]),
            "checkpoint_reused": True,
        }])
        print(json.dumps({"reused_checkpoint": str(checkpoint)}), flush=True)
        return matrix, curve
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    amp = bool(settings.get("amp", False))
    accumulation = int(settings.get("gradient_accumulation_steps", 1))
    scaler = torch.amp.GradScaler(device.type, enabled=amp and device.type == "cuda")
    best_score, best_state, patience = -np.inf, None, 0
    curves = []
    for epoch in range(int(settings["max_epochs"])):
        model.train()
        losses = []
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(loader):
            volume = batch["volume"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            weight = batch["weight"].to(device, non_blocking=True)
            with torch.amp.autocast(
                device_type=device.type, enabled=amp and device.type == "cuda"
            ):
                logits, _ = model(volume)
                loss_values = F.cross_entropy(logits.float(), label, reduction="none")
                loss = (loss_values * weight).sum() / weight.sum()
            scaler.scale(loss / accumulation).backward()
            should_step = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader)
            if should_step:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), float(settings["gradient_clip"]))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
        rows, probabilities, _ = evaluate_sfcn(
            model, tune_dataset, int(settings["batch_size"]), device,
            int(settings["num_workers"]), amp,
        )
        score = weighted_macro_auc(
            frame.iloc[rows].label.to_numpy(), probabilities, base_weights[rows]
        )
        curves.append({"fold": fold, "seed": seed, "epoch": epoch,
                       "train_loss": float(np.mean(losses)), "tune_macro_auc": score})
        print(json.dumps(curves[-1]), flush=True)
        if score > best_score + 1e-5:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= int(settings["patience"]):
                break
    if best_state is None:
        raise RuntimeError("No SFCN checkpoint selected")
    model.load_state_dict(best_state)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "fold": fold, "seed": seed,
                "tune_macro_auc": best_score}, checkpoint)
    all_indices = np.arange(len(frame))
    rows, _, embedding = evaluate_sfcn(
        model, VolumeDataset(frame, all_indices, volume_path, base_weights),
        int(settings["batch_size"]), device,
        int(settings.get("export_num_workers", settings["num_workers"])), amp,
    )
    matrix = np.empty((len(frame), embedding.shape[1]), dtype=np.float32)
    matrix[rows] = embedding
    return matrix, pd.DataFrame(curves)


def fit_classifier(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    weights: np.ndarray,
    config: dict[str, Any],
) -> Pipeline:
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=float(config["logistic"]["C"]), max_iter=int(config["logistic"]["max_iter"]),
            class_weight="balanced", multi_class="auto", solver="lbfgs",
        )),
    ])
    model.fit(x_fit, y_fit, classifier__sample_weight=weights)
    return model


def calibrated_probabilities(model: Pipeline, x: np.ndarray, temperature: float) -> np.ndarray:
    logits = model.decision_function(x)
    return softmax(logits / float(temperature), axis=1)


def select_temperature(
    model: Pipeline, x: np.ndarray, y: np.ndarray, weights: np.ndarray, grid: list[float]
) -> float:
    scores = [(float(t), log_loss(y, calibrated_probabilities(model, x, t),
                                  labels=[0, 1, 2], sample_weight=weights)) for t in grid]
    return min(scores, key=lambda row: (row[1], abs(row[0] - 1.0)))[0]


def shuffled_indices(frame: pd.DataFrame, test_indices: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    subjects = frame.iloc[test_indices].subject_id.astype(str).to_numpy()
    order = np.arange(len(test_indices))
    for _ in range(1000):
        rng.shuffle(order)
        if np.all(subjects[order] != subjects):
            return test_indices[order]
    # Deterministic cyclic fallback at the subject-block level.
    unique = np.unique(subjects)
    mapping = {subject: unique[(index + 1) % len(unique)] for index, subject in enumerate(unique)}
    by_subject = {subject: test_indices[subjects == subject] for subject in unique}
    result = []
    for subject in subjects:
        candidates = by_subject[mapping[subject]]
        result.append(int(candidates[int(rng.integers(0, len(candidates)))]))
    return np.asarray(result)


def prediction_rows(
    frame: pd.DataFrame, indices: np.ndarray, probabilities: np.ndarray,
    arm: str, fold: int, seed: int,
) -> pd.DataFrame:
    result = frame.iloc[indices][[
        "scan_id", "subject_id", "cutoff_month", "label", "dxsum_diagnosis_label",
        "dxsum_phase", "age", "sex", "cognitive_missing_fraction",
    ]].copy()
    result["row_index"] = indices
    result["arm"] = arm
    result["fold"] = fold
    result["seed"] = seed
    for index, name in enumerate(["CN", "MCI", "AD"]):
        result[f"prob_{name}"] = probabilities[:, index]
    return result


def run_crossfit(config: dict[str, Any], output: Path) -> None:
    frame, volume_path = build_frame(config)
    global_matrix = global_features(frame)
    cognitive_matrix, cognitive_names = cognitive_history_features(frame, list(config["targets"]))
    demo = frame[["age", "sex_numeric"]].to_numpy(dtype=np.float32)
    folds = make_folds(frame, config, output)
    fold_lookup = folds.set_index("subject_id").fold
    frame["fold"] = frame.subject_id.map(fold_lookup).astype(int)
    frame.to_csv(output / "diagnostic_manifest.csv", index=False)
    (output / "feature_contract.json").write_text(json.dumps({
        "demographics": ["age", "sex_numeric"],
        "cognitive_history": cognitive_names,
        "global_mri_dimension": int(global_matrix.shape[1]),
        "sfcn_embedding_dimension": 128,
        "volume_cache": config.get("volume_cache"),
        "diagnosis_input": False,
    }, indent=2) + "\n")
    base_weights = subject_weights(frame)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outputs, curves = [], []
    all_subjects = folds.subject_id.astype(str).to_numpy()
    subject_strata = folds.stratification_label.to_numpy()
    folds_to_run = config.get("run_folds", list(range(int(config["outer_folds"]))))
    for fold in [int(value) for value in folds_to_run]:
        test_subjects = set(folds.loc[folds.fold.eq(fold), "subject_id"].astype(str))
        outer_subjects = np.asarray([subject for subject in all_subjects if subject not in test_subjects])
        outer_labels = np.asarray([
            int(folds.set_index("subject_id").loc[subject, "stratification_label"])
            for subject in outer_subjects
        ])
        split = StratifiedShuffleSplit(
            n_splits=1, test_size=float(config["inner_tune_fraction"]),
            random_state=int(config["fold_seed"]) + fold,
        )
        fit_local, tune_local = next(split.split(outer_subjects, outer_labels))
        fit_subjects, tune_subjects = set(outer_subjects[fit_local]), set(outer_subjects[tune_local])
        fit_indices = np.flatnonzero(frame.subject_id.astype(str).isin(fit_subjects))
        tune_indices = np.flatnonzero(frame.subject_id.astype(str).isin(tune_subjects))
        test_indices = np.flatnonzero(frame.subject_id.astype(str).isin(test_subjects))
        if set(frame.iloc[fit_indices].subject_id) & set(frame.iloc[test_indices].subject_id):
            raise RuntimeError("Outer-fold subject leakage")
        fixed_arms = {
            "demographics": demo,
            "global_mri": np.concatenate([demo, global_matrix], axis=1),
            "cognition": np.concatenate([demo, cognitive_matrix], axis=1),
            "cognition_global_mri": np.concatenate([demo, cognitive_matrix, global_matrix], axis=1),
        }
        for arm, matrix in fixed_arms.items():
            model = fit_classifier(matrix[fit_indices], frame.iloc[fit_indices].label.to_numpy(),
                                   base_weights[fit_indices], config)
            temperature = select_temperature(
                model, matrix[tune_indices], frame.iloc[tune_indices].label.to_numpy(),
                base_weights[tune_indices], list(config["temperature_grid"]),
            )
            probability = calibrated_probabilities(model, matrix[test_indices], temperature)
            block = prediction_rows(frame, test_indices, probability, arm, fold, -1)
            block["temperature"] = temperature
            outputs.append(block)
        for seed in config["neural_seeds"]:
            embedding, curve = train_sfcn(
                fold, int(seed), frame, fit_indices, tune_indices, volume_path,
                config, output, device,
            )
            curves.append(curve)
            sfcn_arms = {
                "sfcn_mri": np.concatenate([demo, embedding], axis=1),
                "cognition_sfcn_mri": np.concatenate([demo, cognitive_matrix, embedding], axis=1),
            }
            fitted = {}
            for arm, matrix in sfcn_arms.items():
                model = fit_classifier(
                    matrix[fit_indices], frame.iloc[fit_indices].label.to_numpy(),
                    base_weights[fit_indices], config,
                )
                temperature = select_temperature(
                    model, matrix[tune_indices], frame.iloc[tune_indices].label.to_numpy(),
                    base_weights[tune_indices], list(config["temperature_grid"]),
                )
                probability = calibrated_probabilities(model, matrix[test_indices], temperature)
                block = prediction_rows(frame, test_indices, probability, arm, fold, int(seed))
                block["temperature"] = temperature
                outputs.append(block)
                fitted[arm] = (model, temperature)
            shuffled = shuffled_indices(frame, test_indices, int(seed) + fold * 10000)
            shuffled_matrix = np.concatenate(
                [demo[test_indices], cognitive_matrix[test_indices], embedding[shuffled]], axis=1
            )
            fusion_model, fusion_temperature = fitted["cognition_sfcn_mri"]
            probability = calibrated_probabilities(
                fusion_model, shuffled_matrix, fusion_temperature
            )
            block = prediction_rows(
                frame, test_indices, probability, "cognition_shuffled_sfcn", fold, int(seed)
            )
            block["temperature"] = fusion_temperature
            outputs.append(block)
        print(json.dumps({"completed_fold": fold, "test_subjects": len(test_subjects),
                          "test_scans": len(test_indices)}), flush=True)
    pd.concat(curves, ignore_index=True).to_csv(output / "sfcn_training_curves.csv", index=False)
    seed_predictions = pd.concat(outputs, ignore_index=True)
    seed_predictions.to_csv(output / "crossfit_seed_predictions.csv", index=False)
    keys = [
        "row_index", "scan_id", "subject_id", "cutoff_month", "label",
        "dxsum_diagnosis_label", "dxsum_phase", "age", "sex",
        "cognitive_missing_fraction", "arm", "fold",
    ]
    ensemble = seed_predictions.groupby(keys, as_index=False)[["prob_CN", "prob_MCI", "prob_AD"]].mean()
    ensemble.to_csv(output / "crossfit_predictions.csv", index=False)


def multiclass_brier(y: np.ndarray, p: np.ndarray, weights: np.ndarray) -> float:
    one_hot = np.eye(3)[y]
    return float(np.average(np.sum((p - one_hot) ** 2, axis=1), weights=weights))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, weights: np.ndarray, bins: int = 10) -> float:
    confidence = p.max(1)
    correct = p.argmax(1) == y
    total = weights.sum()
    value = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence >= low) & (confidence < high if high < 1 else confidence <= high)
        if mask.any():
            mass = weights[mask].sum()
            value += mass / total * abs(
                np.average(correct[mask], weights=weights[mask])
                - np.average(confidence[mask], weights=weights[mask])
            )
    return float(value)


def arm_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm, group in frame.groupby("arm", sort=True):
        y = group.label.to_numpy(dtype=int)
        p = group[["prob_CN", "prob_MCI", "prob_AD"]].to_numpy(dtype=float)
        predicted = p.argmax(1)
        weights = subject_weights(group)
        class_sensitivity, class_specificity, class_f1 = [], [], []
        for label, name in enumerate(["CN", "MCI", "AD"]):
            positive = y == label
            negative = ~positive
            sensitivity = float(np.average(predicted[positive] == label, weights=weights[positive]))
            specificity = float(np.average(predicted[negative] != label, weights=weights[negative]))
            label_f1 = float(f1_score(
                positive, predicted == label, sample_weight=weights, zero_division=0
            ))
            class_sensitivity.append(sensitivity)
            class_specificity.append(specificity)
            class_f1.append(label_f1)
        row = {
            "arm": arm, "scans": len(group), "subjects": group.subject_id.nunique(),
            "accuracy": float(accuracy_score(y, predicted, sample_weight=weights)),
            "macro_auc": weighted_macro_auc(y, p, weights),
            "macro_sensitivity": float(np.mean(class_sensitivity)),
            "macro_specificity": float(np.mean(class_specificity)),
            "balanced_accuracy": float(balanced_accuracy_score(y, predicted, sample_weight=weights)),
            "macro_f1": float(f1_score(y, predicted, average="macro", sample_weight=weights)),
            "log_loss": float(log_loss(y, p, labels=[0, 1, 2], sample_weight=weights)),
            "brier": multiclass_brier(y, p, weights),
            "ece": expected_calibration_error(y, p, weights),
        }
        for index, name in enumerate(["CN", "MCI", "AD"]):
            row[f"sensitivity_{name}"] = class_sensitivity[index]
            row[f"specificity_{name}"] = class_specificity[index]
            row[f"f1_{name}"] = class_f1[index]
        rows.append(row)
    return pd.DataFrame(rows)


def requested_classification_metrics(
    y: np.ndarray, p: np.ndarray, weights: np.ndarray
) -> dict[str, float]:
    predicted = p.argmax(1)
    sensitivities, specificities = [], []
    for label in range(3):
        positive = y == label
        negative = ~positive
        sensitivities.append(float(np.average(predicted[positive] == label, weights=weights[positive])))
        specificities.append(float(np.average(predicted[negative] != label, weights=weights[negative])))
    return {
        "accuracy": float(accuracy_score(y, predicted, sample_weight=weights)),
        "macro_auc": weighted_macro_auc(y, p, weights),
        "macro_sensitivity": float(np.mean(sensitivities)),
        "macro_specificity": float(np.mean(specificities)),
        "macro_f1": float(f1_score(y, predicted, average="macro", sample_weight=weights)),
    }


def bootstrap_metric_intervals(
    predictions: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    rows = []
    metric_names = ["accuracy", "macro_auc", "macro_sensitivity", "macro_specificity", "macro_f1"]
    for arm_index, (arm, group) in enumerate(predictions.groupby("arm", sort=True)):
        group = group.sort_values("row_index")
        subject_values = group.subject_id.astype(str).to_numpy()
        subjects, subject_codes = np.unique(subject_values, return_inverse=True)
        scans_per_subject = np.bincount(subject_codes).astype(float)
        y = group.label.to_numpy(dtype=int)
        p = group[["prob_CN", "prob_MCI", "prob_AD"]].to_numpy(dtype=float)
        rng = np.random.default_rng(int(config["statistics"]["bootstrap_seed"]) + arm_index)
        draws = {name: [] for name in metric_names}
        for _ in range(int(config["statistics"]["bootstrap_replicates"])):
            sampled = rng.integers(0, len(subjects), size=len(subjects))
            multiplicity = np.bincount(sampled, minlength=len(subjects))
            weights = multiplicity[subject_codes] / scans_per_subject[subject_codes]
            values = requested_classification_metrics(y, p, weights)
            for name in metric_names:
                draws[name].append(values[name])
        point = requested_classification_metrics(y, p, 1.0 / scans_per_subject[subject_codes])
        for name in metric_names:
            rows.append({
                "arm": arm,
                "metric": name,
                "value": point[name],
                "ci_low": float(np.quantile(draws[name], .025)),
                "ci_high": float(np.quantile(draws[name], .975)),
                "replicates": int(config["statistics"]["bootstrap_replicates"]),
                "subjects": len(subjects),
            })
    return pd.DataFrame(rows)


def bootstrap_contrast(
    predictions: pd.DataFrame, candidate: str, baseline: str, config: dict[str, Any]
) -> list[dict[str, Any]]:
    candidate_frame = predictions[predictions.arm.eq(candidate)].sort_values("row_index")
    baseline_frame = predictions[predictions.arm.eq(baseline)].sort_values("row_index")
    if candidate_frame.row_index.tolist() != baseline_frame.row_index.tolist():
        raise RuntimeError("Unpaired diagnostic predictions")
    # Sampling a subject more than once is exactly equivalent to multiplying
    # every one of that subject's scan weights by its bootstrap multiplicity.
    # Keeping the paired rows in place avoids thousands of repeated DataFrame
    # filters/concatenations while preserving the subject-cluster bootstrap.
    subject_values = candidate_frame.subject_id.astype(str).to_numpy()
    subjects, subject_codes = np.unique(subject_values, return_inverse=True)
    scans_per_subject = np.bincount(subject_codes).astype(float)
    y = candidate_frame.label.to_numpy(dtype=int)
    p_candidate = candidate_frame[["prob_CN", "prob_MCI", "prob_AD"]].to_numpy(dtype=float)
    p_baseline = baseline_frame[["prob_CN", "prob_MCI", "prob_AD"]].to_numpy(dtype=float)
    predicted_candidate = p_candidate.argmax(1)
    predicted_baseline = p_baseline.argmax(1)
    rng = np.random.default_rng(int(config["statistics"]["bootstrap_seed"]))
    n = int(config["statistics"]["bootstrap_replicates"])
    draws = {"macro_auc": [], "balanced_accuracy": []}
    for _ in range(n):
        sampled_codes = rng.integers(0, len(subjects), size=len(subjects))
        multiplicity = np.bincount(sampled_codes, minlength=len(subjects))
        weight = multiplicity[subject_codes] / scans_per_subject[subject_codes]
        draws["macro_auc"].append(
            weighted_macro_auc(y, p_candidate, weight) - weighted_macro_auc(y, p_baseline, weight)
        )
        draws["balanced_accuracy"].append(
            balanced_accuracy_score(y, predicted_candidate, sample_weight=weight)
            - balanced_accuracy_score(y, predicted_baseline, sample_weight=weight)
        )
    rows = []
    for metric, values in draws.items():
        values = np.asarray(values)
        rows.append({
            "comparison": f"{candidate}_minus_{baseline}", "metric": metric,
            "difference": float(values.mean()), "ci_low": float(np.quantile(values, .025)),
            "ci_high": float(np.quantile(values, .975)), "replicates": n,
            "subjects": int(len(subjects)),
        })
    return rows


def decision_curves(predictions: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    thresholds = [float(value) for value in config["decision_curve"]["thresholds"]]
    for arm in ["cognition", "sfcn_mri", "cognition_sfcn_mri", "cognition_shuffled_sfcn"]:
        group = predictions[predictions.arm.eq(arm)]
        truth = group.label.eq(2).to_numpy()
        probability = group.prob_AD.to_numpy(dtype=float)
        weights = subject_weights(group)
        total = weights.sum()
        prevalence = float(np.average(truth, weights=weights))
        for threshold in thresholds:
            positive = probability >= threshold
            tp = weights[positive & truth].sum() / total
            fp = weights[positive & ~truth].sum() / total
            net = tp - fp * threshold / (1 - threshold)
            treat_all = prevalence - (1 - prevalence) * threshold / (1 - threshold)
            rows.append({"arm": arm, "threshold": threshold, "net_benefit": float(net),
                         "treat_all": float(treat_all), "treat_none": 0.0})
    return pd.DataFrame(rows)


def subgroup_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    median_age = float(frame.age.median())
    frame["age_group"] = np.where(frame.age < median_age, "younger", "older")
    frame["missingness_group"] = np.where(
        frame.cognitive_missing_fraction <= 0.2, "low_missingness", "high_missingness"
    )
    rows = []
    for column in ["dxsum_phase", "sex", "age_group", "missingness_group"]:
        for value, subset in frame.groupby(column, dropna=False):
            for arm in ["cognition", "cognition_sfcn_mri"]:
                group = subset[subset.arm.eq(arm)]
                if group.label.nunique() < 3 or group.subject_id.nunique() < 10:
                    continue
                y = group.label.to_numpy(dtype=int)
                p = group[["prob_CN", "prob_MCI", "prob_AD"]].to_numpy(dtype=float)
                weights = subject_weights(group)
                rows.append({"stratum_type": column, "stratum": str(value), "arm": arm,
                             "scans": len(group), "subjects": group.subject_id.nunique(),
                             "macro_auc": weighted_macro_auc(y, p, weights),
                             "balanced_accuracy": float(balanced_accuracy_score(
                                 y, p.argmax(1), sample_weight=weights))})
    return pd.DataFrame(rows)


def summarize(config: dict[str, Any], output: Path) -> None:
    predictions = pd.read_csv(output / "crossfit_predictions.csv")
    metrics = arm_metrics(predictions)
    metrics.to_csv(output / "diagnostic_metrics.csv", index=False)
    requested = metrics[[
        "arm", "scans", "subjects", "accuracy", "macro_auc",
        "macro_sensitivity", "macro_specificity", "macro_f1",
    ]]
    requested.to_csv(output / "diagnostic_requested_metrics.csv", index=False)
    metric_intervals = bootstrap_metric_intervals(predictions, config)
    metric_intervals.to_csv(output / "diagnostic_metric_intervals.csv", index=False)
    per_class_rows = []
    for row in metrics.itertuples(index=False):
        for label in ["CN", "MCI", "AD"]:
            per_class_rows.append({
                "arm": row.arm, "class": label,
                "sensitivity": getattr(row, f"sensitivity_{label}"),
                "specificity": getattr(row, f"specificity_{label}"),
                "f1": getattr(row, f"f1_{label}"),
            })
    pd.DataFrame(per_class_rows).to_csv(output / "diagnostic_per_class_metrics.csv", index=False)
    comparisons = []
    for candidate, baseline in [
        ("sfcn_mri", "demographics"),
        ("cognition_sfcn_mri", "cognition"),
        ("cognition_sfcn_mri", "cognition_shuffled_sfcn"),
    ]:
        comparisons.extend(bootstrap_contrast(predictions, candidate, baseline, config))
    bootstrap = pd.DataFrame(comparisons)
    bootstrap.to_csv(output / "diagnostic_bootstrap.csv", index=False)
    curves = decision_curves(predictions, config)
    curves.to_csv(output / "decision_curves.csv", index=False)
    subgroup_metrics(predictions).to_csv(output / "diagnostic_subgroups.csv", index=False)
    confusion_rows = []
    for arm, group in predictions.groupby("arm", sort=True):
        matrix = confusion_matrix(group.label, group[["prob_CN", "prob_MCI", "prob_AD"]].to_numpy().argmax(1),
                                  labels=[0, 1, 2])
        for truth in range(3):
            for predicted in range(3):
                confusion_rows.append({"arm": arm, "truth": config["classes"][truth],
                                       "predicted": config["classes"][predicted],
                                       "count": int(matrix[truth, predicted])})
    pd.DataFrame(confusion_rows).to_csv(output / "confusion_matrix.csv", index=False)
    indexed = metrics.set_index("arm")
    macro = bootstrap[bootstrap.metric.eq("macro_auc")].set_index("comparison")
    gate_config = config["gate"]
    mri = macro.loc["sfcn_mri_minus_demographics"]
    increment = macro.loc["cognition_sfcn_mri_minus_cognition"]
    mechanism = macro.loc["cognition_sfcn_mri_minus_cognition_shuffled_sfcn"]
    curve_pivot = curves.pivot(index="threshold", columns="arm", values="net_benefit")
    positive_thresholds = int((
        curve_pivot.cognition_sfcn_mri > curve_pivot.cognition
    ).sum())
    criteria = {
        "mri_signal_effect": bool(mri.difference >= gate_config["minimum_mri_vs_demographics_macro_auc_difference"]),
        "mri_signal_ci": bool(mri.ci_low > gate_config["minimum_ci_low"]),
        "fusion_increment_effect": bool(increment.difference >= gate_config["minimum_fusion_vs_cognition_macro_auc_difference"]),
        "fusion_increment_ci": bool(increment.ci_low > gate_config["minimum_ci_low"]),
        "mechanism_effect": bool(mechanism.difference >= gate_config["minimum_fusion_vs_shuffled_macro_auc_difference"]),
        "mechanism_ci": bool(mechanism.ci_low > gate_config["minimum_ci_low"]),
        "balanced_accuracy_noninferiority": bool(
            indexed.loc["cognition_sfcn_mri", "balanced_accuracy"]
            - indexed.loc["cognition", "balanced_accuracy"]
            >= gate_config["minimum_balanced_accuracy_difference"]
        ),
        "brier_noninferiority": bool(
            indexed.loc["cognition_sfcn_mri", "brier"]
            - indexed.loc["cognition", "brier"] <= gate_config["maximum_brier_increase"]
        ),
        "decision_curve_gain": bool(
            positive_thresholds >= gate_config["require_positive_ad_net_benefit_gain_at_thresholds"]
        ),
    }
    decision = {
        "run_id": config["run_id"], "official_validation_loaded": False,
        "official_test_loaded": False, "training_subject_source": "original_train_only",
        "mri_concurrent_signal_supported": bool(criteria["mri_signal_effect"] and criteria["mri_signal_ci"]),
        "mri_increment_beyond_cognition_supported": bool(all(criteria.values())),
        "criteria": criteria, "ad_net_benefit_gain_thresholds": positive_thresholds,
        "mci_conversion_open": bool(all(criteria.values())),
        "external_testing_required": True,
    }
    (output / "development_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    manifest = pd.read_csv(output / "diagnostic_manifest.csv")
    cropped_input = bool(config.get("volume_cache"))
    report_title = config.get("report_title") or (
        "# Phase 3K center-cropped T1 diagnostic cross-fitting"
        if cropped_input else "# Phase 3G MRI diagnostic incremental-value cross-fitting"
    )
    report = [
        report_title,
        "",
        "Status: completed on original-train subjects only with five-fold subject-wise cross-fitting. Official validation and test subjects were not loaded.",
        "",
        "## Cohort",
        "",
        f"- MRI-linked diagnostic visits: {len(manifest)}",
        f"- Subjects: {manifest.subject_id.nunique()}",
        f"- CN/MCI/AD visits: {manifest.dxsum_diagnosis_label.value_counts().to_dict()}",
        f"- CN/MCI/AD subjects: {manifest.groupby('dxsum_diagnosis_label').subject_id.nunique().to_dict()}",
        "- Diagnosis-to-scan window: <= 90 days",
        *(["- MRI input: interpolation-free 160x192x160 center crop at 1 mm."] if cropped_input else []),
        "- Cross-fitted SFCNs: 5 folds x 3 seeds = 15 checkpoints",
        "",
        "## Subject-balanced out-of-fold performance",
        "",
        "ACC is subject-balanced accuracy. AUC, SEN, SEC, and F1 are macro one-vs-rest values across CN/MCI/AD.",
        "",
        markdown_table(requested),
        "",
        "Per-class sensitivity, specificity, and F1 are available in `diagnostic_per_class_metrics.csv`; 2,000-replicate confidence intervals are in `diagnostic_metric_intervals.csv`.",
        "",
        "## Paired subject-cluster bootstrap",
        "",
        markdown_table(bootstrap),
        "",
        "## Diagnostic performance figure",
        "",
        "![Phase 3G diagnosis performance and incremental-value tests](figures/fig_phase3g_diagnosis_increment.png)",
        "",
        "## Frozen decision",
        "",
        "```json",
        json.dumps(decision, indent=2),
        "```",
        "",
        "MRI signal relative to demographics and MRI increment beyond cognition are separate claims. The conversion task opens only if every frozen increment criterion passes. Internal cross-fitting is not external testing and does not establish clinical deployment value.",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n")


def run(config_path: Path = CONFIG) -> None:
    config = load_config(config_path)
    output = output_dir(config)
    (output / "resolved_config.json").write_text(json.dumps(config, indent=2) + "\n")
    run_crossfit(config, output)
    if not bool(config.get("defer_summary", False)):
        summarize(config, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
