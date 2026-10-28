"""Train-only cross-fitted MRI-inclusive MCI-to-AD conversion experiment."""

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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
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
from .phase3g_mri_diagnosis import cognitive_history_features, global_features


CONFIG = ROOT / "configs/phase3h_mci_conversion_crossfit.yaml"
MONTH_DAYS = 365.25 / 12.0


def load_config(path: Path = CONFIG) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def output_dir(config: dict[str, Any]) -> Path:
    path = ROOT / "outputs" / config["run_id"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_horizon_frame(config: dict[str, Any], horizon: int) -> tuple[pd.DataFrame, Path]:
    diagnosis_output = ROOT / "outputs" / config["source_diagnostic_run_id"]
    frame = pd.read_csv(diagnosis_output / "diagnostic_manifest.csv", low_memory=False)
    frame = frame[frame.dxsum_diagnosis_label.eq(config["baseline_diagnosis"])].copy()
    frame["index_diagnosis_date"] = pd.to_datetime(frame.dxsum_visit_date, errors="coerce")

    clinical = pd.read_csv(
        ROOT / "splits/train.csv", low_memory=False,
        usecols=["subject_id", "dxsum_visit_date", "dxsum_diagnosis_label"],
    )
    clinical["diagnosis_date"] = pd.to_datetime(clinical.dxsum_visit_date, errors="coerce")
    clinical = clinical[
        clinical.dxsum_diagnosis_label.isin(["CN", "MCI", "AD"])
        & clinical.diagnosis_date.notna()
    ].sort_values(["subject_id", "diagnosis_date"])
    diagnoses = {str(subject): group for subject, group in clinical.groupby("subject_id", sort=False)}

    records = []
    for index, row in frame.iterrows():
        history = diagnoses[str(row.subject_id)]
        future = history[history.diagnosis_date > row.index_diagnosis_date].copy()
        delta = (future.diagnosis_date - row.index_diagnosis_date).dt.days / MONTH_DAYS
        within = future[(future.dxsum_diagnosis_label.eq("AD")) & delta.le(float(horizon))]
        converted = not within.empty
        full_followup = bool(delta.ge(float(horizon)).any())
        if not (converted or full_followup):
            continue
        record = row.to_dict()
        record.update({
            "horizon_months": int(horizon),
            "conversion_label": int(converted),
            "conversion_date": within.diagnosis_date.min() if converted else pd.NaT,
            "conversion_time_months": (
                float((within.diagnosis_date.min() - row.index_diagnosis_date).days / MONTH_DAYS)
                if converted else np.nan
            ),
            "last_followup_date": future.diagnosis_date.max(),
            "followup_months": float(delta.max()),
        })
        records.append(record)
    result = pd.DataFrame(records)
    result = result.sort_values(["subject_id", "index_diagnosis_date", "cutoff_month"])
    result = result.drop_duplicates("subject_id", keep="first").reset_index(drop=True)
    result["label"] = result.conversion_label.astype(int)
    if result.subject_id.duplicated().any():
        raise RuntimeError("More than one MCI baseline per subject")
    if result.label.nunique() != 2:
        raise RuntimeError(f"Horizon {horizon} does not contain both outcome classes")
    official_train = set(pd.read_csv(ROOT / "splits/train.csv", usecols=["subject_id"]).subject_id.astype(str))
    if not set(result.subject_id.astype(str)) <= official_train:
        raise RuntimeError("Non-train subject entered Phase 3H")
    if config.get("volume_cache"):
        settings = config["volume_cache"]
        durable_path = ROOT / settings["path"]
        runtime_path = Path(settings.get("runtime_path", durable_path))
        volume_path = runtime_path if runtime_path.exists() else durable_path
    else:
        volume_path = ROOT / "outputs" / config["source_mri_run_id"] / "volumes_48x56x48.npy"
    return result, volume_path


def make_folds(frame: pd.DataFrame, config: dict[str, Any], horizon: int) -> pd.DataFrame:
    if config.get("reuse_subject_folds"):
        source = pd.read_csv(ROOT / config["reuse_subject_folds"])
        folds = source[source.horizon_months.eq(int(horizon))].copy()
        if set(folds.subject_id.astype(str)) != set(frame.subject_id.astype(str)):
            raise RuntimeError(f"Reused conversion folds do not match horizon {horizon}")
        return folds
    splitter = StratifiedKFold(
        n_splits=int(config["outer_folds"]), shuffle=True,
        random_state=int(config["fold_seed"]) + int(horizon),
    )
    rows = []
    for fold, (_, test_index) in enumerate(splitter.split(frame.subject_id, frame.label)):
        for index in test_index:
            rows.append({
                "subject_id": frame.iloc[index].subject_id,
                "horizon_months": int(horizon),
                "fold": int(fold),
                "conversion_label": int(frame.iloc[index].label),
            })
    folds = pd.DataFrame(rows)
    if folds.subject_id.duplicated().any() or len(folds) != len(frame):
        raise RuntimeError("Invalid Phase 3H subject folds")
    return folds


class VolumeDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, indices: np.ndarray, volume_path: Path, weights: np.ndarray):
        self.frame = frame
        self.indices = np.asarray(indices, dtype=int)
        self.volume_path = volume_path
        self.weights = np.asarray(weights, dtype=np.float32)
        self._volumes = None

    @property
    def volumes(self):
        if self._volumes is None:
            self._volumes = np.load(self.volume_path, mmap_mode="r")
        return self._volumes

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        row = int(self.indices[item])
        volume_index = int(self.frame.iloc[row].volume_index)
        volume = np.asarray(self.volumes[volume_index], dtype=np.float32)[None]
        return {
            "row": torch.tensor(row, dtype=torch.long),
            "volume": torch.from_numpy(volume),
            "label": torch.tensor(int(self.frame.iloc[row].label), dtype=torch.long),
            "weight": torch.tensor(float(self.weights[row]), dtype=torch.float32),
        }


class BinarySFCN(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.encoder = SFCNEncoder(dropout)
        self.head = nn.Linear(128, 2)

    def forward(self, volume: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(volume)
        return self.head(embedding), embedding


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
            rows.append(batch["row"].numpy())
            probabilities.append(torch.softmax(logits.float(), 1).cpu().numpy())
            embeddings.append(embedding.float().cpu().numpy())
    return np.concatenate(rows), np.concatenate(probabilities), np.concatenate(embeddings)


def train_sfcn(
    horizon: int,
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
    y_fit = frame.iloc[fit_indices].label.to_numpy(dtype=int)
    class_count = np.bincount(y_fit, minlength=2).astype(float)
    class_factor = len(y_fit) / (2.0 * class_count)
    weights = np.ones(len(frame), dtype=np.float32)
    weights[fit_indices] = class_factor[y_fit]
    fit_dataset = VolumeDataset(frame, fit_indices, volume_path, weights)
    tune_dataset = VolumeDataset(frame, tune_indices, volume_path, np.ones(len(frame)))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        fit_dataset, batch_size=int(settings["batch_size"]), shuffle=True,
        generator=generator, num_workers=int(settings["num_workers"]),
        pin_memory=device.type == "cuda",
        # Release the training workers before evaluation starts to bound host
        # memory and shared-memory use for full-resolution crops.
        persistent_workers=False,
    )
    model = BinarySFCN(float(settings["dropout"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    amp = bool(settings.get("amp", False))
    accumulation = int(settings.get("gradient_accumulation_steps", 1))
    scaler = torch.amp.GradScaler(device.type, enabled=amp and device.type == "cuda")
    best_auc, best_state, patience = -np.inf, None, 0
    curves = []
    for epoch in range(int(settings["max_epochs"])):
        model.train()
        losses = []
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(loader):
            volume = batch["volume"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            batch_weight = batch["weight"].to(device, non_blocking=True)
            with torch.amp.autocast(
                device_type=device.type, enabled=amp and device.type == "cuda"
            ):
                logits, _ = model(volume)
                loss_values = F.cross_entropy(logits.float(), label, reduction="none")
                loss = (loss_values * batch_weight).sum() / batch_weight.sum()
            scaler.scale(loss / accumulation).backward()
            should_step = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader)
            if should_step:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), float(settings["gradient_clip"]))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
        rows, probability, _ = evaluate_sfcn(
            model, tune_dataset, int(settings["batch_size"]), device,
            int(settings["num_workers"]), amp,
        )
        auc = float(roc_auc_score(frame.iloc[rows].label, probability[:, 1]))
        curve_row = {
            "horizon_months": horizon, "fold": fold, "seed": seed, "epoch": epoch,
            "train_loss": float(np.mean(losses)), "tune_auc": auc,
        }
        curves.append(curve_row)
        print(json.dumps(curve_row), flush=True)
        if auc > best_auc + 1e-5:
            best_auc = auc
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= int(settings["patience"]):
                break
    if best_state is None:
        raise RuntimeError("No binary SFCN checkpoint selected")
    model.load_state_dict(best_state)
    checkpoint = output / "checkpoints" / f"horizon{horizon}" / f"fold{fold}" / f"seed{seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": best_state, "horizon_months": horizon, "fold": fold,
        "seed": seed, "tune_auc": best_auc,
    }, checkpoint)
    all_indices = np.arange(len(frame))
    rows, _, embedding = evaluate_sfcn(
        model, VolumeDataset(frame, all_indices, volume_path, np.ones(len(frame))),
        int(settings["batch_size"]), device, 0, amp,
    )
    matrix = np.empty((len(frame), embedding.shape[1]), dtype=np.float32)
    matrix[rows] = embedding
    return matrix, pd.DataFrame(curves)


def fit_classifier(x: np.ndarray, y: np.ndarray, config: dict[str, Any]) -> Pipeline:
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=float(config["logistic"]["C"]), max_iter=int(config["logistic"]["max_iter"]),
            class_weight="balanced", solver="lbfgs",
        )),
    ])
    model.fit(x, y)
    return model


def shuffled_indices(frame: pd.DataFrame, indices: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = np.arange(len(indices))
    rng.shuffle(order)
    if len(indices) > 1 and np.any(order == np.arange(len(indices))):
        order = np.roll(np.arange(len(indices)), 1)
    return indices[order]


def prediction_rows(
    frame: pd.DataFrame, indices: np.ndarray, probability: np.ndarray,
    arm: str, fold: int, seed: int,
) -> pd.DataFrame:
    block = frame.iloc[indices][[
        "scan_id", "subject_id", "cutoff_month", "horizon_months", "label",
        "age", "sex", "dxsum_phase", "index_diagnosis_date", "last_followup_date",
        "conversion_date", "conversion_time_months", "followup_months",
        "cognitive_missing_fraction",
    ]].copy()
    block["row_index"] = indices
    block["arm"] = arm
    block["fold"] = fold
    block["seed"] = seed
    block["prob_conversion"] = probability
    return block


def run_crossfit(config: dict[str, Any], output: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_predictions, all_curves, all_manifests, all_folds = [], [], [], []
    feature_contract = None
    for horizon in config["horizons_months"]:
        horizon = int(horizon)
        frame, volume_path = build_horizon_frame(config, horizon)
        global_matrix = global_features(frame)
        cognitive_matrix, cognitive_names = cognitive_history_features(frame, list(config["targets"]))
        demo = frame[["age", "sex_numeric"]].to_numpy(dtype=np.float32)
        folds = make_folds(frame, config, horizon)
        fold_lookup = folds.set_index("subject_id").fold
        frame["fold"] = frame.subject_id.map(fold_lookup).astype(int)
        all_manifests.append(frame)
        all_folds.append(folds)
        if feature_contract is None:
            feature_contract = {
                "demographics": ["age", "sex_numeric"],
                "cognitive_history": cognitive_names,
                "global_mri_dimension": int(global_matrix.shape[1]),
                "sfcn_embedding_dimension": 128,
                "volume_cache": config.get("volume_cache"),
                "post_baseline_input": False,
                "diagnosis_input": False,
            }
        for fold in range(int(config["outer_folds"])):
            test_indices = np.flatnonzero(frame.fold.eq(fold))
            outer_indices = np.flatnonzero(frame.fold.ne(fold))
            split = StratifiedShuffleSplit(
                n_splits=1, test_size=float(config["inner_tune_fraction"]),
                random_state=int(config["fold_seed"]) + horizon * 10 + fold,
            )
            fit_local, tune_local = next(split.split(outer_indices, frame.iloc[outer_indices].label))
            fit_indices, tune_indices = outer_indices[fit_local], outer_indices[tune_local]
            fixed_arms = {
                "demographics": demo,
                "global_mri": np.concatenate([demo, global_matrix], axis=1),
                "cognition": np.concatenate([demo, cognitive_matrix], axis=1),
                "cognition_global_mri": np.concatenate([demo, cognitive_matrix, global_matrix], axis=1),
            }
            for arm, matrix in fixed_arms.items():
                model = fit_classifier(matrix[fit_indices], frame.iloc[fit_indices].label.to_numpy(), config)
                probability = model.predict_proba(matrix[test_indices])[:, 1]
                all_predictions.append(prediction_rows(frame, test_indices, probability, arm, fold, -1))
            for seed in config["neural_seeds"]:
                embedding, curves = train_sfcn(
                    horizon, fold, int(seed), frame, fit_indices, tune_indices,
                    volume_path, config, output, device,
                )
                all_curves.append(curves)
                matrices = {
                    "sfcn_mri": np.concatenate([demo, embedding], axis=1),
                    "cognition_sfcn_mri": np.concatenate([demo, cognitive_matrix, embedding], axis=1),
                }
                fitted = {}
                for arm, matrix in matrices.items():
                    model = fit_classifier(matrix[fit_indices], frame.iloc[fit_indices].label.to_numpy(), config)
                    probability = model.predict_proba(matrix[test_indices])[:, 1]
                    all_predictions.append(
                        prediction_rows(frame, test_indices, probability, arm, fold, int(seed))
                    )
                    fitted[arm] = model
                shuffled = shuffled_indices(frame, test_indices, int(seed) + horizon * 1000 + fold)
                shuffled_matrix = np.concatenate(
                    [demo[test_indices], cognitive_matrix[test_indices], embedding[shuffled]], axis=1
                )
                probability = fitted["cognition_sfcn_mri"].predict_proba(shuffled_matrix)[:, 1]
                all_predictions.append(prediction_rows(
                    frame, test_indices, probability, "cognition_shuffled_sfcn", fold, int(seed)
                ))
            print(json.dumps({
                "completed_horizon": horizon, "completed_fold": fold,
                "test_subjects": len(test_indices), "events": int(frame.iloc[test_indices].label.sum()),
            }), flush=True)
    manifest = pd.concat(all_manifests, ignore_index=True)
    manifest.to_csv(output / "conversion_manifest.csv", index=False)
    pd.concat(all_folds, ignore_index=True).to_csv(output / "subject_folds.csv", index=False)
    pd.concat(all_curves, ignore_index=True).to_csv(output / "sfcn_training_curves.csv", index=False)
    (output / "feature_contract.json").write_text(json.dumps(feature_contract, indent=2) + "\n")
    seed_predictions = pd.concat(all_predictions, ignore_index=True)
    seed_predictions.to_csv(output / "crossfit_seed_predictions.csv", index=False)
    keys = [
        "row_index", "scan_id", "subject_id", "cutoff_month", "horizon_months", "label",
        "age", "sex", "dxsum_phase", "index_diagnosis_date", "last_followup_date",
        "conversion_date", "conversion_time_months", "followup_months",
        "cognitive_missing_fraction", "arm", "fold",
    ]
    ensemble = seed_predictions.groupby(keys, as_index=False, dropna=False).prob_conversion.mean()
    ensemble.to_csv(output / "crossfit_predictions.csv", index=False)


def metric_values(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "auc": float(roc_auc_score(y, probability)),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def bootstrap_metrics(
    y: np.ndarray, probability: np.ndarray, threshold: float, replicates: int, seed: int
) -> dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    draws = {name: [] for name in ["accuracy", "auc", "sensitivity", "specificity", "f1"]}
    for _ in range(replicates):
        indices = rng.integers(0, len(y), size=len(y))
        if np.unique(y[indices]).size < 2:
            continue
        values = metric_values(y[indices], probability[indices], threshold)
        for name in draws:
            draws[name].append(values[name])
    return {
        name: (float(np.quantile(values, .025)), float(np.quantile(values, .975)))
        for name, values in draws.items()
    }


def paired_auc_contrast(
    predictions: pd.DataFrame, horizon: int, candidate: str, baseline: str,
    replicates: int, seed: int,
) -> dict[str, Any]:
    columns = ["subject_id", "label", "prob_conversion"]
    candidate_frame = predictions[
        predictions.horizon_months.eq(horizon) & predictions.arm.eq(candidate)
    ][columns].sort_values("subject_id")
    baseline_frame = predictions[
        predictions.horizon_months.eq(horizon) & predictions.arm.eq(baseline)
    ][columns].sort_values("subject_id")
    if candidate_frame.subject_id.tolist() != baseline_frame.subject_id.tolist():
        raise RuntimeError("Unpaired Phase 3H predictions")
    y = candidate_frame.label.to_numpy(dtype=int)
    pc = candidate_frame.prob_conversion.to_numpy(dtype=float)
    pb = baseline_frame.prob_conversion.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(replicates):
        indices = rng.integers(0, len(y), size=len(y))
        if np.unique(y[indices]).size < 2:
            continue
        draws.append(roc_auc_score(y[indices], pc[indices]) - roc_auc_score(y[indices], pb[indices]))
    return {
        "horizon_months": horizon,
        "comparison": f"{candidate}_minus_{baseline}",
        "auc_difference": float(roc_auc_score(y, pc) - roc_auc_score(y, pb)),
        "ci_low": float(np.quantile(draws, .025)),
        "ci_high": float(np.quantile(draws, .975)),
        "replicates": len(draws),
        "subjects": len(y),
    }


def summarize(config: dict[str, Any], output: Path) -> None:
    predictions = pd.read_csv(output / "crossfit_predictions.csv")
    threshold = float(config["classification_threshold"])
    replicates = int(config["statistics"]["bootstrap_replicates"])
    base_seed = int(config["statistics"]["bootstrap_seed"])
    rows = []
    for horizon in config["horizons_months"]:
        for arm_index, arm in enumerate(config["arms"]):
            group = predictions[
                predictions.horizon_months.eq(int(horizon)) & predictions.arm.eq(arm)
            ].sort_values("subject_id")
            y = group.label.to_numpy(dtype=int)
            probability = group.prob_conversion.to_numpy(dtype=float)
            point = metric_values(y, probability, threshold)
            intervals = bootstrap_metrics(
                y, probability, threshold, replicates, base_seed + int(horizon) * 100 + arm_index
            )
            row = {
                "horizon_months": int(horizon), "arm": arm,
                "subjects": len(group), "converters": int(y.sum()),
                "threshold": threshold, **point,
            }
            for metric, (low, high) in intervals.items():
                row[f"{metric}_ci_low"] = low
                row[f"{metric}_ci_high"] = high
            rows.append(row)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "conversion_metrics.csv", index=False)

    contrasts = []
    comparisons = [
        ("sfcn_mri", "demographics"),
        ("cognition_sfcn_mri", "cognition"),
        ("cognition_sfcn_mri", "cognition_shuffled_sfcn"),
    ]
    for horizon in config["horizons_months"]:
        for comparison_index, (candidate, baseline) in enumerate(comparisons):
            contrasts.append(paired_auc_contrast(
                predictions, int(horizon), candidate, baseline, replicates,
                base_seed + int(horizon) * 1000 + comparison_index,
            ))
    contrast_frame = pd.DataFrame(contrasts)
    contrast_frame.to_csv(output / "conversion_auc_contrasts.csv", index=False)

    decision = {"analysis_status": config["analysis_status"], "horizons": {}}
    for horizon in config["horizons_months"]:
        current = contrast_frame[contrast_frame.horizon_months.eq(int(horizon))].set_index("comparison")
        mri = current.loc["sfcn_mri_minus_demographics"]
        fusion = current.loc["cognition_sfcn_mri_minus_cognition"]
        mechanism = current.loc["cognition_sfcn_mri_minus_cognition_shuffled_sfcn"]
        decision["horizons"][str(horizon)] = {
            "mri_vs_demographics_auc_supported": bool(mri.auc_difference > 0 and mri.ci_low > 0),
            "mri_increment_beyond_cognition_supported": bool(
                fusion.auc_difference > 0 and fusion.ci_low > 0
            ),
            "correct_image_mechanism_supported": bool(
                mechanism.auc_difference > 0 and mechanism.ci_low > 0
            ),
        }
    decision.update({
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "external_confirmation_required": True,
    })
    (output / "development_decision.json").write_text(json.dumps(decision, indent=2) + "\n")

    manifest = pd.read_csv(output / "conversion_manifest.csv")
    display_columns = [
        "horizon_months", "arm", "subjects", "converters", "accuracy", "auc",
        "sensitivity", "specificity", "f1", "balanced_accuracy",
    ]
    cropped_input = bool(config.get("volume_cache"))
    report_title = (
        "# Phase 3L center-cropped T1 MCI-to-AD conversion classification"
        if cropped_input else "# Phase 3H MCI-to-AD conversion classification"
    )
    report = [
        report_title,
        "",
        "Status: completed as a train-only five-fold cross-fitting experiment. Official validation and test subjects were not loaded.",
        "",
        "## Cohort",
        "",
    ]
    for horizon in config["horizons_months"]:
        group = manifest[manifest.horizon_months.eq(int(horizon))]
        report.append(
            f"- {horizon} months: {len(group)} subjects; {int(group.label.sum())} converters "
            f"({group.label.mean():.1%}) and {int((1-group.label).sum())} nonconverters."
        )
    report.extend([
        "- One earliest eligible MRI-linked MCI baseline per subject and horizon.",
        "- Exact horizon labeling; nonconverters require observed diagnosis follow-up at or beyond the horizon.",
        *(["- MRI input: interpolation-free 160x192x160 center crop at 1 mm."] if cropped_input else []),
        "- Five outer folds x three SFCN seeds x two horizons = 30 checkpoints.",
        "- Official validation and test subjects were not loaded.",
        "",
        "## Out-of-fold classification metrics",
        "",
        markdown_table(metrics[display_columns]),
        "",
        "ACC is accuracy; AUC is ROC-AUC; sensitivity treats conversion as positive; specificity treats nonconversion as negative; F1 is the positive-class F1. Threshold-dependent metrics use 0.5.",
        "",
        "## Paired AUC contrasts",
        "",
        markdown_table(contrast_frame),
        "",
        "## Requested classification figure",
        "",
        "![ACC, AUC, sensitivity, specificity, and F1 across both classification experiments](figures/fig_classification_metrics.png)",
        "",
        "The joint machine-readable table is `requested_classification_metrics.csv`.",
        "",
        "## Development decision",
        "",
        "```json",
        json.dumps(decision, indent=2),
        "```",
        "",
        "These are internal cross-fitted prognostic results, not external clinical validation.",
    ])
    (output / "report.md").write_text("\n".join(report) + "\n")


def run(config_path: Path = CONFIG) -> None:
    config = load_config(config_path)
    output = output_dir(config)
    (output / "resolved_config.json").write_text(json.dumps(config, indent=2) + "\n")
    run_crossfit(config, output)
    summarize(config, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
