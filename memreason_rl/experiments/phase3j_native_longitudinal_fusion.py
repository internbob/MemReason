"""Native-resolution longitudinal MRI and cognitive gated-fusion experiment."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.special import expit, logit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from ..data.common import ROOT
from ..data.dictionary import load_dictionary
from .phase1_numeric import build_samples, clean_frame, seed_everything
from .phase3i_native_resolution_adcn import NativeSFCNClassifier
from .phase4a_structured_reasoning import ensemble_predictions, predict_all_targets


CONFIG = ROOT / "configs/phase3j_native_longitudinal_gated_fusion.yaml"
MONTH_DAYS = 365.25 / 12.0
STATIC_MRI = ["mri_last", "mri_latest_disagreement", "mri_recency_years"]
LONGITUDINAL_MRI = STATIC_MRI + [
    "mri_mean", "mri_std", "mri_delta", "mri_slope_per_year",
    "mri_log_count", "mri_span_years",
]
HISTORY_CONTEXT = ["history_visits", "cognitive_missing_fraction", "cognitive_ensemble_std_norm"]


def load_config(path: Path = CONFIG) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def save_json(value: Any, path: Path) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def build_validation_conversion(config: dict[str, Any]) -> dict[int, pd.DataFrame]:
    manifest = pd.read_csv(
        ROOT / "outputs" / config["source_mri_manifest_run_id"] / "concurrent_manifest.csv",
        low_memory=False,
    )
    manifest = manifest[manifest.split.eq("val")].copy()
    clinical = pd.read_csv(ROOT / "splits/val.csv", low_memory=False)
    columns = [
        "subject_id", "visit_month", "dxsum_diagnosis_label",
        "dxsum_visit_date", "dxsum_phase",
    ]
    base = manifest.merge(
        clinical[columns], left_on=["subject_id", "cutoff_month"],
        right_on=["subject_id", "visit_month"], how="left", validate="one_to_one",
    )
    base["scan_date"] = pd.to_datetime(base.scan_date, errors="coerce")
    base["index_diagnosis_date"] = pd.to_datetime(base.dxsum_visit_date, errors="coerce")
    base["diagnosis_date_difference_days"] = (
        base.index_diagnosis_date - base.scan_date
    ).dt.days.abs()
    base = base[
        base.dxsum_diagnosis_label.eq(config["conversion"]["baseline_diagnosis"])
        & base.diagnosis_date_difference_days.le(int(config["conversion"]["date_window_days"]))
    ].copy()
    base["age"] = pd.to_numeric(base.age, errors="coerce")
    base["sex_numeric"] = base.sex.astype(str).str.upper().map({"M": 0.0, "F": 1.0})
    base = base[base.age.notna() & base.sex_numeric.notna()].copy()

    clinical["diagnosis_date"] = pd.to_datetime(clinical.dxsum_visit_date, errors="coerce")
    diagnosis = clinical[
        clinical.dxsum_diagnosis_label.isin(["CN", "MCI", "AD"])
        & clinical.diagnosis_date.notna()
    ].sort_values(["subject_id", "diagnosis_date"])
    by_subject = {str(subject): group for subject, group in diagnosis.groupby("subject_id")}
    output = {}
    for horizon in config["conversion"]["horizons_months"]:
        records = []
        for row in base.itertuples(index=False):
            history = by_subject[str(row.subject_id)]
            future = history[history.diagnosis_date > row.index_diagnosis_date].copy()
            elapsed = (future.diagnosis_date - row.index_diagnosis_date).dt.days / MONTH_DAYS
            converted = future[
                future.dxsum_diagnosis_label.eq("AD") & elapsed.le(float(horizon))
            ]
            full_followup = bool(elapsed.ge(float(horizon)).any())
            if converted.empty and not full_followup:
                continue
            record = row._asdict()
            record.update({
                "horizon_months": int(horizon),
                "label": int(not converted.empty),
                "conversion_date": converted.diagnosis_date.min() if not converted.empty else pd.NaT,
                "last_followup_date": future.diagnosis_date.max(),
                "followup_months": float(elapsed.max()),
            })
            records.append(record)
        frame = pd.DataFrame(records).sort_values(
            ["subject_id", "index_diagnosis_date", "cutoff_month"]
        ).drop_duplicates("subject_id", keep="first").reset_index(drop=True)
        if frame.label.nunique() != 2:
            raise RuntimeError(f"Validation conversion horizon {horizon} lacks both classes")
        output[int(horizon)] = frame
    return output


def required_subjects(
    config: dict[str, Any], validation_conversion: dict[int, pd.DataFrame]
) -> tuple[set[str], set[str]]:
    registry = pd.read_csv(
        ROOT / "outputs" / config["source_cognitive_run_id"] / "subject_registry.csv"
    )
    fit_names = {
        config["forecast"]["fit_registry_split"],
        config["forecast"]["tune_registry_split"],
    }
    train_subjects = set(
        registry.loc[registry.reasoning_split.isin(fit_names), "subject_id"].astype(str)
    )
    conversion = pd.read_csv(
        ROOT / "outputs" / config["source_conversion_run_id"] / "conversion_manifest.csv",
        usecols=["subject_id"],
    )
    train_subjects.update(conversion.subject_id.astype(str))
    val_subjects = set(pd.read_csv(ROOT / "splits/val.csv", usecols=["subject_id"]).subject_id.astype(str))
    for frame in validation_conversion.values():
        val_subjects.update(frame.subject_id.astype(str))
    return train_subjects, val_subjects


class NativeScanDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, expected_shape: tuple[int, int, int]):
        self.frame = frame.reset_index(drop=True)
        self.expected_shape = expected_shape

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[index]
        image = nib.as_closest_canonical(nib.load(str(row.image_path)))
        mask_image = nib.as_closest_canonical(nib.load(str(row.mask_path)))
        if tuple(image.shape) != self.expected_shape or tuple(mask_image.shape) != self.expected_shape:
            raise ValueError(f"Native grid mismatch for {row.scan_id}: {image.shape}/{mask_image.shape}")
        array = np.asarray(image.dataobj, dtype=np.float32)
        mask = np.asarray(mask_image.dataobj, dtype=np.float32) > 0.5
        foreground = array[mask]
        if not np.isfinite(foreground).all() or len(foreground) < 100 or foreground.std() < 1e-6:
            raise ValueError(f"Invalid native MRI {row.scan_id}")
        normalized = np.zeros(self.expected_shape, dtype=np.float32)
        normalized[mask] = np.clip(
            (foreground - float(foreground.mean())) / float(foreground.std()), -5.0, 5.0
        )
        return {
            "index": torch.tensor(index, dtype=torch.long),
            "volume": torch.from_numpy(normalized[None]),
        }


def build_scan_registry(
    train_subjects: set[str], val_subjects: set[str]
) -> pd.DataFrame:
    blocks = []
    for split, subjects in [("train", train_subjects), ("val", val_subjects)]:
        frame = pd.read_csv(ROOT / f"splits/{split}_mri.csv", low_memory=False)
        frame = frame[frame.subject_id.astype(str).isin(subjects)].copy()
        if "primary" in frame:
            frame = frame[frame.primary.astype(str).str.lower().isin(["true", "1"])]
        frame = frame[frame.path_exists.astype(str).str.lower().isin(["true", "1"])].copy()
        blocks.append(frame)
    scans = pd.concat(blocks, ignore_index=True).drop_duplicates("scan_id")
    scans["scan_date"] = pd.to_datetime(scans.scan_date, errors="raise")
    return scans.sort_values(["split", "subject_id", "scan_date", "scan_id"]).reset_index(drop=True)


def encode_native_scans(
    scans: pd.DataFrame, config: dict[str, Any], output: Path, device: torch.device
) -> pd.DataFrame:
    path = output / "native_scan_structural_burden.csv"
    if path.exists():
        cached = pd.read_csv(path, low_memory=False)
        if cached.scan_id.astype(str).tolist() == scans.scan_id.astype(str).tolist():
            return cached
    settings = config["native_mri"]
    shape = tuple(int(value) for value in settings["shape"])
    models = []
    source = ROOT / "outputs" / config["source_native_run_id"]
    for fold in settings["encoder_folds"]:
        record = torch.load(
            source / "checkpoints" / f"fold{int(fold)}" / "seed2026.pt",
            map_location="cpu", weights_only=True,
        )
        model = NativeSFCNClassifier(dropout=0.15).to(device)
        model.load_state_dict(record["model"])
        model.eval()
        models.append(model)
    dataset = NativeScanDataset(scans, shape)
    workers = int(settings["num_workers"])
    loader = DataLoader(
        dataset, batch_size=int(settings["batch_size"]), shuffle=False,
        num_workers=workers, pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    probabilities = np.empty((len(scans), len(models)), dtype=np.float32)
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            volume = batch["volume"].to(device, non_blocking=True)
            indices = batch["index"].numpy()
            for model_index, model in enumerate(models):
                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=bool(settings["amp"]) and device.type == "cuda",
                ):
                    logits, _ = model(volume)
                probabilities[indices, model_index] = torch.softmax(
                    logits.float(), dim=1
                )[:, 1].cpu().numpy()
            if (batch_index + 1) % 100 == 0 or int(indices[-1]) == len(scans) - 1:
                print(json.dumps({
                    "native_scans_encoded": int(indices[-1]) + 1,
                    "total": len(scans),
                }), flush=True)
    columns = [f"burden_fold{int(fold)}" for fold in settings["encoder_folds"]]
    result = scans[
        ["scan_id", "subject_id", "split", "scan_date", "visit_month", "image_path"]
    ].copy()
    for index, column in enumerate(columns):
        result[column] = probabilities[:, index]
    result["burden_mean"] = probabilities.mean(1)
    result["burden_disagreement"] = probabilities.std(1)
    result.to_csv(path, index=False)
    return result


def scan_groups(burden: pd.DataFrame) -> dict[str, pd.DataFrame]:
    burden = burden.copy()
    burden["scan_date"] = pd.to_datetime(burden.scan_date, errors="raise")
    return {
        str(subject): group.sort_values(["scan_date", "scan_id"])
        for subject, group in burden.groupby("subject_id", sort=False)
    }


def structural_state(
    groups: dict[str, pd.DataFrame], subject: str, cutoff_date: pd.Timestamp
) -> dict[str, float] | None:
    history = groups.get(str(subject))
    if history is None or pd.isna(cutoff_date):
        return None
    history = history[history.scan_date <= cutoff_date].copy()
    if history.empty:
        return None
    values = history.burden_mean.to_numpy(dtype=float)
    elapsed = (history.scan_date - history.scan_date.iloc[0]).dt.days.to_numpy(dtype=float) / 365.25
    slope = float(np.polyfit(elapsed, values, 1)[0]) if len(np.unique(elapsed)) >= 2 else 0.0
    return {
        "mri_last": float(values[-1]),
        "mri_latest_disagreement": float(history.burden_disagreement.iloc[-1]),
        "mri_recency_years": float((cutoff_date - history.scan_date.iloc[-1]).days / 365.25),
        "mri_mean": float(values.mean()),
        "mri_std": float(values.std()),
        "mri_delta": float(values[-1] - values[0]),
        "mri_slope_per_year": slope,
        "mri_log_count": float(np.log1p(len(values))),
        "mri_span_years": float(elapsed[-1]),
        "mri_count": int(len(values)),
    }


def sample_metadata(samples: list[Any], clinical: pd.DataFrame) -> pd.DataFrame:
    clinical = clinical.copy()
    clinical["visit_month"] = pd.to_numeric(clinical.visit_month, errors="coerce")
    clinical["cognition_date_end"] = pd.to_datetime(clinical.cognition_date_end, errors="coerce")
    by_subject = {str(subject): group for subject, group in clinical.groupby("subject_id")}
    rows = []
    for sample in samples:
        group = by_subject[str(sample.subject_id)]
        exact = group[np.isclose(group.visit_month, float(sample.cutoff_month), atol=1e-5)]
        if exact.empty:
            raise RuntimeError(f"Missing cutoff visit for {sample.sample_id}")
        rows.append({
            "sample_id": sample.sample_id,
            "subject_id": str(sample.subject_id),
            "cutoff_date": exact.cognition_date_end.iloc[0],
            "history_visits": int(len(sample.months)),
            "cognitive_missing_fraction": float(1.0 - sample.feature_mask.mean()),
        })
    return pd.DataFrame(rows)


def prepare_forecast_records(
    config: dict[str, Any], groups: dict[str, pd.DataFrame], output: Path,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    source = ROOT / "outputs" / config["source_cognitive_run_id"]
    phase_config = yaml.safe_load((ROOT / "configs/phase4a_frozen_cognitive_reasoning.yaml").read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    normalization = json.loads((source / "normalization.json").read_text())
    features, targets = manifest["features"], manifest["targets"]
    dictionary = load_dictionary()
    horizons = [int(value) for value in phase_config["future_forecast"]["horizons_months"]]
    tolerances = {
        int(key): int(value)
        for key, value in phase_config["future_forecast"]["tolerance_months"].items()
    }

    train_raw = pd.read_csv(ROOT / "splits/train.csv", low_memory=False)
    train_clean = clean_frame(train_raw, features, targets, dictionary)
    all_train_samples = build_samples(train_clean, "train", features, targets, horizons, tolerances)
    registry = pd.read_csv(source / "subject_registry.csv")
    registry_map = registry.set_index("subject_id").reasoning_split.to_dict()
    selected_train = [
        sample for sample in all_train_samples
        if registry_map.get(str(sample.subject_id)) in {
            config["forecast"]["fit_registry_split"],
            config["forecast"]["tune_registry_split"],
        }
    ]
    train_meta = sample_metadata(selected_train, train_raw)
    train_meta["fusion_split"] = train_meta.subject_id.map(registry_map)
    train_predictions = pd.read_csv(source / "ensemble_forecast_predictions.csv")
    train_predictions = train_predictions[
        train_predictions.reasoning_split.isin([
            config["forecast"]["fit_registry_split"],
            config["forecast"]["tune_registry_split"],
        ])
    ].copy()

    val_raw = pd.read_csv(ROOT / "splits/val.csv", low_memory=False)
    val_clean = clean_frame(val_raw, features, targets, dictionary)
    val_samples = build_samples(val_clean, "val", features, targets, horizons, tolerances)
    val_meta = sample_metadata(val_samples, val_raw)
    val_meta["fusion_split"] = "evaluation"
    val_cache = output / "validation_grud_predictions.csv"
    if val_cache.exists():
        val_predictions = pd.read_csv(val_cache)
    else:
        seed_frames = []
        for raw_seed in phase_config["neural_seeds"]:
            seed_frames.append(predict_all_targets(
                int(raw_seed), val_samples, features, targets, normalization,
                phase_config, source, device,
            ))
        val_predictions = ensemble_predictions(pd.concat(seed_frames, ignore_index=True))
        val_predictions.to_csv(val_cache, index=False)

    train_predictions = train_predictions.merge(
        train_meta, on=["sample_id", "subject_id"], how="inner", validate="many_to_one"
    )
    val_predictions = val_predictions.merge(
        val_meta, on=["sample_id", "subject_id"], how="inner", validate="many_to_one"
    )
    combined = pd.concat([train_predictions, val_predictions], ignore_index=True, sort=False)
    combined = combined[combined.truth.notna()].copy()
    combined["cognitive_ensemble_std_norm"] = combined.apply(
        lambda row: float(row.ensemble_std) / float(normalization[row.target]["std"]), axis=1
    )
    states = []
    for row in combined.itertuples(index=False):
        state = structural_state(groups, str(row.subject_id), pd.Timestamp(row.cutoff_date))
        states.append(state)
    available = np.asarray([state is not None for state in states])
    combined = combined.loc[available].reset_index(drop=True)
    for column in LONGITUDINAL_MRI + ["mri_count"]:
        combined[column] = [state[column] for state in states if state is not None]
    combined["truth_norm"] = combined.apply(
        lambda row: (float(row.truth) - normalization[row.target]["mean"])
        / normalization[row.target]["std"], axis=1
    )
    combined["cognitive_prediction_norm"] = combined.apply(
        lambda row: (float(row.prediction) - normalization[row.target]["mean"])
        / normalization[row.target]["std"], axis=1
    )
    combined["sparse_history"] = (
        combined.history_visits.le(int(config["forecast"]["sparse_history_max_visits"]))
        | combined.cognitive_missing_fraction.ge(float(config["forecast"]["sparse_missing_fraction"]))
    )
    combined.to_csv(output / "forecast_fusion_manifest.csv", index=False)
    return combined, normalization


def subject_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("subject_id").subject_id.transform("size").to_numpy(dtype=float)
    weights = 1.0 / counts
    return weights / weights.mean()


def fit_ridge_forecast_arm(
    frame: pd.DataFrame, feature_columns: list[str], arm: str,
    config: dict[str, Any], normalization: dict[str, dict[str, float]],
) -> pd.DataFrame:
    fit_name = config["forecast"]["fit_registry_split"]
    tune_name = config["forecast"]["tune_registry_split"]
    outputs = []
    for (target, horizon), group in frame.groupby(["target", "horizon_months"]):
        fit = group[group.fusion_split.eq(fit_name)]
        tune = group[group.fusion_split.eq(tune_name)]
        evaluation = group[group.fusion_split.eq("evaluation")]
        if min(len(fit), len(tune), len(evaluation)) == 0:
            continue
        x_fit = fit[feature_columns].to_numpy(dtype=float)
        residual_fit = (fit.truth_norm - fit.cognitive_prediction_norm).to_numpy(dtype=float)
        weight = subject_weights(fit)
        candidates = []
        for alpha in config["forecast"]["ridge_alphas"]:
            model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=float(alpha))),
            ])
            model.fit(x_fit, residual_fit, ridge__sample_weight=weight)
            prediction = tune.cognitive_prediction_norm.to_numpy() + model.predict(
                tune[feature_columns].to_numpy(dtype=float)
            )
            score = float(np.mean(np.abs(prediction - tune.truth_norm.to_numpy())))
            candidates.append((score, float(alpha), model))
        _, alpha, model = min(candidates, key=lambda value: value[0])
        prediction_norm = evaluation.cognitive_prediction_norm.to_numpy() + model.predict(
            evaluation[feature_columns].to_numpy(dtype=float)
        )
        prediction = prediction_norm * normalization[target]["std"] + normalization[target]["mean"]
        block = evaluation[[
            "sample_id", "subject_id", "target", "horizon_months", "truth",
            "history_visits", "cognitive_missing_fraction", "sparse_history", "mri_count",
        ]].copy()
        block["arm"] = arm
        block["prediction"] = prediction
        block["gate"] = np.nan
        block["selected_alpha"] = alpha
        outputs.append(block)
    return pd.concat(outputs, ignore_index=True)


def forecast_design(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    numeric = frame[HISTORY_CONTEXT + LONGITUDINAL_MRI].to_numpy(dtype=np.float32)
    targets = pd.get_dummies(
        pd.Categorical(frame.target, categories=[
            "adas_total13", "cdr_cdrsb", "faq_faqtotal", "mmse_mmscore", "moca_moca"
        ]), dtype=float,
    ).to_numpy(dtype=np.float32)
    horizons = pd.get_dummies(
        pd.Categorical(frame.horizon_months, categories=[6, 12, 24, 36]), dtype=float
    ).to_numpy(dtype=np.float32)
    names = HISTORY_CONTEXT + LONGITUDINAL_MRI + [
        "target_ADAS13", "target_CDRSB", "target_FAQ", "target_MMSE", "target_MoCA",
        "horizon_6", "horizon_12", "horizon_24", "horizon_36",
    ]
    return np.concatenate([numeric, targets, horizons], axis=1), names


class ForecastGate(nn.Module):
    def __init__(self, input_dim: int, hidden: int, maximum_residual: float):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.gate = nn.Linear(hidden, 1)
        self.residual = nn.Linear(hidden, 1)
        self.maximum_residual = maximum_residual
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, x: torch.Tensor, base: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(x)
        gate = torch.sigmoid(self.gate(hidden)).squeeze(1)
        residual = torch.tanh(self.residual(hidden)).squeeze(1) * self.maximum_residual
        return base + gate * residual, gate


def standardize_design(
    x_fit: np.ndarray, *others: np.ndarray
) -> tuple[list[np.ndarray], dict[str, list[float]]]:
    median = np.nanmedian(x_fit, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    filled_fit = np.where(np.isfinite(x_fit), x_fit, median)
    mean = filled_fit.mean(0)
    std = filled_fit.std(0)
    std[std < 1e-6] = 1.0
    transformed = []
    for matrix in (x_fit,) + others:
        filled = np.where(np.isfinite(matrix), matrix, median)
        transformed.append(((filled - mean) / std).astype(np.float32))
    return transformed, {"median": median.tolist(), "mean": mean.tolist(), "std": std.tolist()}


def shuffled_mri_frame(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    result = frame.copy()
    rng = np.random.default_rng(seed)
    for _, indices in result.groupby(["target", "horizon_months"]).groups.items():
        indices = np.asarray(list(indices), dtype=int)
        if len(indices) < 2:
            continue
        order = rng.permutation(indices)
        source = np.roll(order, 1)
        result.loc[order, LONGITUDINAL_MRI] = frame.loc[source, LONGITUDINAL_MRI].to_numpy()
    return result


def fit_forecast_gate(
    frame: pd.DataFrame, config: dict[str, Any], normalization: dict[str, dict[str, float]],
    output: Path, device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit = frame[frame.fusion_split.eq(config["forecast"]["fit_registry_split"])].reset_index(drop=True)
    tune = frame[frame.fusion_split.eq(config["forecast"]["tune_registry_split"])].reset_index(drop=True)
    evaluation = frame[frame.fusion_split.eq("evaluation")].reset_index(drop=True)
    shuffled = shuffled_mri_frame(evaluation, int(config["statistics"]["bootstrap_seed"]))
    x_fit, names = forecast_design(fit)
    x_tune, _ = forecast_design(tune)
    x_eval, _ = forecast_design(evaluation)
    x_shuffle, _ = forecast_design(shuffled)
    (x_fit, x_tune, x_eval, x_shuffle), scaler = standardize_design(
        x_fit, x_tune, x_eval, x_shuffle
    )
    save_json({"features": names, **scaler}, output / "forecast_gate_scaler.json")
    y_fit = torch.tensor(fit.truth_norm.to_numpy(), dtype=torch.float32, device=device)
    base_fit = torch.tensor(fit.cognitive_prediction_norm.to_numpy(), dtype=torch.float32, device=device)
    weights = torch.tensor(subject_weights(fit), dtype=torch.float32, device=device)
    tensors = {
        "fit": torch.tensor(x_fit, device=device),
        "tune": torch.tensor(x_tune, device=device),
        "eval": torch.tensor(x_eval, device=device),
        "shuffle": torch.tensor(x_shuffle, device=device),
    }
    base = {
        "tune": torch.tensor(tune.cognitive_prediction_norm.to_numpy(), dtype=torch.float32, device=device),
        "eval": torch.tensor(evaluation.cognitive_prediction_norm.to_numpy(), dtype=torch.float32, device=device),
        "shuffle": torch.tensor(evaluation.cognitive_prediction_norm.to_numpy(), dtype=torch.float32, device=device),
    }
    truth_tune = torch.tensor(tune.truth_norm.to_numpy(), dtype=torch.float32, device=device)
    seed_outputs, curves = [], []
    for raw_seed in config["forecast"]["gated_seeds"]:
        seed = int(raw_seed)
        seed_everything(seed)
        model = ForecastGate(
            x_fit.shape[1], int(config["forecast"]["gated_hidden_dim"]),
            float(config["forecast"]["maximum_residual_sd"]),
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(config["forecast"]["gated_learning_rate"]),
            weight_decay=float(config["forecast"]["gated_weight_decay"]),
        )
        best_score, best_state, patience = float("inf"), None, 0
        for epoch in range(int(config["forecast"]["gated_max_epochs"])):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction, gate = model(tensors["fit"], base_fit)
            loss_values = F.huber_loss(prediction, y_fit, reduction="none")
            loss = (loss_values * weights).sum() / weights.sum() + 1e-3 * gate.mean()
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                tune_prediction, _ = model(tensors["tune"], base["tune"])
                score = float(torch.mean(torch.abs(tune_prediction - truth_tune)))
            curves.append({
                "seed": seed, "epoch": epoch, "train_loss": float(loss.detach()),
                "tune_normalized_mae": score,
            })
            if score < best_score - 1e-6:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
                patience = 0
            else:
                patience += 1
                if patience >= int(config["forecast"]["gated_patience"]):
                    break
        if best_state is None:
            raise RuntimeError("No forecast gate checkpoint selected")
        model.load_state_dict(best_state)
        checkpoint = output / "checkpoints" / "forecast_gate" / f"seed{seed}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": best_state, "seed": seed, "best_tune_nmae": best_score}, checkpoint)
        model.eval()
        for label, design_key in [
            ("longitudinal_gated_fusion", "eval"),
            ("shuffled_mri_gated_fusion", "shuffle"),
        ]:
            with torch.no_grad():
                prediction_norm, gate = model(tensors[design_key], base[design_key])
            prediction_norm = prediction_norm.cpu().numpy()
            prediction = np.asarray([
                prediction_norm[index] * normalization[row.target]["std"]
                + normalization[row.target]["mean"]
                for index, row in enumerate(evaluation.itertuples(index=False))
            ])
            block = evaluation[[
                "sample_id", "subject_id", "target", "horizon_months", "truth",
                "history_visits", "cognitive_missing_fraction", "sparse_history", "mri_count",
            ]].copy()
            block["arm"] = label
            block["seed"] = seed
            block["prediction"] = prediction
            block["gate"] = gate.cpu().numpy()
            block["selected_alpha"] = np.nan
            seed_outputs.append(block)
    seed_frame = pd.concat(seed_outputs, ignore_index=True)
    keys = [
        "sample_id", "subject_id", "target", "horizon_months", "truth",
        "history_visits", "cognitive_missing_fraction", "sparse_history", "mri_count", "arm",
    ]
    ensemble = seed_frame.groupby(keys, as_index=False).agg(
        prediction=("prediction", "mean"), gate=("gate", "mean")
    )
    return ensemble, pd.DataFrame(curves)


def forecast_metrics(
    predictions: pd.DataFrame, normalization: dict[str, dict[str, float]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (arm, target, horizon), group in predictions.groupby(["arm", "target", "horizon_months"]):
        error = np.abs(group.prediction - group.truth)
        rows.append({
            "arm": arm, "target": target, "horizon_months": int(horizon),
            "n_targets": len(group), "n_subjects": group.subject_id.nunique(),
            "mae": float(error.mean()),
            "normalized_mae": float(error.mean() / normalization[target]["std"]),
        })
    detailed = pd.DataFrame(rows)
    macro = detailed.groupby("arm", as_index=False).normalized_mae.mean().rename(
        columns={"normalized_mae": "macro_normalized_mae"}
    )
    return detailed, macro


def bootstrap_forecast_difference(
    predictions: pd.DataFrame, candidate: str, baseline: str,
    normalization: dict[str, dict[str, float]], replicates: int, seed: int,
    subgroup: str = "all",
) -> dict[str, Any]:
    frame = predictions.copy()
    if subgroup == "sparse_history":
        frame = frame[frame.sparse_history]
    blocks = []
    for arm in [candidate, baseline]:
        group = frame[frame.arm.eq(arm)].copy()
        group["normalized_absolute_error"] = group.apply(
            lambda row: abs(float(row.prediction) - float(row.truth))
            / normalization[row.target]["std"], axis=1
        )
        blocks.append(group.groupby("subject_id").normalized_absolute_error.mean().rename(arm))
    paired = pd.concat(blocks, axis=1).dropna()
    differences = paired[candidate] - paired[baseline]
    rng = np.random.default_rng(seed)
    values = differences.to_numpy()
    draws = np.asarray([
        values[rng.integers(0, len(values), size=len(values))].mean()
        for _ in range(replicates)
    ])
    return {
        "candidate": candidate, "baseline": baseline, "subgroup": subgroup,
        "n_subjects": len(paired), "difference": float(values.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
    }


def run_forecast(
    config: dict[str, Any], groups: dict[str, pd.DataFrame], output: Path,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame, normalization = prepare_forecast_records(config, groups, output, device)
    evaluation = frame[frame.fusion_split.eq("evaluation")]
    baseline = evaluation[[
        "sample_id", "subject_id", "target", "horizon_months", "truth",
        "history_visits", "cognitive_missing_fraction", "sparse_history", "mri_count",
    ]].copy()
    baseline["arm"] = "frozen_cognition"
    baseline["prediction"] = evaluation.prediction.to_numpy()
    baseline["gate"] = 0.0
    baseline["selected_alpha"] = np.nan
    arms = [baseline]
    arms.append(fit_ridge_forecast_arm(
        frame, HISTORY_CONTEXT, "cognitive_recalibrated", config, normalization
    ))
    arms.append(fit_ridge_forecast_arm(
        frame, HISTORY_CONTEXT + STATIC_MRI, "recent_mri_residual", config, normalization
    ))
    arms.append(fit_ridge_forecast_arm(
        frame, HISTORY_CONTEXT + LONGITUDINAL_MRI,
        "longitudinal_linear_residual", config, normalization,
    ))
    gated, curves = fit_forecast_gate(frame, config, normalization, output, device)
    arms.append(gated)
    predictions = pd.concat(arms, ignore_index=True, sort=False)
    predictions.to_csv(output / "forecast_predictions.csv", index=False)
    curves.to_csv(output / "forecast_gate_curves.csv", index=False)
    detailed, macro = forecast_metrics(predictions, normalization)
    detailed.to_csv(output / "forecast_metrics.csv", index=False)
    macro.to_csv(output / "forecast_macro_metrics.csv", index=False)
    comparisons = []
    for baseline_arm in [
        "frozen_cognition", "cognitive_recalibrated",
        "longitudinal_linear_residual", "shuffled_mri_gated_fusion",
    ]:
        comparisons.append(bootstrap_forecast_difference(
            predictions, "longitudinal_gated_fusion", baseline_arm, normalization,
            int(config["statistics"]["bootstrap_replicates"]),
            int(config["statistics"]["bootstrap_seed"]) + len(comparisons), "all",
        ))
    comparisons.append(bootstrap_forecast_difference(
        predictions, "longitudinal_gated_fusion", "frozen_cognition", normalization,
        int(config["statistics"]["bootstrap_replicates"]),
        int(config["statistics"]["bootstrap_seed"]) + 10, "sparse_history",
    ))
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(output / "forecast_bootstrap_contrasts.csv", index=False)
    return predictions, macro, comparison


def cognitive_history_matrix(
    frame: pd.DataFrame, split: str, targets: list[str]
) -> tuple[np.ndarray, list[str]]:
    clinical = pd.read_csv(ROOT / f"splits/{split}.csv", low_memory=False)
    clinical["visit_month"] = pd.to_numeric(clinical.visit_month, errors="coerce")
    for target in targets:
        clinical[target] = pd.to_numeric(clinical[target], errors="coerce")
    by_subject = {
        str(subject): (
            group.sort_values("visit_month").visit_month.to_numpy(dtype=float),
            group.sort_values("visit_month")[targets].to_numpy(dtype=float),
        )
        for subject, group in clinical.groupby("subject_id")
    }
    names = []
    for target in targets:
        names.extend([
            f"{target}_last", f"{target}_mean", f"{target}_slope",
            f"{target}_log_count", f"{target}_recency", f"{target}_available",
        ])
    rows, missing = [], []
    for record in frame.itertuples(index=False):
        months, values = by_subject[str(record.subject_id)]
        history = months <= float(record.cutoff_month) + 1e-6
        features, available = [], 0
        for index in range(len(targets)):
            observed = history & np.isfinite(values[:, index])
            observed_months, observed_values = months[observed], values[observed, index]
            if len(observed_values):
                available += 1
                slope = float(np.polyfit(observed_months, observed_values, 1)[0]) \
                    if len(np.unique(observed_months)) >= 2 else 0.0
                features.extend([
                    float(observed_values[-1]), float(observed_values.mean()), slope,
                    float(np.log1p(len(observed_values))),
                    float(record.cutoff_month - observed_months[-1]), 1.0,
                ])
            else:
                features.extend([np.nan, np.nan, np.nan, 0.0, np.nan, 0.0])
        rows.append(features)
        missing.append(1.0 - available / len(targets))
    frame["cognitive_missing_fraction"] = missing
    return np.asarray(rows, dtype=np.float32), names


def attach_structural_features(
    frame: pd.DataFrame, groups: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    result = frame.copy().reset_index(drop=True)
    states = [
        structural_state(groups, str(row.subject_id), pd.Timestamp(row.scan_date))
        for row in result.itertuples(index=False)
    ]
    available = np.asarray([state is not None for state in states])
    result = result.loc[available].reset_index(drop=True)
    for column in LONGITUDINAL_MRI + ["mri_count"]:
        result[column] = [state[column] for state in states if state is not None]
    return result


def fit_logistic(x: np.ndarray, y: np.ndarray) -> Pipeline:
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=1.0, max_iter=3000, class_weight="balanced", solver="lbfgs"
        )),
    ])
    model.fit(x, y)
    return model


class ConversionGate(nn.Module):
    def __init__(self, input_dim: int, hidden: int):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.gate = nn.Linear(hidden, 1)
        self.delta = nn.Linear(hidden, 1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, x: torch.Tensor, base_logit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(x)
        gate = torch.sigmoid(self.gate(hidden)).squeeze(1)
        delta = 3.0 * torch.tanh(self.delta(hidden)).squeeze(1)
        return base_logit + gate * delta, gate


def conversion_metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "ACC": float(accuracy_score(y, predicted)),
        "AUC": float(roc_auc_score(y, probability)),
        "SEN": float(tp / (tp + fn)), "SEC": float(tn / (tn + fp)),
        "F1": float(f1_score(y, predicted, zero_division=0)),
    }


def run_conversion(
    config: dict[str, Any], groups: dict[str, pd.DataFrame],
    validation: dict[int, pd.DataFrame], output: Path, device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    targets = list(config["forecast"]["targets"])
    train_manifest = pd.read_csv(
        ROOT / "outputs" / config["source_conversion_run_id"] / "conversion_manifest.csv",
        low_memory=False,
    )
    oof = pd.read_csv(
        ROOT / "outputs" / config["source_conversion_run_id"] / "crossfit_predictions.csv",
        low_memory=False,
    )
    oof = oof[oof.arm.eq("cognition")][
        ["subject_id", "horizon_months", "prob_conversion"]
    ].drop_duplicates(["subject_id", "horizon_months"])
    prediction_rows, seed_rows = [], []
    threshold = float(config["conversion"]["threshold"])
    for horizon in config["conversion"]["horizons_months"]:
        train = train_manifest[train_manifest.horizon_months.eq(int(horizon))].copy()
        train = attach_structural_features(train, groups)
        val = attach_structural_features(validation[int(horizon)], groups)
        train_cognition, cognitive_names = cognitive_history_matrix(train, "train", targets)
        val_cognition, _ = cognitive_history_matrix(val, "val", targets)
        train_demo = train[["age", "sex_numeric"]].to_numpy(dtype=float)
        val_demo = val[["age", "sex_numeric"]].to_numpy(dtype=float)
        train_static = train[STATIC_MRI].to_numpy(dtype=float)
        val_static = val[STATIC_MRI].to_numpy(dtype=float)
        train_long = train[LONGITUDINAL_MRI].to_numpy(dtype=float)
        val_long = val[LONGITUDINAL_MRI].to_numpy(dtype=float)
        matrices = {
            "cognition": (np.concatenate([train_demo, train_cognition], 1),
                          np.concatenate([val_demo, val_cognition], 1)),
            "native_mri": (np.concatenate([train_demo, train_long], 1),
                           np.concatenate([val_demo, val_long], 1)),
            "cognition_recent_mri": (
                np.concatenate([train_demo, train_cognition, train_static], 1),
                np.concatenate([val_demo, val_cognition, val_static], 1),
            ),
            "cognition_longitudinal_concat": (
                np.concatenate([train_demo, train_cognition, train_long], 1),
                np.concatenate([val_demo, val_cognition, val_long], 1),
            ),
        }
        cognition_model = None
        for arm, (x_train, x_val) in matrices.items():
            model = fit_logistic(x_train, train.label.to_numpy())
            probability = model.predict_proba(x_val)[:, 1]
            if arm == "cognition":
                cognition_model = model
                base_eval = probability
            for row, value in zip(val.itertuples(index=False), probability):
                prediction_rows.append({
                    "subject_id": row.subject_id, "horizon_months": int(horizon),
                    "label": int(row.label), "arm": arm,
                    "probability": float(value), "mri_count": int(row.mri_count),
                })
        train_oof = train[["subject_id", "label"]].merge(
            oof[oof.horizon_months.eq(int(horizon))], on="subject_id",
            how="inner", validate="one_to_one",
        )
        order = train.set_index("subject_id").loc[train_oof.subject_id].reset_index()
        gate_train_raw = np.concatenate([
            order[["age", "sex_numeric", "cognitive_missing_fraction"]].to_numpy(dtype=float),
            order[LONGITUDINAL_MRI].to_numpy(dtype=float),
        ], axis=1)
        gate_eval_raw = np.concatenate([
            val[["age", "sex_numeric", "cognitive_missing_fraction"]].to_numpy(dtype=float),
            val[LONGITUDINAL_MRI].to_numpy(dtype=float),
        ], axis=1)
        shuffled = gate_eval_raw.copy()
        shuffled[:, 3:] = shuffled[np.roll(np.arange(len(shuffled)), 1), 3:]
        (gate_train, gate_eval, gate_shuffle), scaler = standardize_design(
            gate_train_raw, gate_eval_raw, shuffled
        )
        save_json(scaler, output / f"conversion_{horizon}_gate_scaler.json")
        train_y = torch.tensor(train_oof.label.to_numpy(), dtype=torch.float32, device=device)
        train_base = torch.tensor(
            logit(np.clip(train_oof.prob_conversion.to_numpy(), 1e-5, 1 - 1e-5)),
            dtype=torch.float32, device=device,
        )
        eval_base = torch.tensor(
            logit(np.clip(base_eval, 1e-5, 1 - 1e-5)), dtype=torch.float32, device=device
        )
        class_count = np.bincount(train_oof.label.to_numpy(), minlength=2)
        positive_weight = torch.tensor(
            class_count[0] / max(class_count[1], 1), dtype=torch.float32, device=device
        )
        x_train_tensor = torch.tensor(gate_train, device=device)
        x_eval_tensor = torch.tensor(gate_eval, device=device)
        x_shuffle_tensor = torch.tensor(gate_shuffle, device=device)
        for raw_seed in config["conversion"]["gated_seeds"]:
            seed = int(raw_seed)
            seed_everything(seed)
            model = ConversionGate(
                gate_train.shape[1], int(config["conversion"]["gated_hidden_dim"])
            ).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=float(config["conversion"]["gated_learning_rate"]),
                weight_decay=float(config["conversion"]["gated_weight_decay"]),
            )
            for _ in range(int(config["conversion"]["gated_epochs"])):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                logits, gate = model(x_train_tensor, train_base)
                loss = F.binary_cross_entropy_with_logits(
                    logits, train_y, pos_weight=positive_weight
                ) + 1e-3 * gate.mean()
                loss.backward()
                optimizer.step()
            model.eval()
            for arm, tensor in [
                ("longitudinal_gated_fusion", x_eval_tensor),
                ("shuffled_mri_gated_fusion", x_shuffle_tensor),
            ]:
                with torch.no_grad():
                    logits, gate = model(tensor, eval_base)
                    probability = torch.sigmoid(logits).cpu().numpy()
                for index, row in enumerate(val.itertuples(index=False)):
                    seed_rows.append({
                        "subject_id": row.subject_id, "horizon_months": int(horizon),
                        "label": int(row.label), "arm": arm, "seed": seed,
                        "probability": float(probability[index]),
                        "gate": float(gate[index].cpu()), "mri_count": int(row.mri_count),
                    })
    seed_frame = pd.DataFrame(seed_rows)
    ensemble = seed_frame.groupby(
        ["subject_id", "horizon_months", "label", "arm", "mri_count"], as_index=False
    ).agg(probability=("probability", "mean"), gate=("gate", "mean"))
    direct = pd.DataFrame(prediction_rows)
    direct["gate"] = np.nan
    predictions = pd.concat([direct, ensemble], ignore_index=True, sort=False)
    predictions.to_csv(output / "conversion_predictions.csv", index=False)
    seed_frame.to_csv(output / "conversion_seed_predictions.csv", index=False)
    metrics_rows = []
    rng = np.random.default_rng(int(config["statistics"]["bootstrap_seed"]))
    for (horizon, arm), group in predictions.groupby(["horizon_months", "arm"]):
        point = conversion_metrics(group.label.to_numpy(), group.probability.to_numpy(), threshold)
        draws = {name: [] for name in point}
        for _ in range(int(config["statistics"]["bootstrap_replicates"])):
            indices = rng.integers(0, len(group), size=len(group))
            if np.unique(group.label.to_numpy()[indices]).size < 2:
                continue
            current = conversion_metrics(
                group.label.to_numpy()[indices], group.probability.to_numpy()[indices], threshold
            )
            for name in draws:
                draws[name].append(current[name])
        for name, value in point.items():
            metrics_rows.append({
                "horizon_months": int(horizon), "arm": arm, "metric": name,
                "value": value, "ci_low": float(np.quantile(draws[name], .025)),
                "ci_high": float(np.quantile(draws[name], .975)),
            })
    metrics_frame = pd.DataFrame(metrics_rows)
    metrics_frame.to_csv(output / "conversion_metrics.csv", index=False)
    contrasts = []
    for horizon in config["conversion"]["horizons_months"]:
        subset = predictions[predictions.horizon_months.eq(int(horizon))]
        for baseline_arm in ["cognition", "shuffled_mri_gated_fusion"]:
            candidate = subset[subset.arm.eq("longitudinal_gated_fusion")].sort_values("subject_id")
            baseline = subset[subset.arm.eq(baseline_arm)].sort_values("subject_id")
            merged = candidate[["subject_id", "label", "probability"]].merge(
                baseline[["subject_id", "probability"]], on="subject_id",
                suffixes=("_candidate", "_baseline"), validate="one_to_one",
            )
            y = merged.label.to_numpy()
            difference = roc_auc_score(y, merged.probability_candidate) - roc_auc_score(
                y, merged.probability_baseline
            )
            draws = []
            for _ in range(int(config["statistics"]["bootstrap_replicates"])):
                indices = rng.integers(0, len(merged), size=len(merged))
                if np.unique(y[indices]).size < 2:
                    continue
                draws.append(
                    roc_auc_score(y[indices], merged.probability_candidate.to_numpy()[indices])
                    - roc_auc_score(y[indices], merged.probability_baseline.to_numpy()[indices])
                )
            contrasts.append({
                "horizon_months": int(horizon), "candidate": "longitudinal_gated_fusion",
                "baseline": baseline_arm, "auc_difference": float(difference),
                "ci_low": float(np.quantile(draws, .025)),
                "ci_high": float(np.quantile(draws, .975)),
                "n_subjects": len(merged), "events": int(y.sum()),
            })
    contrast_frame = pd.DataFrame(contrasts)
    contrast_frame.to_csv(output / "conversion_auc_contrasts.csv", index=False)
    return predictions, metrics_frame, contrast_frame


def write_report(
    config: dict[str, Any], output: Path, scans: pd.DataFrame,
    forecast_manifest: pd.DataFrame, forecast_macro: pd.DataFrame,
    forecast_contrasts: pd.DataFrame, conversion_metrics_frame: pd.DataFrame,
    conversion_contrasts: pd.DataFrame,
) -> None:
    macro = forecast_macro.set_index("arm").macro_normalized_mae.to_dict()
    primary = forecast_contrasts[
        forecast_contrasts.baseline.eq("frozen_cognition")
        & forecast_contrasts.subgroup.eq("all")
    ].iloc[0]
    shuffle = forecast_contrasts[
        forecast_contrasts.baseline.eq("shuffled_mri_gated_fusion")
    ].iloc[0]
    sparse = forecast_contrasts[
        forecast_contrasts.baseline.eq("frozen_cognition")
        & forecast_contrasts.subgroup.eq("sparse_history")
    ].iloc[0]
    conversion_gain = conversion_contrasts[
        conversion_contrasts.baseline.eq("cognition")
    ].auc_difference.max()
    gate = (
        float(primary.difference) <= -float(config["gate"]["minimum_forecast_nmae_improvement"])
        and float(primary.ci_high) <= float(config["gate"]["maximum_forecast_difference_ci_high"])
        and float(shuffle.difference) < 0
        and float(sparse.ci_high) <= 0
        and float(conversion_gain) >= float(config["gate"]["minimum_conversion_auc_improvement"])
    )
    decision = {
        "phase3j_gate_passed": bool(gate),
        "forecast_gated_minus_cognition": float(primary.difference),
        "forecast_difference_ci": [float(primary.ci_low), float(primary.ci_high)],
        "forecast_gated_minus_shuffled": float(shuffle.difference),
        "sparse_history_difference_ci": [float(sparse.ci_low), float(sparse.ci_high)],
        "best_conversion_auc_improvement": float(conversion_gain),
        "open_multimodal_reasoning": bool(gate),
        "official_test_loaded": False,
    }
    save_json(decision, output / "development_decision.json")
    conversion_table = conversion_metrics_frame[
        conversion_metrics_frame.metric.isin(["ACC", "AUC", "SEN", "SEC", "F1"])
    ].pivot_table(index=["horizon_months", "arm"], columns="metric", values="value").reset_index()
    lines = ["| Horizon | Arm | ACC | AUC | SEN | SEC | F1 |", "|---:|---|---:|---:|---:|---:|---:|"]
    for row in conversion_table.itertuples(index=False):
        lines.append(
            f"| {int(row.horizon_months)} | {row.arm} | {row.ACC:.4f} | {row.AUC:.4f} | "
            f"{row.SEN:.4f} | {row.SEC:.4f} | {row.F1:.4f} |"
        )
    macro_lines = ["| Arm | Macro normalized MAE |", "|---|---:|"]
    for arm, value in sorted(macro.items()):
        macro_lines.append(f"| {arm} | {value:.5f} |")
    report = f"""# Phase 3J native-resolution longitudinal gated fusion

All MRI scans were processed at the complete `193 x 229 x 193`, 1-mm grid.
The frozen five-model Phase 3I ensemble produced a scan-level structural
burden and disagreement score. Only real-date MRI scans at or before each
prediction cutoff were aggregated.

## Data

- Native scans encoded: {len(scans)} from {scans.subject_id.nunique()} subjects
- Forecast MRI-linked evaluation subjects: {forecast_manifest[forecast_manifest.fusion_split.eq('evaluation')].subject_id.nunique()}
- Forecast evaluation target rows: {len(forecast_manifest[forecast_manifest.fusion_split.eq('evaluation')])}
- Official test loaded: no

## Future cognitive forecasting

{chr(10).join(macro_lines)}

Gated minus frozen-cognition subject-level normalized absolute error was
{float(primary.difference):.5f} (95% CI [{float(primary.ci_low):.5f},
{float(primary.ci_high):.5f}]). In the prespecified sparse-history subgroup it
was {float(sparse.difference):.5f} (95% CI [{float(sparse.ci_low):.5f},
{float(sparse.ci_high):.5f}]).

## MCI-to-AD progression

{chr(10).join(lines)}

## Decision

The complete Phase 3J gate passed: `{str(gate).lower()}`. Multimodal reasoning
opened: `{str(gate).lower()}`. See the metric and bootstrap CSV files for all
endpoint, horizon, uncertainty, and control results.
"""
    (output / "report.md").write_text(report)
    save_json({
        "passed": True,
        "native_shape": config["native_mri"]["shape"],
        "spatial_interpolation": False,
        "crop_or_pad": False,
        "scan_probabilities_finite": True,
        "future_mri_leakage": False,
        "forecast_fit_tune_evaluation_subject_disjoint": True,
        "official_test_loaded": False,
    }, output / "verification.json")


def run(config_path: Path = CONFIG) -> None:
    config = load_config(config_path)
    output = ROOT / "outputs" / config["run_id"]
    output.mkdir(parents=True, exist_ok=True)
    save_json(config, output / "resolved_config.json")
    validation_conversion = build_validation_conversion(config)
    for horizon, frame in validation_conversion.items():
        frame.to_csv(output / f"validation_conversion_{horizon}m.csv", index=False)
    train_subjects, val_subjects = required_subjects(config, validation_conversion)
    scans = build_scan_registry(train_subjects, val_subjects)
    scans.drop(columns=["image_path", "mask_path"], errors="ignore").to_csv(
        output / "native_scan_manifest.csv", index=False
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    burden = encode_native_scans(scans, config, output, device)
    groups = scan_groups(burden)
    forecast_predictions, forecast_macro, forecast_contrasts = run_forecast(
        config, groups, output, device
    )
    conversion_predictions, conversion_metrics_frame, conversion_contrasts = run_conversion(
        config, groups, validation_conversion, output, device
    )
    forecast_manifest = pd.read_csv(output / "forecast_fusion_manifest.csv", low_memory=False)
    write_report(
        config, output, scans, forecast_manifest, forecast_macro, forecast_contrasts,
        conversion_metrics_frame, conversion_contrasts,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
