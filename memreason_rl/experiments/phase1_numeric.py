"""Formal Phase 1A: classical and deep numeric longitudinal baselines."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
# Required by PyTorch deterministic CUDA matrix operations; set before CUDA initializes.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from scipy import stats
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import yaml

from ..data.common import ROOT, read_csv, save_csv, save_json, valid_value
from ..data.dictionary import load_dictionary


@dataclass
class ForecastSample:
    sample_id: str
    subject_id: str
    split: str
    cutoff_month: float
    horizon_months: int
    months: np.ndarray
    features: np.ndarray
    feature_mask: np.ndarray
    history_targets: np.ndarray
    history_target_mask: np.ndarray
    targets: np.ndarray
    target_mask: np.ndarray
    target_months: np.ndarray


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        # Fused attention backward kernels are non-deterministic in torch 2.5.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def select_numeric_features(config: dict[str, Any], dictionary: dict[str, Any]) -> list[str]:
    patterns = [re.compile(pattern) for pattern in config["features"]["exclusions"]]
    targets = set(config["targets"])
    selected = []
    for key, entry in dictionary.items():
        if key in targets or key.startswith(("dxsum_", "faq_")):
            continue
        if config["features"]["require_use_for_state"] and not entry["use_for_state"]:
            continue
        if entry.get("default_drop") or any(pattern.search(key) for pattern in patterns):
            continue
        if entry["value_type"] == "categorical":
            continue
        selected.append(key)
    if not selected:
        raise ValueError("No numeric features selected")
    return selected


def clean_frame(
    frame: pd.DataFrame,
    features: list[str],
    targets: list[str],
    dictionary: dict[str, Any],
) -> pd.DataFrame:
    result = frame.copy()
    for column in features + targets:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        if column in dictionary:
            valid = result[column].map(lambda value: valid_value(value, dictionary[column]))
            result.loc[~valid, column] = np.nan
    result["visit_month"] = pd.to_numeric(result["visit_month"], errors="raise")
    return result.sort_values(["subject_id", "visit_month"], kind="stable")


def fit_normalization(
    train: pd.DataFrame, features: list[str], targets: list[str]
) -> dict[str, dict[str, float]]:
    output = {}
    for column in features + targets:
        values = train[column].dropna().to_numpy(dtype=float)
        if not len(values):
            raise ValueError(f"No training observations for {column}")
        std = float(np.std(values))
        output[column] = {
            "mean": float(np.mean(values)),
            "std": max(std, 1e-6),
            "n": int(len(values)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }
    return output


def build_samples(
    frame: pd.DataFrame,
    split: str,
    features: list[str],
    targets: list[str],
    horizons: list[int],
    tolerances: dict[int, int],
) -> list[ForecastSample]:
    samples: list[ForecastSample] = []
    for subject, group in frame.groupby("subject_id", sort=True):
        group = group.sort_values("visit_month", kind="stable").reset_index(drop=True)
        months = group["visit_month"].to_numpy(dtype=float)
        feature_values = group[features].to_numpy(dtype=float)
        feature_mask = np.isfinite(feature_values)
        target_history = group[targets].to_numpy(dtype=float)
        target_history_mask = np.isfinite(target_history)
        for cutoff_index, cutoff in enumerate(months):
            if not feature_mask[: cutoff_index + 1].any():
                continue
            future = months > cutoff
            if not future.any():
                continue
            for horizon in horizons:
                truth = np.full(len(targets), np.nan, dtype=np.float32)
                truth_month = np.full(len(targets), np.nan, dtype=np.float32)
                truth_mask = np.zeros(len(targets), dtype=bool)
                center = cutoff + horizon
                tolerance = tolerances[horizon]
                for target_index in range(len(targets)):
                    candidates = np.flatnonzero(
                        future
                        & target_history_mask[:, target_index]
                        & (np.abs(months - center) <= tolerance)
                    )
                    if not len(candidates):
                        continue
                    distance = np.abs(months[candidates] - center)
                    candidate = candidates[np.lexsort((months[candidates], distance))[0]]
                    truth[target_index] = target_history[candidate, target_index]
                    truth_month[target_index] = months[candidate]
                    truth_mask[target_index] = True
                if not truth_mask.any():
                    continue
                samples.append(
                    ForecastSample(
                        sample_id=f"{subject}:m{cutoff:g}:h{horizon}",
                        subject_id=subject,
                        split=split,
                        cutoff_month=float(cutoff),
                        horizon_months=int(horizon),
                        months=months[: cutoff_index + 1].copy(),
                        features=feature_values[: cutoff_index + 1].astype(np.float32),
                        feature_mask=feature_mask[: cutoff_index + 1].copy(),
                        history_targets=target_history[: cutoff_index + 1].astype(np.float32),
                        history_target_mask=target_history_mask[: cutoff_index + 1].copy(),
                        targets=truth,
                        target_mask=truth_mask,
                        target_months=truth_month,
                    )
                )
    return samples


class NumericForecastDataset(Dataset):
    def __init__(self, samples: list[ForecastSample]):
        self.samples = samples
    def __len__(self) -> int:
        return len(self.samples)
    def __getitem__(self, index: int) -> ForecastSample:
        return self.samples[index]


class NumericCollator:
    def __init__(
        self,
        features: list[str],
        targets: list[str],
        normalization: dict[str, dict[str, float]],
    ):
        self.features = features
        self.targets = targets
        self.feature_mean = np.array([normalization[k]["mean"] for k in features], dtype=np.float32)
        self.feature_std = np.array([normalization[k]["std"] for k in features], dtype=np.float32)
        self.target_mean = np.array([normalization[k]["mean"] for k in targets], dtype=np.float32)
        self.target_std = np.array([normalization[k]["std"] for k in targets], dtype=np.float32)

    def __call__(self, samples: list[ForecastSample]) -> dict[str, Any]:
        batch = len(samples)
        length = max(len(sample.months) for sample in samples)
        n_features = len(self.features)
        values = np.zeros((batch, length, n_features), dtype=np.float32)
        mask = np.zeros((batch, length, n_features), dtype=bool)
        times = np.zeros((batch, length), dtype=np.float32)
        event_mask = np.zeros((batch, length), dtype=bool)
        targets = np.zeros((batch, len(self.targets)), dtype=np.float32)
        target_mask = np.zeros((batch, len(self.targets)), dtype=bool)
        for b, sample in enumerate(samples):
            n = len(sample.months)
            normalized = (sample.features - self.feature_mean) / self.feature_std
            values[b, :n] = np.where(sample.feature_mask, normalized, 0.0)
            mask[b, :n] = sample.feature_mask
            times[b, :n] = sample.months / 12.0
            event_mask[b, :n] = True
            targets[b] = np.where(
                sample.target_mask,
                (sample.targets - self.target_mean) / self.target_std,
                0.0,
            )
            target_mask[b] = sample.target_mask
        return {
            "values": torch.from_numpy(values),
            "mask": torch.from_numpy(mask),
            "times": torch.from_numpy(times),
            "event_mask": torch.from_numpy(event_mask),
            "horizon": torch.tensor([s.horizon_months / 12 for s in samples], dtype=torch.float32),
            "targets": torch.from_numpy(targets),
            "target_mask": torch.from_numpy(target_mask),
            "samples": samples,
        }


class GRUDForecaster(nn.Module):
    def __init__(self, n_features: int, n_targets: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.decay_weight = nn.Parameter(torch.zeros(n_features))
        self.decay_bias = nn.Parameter(torch.zeros(n_features))
        self.cells = nn.ModuleList(
            [nn.GRUCell(n_features * 2 + 2, hidden)]
            + [nn.GRUCell(hidden, hidden) for _ in range(layers - 1)]
        )
        self.dropout = nn.Dropout(dropout)
        self.horizon = nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.output = nn.Linear(hidden * 2, n_targets)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x, observed, times, valid = batch["values"], batch["mask"], batch["times"], batch["event_mask"]
        batch_size, steps, features = x.shape
        hidden = [x.new_zeros(batch_size, cell.hidden_size) for cell in self.cells]
        last_value = x.new_zeros(batch_size, features)
        last_observed_time = times[:, :1].expand(-1, features).clone()
        previous_time = times[:, 0]
        for step in range(steps):
            event_valid = valid[:, step]
            current_time = times[:, step]
            elapsed = (current_time[:, None] - last_observed_time).clamp_min(0)
            gamma = torch.exp(-F.relu(self.decay_weight[None] * elapsed + self.decay_bias[None]))
            imputed = torch.where(observed[:, step], x[:, step], gamma * last_value)
            gap = (current_time - previous_time).clamp_min(0)
            cell_input = torch.cat([imputed, observed[:, step].to(x.dtype), current_time[:, None], gap[:, None]], dim=-1)
            for layer, cell in enumerate(self.cells):
                proposed = cell(cell_input, hidden[layer])
                hidden[layer] = torch.where(event_valid[:, None], proposed, hidden[layer])
                cell_input = self.dropout(hidden[layer])
            last_value = torch.where(observed[:, step] & event_valid[:, None], x[:, step], last_value)
            last_observed_time = torch.where(
                observed[:, step] & event_valid[:, None], current_time[:, None], last_observed_time
            )
            previous_time = torch.where(event_valid, current_time, previous_time)
        horizon = self.horizon(batch["horizon"][:, None])
        return self.output(torch.cat([hidden[-1], horizon], dim=-1))


class TimeAwareTransformerForecaster(nn.Module):
    def __init__(self, n_features: int, n_targets: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.event = nn.Sequential(nn.Linear(n_features * 2, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.time = nn.Sequential(nn.Linear(4, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        layer = nn.TransformerEncoderLayer(
            hidden, 4, hidden * 4, dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.horizon = nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.output = nn.Linear(hidden * 2, n_targets)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        values, mask, times, valid = batch["values"], batch["mask"], batch["times"], batch["event_mask"]
        gaps = torch.cat([torch.zeros_like(times[:, :1]), (times[:, 1:] - times[:, :-1]).clamp_min(0)], dim=1)
        time_features = torch.stack([times, torch.sin(times), torch.cos(times), torch.log1p(gaps)], dim=-1)
        tokens = self.event(torch.cat([values, mask.to(values.dtype)], dim=-1)) + self.time(time_features)
        encoded = self.encoder(tokens, src_key_padding_mask=~valid)
        last = valid.sum(1) - 1
        summary = encoded[torch.arange(len(encoded), device=encoded.device), last]
        horizon = self.horizon(batch["horizon"][:, None])
        return self.output(torch.cat([summary, horizon], dim=-1))


def masked_huber(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not mask.any():
        return prediction.sum() * 0
    return F.huber_loss(prediction[mask], target[mask], delta=1.0)


def prediction_rows(
    samples: list[ForecastSample],
    prediction: np.ndarray,
    model: str,
    seed: int | None,
    target_names: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for sample, values in zip(samples, prediction):
        for target_index, target in enumerate(target_names):
            if not sample.target_mask[target_index]:
                continue
            value = values[target_index]
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "subject_id": sample.subject_id,
                    "split": sample.split,
                    "model": model,
                    "seed": seed,
                    "target": target,
                    "horizon_months": sample.horizon_months,
                    "cutoff_month": sample.cutoff_month,
                    "target_month": float(sample.target_months[target_index]),
                    "truth": float(sample.targets[target_index]),
                    "prediction": float(value) if np.isfinite(value) else np.nan,
                }
            )
    return rows


def evaluate_model(
    model: nn.Module,
    samples: list[ForecastSample],
    collator: NumericCollator,
    batch_size: int,
    device: torch.device,
    target_names: list[str],
    model_name: str,
    seed: int,
) -> pd.DataFrame:
    loader = DataLoader(NumericForecastDataset(samples), batch_size=batch_size, shuffle=False, collate_fn=collator)
    model.eval()
    predictions = []
    ordered_samples = []
    means = np.array([collator.target_mean], dtype=np.float32)
    stds = np.array([collator.target_std], dtype=np.float32)
    with torch.no_grad():
        for batch in loader:
            local_samples = batch.pop("samples")
            batch = {k: v.to(device) for k, v in batch.items()}
            prediction = model(batch).cpu().numpy() * stds + means
            predictions.append(prediction)
            ordered_samples.extend(local_samples)
    return pd.DataFrame(
        prediction_rows(ordered_samples, np.concatenate(predictions), model_name, seed, target_names)
    )


def compute_metrics(predictions: pd.DataFrame, normalization: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows = []
    keys = ["split", "model", "seed", "target", "horizon_months"]
    for key, group in predictions.groupby(keys, dropna=False, sort=True):
        available = group.prediction.notna() & np.isfinite(group.prediction)
        observed = group.loc[available]
        error = observed.prediction - observed.truth
        target = key[3]
        rows.append(
            dict(
                zip(keys, key),
                n_targets=int(len(group)),
                n_predictions=int(available.sum()),
                n_subjects=int(observed.subject_id.nunique()),
                coverage=float(available.mean()),
                mae=float(error.abs().mean()) if len(observed) else None,
                normalized_mae=float(error.abs().mean() / normalization[target]["std"]) if len(observed) else None,
                rmse=float(np.sqrt(np.mean(error**2))) if len(observed) else None,
                pearson=float(observed.prediction.corr(observed.truth)) if len(observed) > 2 and observed.prediction.std() > 0 and observed.truth.std() > 0 else None,
                spearman=float(observed.prediction.corr(observed.truth, method="spearman")) if len(observed) > 2 and observed.prediction.nunique() > 1 and observed.truth.nunique() > 1 else None,
            )
        )
    return pd.DataFrame(rows)


def macro_score(predictions: pd.DataFrame, normalization: dict[str, dict[str, float]]) -> float:
    metrics = compute_metrics(predictions, normalization)
    return float(metrics.normalized_mae.mean())


def train_neural(
    architecture: str,
    seed: int,
    train_samples: list[ForecastSample],
    val_samples: list[ForecastSample],
    test_samples: list[ForecastSample],
    features: list[str],
    targets: list[str],
    normalization: dict[str, dict[str, float]],
    config: dict[str, Any],
    output: Path,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_everything(seed)
    settings = config["training"]
    collator = NumericCollator(features, targets, normalization)
    train_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        NumericForecastDataset(train_samples), batch_size=settings["batch_size"], shuffle=True,
        collate_fn=collator, generator=train_generator, num_workers=0
    )
    cls = GRUDForecaster if architecture == "GRU-D" else TimeAwareTransformerForecaster
    model = cls(
        len(features), len(targets), settings["hidden_dim"], settings["layers"], settings["dropout"]
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    best_score = float("inf")
    best_state = None
    patience = 0
    curves = []
    for epoch in range(settings["max_epochs"]):
        model.train()
        losses = []
        for batch in train_loader:
            batch.pop("samples")
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            loss = masked_huber(prediction, batch["targets"], batch["target_mask"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite {architecture} loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"])
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = evaluate_model(
            model, val_samples, collator, settings["batch_size"], device, targets, architecture, seed
        )
        score = macro_score(validation, normalization)
        curves.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_macro_normalized_mae": score})
        print(json.dumps({"model": architecture, "seed": seed, **curves[-1]}), flush=True)
        if score < best_score - 1e-5:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= settings["patience"]:
                break
    if best_state is None:
        raise RuntimeError("No neural checkpoint selected")
    model.load_state_dict(best_state)
    run_dir = output / "checkpoints" / architecture.lower().replace("-", "_") / str(seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "features": features, "targets": targets, "seed": seed, "architecture": architecture}, run_dir / "best.pt")
    save_csv(pd.DataFrame(curves), run_dir / "curves.csv")
    val_predictions = evaluate_model(model, val_samples, collator, settings["batch_size"], device, targets, architecture, seed)
    test_predictions = evaluate_model(model, test_samples, collator, settings["batch_size"], device, targets, architecture, seed)
    return val_predictions, test_predictions


def locf_predictions(samples: list[ForecastSample], targets: list[str]) -> pd.DataFrame:
    prediction = np.full((len(samples), len(targets)), np.nan, dtype=float)
    for row, sample in enumerate(samples):
        for target_index in range(len(targets)):
            indices = np.flatnonzero(sample.history_target_mask[:, target_index])
            if len(indices):
                prediction[row, target_index] = sample.history_targets[indices[-1], target_index]
    return pd.DataFrame(prediction_rows(samples, prediction, "LOCF", None, targets))


class MixedEffectsForecaster:
    def __init__(self, target: str):
        self.target = target
        self.fixed = None
        self.random_cov = None
        self.residual_variance = None
        self.random_slope = True
        self.diagnostics: dict[str, Any] = {}

    def fit(self, frame: pd.DataFrame) -> "MixedEffectsForecaster":
        import statsmodels.api as sm
        data = frame[["subject_id", "visit_month", self.target]].dropna().copy()
        data["time_years"] = data.visit_month / 12.0
        exog = sm.add_constant(data[["time_years"]], has_constant="add")
        attempts = [(True, exog.to_numpy()), (False, np.ones((len(data), 1)))]
        result = None
        messages = []
        for random_slope, exog_re in attempts:
            try:
                model = sm.MixedLM(data[self.target].to_numpy(), exog, groups=data.subject_id, exog_re=exog_re)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    fitted = model.fit(reml=True, method="lbfgs", maxiter=300, disp=False)
                messages.extend(str(item.message) for item in caught)
                if np.isfinite(fitted.fe_params).all() and np.isfinite(fitted.scale):
                    result = fitted
                    self.random_slope = random_slope
                    break
            except Exception as error:  # diagnostics preserve the fallback reason
                messages.append(repr(error))
        if result is None:
            raise RuntimeError(f"LME failed for {self.target}: {messages}")
        self.fixed = np.asarray(result.fe_params, dtype=float)
        self.random_cov = np.asarray(result.cov_re, dtype=float)
        self.residual_variance = max(float(result.scale), 1e-8)
        residual = np.asarray(result.resid, dtype=float)
        sample = residual if len(residual) <= 5000 else np.random.default_rng(2026).choice(residual, 5000, replace=False)
        shapiro = stats.shapiro(sample)
        self.diagnostics = {
            "target": self.target,
            "n_observations": int(len(data)),
            "n_subjects": int(data.subject_id.nunique()),
            "converged": bool(result.converged),
            "random_effects": "intercept+slope" if self.random_slope else "intercept_only",
            "fixed_intercept": float(self.fixed[0]),
            "fixed_time_slope_per_year": float(self.fixed[1]),
            "residual_variance": self.residual_variance,
            "random_covariance": self.random_cov.tolist(),
            "residual_skew": float(stats.skew(residual)),
            "residual_excess_kurtosis": float(stats.kurtosis(residual)),
            "shapiro_sample_n": int(len(sample)),
            "shapiro_w": float(shapiro.statistic),
            "shapiro_p": float(shapiro.pvalue),
            "fit_warnings_or_errors": messages,
        }
        return self

    def predict(self, months: np.ndarray, values: np.ndarray, mask: np.ndarray, future_month: float) -> float:
        time = months[mask] / 12.0
        y = values[mask]
        x = np.column_stack([np.ones(len(time)), time])
        if self.random_slope:
            z = x
            z_future = np.array([1.0, future_month / 12.0])
        else:
            z = np.ones((len(time), 1))
            z_future = np.ones(1)
        fixed_future = np.array([1.0, future_month / 12.0]) @ self.fixed
        if not len(y):
            return float(fixed_future)
        variance = z @ self.random_cov @ z.T + self.residual_variance * np.eye(len(y))
        random_effect = self.random_cov @ z.T @ np.linalg.pinv(variance) @ (y - x @ self.fixed)
        return float(fixed_future + z_future @ random_effect)


def fit_lme_models(
    train: pd.DataFrame, targets: list[str]
) -> list[MixedEffectsForecaster]:
    return [MixedEffectsForecaster(target).fit(train) for target in targets]


def lme_predictions(
    models: list[MixedEffectsForecaster],
    samples: list[ForecastSample],
    targets: list[str],
) -> pd.DataFrame:
    prediction = np.full((len(samples), len(targets)), np.nan, dtype=float)
    for row, sample in enumerate(samples):
        for target_index, model in enumerate(models):
            prediction[row, target_index] = model.predict(
                sample.months,
                sample.history_targets[:, target_index],
                sample.history_target_mask[:, target_index],
                sample.cutoff_month + sample.horizon_months,
            )
    return pd.DataFrame(prediction_rows(samples, prediction, "LME", None, targets))


def ensemble_predictions(predictions: pd.DataFrame, architecture: str) -> pd.DataFrame:
    selected = predictions[predictions.model.eq(architecture)].copy()
    keys = ["sample_id", "subject_id", "split", "target", "horizon_months", "cutoff_month", "target_month", "truth"]
    ensemble = selected.groupby(keys, as_index=False).prediction.mean()
    ensemble["model"] = architecture + " ensemble"
    ensemble["seed"] = pd.NA
    return ensemble


def subject_bootstrap_mae_difference(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    keys = ["sample_id", "subject_id", "target", "horizon_months", "truth"]
    paired = candidate[keys + ["prediction"]].merge(
        baseline[keys + ["prediction"]], on=keys, suffixes=("_candidate", "_baseline"), validate="one_to_one"
    ).dropna()
    paired["candidate_error"] = (paired.prediction_candidate - paired.truth).abs()
    paired["baseline_error"] = (paired.prediction_baseline - paired.truth).abs()
    subjects = paired.subject_id.unique()
    rng = np.random.default_rng(seed)
    differences = []
    by_subject = {subject: group for subject, group in paired.groupby("subject_id")}
    for _ in range(replicates):
        sampled = rng.choice(subjects, len(subjects), replace=True)
        values = pd.concat([by_subject[subject] for subject in sampled], ignore_index=True)
        differences.append(float(values.candidate_error.mean() - values.baseline_error.mean()))
    alpha = 0.025
    point = float(paired.candidate_error.mean() - paired.baseline_error.mean())
    return {
        "candidate": str(candidate.model.iloc[0]),
        "baseline": str(baseline.model.iloc[0]),
        "n_paired_targets": int(len(paired)),
        "n_subjects": int(len(subjects)),
        "mae_difference": point,
        "ci_lower": float(np.quantile(differences, alpha)),
        "ci_upper": float(np.quantile(differences, 1 - alpha)),
        "bootstrap_probability_candidate_better": float(np.mean(np.asarray(differences) < 0)),
        "bootstrap_unit": "subject",
        "replicates": replicates,
    }


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without the optional tabulate dependency."""
    if frame.empty:
        return "No rows."
    printable = frame.copy()
    printable = printable.map(
        lambda value: "" if pd.isna(value) else (f"{value:.4g}" if isinstance(value, float) else str(value))
    )
    headers = [str(column).replace("|", "\\|") for column in printable.columns]
    rows = [headers, ["---"] * len(headers)]
    rows.extend(
        [[str(value).replace("|", "\\|") for value in record] for record in printable.itertuples(index=False, name=None)]
    )
    return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def write_report(
    output: Path,
    config: dict[str, Any],
    features: list[str],
    sample_counts: dict[str, int],
    validation_metrics: pd.DataFrame,
    test_metrics: pd.DataFrame,
    selection: dict[str, Any],
    bootstrap: list[dict[str, Any]],
) -> None:
    summary = test_metrics[
        test_metrics.model.isin(["LOCF", "LME", "GRU-D ensemble", "Transformer ensemble"])
    ][["model", "target", "horizon_months", "n_targets", "coverage", "mae", "rmse", "spearman"]]
    text = f"""# Phase 1A numeric forecasting results

Status: completed according to the predeclared protocol. This is a total-score forecasting baseline experiment, not evidence for semantic, MRI or RL contributions.

## Data and model selection

- Numeric input features: {len(features)}
- Samples: {sample_counts}
- Neural seeds: {config['neural_seeds']}
- Validation macro normalized MAE selected architecture: **{selection['selected_architecture']}**
- Test data were evaluated after validation training and configuration freeze.

## Test metrics

{markdown_table(summary)}

## Paired subject bootstrap

Negative pooled raw-MAE differences favor the candidate. Confidence intervals account for repeated targets within subjects by resampling subjects. Because endpoint units differ, the per-endpoint metrics and macro normalized MAE are the primary summaries; this pooled comparison is supportive.

{markdown_table(pd.DataFrame(bootstrap))}

## Limits

- Forecast time is nominal ADNI visit month; it is unrelated to exact MRI scan-date matching.
- LOCF coverage is below 100% when the target scale has no historical observation.
- LME residual diagnostics are reported separately; primary comparison uses MAE and subject-cluster bootstrap rather than normal-theory p-values.
- Endpoint/horizon groups with very small n must not be generalized; inspect `n_targets` before interpreting correlation or error estimates.
- Items with unresolved encodings (MoCA delayed-word/serial/letter fields, ADAS q13, RAVLT and Logical Memory) were excluded before training.
- Test metrics must not be used to change this experiment's model or hyperparameters. Any revised experiment requires a new run identifier.
"""
    (output / "report.md").write_text(text)


def run(config_path: Path, smoke: bool = False) -> None:
    config = load_config(config_path)
    output = ROOT / "outputs" / (config["run_id"] + ("_smoke" if smoke else ""))
    output.mkdir(parents=True, exist_ok=True)
    dictionary = load_dictionary()
    features = select_numeric_features(config, dictionary)
    targets = list(config["targets"])
    frames = {
        split: clean_frame(read_csv(ROOT / f"splits/{split}.csv"), features, targets, dictionary)
        for split in ["train", "val", "test"]
    }
    normalization = fit_normalization(frames["train"], features, targets)
    horizons = config["forecast"]["horizons_months"]
    tolerances = {int(k): int(v) for k, v in config["forecast"]["tolerance_months"].items()}
    samples = {
        split: build_samples(frames[split], split, features, targets, horizons, tolerances)
        for split in frames
    }
    if smoke:
        samples = {key: value[: min(len(value), 256)] for key, value in samples.items()}
        config = copy.deepcopy(config)
        config["neural_seeds"] = [config["neural_seeds"][0]]
        config["training"]["max_epochs"] = 2
        config["training"]["patience"] = 2
        config["training"]["batch_size"] = 64
    save_json(
        {
            "run_id": config["run_id"],
            "engineering_smoke_only": smoke,
            "features": features,
            "targets": targets,
            "sample_counts": {key: len(value) for key, value in samples.items()},
            "normalization": normalization,
            "diagnosis_input": False,
            "deterministic_algorithms": True,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "cuda_attention_backend": "math_sdp",
            "test_evaluated": not smoke,
        },
        output / "manifest.json",
    )
    save_json(config, output / "resolved_config.json")
    print(json.dumps({"features": len(features), "samples": {k: len(v) for k, v in samples.items()}}), flush=True)

    classical_val = [locf_predictions(samples["val"], targets)]
    classical_test = [locf_predictions(samples["test"], targets)]
    lme_models = fit_lme_models(frames["train"], targets)
    lme_val = lme_predictions(lme_models, samples["val"], targets)
    lme_test = lme_predictions(lme_models, samples["test"], targets)
    classical_val.append(lme_val)
    classical_test.append(lme_test)
    save_json({"models": [model.diagnostics for model in lme_models]}, output / "lme_diagnostics.json")

    device = torch.device("cuda" if torch.cuda.is_available() and not smoke else "cpu")
    neural_val, neural_test = [], []
    for architecture in ["GRU-D", "Transformer"]:
        for seed in config["neural_seeds"]:
            val_predictions, test_predictions = train_neural(
                architecture, seed, samples["train"], samples["val"], samples["test"],
                features, targets, normalization, config, output, device
            )
            neural_val.append(val_predictions)
            neural_test.append(test_predictions)
    validation = pd.concat(classical_val + neural_val, ignore_index=True)
    test = pd.concat(classical_test + neural_test, ignore_index=True)
    for architecture in ["GRU-D", "Transformer"]:
        validation = pd.concat([validation, ensemble_predictions(validation, architecture)], ignore_index=True)
        test = pd.concat([test, ensemble_predictions(test, architecture)], ignore_index=True)
    save_csv(validation, output / "validation_predictions.csv")
    save_csv(test, output / "test_predictions.csv")
    validation_metrics = compute_metrics(validation, normalization)
    test_metrics = compute_metrics(test, normalization)
    save_csv(validation_metrics, output / "validation_metrics.csv")
    save_csv(test_metrics, output / "test_metrics.csv")

    architecture_scores = {}
    for architecture in ["GRU-D", "Transformer"]:
        per_seed = {
            int(seed): macro_score(frame, normalization)
            for seed, frame in validation[validation.model.eq(architecture)].groupby("seed", dropna=False)
        }
        score_values = np.asarray(list(per_seed.values()), dtype=float)
        architecture_scores[architecture] = {
            "seed_scores": {str(k): float(v) for k, v in per_seed.items()},
            "mean": float(score_values.mean()),
            "std": float(score_values.std(ddof=1)) if len(score_values) > 1 else 0.0,
        }
    selected = min(architecture_scores, key=lambda key: architecture_scores[key]["mean"])
    selection = {"metric": config["selection"]["metric"], "architectures": architecture_scores, "selected_architecture": selected}
    save_json(selection, output / "selection.json")

    bootstrap = []
    if not smoke:
        for architecture in ["GRU-D", "Transformer"]:
            candidate = test[test.model.eq(architecture + " ensemble")]
            for baseline in ["LOCF", "LME"]:
                bootstrap.append(
                    subject_bootstrap_mae_difference(
                        candidate, test[test.model.eq(baseline)], config["statistics"]["bootstrap_replicates"], config["seed"]
                    )
                )
    save_csv(pd.DataFrame(bootstrap), output / "paired_subject_bootstrap.csv")
    write_report(
        output, config, features, {key: len(value) for key, value in samples.items()},
        validation_metrics, test_metrics, selection, bootstrap
    )
    save_json(
        {
            "run_id": config["run_id"],
            "engineering_smoke_only": smoke,
            "completed": True,
            "selected_architecture": selected,
            "test_evaluated": not smoke,
            "artifacts": [
                str(output / "validation_metrics.csv"), str(output / "test_metrics.csv"),
                str(output / "lme_diagnostics.json"), str(output / "report.md")
            ],
        },
        output / "status.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/phase1_numeric_formal.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.config, args.smoke)


if __name__ == "__main__":
    main()
