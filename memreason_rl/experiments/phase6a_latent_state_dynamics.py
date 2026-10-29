"""Exploratory continuous-time forecasting of Phase 5G Memory posteriors."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.nn import functional as F

from ..data.common import ROOT
from .phase5c_memory_measurement import clean_matrix, subject_fold
from .phase5d_anchored_mirt import response_representation


CONFIG = ROOT / "configs/phase6a_latent_state_dynamics_dev.yaml"


def precision_update(
    prior_mean: torch.Tensor,
    prior_var: torch.Tensor,
    evidence_mean: torch.Tensor,
    evidence_var: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    prior_precision = prior_var.clamp_min(1e-8).reciprocal()
    evidence_precision = evidence_var.clamp_min(1e-8).reciprocal()
    variance = (prior_precision + evidence_precision).reciprocal()
    mean = variance * (prior_precision * prior_mean + evidence_precision * evidence_mean)
    return mean, variance


def analytic_transition(
    mean: torch.Tensor,
    variance: torch.Tensor,
    delta_years: torch.Tensor,
    decay: torch.Tensor,
    center: torch.Tensor,
    drift: torch.Tensor,
    process_rate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    delta = delta_years.clamp_min(0)
    rate = decay.clamp_min(1e-6)
    phi = torch.exp(-rate * delta)
    equilibrium = center + drift / rate
    next_mean = equilibrium + phi * (mean - equilibrium)
    accumulated = process_rate.clamp_min(1e-8) * (-torch.expm1(-2 * rate * delta)) / (2 * rate)
    next_variance = phi.square() * variance + accumulated
    return next_mean, next_variance


@dataclass
class LatentSample:
    sample_id: str
    subject_id: str
    fold: int
    horizon_months: int
    history_mean: np.ndarray
    history_var: np.ndarray
    history_time: np.ndarray
    target_mean: float
    target_var: float
    target_time: float
    target_composite: float
    mri_features: np.ndarray
    mri_available: bool


def reconstruct_state_records(config: dict[str, Any], items: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.read_csv(ROOT / "splits/train.csv", low_memory=False)
    raw = clean_matrix(frame, items); mask = np.isfinite(raw)
    keep = mask.sum(1) >= 2
    frame = frame.loc[keep].reset_index(drop=True); raw = raw[keep]; mask = mask[keep]
    sample_ids = [f"{row.subject_id}:{row.visit_code}:{index}"
                  for index, row in enumerate(frame.itertuples())]
    fraction, _, _ = response_representation(raw, items)
    memory = np.asarray([item["domain"] == "Memory" for item in items])
    memory_mask = mask[:, memory]
    memory_composite = np.nansum(fraction[:, memory], axis=1) / np.maximum(memory_mask.sum(1), 1)
    memory_composite[memory_mask.sum(1) == 0] = np.nan
    metadata = pd.DataFrame({
        "sample_id": sample_ids,
        "subject_id": frame.subject_id.astype(str),
        "visit_month": pd.to_numeric(frame.visit_month, errors="coerce"),
        "diagnosis": frame.get("dxsum_diagnosis_label", pd.Series("", index=frame.index)).astype(str),
        "memory_composite": memory_composite,
    })
    states = pd.read_csv(ROOT / config["phase5g_states"])
    states = states[states.arm.eq(config["phase5g_selected_arm"])].copy()
    states["state_mean"] = 0.5 * (states.view_a + states.view_b)
    states["state_var"] = (0.25 * (states.view_a_sd ** 2 + states.view_b_sd ** 2)
                           + 0.25 * (states.view_a - states.view_b) ** 2)
    columns = ["sample_id", "subject_id", "fold", "state_mean", "state_var"]
    result = states[columns].merge(metadata, on=["sample_id", "subject_id"], how="left", validate="one_to_one")
    if result.visit_month.isna().any():
        raise RuntimeError("Phase 5G state could not be mapped back to its visit month")
    return result.sort_values(["subject_id", "visit_month", "sample_id"], kind="stable").reset_index(drop=True)


def aggregate_mri_features(records: pd.DataFrame, config: dict[str, Any]) -> dict[str, np.ndarray]:
    source = pd.read_csv(ROOT / config["mri_burden"], low_memory=False)
    source = source[source.split.eq("train")].copy()
    source.visit_month = pd.to_numeric(source.visit_month, errors="coerce")
    result: dict[str, np.ndarray] = {}
    for row in records.itertuples(index=False):
        scans = source[(source.subject_id.astype(str) == str(row.subject_id))
                       & (source.visit_month <= float(row.visit_month))].sort_values("visit_month")
        if len(scans) < int(config["mri"]["minimum_scans"]):
            result[row.sample_id] = np.asarray([0, 0, 0, 0, 0], np.float32); continue
        times = scans.visit_month.to_numpy(float) / 12.0
        burden = scans.burden_mean.to_numpy(float)
        slope = (float(np.polyfit(times, burden, 1)[0]) if len(scans) >= 2 and np.ptp(times) > 1e-6 else 0.0)
        result[row.sample_id] = np.asarray([
            burden[-1], float(scans.burden_disagreement.iloc[-1]), slope,
            max(float(row.visit_month) / 12.0 - times[-1], 0.0), math.log1p(len(scans)),
        ], np.float32)
    return result


def build_samples(records: pd.DataFrame, mri: dict[str, np.ndarray], config: dict[str, Any]) -> list[LatentSample]:
    tolerances = {int(key): float(value) for key, value in config["forecast"]["tolerance_months"].items()}
    samples: list[LatentSample] = []
    for subject, group in records.groupby("subject_id", sort=True):
        group = group.sort_values("visit_month", kind="stable").drop_duplicates("visit_month", keep="last").reset_index(drop=True)
        if len(group) < 2:
            continue
        months = group.visit_month.to_numpy(float)
        for cutoff in range(len(group) - 1):
            for horizon in map(int, config["forecast"]["horizons_months"]):
                desired = months[cutoff] + horizon
                candidates = np.arange(cutoff + 1, len(group))
                nearest = candidates[np.argmin(np.abs(months[candidates] - desired))]
                if abs(months[nearest] - desired) > tolerances[horizon]:
                    continue
                history = group.iloc[:cutoff + 1]
                target = group.iloc[nearest]
                features = mri[str(group.iloc[cutoff].sample_id)]
                samples.append(LatentSample(
                    sample_id=f"{subject}:m{months[cutoff]:g}:h{horizon}", subject_id=str(subject),
                    fold=int(group.fold.iloc[0]), horizon_months=horizon,
                    history_mean=history.state_mean.to_numpy(np.float32),
                    history_var=history.state_var.to_numpy(np.float32),
                    history_time=history.visit_month.to_numpy(np.float32) / 12.0,
                    target_mean=float(target.state_mean), target_var=float(target.state_var),
                    target_time=float(target.visit_month) / 12.0,
                    target_composite=float(target.memory_composite), mri_features=features,
                    mri_available=bool(np.any(features != 0)),
                ))
    return samples


def tensorize(samples: list[LatentSample], device: torch.device) -> dict[str, torch.Tensor]:
    steps = max(len(sample.history_mean) for sample in samples); count = len(samples)
    mean = np.zeros((count, steps), np.float32); var = np.ones((count, steps), np.float32)
    time = np.zeros((count, steps), np.float32); mask = np.zeros((count, steps), bool)
    for row, sample in enumerate(samples):
        length = len(sample.history_mean); mean[row, :length] = sample.history_mean
        var[row, :length] = sample.history_var; time[row, :length] = sample.history_time; mask[row, :length] = True
    return {
        "history_mean": torch.as_tensor(mean, device=device),
        "history_var": torch.as_tensor(var, device=device),
        "history_time": torch.as_tensor(time, device=device),
        "history_mask": torch.as_tensor(mask, device=device),
        "target_mean": torch.tensor([sample.target_mean for sample in samples], device=device),
        "target_var": torch.tensor([sample.target_var for sample in samples], device=device),
        "target_time": torch.tensor([sample.target_time for sample in samples], device=device),
        "mri": torch.as_tensor(np.stack([sample.mri_features for sample in samples]), device=device),
        "mri_available": torch.tensor([sample.mri_available for sample in samples], device=device),
    }


class GaussianStateDynamics(nn.Module):
    def __init__(self, use_filter: bool, use_reference: bool, use_mri: bool, minimum_variance: float):
        super().__init__(); self.use_filter = use_filter; self.use_reference = use_reference
        self.use_mri = use_mri; self.minimum_variance = float(minimum_variance)
        self.raw_decay = nn.Parameter(torch.tensor(-2.0)); self.center = nn.Parameter(torch.tensor(0.0))
        self.drift = nn.Parameter(torch.tensor(0.0)); self.raw_process = nn.Parameter(torch.tensor(-3.0))
        self.raw_evidence_temperature = nn.Parameter(torch.tensor(0.0))
        self.mri_weight = nn.Parameter(torch.zeros(5)); self.mri_intercept = nn.Parameter(torch.tensor(0.0))
        self.raw_mri_variance = nn.Parameter(torch.tensor(0.0))

    def _transition(self, mean: torch.Tensor, var: torch.Tensor, delta: torch.Tensor):
        return analytic_transition(mean, var, delta, F.softplus(self.raw_decay) + 1e-4,
                                   self.center, self.drift, F.softplus(self.raw_process) + 1e-5)

    def forward(self, batch: dict[str, torch.Tensor], reference: tuple[float, float]) -> tuple[torch.Tensor, torch.Tensor]:
        mean_evidence = batch["history_mean"]
        times = batch["history_time"]; valid = batch["history_mask"]
        intercept, slope = reference if self.use_reference else (0.0, 0.0)
        evidence = mean_evidence - (intercept + slope * times)
        target_reference = intercept + slope * batch["target_time"]
        evidence_scale = F.softplus(self.raw_evidence_temperature) + 0.25
        lengths = valid.sum(1).long(); rows = torch.arange(len(valid), device=valid.device)
        if not self.use_filter:
            last = lengths - 1; mean = evidence[rows, last]
            var = batch["history_var"][rows, last] * evidence_scale
            previous = times[rows, last]
        else:
            mean = evidence[:, 0]; var = batch["history_var"][:, 0] * evidence_scale
            previous = times[:, 0]
            for step in range(1, times.shape[1]):
                active = valid[:, step]; delta = (times[:, step] - previous).clamp_min(0)
                prior_mean, prior_var = self._transition(mean, var, delta)
                update_mean, update_var = precision_update(
                    prior_mean, prior_var, evidence[:, step], batch["history_var"][:, step] * evidence_scale)
                mean = torch.where(active, update_mean, mean); var = torch.where(active, update_var, var)
                previous = torch.where(active, times[:, step], previous)
        if self.use_mri:
            mri_mean = self.mri_intercept + batch["mri"] @ self.mri_weight
            mri_var = F.softplus(self.raw_mri_variance) + self.minimum_variance
            updated_mean, updated_var = precision_update(mean, var, mri_mean, mri_var.expand_as(var))
            mean = torch.where(batch["mri_available"], updated_mean, mean)
            var = torch.where(batch["mri_available"], updated_var, var)
        mean, var = self._transition(mean, var, (batch["target_time"] - previous).clamp_min(0))
        return mean + target_reference, var.clamp_min(self.minimum_variance)


def fit_reference(records: pd.DataFrame, subjects: set[str], config: dict[str, Any]) -> tuple[float, float]:
    local = records[records.subject_id.astype(str).isin(subjects)].copy()
    labels = tuple(str(label).upper() for label in config["reference"]["cn_labels"])
    cn = local[local.diagnosis.str.upper().apply(lambda value: any(label in value for label in labels))]
    if len(cn) < int(config["reference"]["minimum_cn_records"]):
        cn = local
    x = cn.visit_month.to_numpy(float) / 12.0; y = cn.state_mean.to_numpy(float)
    slope, intercept = np.polyfit(x, y, 1) if len(cn) >= 2 and np.ptp(x) > 0 else (0.0, float(np.mean(y)))
    return float(intercept), float(slope)


def normalize_mri(fit: list[LatentSample], groups: list[list[LatentSample]]) -> None:
    available = np.stack([sample.mri_features for sample in fit if sample.mri_available])
    mean = available.mean(0) if len(available) else np.zeros(5); std = available.std(0) if len(available) else np.ones(5)
    std = np.where(std > 1e-6, std, 1.0)
    for group in groups:
        for sample in group:
            if sample.mri_available:
                sample.mri_features = ((sample.mri_features - mean) / std).astype(np.float32)


def shuffle_mri(samples: list[LatentSample], seed: int) -> list[LatentSample]:
    result = copy.deepcopy(samples); rng = np.random.default_rng(seed)
    features = [sample.mri_features.copy() for sample in result]
    available = [sample.mri_available for sample in result]
    order = rng.permutation(len(result))
    for row, source in enumerate(order):
        result[row].mri_features = features[source]; result[row].mri_available = available[source]
    return result


def loss_function(pred_mean: torch.Tensor, pred_var: torch.Tensor, batch: dict[str, torch.Tensor], w2_weight: float) -> torch.Tensor:
    target_mean, target_var = batch["target_mean"], batch["target_var"].clamp_min(1e-5)
    total_var = pred_var + target_var
    nll = 0.5 * (torch.log(2 * torch.pi * total_var) + (target_mean - pred_mean).square() / total_var)
    w2 = (target_mean - pred_mean).square() + (torch.sqrt(target_var) - torch.sqrt(pred_var)).square()
    return nll.mean() + w2_weight * w2.mean()


def train_model(arm: str, fit: list[LatentSample], dev: list[LatentSample], reference: tuple[float, float],
                config: dict[str, Any], device: torch.device, seed: int) -> GaussianStateDynamics:
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = GaussianStateDynamics(
        use_filter=arm not in {"exact_gaussian"}, use_reference=arm in {"reference_filter", "reference_mri_filter", "shuffled_mri_filter"},
        use_mri=arm in {"reference_mri_filter", "shuffled_mri_filter"},
        minimum_variance=float(config["training"]["minimum_variance"]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]),
                                  weight_decay=float(config["training"]["weight_decay"]))
    fit_batch, dev_batch = tensorize(fit, device), tensorize(dev, device)
    w2 = float(config["training"]["wasserstein_weight"]) if arm in {"latent_wasserstein", "reference_filter", "reference_mri_filter", "shuffled_mri_filter"} else 0.0
    best, best_loss, stale = copy.deepcopy(model.state_dict()), float("inf"), 0
    for _ in range(int(config["training"]["max_epochs"])):
        model.train(); optimizer.zero_grad(); mean, var = model(fit_batch, reference)
        loss = loss_function(mean, var, fit_batch, w2); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip"])); optimizer.step()
        model.eval()
        with torch.no_grad():
            mean, var = model(dev_batch, reference); score = float(loss_function(mean, var, dev_batch, w2))
        if score < best_loss - 1e-6:
            best, best_loss, stale = copy.deepcopy(model.state_dict()), score, 0
        else:
            stale += 1
            if stale >= int(config["training"]["patience"]): break
    model.load_state_dict(best); model.best_dev_loss = best_loss; return model


def locf_predictions(samples: list[LatentSample], process_rate: float) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray([sample.history_mean[-1] for sample in samples], float)
    gap = np.asarray([sample.target_time - sample.history_time[-1] for sample in samples], float)
    var = np.asarray([sample.history_var[-1] for sample in samples], float) + process_rate * np.maximum(gap, 0)
    return mean, np.maximum(var, 1e-4)


def estimate_locf_process(samples: list[LatentSample]) -> float:
    gap = np.asarray([sample.target_time - sample.history_time[-1] for sample in samples], float)
    change = np.asarray([sample.target_mean - sample.history_mean[-1] for sample in samples], float)
    rate = np.median(np.maximum(change ** 2 - np.asarray([s.target_var + s.history_var[-1] for s in samples]), 0) / np.maximum(gap, 1e-3))
    return float(max(rate, 1e-4))


def fit_composite_decoder(records: pd.DataFrame, subjects: set[str]) -> tuple[float, float]:
    local = records[records.subject_id.astype(str).isin(subjects)].dropna(subset=["memory_composite"])
    if len(local) < 2 or local.state_mean.std() < 1e-8:
        return float(local.memory_composite.mean()), 0.0
    slope, intercept = np.polyfit(local.state_mean, local.memory_composite, 1)
    return float(intercept), float(slope)


def prediction_rows(arm: str, samples: list[LatentSample], mean: np.ndarray, var: np.ndarray,
                    composite_decoder: tuple[float, float]) -> list[dict[str, Any]]:
    intercept, slope = composite_decoder; rows = []
    for sample, prediction, variance in zip(samples, mean, var):
        total_var = max(float(variance + sample.target_var), 1e-8); error = sample.target_mean - prediction
        rows.append({
            "sample_id": sample.sample_id, "subject_id": sample.subject_id, "fold": sample.fold,
            "arm": arm, "horizon_months": sample.horizon_months,
            "truth_state": sample.target_mean, "prediction_state": float(prediction),
            "prediction_variance": float(variance), "teacher_variance": sample.target_var,
            "latent_absolute_error": float(abs(error)),
            "latent_nll": float(0.5 * (math.log(2 * math.pi * total_var) + error * error / total_var)),
            "wasserstein2": float(error * error + (math.sqrt(max(variance, 0)) - math.sqrt(max(sample.target_var, 0))) ** 2),
            "covered_95": bool(abs(error) <= 1.96 * math.sqrt(max(variance, 1e-8))),
            "variance_growth": float(variance - sample.history_var[-1]),
            "truth_composite": sample.target_composite,
            "prediction_composite": float(intercept + slope * prediction),
            "composite_absolute_error": float(abs((intercept + slope * prediction) - sample.target_composite)),
            "mri_available": sample.mri_available,
        })
    return rows


def bootstrap_comparisons(predictions: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    baseline = config["gate"]["baseline"]; rng = np.random.default_rng(config["statistics"]["bootstrap_seed"])
    repetitions = int(config["statistics"]["bootstrap_replicates"]); rows = []
    for arm in sorted(set(predictions.arm) - {baseline}):
        pair = predictions[predictions.arm.isin([arm, baseline])].pivot_table(
            index=["sample_id", "subject_id"], columns="arm", values="latent_absolute_error").dropna()
        difference = (pair[arm] - pair[baseline]).groupby("subject_id").mean().to_numpy()
        draws = difference[rng.integers(0, len(difference), size=(repetitions, len(difference)))].mean(1)
        rows.append({"candidate": arm, "baseline": baseline, "difference": float(difference.mean()),
                     "ci_low": float(np.quantile(draws, .025)), "ci_high": float(np.quantile(draws, .975)),
                     "subjects": len(difference), "replicates": repetitions})
    return pd.DataFrame(rows)


def run(config_path: Path = CONFIG) -> Path:
    config = yaml.safe_load(config_path.read_text()); contract_path = ROOT / config["item_contract"]
    items = yaml.safe_load(contract_path.read_text())["primary_state_items"]
    records = reconstruct_state_records(config, items); mri = aggregate_mri_features(records, config)
    samples = build_samples(records, mri, config)
    if not samples: raise RuntimeError("No latent forecast samples were constructed")
    output = ROOT / "outputs" / config["run_id"]; output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_rows = []; manifests = []
    for fold in range(int(config["folds"])):
        # MRI normalization is fold-specific and mutates only these isolated copies.
        evaluation = copy.deepcopy([sample for sample in samples if sample.fold == fold])
        outer_fit = copy.deepcopy([sample for sample in samples if sample.fold != fold])
        dev_subjects = {sample.subject_id for sample in outer_fit
                        if int.from_bytes(hashlib.sha256(f"inner:{fold}:{sample.subject_id}".encode()).digest()[:8], "big") / 2**64
                        < float(config["inner_dev_fraction"])}
        development = [sample for sample in outer_fit if sample.subject_id in dev_subjects]
        fit = [sample for sample in outer_fit if sample.subject_id not in dev_subjects]
        fit_subjects = {sample.subject_id for sample in fit}; reference = fit_reference(records, fit_subjects, config)
        normalize_mri(fit, [fit, development, evaluation])
        decoder = fit_composite_decoder(records, fit_subjects)
        process = estimate_locf_process(fit); mean, var = locf_predictions(evaluation, process)
        all_rows.extend(prediction_rows("latent_locf", evaluation, mean, var, decoder))
        checkpoint_root = output / "checkpoints"; checkpoint_root.mkdir(exist_ok=True)
        for arm in [name for name in config["arms"] if name != "latent_locf"]:
            local_fit, local_dev, local_eval = fit, development, evaluation
            if arm == "shuffled_mri_filter":
                local_fit = shuffle_mri(fit, 1000 + fold); local_dev = shuffle_mri(development, 2000 + fold)
                local_eval = shuffle_mri(evaluation, 3000 + fold)
            model = train_model(arm, local_fit, local_dev, reference, config, device, 20261029 + fold * 101 + len(arm))
            model.eval()
            with torch.no_grad():
                batch = tensorize(local_eval, device); predicted, variance = model(batch, reference)
            all_rows.extend(prediction_rows(arm, local_eval, predicted.cpu().numpy(), variance.cpu().numpy(), decoder))
            destination = checkpoint_root / arm; destination.mkdir(exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "fold": fold, "reference": reference,
                        "best_dev_loss": model.best_dev_loss}, destination / f"fold_{fold}.pt")
        manifests.append({"fold": fold, "fit_subjects": len(fit_subjects), "dev_subjects": len(dev_subjects),
                          "evaluation_subjects": len({sample.subject_id for sample in evaluation}),
                          "fit_samples": len(fit), "dev_samples": len(development), "evaluation_samples": len(evaluation),
                          "reference_intercept": reference[0], "reference_slope_per_year": reference[1],
                          "locf_process_rate": process})
    predictions = pd.DataFrame(all_rows); predictions.to_csv(output / "crossfit_predictions.csv", index=False)
    pd.DataFrame(manifests).to_csv(output / "fold_manifest.csv", index=False)
    metrics = predictions.groupby(["arm", "horizon_months"], as_index=False).agg(
        samples=("sample_id", "size"), subjects=("subject_id", "nunique"), latent_mae=("latent_absolute_error", "mean"),
        latent_nll=("latent_nll", "mean"), wasserstein2=("wasserstein2", "mean"), coverage_95=("covered_95", "mean"),
        composite_mae=("composite_absolute_error", "mean"), variance_growth=("variance_growth", "mean"))
    aggregate = predictions.groupby("arm", as_index=False).agg(
        samples=("sample_id", "size"), subjects=("subject_id", "nunique"), latent_mae=("latent_absolute_error", "mean"),
        latent_nll=("latent_nll", "mean"), wasserstein2=("wasserstein2", "mean"), coverage_95=("covered_95", "mean"),
        composite_mae=("composite_absolute_error", "mean"), variance_growth=("variance_growth", "mean"))
    metrics.to_csv(output / "metrics_by_horizon.csv", index=False); aggregate.to_csv(output / "aggregate_metrics.csv", index=False)
    comparisons = bootstrap_comparisons(predictions, config); comparisons.to_csv(output / "subject_bootstrap.csv", index=False)
    table = aggregate.set_index("arm"); candidate = config["gate"]["candidate"]; shuffled = "shuffled_mri_filter"
    comparison_table = comparisons.set_index("candidate")
    candidate_pass = bool(comparison_table.loc[candidate, "ci_high"] < float(config["gate"]["maximum_latent_mae_difference"]))
    mri_gain = float(table.loc[candidate, "latent_mae"] - table.loc[shuffled, "latent_mae"])
    coverage_low, coverage_high = map(float, config["gate"]["coverage_95_bounds"])
    coverage_pass = bool(coverage_low <= table.loc[candidate, "coverage_95"] <= coverage_high)
    decision = {"run_id": config["run_id"], "exploratory_only": True, "original_train_only": True,
                "validation_or_test_loaded": False, "candidate": candidate, "candidate_vs_locf_pass": candidate_pass,
                "real_minus_shuffled_mri_latent_mae": mri_gain, "real_mri_better_than_shuffled": mri_gain < 0,
                "coverage_gate": coverage_pass, "measurement_gate_reopened": False,
                "confirmatory_continuous_time_claim_open": False}
    (output / "development_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    report_lines = ["# Phase 6A: Exploratory latent-state dynamics", "", "Original-train-only five-fold subject cross-fitting; validation and test were not loaded.", "",
                    "| Arm | Latent MAE | NLL | W2 | 95% coverage | Memory composite MAE |", "|---|---:|---:|---:|---:|---:|"]
    for row in aggregate.itertuples(index=False):
        report_lines.append(f"| {row.arm} | {row.latent_mae:.4f} | {row.latent_nll:.4f} | {row.wasserstein2:.4f} | {row.coverage_95:.4f} | {row.composite_mae:.4f} |")
    report_lines.extend(["", f"Candidate-minus-LOCF participant latent-MAE difference: **{comparison_table.loc[candidate, 'difference']:+.5f}**, 95% CI **[{comparison_table.loc[candidate, 'ci_low']:+.5f}, {comparison_table.loc[candidate, 'ci_high']:+.5f}]**.",
                         "", f"Real-minus-shuffled MRI latent MAE: **{mri_gain:+.5f}**.", "",
                         "This is exploratory because the Phase 5F measurement gate remains closed; it cannot establish a confirmatory continuous-time state claim."])
    (output / "report.md").write_text("\n".join(report_lines) + "\n")
    manifest = {"config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "protocol_sha256": hashlib.sha256((ROOT / "reports/experiments/phase6a_latent_state_dynamics_protocol.md").read_bytes()).hexdigest(),
                "phase5g_states_sha256": hashlib.sha256((ROOT / config["phase5g_states"]).read_bytes()).hexdigest(),
                "device": str(device), "state_records": len(records), "forecast_samples": len(samples),
                "neural_checkpoints": (len(config["arms"]) - 1) * int(config["folds"])}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args(); print(run(args.config))


if __name__ == "__main__": main()
