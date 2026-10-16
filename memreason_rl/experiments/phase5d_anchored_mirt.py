"""Train-only anchored bifactor IRT and measurement-invariance experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import beta as beta_distribution
from scipy.stats import spearmanr
from torch import nn
from torch.nn import functional as F

from ..data.common import ROOT
from .phase5c_memory_measurement import (
    clean_matrix,
    cross_views,
    deterministic_targets,
    fit_scaling,
    stable_int,
    subject_fold,
    to_severity,
)


CONFIG = ROOT / "configs/phase5d_anchored_mirt_dev.yaml"


def assign_time_bins(months: np.ndarray) -> np.ndarray:
    """Map nominal months to the four frozen longitudinal invariance groups."""
    values = np.nan_to_num(np.asarray(months, float), nan=0.0)
    return np.select([values <= 0, values <= 12, values <= 24], [0, 1, 2], default=3).astype(np.int64)


def response_representation(raw: np.ndarray, items: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return severity-oriented fractions, ordinal category indices, and response types."""
    fraction = np.full(raw.shape, np.nan, np.float32)
    category = np.full(raw.shape, -1, np.int64)
    response_type = np.zeros(len(items), np.int64)  # binary=0, continuous=1, ordinal=2
    for j, item in enumerate(items):
        values = raw[:, j]
        valid = np.isfinite(values)
        if item["value_type"] == "ordinal":
            response_type[j] = 2
            levels = np.asarray(item["valid_values"], float)
            indices = np.full(len(values), -1, np.int64)
            for k, level in enumerate(levels):
                indices[valid & np.isclose(values, level)] = k
            if item["direction"] == "higher=better":
                indices[indices >= 0] = len(levels) - 1 - indices[indices >= 0]
            category[:, j] = indices
            fraction[indices >= 0, j] = indices[indices >= 0] / max(len(levels) - 1, 1)
        else:
            response_type[j] = 0 if item["value_type"] == "binary" else 1
            if item.get("range") is not None:
                lower, upper = map(float, item["range"])
            else:
                levels = np.asarray(item["valid_values"], float)
                lower, upper = float(levels.min()), float(levels.max())
            local = (values - lower) / max(upper - lower, 1e-6)
            if item["direction"] == "higher=better":
                local = 1.0 - local
            fraction[valid, j] = np.clip(local[valid], 0.0, 1.0)
    return fraction, category, response_type


def concordance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 2:
        return float("nan")
    covariance = np.mean((x - x.mean()) * (y - y.mean()))
    denominator = x.var() + y.var() + (x.mean() - y.mean()) ** 2
    return float(2 * covariance / denominator) if denominator > 0 else float("nan")


class AnchoredBifactorIRT(nn.Module):
    """Global-plus-memory typed item-response model with optional time DIF."""

    def __init__(self, items: list[dict[str, Any]], response_type: np.ndarray,
                 dif_mask: np.ndarray, minimum_precision: float):
        super().__init__()
        self.n_items = len(items)
        self.minimum_precision = float(minimum_precision)
        self.register_buffer("response_type", torch.as_tensor(response_type, dtype=torch.long))
        self.register_buffer("memory_mask", torch.as_tensor(
            [item["domain"] == "Memory" for item in items], dtype=torch.float32))
        self.register_buffer("dif_mask", torch.as_tensor(dif_mask, dtype=torch.float32))
        self.log_a_global = nn.Parameter(torch.zeros(self.n_items))
        self.log_a_memory = nn.Parameter(torch.full((self.n_items,), -0.25))
        self.intercept = nn.Parameter(torch.zeros(self.n_items))
        self.log_precision = nn.Parameter(torch.full((self.n_items,), 2.0))
        self.ordinal_center = nn.Parameter(torch.zeros(self.n_items))
        self.ordinal_gap = nn.Parameter(torch.zeros(self.n_items, 4))
        self.dif_free = nn.Parameter(torch.zeros(self.n_items, 3))

    def initialize_intercepts(self, fraction: np.ndarray, mask: np.ndarray) -> None:
        mean = np.nanmean(np.where(mask, fraction, np.nan), axis=0)
        mean = np.clip(np.nan_to_num(mean, nan=0.5), 0.02, 0.98)
        with torch.no_grad():
            self.intercept.copy_(torch.as_tensor(np.log(mean / (1 - mean)), dtype=torch.float32))

    def discriminations(self) -> tuple[torch.Tensor, torch.Tensor]:
        global_loading = F.softplus(self.log_a_global) + 0.05
        memory_loading = (F.softplus(self.log_a_memory) + 0.05) * self.memory_mask
        return global_loading, memory_loading

    def dif_parameters(self) -> torch.Tensor:
        zero = torch.zeros(self.n_items, 1, device=self.dif_free.device)
        return torch.cat([zero, self.dif_free], dim=1) * self.dif_mask[:, None]

    def linear_predictor(self, state: torch.Tensor, time_bin: torch.Tensor) -> torch.Tensor:
        global_loading, memory_loading = self.discriminations()
        time_effect = self.dif_parameters()[:, time_bin].transpose(0, 1)
        return (state[:, 0, None] * global_loading[None]
                + state[:, 1, None] * memory_loading[None]
                + self.intercept[None] + time_effect)

    def ordinal_thresholds(self) -> torch.Tensor:
        gaps = F.softplus(self.ordinal_gap) + 0.10
        cumulative = torch.cumsum(gaps, dim=1)
        return self.ordinal_center[:, None] + cumulative - cumulative.mean(dim=1, keepdim=True)

    def ordinal_probabilities(self, eta: torch.Tensor, item: int) -> torch.Tensor:
        thresholds = self.ordinal_thresholds()[item]
        greater = torch.sigmoid(eta[:, item, None] - thresholds[None])
        return torch.cat([1 - greater[:, :1], greater[:, :-1] - greater[:, 1:], greater[:, -1:]], 1).clamp_min(1e-7)

    def expected_fraction(self, state: torch.Tensor, time_bin: torch.Tensor) -> torch.Tensor:
        eta = self.linear_predictor(state, time_bin)
        expected = torch.sigmoid(eta)
        for item in torch.where(self.response_type == 2)[0].tolist():
            probabilities = self.ordinal_probabilities(eta, item)
            levels = torch.linspace(0, 1, probabilities.shape[1], device=eta.device)
            expected[:, item] = (probabilities * levels[None]).sum(1)
        return expected

    def nll_matrix(self, state: torch.Tensor, time_bin: torch.Tensor, fraction: torch.Tensor,
                   category: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        eta = self.linear_predictor(state, time_bin)
        losses = torch.zeros_like(fraction)
        binary_items = torch.where(self.response_type == 0)[0]
        continuous_items = torch.where(self.response_type == 1)[0]
        if len(binary_items):
            losses[:, binary_items] = F.binary_cross_entropy_with_logits(
                eta[:, binary_items], fraction[:, binary_items].clamp(0, 1), reduction="none")
        if len(continuous_items):
            mean = torch.sigmoid(eta[:, continuous_items]).clamp(1e-4, 1 - 1e-4)
            precision = F.softplus(self.log_precision[continuous_items])[None] + self.minimum_precision
            observed = fraction[:, continuous_items].clamp(0.0025, 0.9975)
            distribution = torch.distributions.Beta(mean * precision, (1 - mean) * precision)
            losses[:, continuous_items] = -distribution.log_prob(observed)
        for item in torch.where(self.response_type == 2)[0].tolist():
            probabilities = self.ordinal_probabilities(eta, item)
            target = category[:, item].clamp_min(0)
            losses[:, item] = -torch.log(probabilities.gather(1, target[:, None]).squeeze(1))
        return torch.where(mask, losses, torch.zeros_like(losses))


def variational_objective(model: AnchoredBifactorIRT, mean: torch.Tensor, logvar: torch.Tensor,
                          time_bin: torch.Tensor, fraction: torch.Tensor, category: torch.Tensor,
                          mask: torch.Tensor, kl_weight: float, dif_weight: float) -> torch.Tensor:
    state = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
    nll = model.nll_matrix(state, time_bin, fraction, category, mask)
    likelihood = nll.sum() / mask.sum().clamp_min(1)
    kl = -0.5 * (1 + logvar - mean.square() - logvar.exp()).sum(1).mean()
    dif_penalty = model.dif_parameters().square().mean()
    return likelihood + kl_weight * kl + dif_weight * dif_penalty


def infer_state(model: AnchoredBifactorIRT, fraction: np.ndarray, category: np.ndarray,
                mask: np.ndarray, time_bin: np.ndarray, config: dict[str, Any],
                device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    settings = config["training"]
    inference_seed = 20261028 + len(fraction) * 17 + int(mask.sum()) % 100003
    torch.manual_seed(inference_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(inference_seed)
    observed_fraction = torch.as_tensor(np.nan_to_num(fraction), dtype=torch.float32, device=device)
    observed_category = torch.as_tensor(category, dtype=torch.long, device=device)
    observed_mask = torch.as_tensor(mask, dtype=torch.bool, device=device)
    time_tensor = torch.as_tensor(time_bin, dtype=torch.long, device=device)
    mean = nn.Parameter(torch.zeros(len(fraction), 2, device=device))
    logvar = nn.Parameter(torch.zeros(len(fraction), 2, device=device))
    optimizer = torch.optim.Adam([mean, logvar], lr=float(settings["inference_learning_rate"]))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for _ in range(int(settings["inference_steps"])):
        optimizer.zero_grad()
        loss = variational_objective(model, mean, logvar, time_tensor, observed_fraction,
                                     observed_category, observed_mask,
                                     float(settings["kl_weight"]), 0.0)
        loss.backward(); optimizer.step()
        with torch.no_grad():
            logvar.clamp_(-6, 2)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return mean.detach().cpu().numpy(), logvar.detach().cpu().numpy()


def evaluate_nll(model: AnchoredBifactorIRT, mean: np.ndarray, fraction: np.ndarray,
                 category: np.ndarray, target_mask: np.ndarray, time_bin: np.ndarray,
                 device: torch.device) -> float:
    with torch.no_grad():
        losses = model.nll_matrix(
            torch.as_tensor(mean, dtype=torch.float32, device=device),
            torch.as_tensor(time_bin, dtype=torch.long, device=device),
            torch.as_tensor(np.nan_to_num(fraction), dtype=torch.float32, device=device),
            torch.as_tensor(category, dtype=torch.long, device=device),
            torch.as_tensor(target_mask, dtype=torch.bool, device=device))
        return float(losses.sum().item() / max(int(target_mask.sum()), 1))


def fit_model(model: AnchoredBifactorIRT, fraction: np.ndarray, category: np.ndarray,
              mask: np.ndarray, time_bin: np.ndarray, dev_fraction: np.ndarray,
              dev_category: np.ndarray, dev_mask: np.ndarray, dev_time: np.ndarray,
              dev_ids: list[str], instruments: np.ndarray, config: dict[str, Any],
              device: torch.device, epochs: int, learning_rate: float,
              seed: int) -> tuple[AnchoredBifactorIRT, np.ndarray, np.ndarray, float]:
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model.to(device); model.initialize_intercepts(fraction, mask)
    fit_fraction = torch.as_tensor(np.nan_to_num(fraction), dtype=torch.float32, device=device)
    fit_category = torch.as_tensor(category, dtype=torch.long, device=device)
    fit_mask = torch.as_tensor(mask, dtype=torch.bool, device=device)
    fit_time = torch.as_tensor(time_bin, dtype=torch.long, device=device)
    mean = nn.Parameter(torch.zeros(len(fraction), 2, device=device))
    logvar = nn.Parameter(torch.zeros(len(fraction), 2, device=device))
    optimizer = torch.optim.AdamW(list(model.parameters()) + [mean, logvar], lr=learning_rate,
                                  weight_decay=float(config["training"]["weight_decay"]))
    dev_input, dev_target = deterministic_targets(
        dev_mask, instruments, dev_ids, "heldout_item", float(config["masking"]["item_fraction"]))
    best_state = copy.deepcopy(model.state_dict()); best_loss = float("inf"); stale = 0
    every = int(config["training"]["evaluation_every"])
    for epoch in range(1, epochs + 1):
        model.train(); optimizer.zero_grad()
        loss = variational_objective(model, mean, logvar, fit_time, fit_fraction, fit_category,
                                     fit_mask, float(config["training"]["kl_weight"]),
                                     float(config["training"]["dif_l2_weight"]))
        loss.backward(); torch.nn.utils.clip_grad_norm_(list(model.parameters()) + [mean, logvar], 5.0)
        optimizer.step()
        with torch.no_grad(): logvar.clamp_(-6, 2)
        if epoch % every == 0 or epoch == epochs:
            dev_mean, _ = infer_state(model, dev_fraction, dev_category, dev_input, dev_time, config, device)
            score = evaluate_nll(model, dev_mean, dev_fraction, dev_category, dev_target, dev_time, device)
            if score < best_loss - 1e-5:
                best_loss = score; best_state = copy.deepcopy(model.state_dict()); stale = 0
            else:
                stale += 1
                if stale >= int(config["training"]["patience_evaluations"]):
                    break
    model.load_state_dict(best_state)
    final_mean, final_logvar = infer_state(model, fraction, category, mask, time_bin, config, device)
    model.best_dev_nll = best_loss
    return model, final_mean, final_logvar, best_loss


def residual_dif_audit(model: AnchoredBifactorIRT, mean: np.ndarray, fraction: np.ndarray,
                       mask: np.ndarray, time_bin: np.ndarray, diagnosis: np.ndarray,
                       phase: np.ndarray, items: list[dict[str, Any]], config: dict[str, Any],
                       fold: int, device: torch.device) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    with torch.no_grad():
        expected = model.expected_fraction(
            torch.as_tensor(mean, dtype=torch.float32, device=device),
            torch.as_tensor(time_bin, dtype=torch.long, device=device)).cpu().numpy()
    residual = fraction - expected
    threshold = float(config["dif"]["residual_effect_threshold"])
    minimum = int(config["dif"]["minimum_group_observations"])
    groups = {"time_bin": time_bin.astype(str), "diagnosis": diagnosis.astype(str), "adni_phase": phase.astype(str)}
    records: list[dict[str, Any]] = []
    flags_by_type = {name: np.zeros(len(items), bool) for name in groups}
    for group_type, labels in groups.items():
        for item, metadata in enumerate(items):
            valid = mask[:, item] & pd.notna(labels) & (labels != "nan") & (labels != "None")
            overall = float(np.nanmean(residual[valid, item])) if valid.any() else float("nan")
            for label in sorted(set(labels[valid])):
                choose = valid & (labels == label)
                if choose.sum() < minimum:
                    continue
                group_mean = float(np.nanmean(residual[choose, item])); effect = group_mean - overall
                flagged = abs(effect) >= threshold
                flags_by_type[group_type][item] |= flagged
                records.append({"fold": fold, "group_type": group_type, "group": label,
                                "item": metadata["field"], "instrument": metadata["instrument"],
                                "domain": metadata["domain"], "observations": int(choose.sum()),
                                "mean_residual": group_mean, "effect_vs_item_mean": effect,
                                "practical_dif": flagged})
    any_flag = np.logical_or.reduce(list(flags_by_type.values()))
    time_flag = flags_by_type["time_bin"]
    return pd.DataFrame(records), time_flag, any_flag


def severity_fraction_to_z(expected: np.ndarray, items: list[dict[str, Any]],
                           scaling: dict[str, np.ndarray]) -> np.ndarray:
    raw_prediction = np.zeros_like(expected, np.float32)
    for item, metadata in enumerate(items):
        if metadata["value_type"] == "ordinal":
            levels = np.asarray(metadata["valid_values"], float)
            position = np.clip(expected[:, item], 0, 1) * (len(levels) - 1)
            if metadata["direction"] == "higher=better":
                position = len(levels) - 1 - position
            raw_prediction[:, item] = np.interp(position, np.arange(len(levels)), levels)
        else:
            if metadata.get("range") is not None:
                lower, upper = map(float, metadata["range"])
            else:
                values = np.asarray(metadata["valid_values"], float); lower, upper = values.min(), values.max()
            local = np.clip(expected[:, item], 0, 1)
            raw_prediction[:, item] = (lower + local * (upper - lower)
                                       if metadata["direction"] == "higher=worse"
                                       else upper - local * (upper - lower))
    return to_severity(raw_prediction, scaling)


def interval_z(model: AnchoredBifactorIRT, mean: np.ndarray, time_bin: np.ndarray,
               items: list[dict[str, Any]], scaling: dict[str, np.ndarray],
               device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Continuous-item beta predictive intervals transformed to the Phase 5C z scale."""
    with torch.no_grad():
        eta = model.linear_predictor(torch.as_tensor(mean, dtype=torch.float32, device=device),
                                     torch.as_tensor(time_bin, dtype=torch.long, device=device))
        probability = torch.sigmoid(eta).cpu().numpy()
        precision = (F.softplus(model.log_precision) + model.minimum_precision).detach().cpu().numpy()
    quantiles = [np.full_like(probability, np.nan) for _ in range(4)]
    continuous = np.where(model.response_type.detach().cpu().numpy() == 1)[0]
    for item in continuous:
        alpha = np.clip(probability[:, item] * precision[item], 1e-4, None)
        beta = np.clip((1 - probability[:, item]) * precision[item], 1e-4, None)
        for array, q in zip(quantiles, [.05, .95, .025, .975]):
            array[:, item] = beta_distribution.ppf(q, alpha, beta)
    transformed = [severity_fraction_to_z(array, items, scaling) for array in quantiles]
    low90 = np.minimum(transformed[0], transformed[1]); high90 = np.maximum(transformed[0], transformed[1])
    low95 = np.minimum(transformed[2], transformed[3]); high95 = np.maximum(transformed[2], transformed[3])
    return low90, high90, low95, high95


def cluster_bootstrap(predictions: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    candidate = config["gate"]["candidate"]; rng = np.random.default_rng(config["statistics"]["bootstrap_seed"])
    repetitions = int(config["statistics"]["bootstrap_replicates"]); rows = []
    keys = ["sample_id", "subject_id", "task", "item"]
    for task in sorted(predictions.task.unique()):
        local = predictions[predictions.task.eq(task)]
        for baseline in config["baseline_arms"] + [x for x in config["arms"] if x != candidate]:
            pair = local[local.arm.isin([candidate, baseline])].pivot_table(
                index=keys, columns="arm", values="absolute_error").dropna()
            pair["difference"] = pair[candidate] - pair[baseline]
            subject = pair.groupby("subject_id").difference.mean().to_numpy()
            draws = subject[rng.integers(0, len(subject), size=(repetitions, len(subject)))].mean(1)
            rows.append({"task": task, "candidate": candidate, "baseline": baseline,
                         "difference": float(subject.mean()), "ci_low": float(np.quantile(draws, .025)),
                         "ci_high": float(np.quantile(draws, .975)), "subjects": len(subject),
                         "paired_decisions": len(pair), "replicates": repetitions})
    return pd.DataFrame(rows)


def ccc_bootstrap(states: pd.DataFrame, config: dict[str, Any]) -> dict[str, float]:
    candidate = states[states.arm.eq(config["gate"]["candidate"])]
    sufficient = candidate.groupby("subject_id").apply(
        lambda x: pd.Series({"n": len(x), "sx": x.view_a.sum(), "sy": x.view_b.sum(),
                             "sxx": np.square(x.view_a).sum(), "syy": np.square(x.view_b).sum(),
                             "sxy": (x.view_a * x.view_b).sum()})).reset_index(drop=True)
    values = sufficient.to_numpy(float); rng = np.random.default_rng(config["statistics"]["bootstrap_seed"] + 1)
    repetitions = int(config["statistics"]["state_bootstrap_replicates"]); estimates = np.empty(repetitions)
    for repeat in range(repetitions):
        total = values[rng.integers(0, len(values), len(values))].sum(0)
        n, sx, sy, sxx, syy, sxy = total
        mx, my = sx / n, sy / n
        covariance = sxy / n - mx * my
        denominator = sxx / n - mx * mx + syy / n - my * my + (mx - my) ** 2
        estimates[repeat] = 2 * covariance / denominator if denominator > 0 else np.nan
    observed = concordance_correlation(candidate.view_a.to_numpy(), candidate.view_b.to_numpy())
    return {"ccc": observed, "ci_low": float(np.nanquantile(estimates, .025)),
            "ci_high": float(np.nanquantile(estimates, .975)), "replicates": repetitions,
            "subjects": int(candidate.subject_id.nunique()), "pairs": len(candidate)}


def connectivity_audit(frame: pd.DataFrame, items: list[dict[str, Any]], raw: np.ndarray,
                       output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    instruments = sorted({item["instrument"] for item in items})
    item_instrument = np.asarray([item["instrument"] for item in items])
    present = {instrument: np.isfinite(raw[:, item_instrument == instrument]).any(1) for instrument in instruments}
    date_columns = {"ADAS": "adas_visit_date", "CDR": "cdr_visit_date", "MMSE": "mmse_visit_date",
                    "MOCA": "moca_visit_date", "LOGMEM": "LogMemIA_ExamDate"}
    rows = []
    for i, left in enumerate(instruments):
        for right in instruments[i + 1:]:
            both = present[left] & present[right]
            left_date = pd.to_datetime(frame.get(date_columns[left]), errors="coerce")
            right_date = pd.to_datetime(frame.get(date_columns[right]), errors="coerce")
            dated = both & left_date.notna().to_numpy() & right_date.notna().to_numpy()
            difference = np.abs((left_date[dated] - right_date[dated]).dt.days.to_numpy())
            rows.append({"instrument_a": left, "instrument_b": right, "concurrent_rows": int(both.sum()),
                         "concurrent_subjects": int(frame.loc[both, "subject_id"].nunique()),
                         "both_dates_available": int(dated.sum()), "same_day": int((difference == 0).sum()),
                         "within_30_days": int((difference <= 30).sum()),
                         "within_90_days": int((difference <= 90).sum())})
    pair_frame = pd.DataFrame(rows); pair_frame.to_csv(output / "instrument_connectivity.csv", index=False)
    pattern = pd.DataFrame(present).astype(int).astype(str).agg("".join, axis=1)
    pattern_frame = pattern.value_counts().rename_axis("presence_pattern").reset_index(name="rows")
    pattern_frame["instrument_order"] = ",".join(instruments)
    pattern_frame.to_csv(output / "instrument_patterns.csv", index=False)
    return pair_frame, pattern_frame


def run() -> Path:
    config = yaml.safe_load(CONFIG.read_text()); contract_path = ROOT / config["item_contract"]
    contract = yaml.safe_load(contract_path.read_text()); items = contract["primary_state_items"]
    fields = [item["field"] for item in items]; instruments = np.asarray([item["instrument"] for item in items])
    memory = np.asarray([item["domain"] == "Memory" for item in items])
    frame = pd.read_csv(ROOT / "splits/train.csv", low_memory=False)
    raw = clean_matrix(frame, items); mask = np.isfinite(raw)
    keep = mask.sum(1) >= 2; frame = frame.loc[keep].reset_index(drop=True); raw = raw[keep]; mask = mask[keep]
    fraction, category, response_type = response_representation(raw, items)
    sample_ids = [f"{row.subject_id}:{row.visit_code}:{i}" for i, row in enumerate(frame.itertuples())]
    folds = int(config["folds"]); seed = int(config["partition_seed"])
    fold_assignment = np.asarray([subject_fold(str(subject), seed, folds) for subject in frame.subject_id])
    time_bin = assign_time_bins(pd.to_numeric(frame.visit_month, errors="coerce").to_numpy())
    diagnosis = frame.dxsum_diagnosis_label.fillna("missing").astype(str).to_numpy()
    phase = frame.dxsum_phase.fillna("missing").astype(str).to_numpy()
    output = ROOT / "outputs" / config["run_id"]; output.mkdir(parents=True, exist_ok=True)
    connectivity_audit(frame, items, raw, output)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions: list[dict[str, Any]] = []; states: list[dict[str, Any]] = []
    dif_records: list[pd.DataFrame] = []; anchor_records: list[dict[str, Any]] = []; manifests = []
    checkpoint_root = output / "checkpoints"; checkpoint_root.mkdir(exist_ok=True)
    for fold in range(folds):
        evaluation = fold_assignment == fold; outer_fit = ~evaluation
        dev_subjects = {subject for subject in frame.loc[outer_fit, "subject_id"].astype(str).unique()
                        if stable_int(f"inner:{seed}:{fold}:{subject}") / 2**64 < float(config["inner_dev_fraction"])}
        development = outer_fit & frame.subject_id.astype(str).isin(dev_subjects).to_numpy(); fit = outer_fit & ~development
        scaling = fit_scaling(raw, fit, items); truth_z = to_severity(raw, scaling)
        dev_ids = [sample_ids[i] for i in np.flatnonzero(development)]
        invariant = AnchoredBifactorIRT(items, response_type, np.zeros(len(items), bool),
                                       config["training"]["minimum_precision"])
        invariant, fit_mean, fit_logvar, _ = fit_model(
            invariant, fraction[fit], category[fit], mask[fit], time_bin[fit],
            fraction[development], category[development], mask[development], time_bin[development],
            dev_ids, instruments, config, device, int(config["training"]["invariant_epochs"]),
            float(config["training"]["learning_rate"]), seed + fold * 101)
        audit, time_dif, any_dif = residual_dif_audit(
            invariant, fit_mean, fraction[fit], mask[fit], time_bin[fit], diagnosis[fit], phase[fit],
            items, config, fold, device); dif_records.append(audit)
        fit_observations = mask[fit].sum(0)
        stable = (~any_dif) & (fit_observations >= int(config["anchor"]["minimum_fit_observations"]))
        for item, metadata in enumerate(items):
            anchor_records.append({"fold": fold, "item": fields[item], "instrument": metadata["instrument"],
                                   "domain": metadata["domain"], "fit_observations": int(fit_observations[item]),
                                   "time_dif": bool(time_dif[item]), "any_audited_dif": bool(any_dif[item]),
                                   "anchor_candidate": bool(stable[item])})
        models: dict[str, AnchoredBifactorIRT] = {"invariant_mirt": invariant}
        for arm, dif_mask in [("partial_invariant_mirt", time_dif),
                              ("no_anchor_mirt", np.ones(len(items), bool))]:
            if arm == "partial_invariant_mirt" and not dif_mask.any():
                # Partial invariance is exactly the invariant model when the
                # frozen inner-fit DIF rule releases no item parameters.
                models[arm] = copy.deepcopy(invariant)
                continue
            model = AnchoredBifactorIRT(items, response_type, dif_mask,
                                       config["training"]["minimum_precision"])
            model.load_state_dict(invariant.state_dict()); model.dif_mask.copy_(torch.as_tensor(dif_mask, dtype=torch.float32))
            model, _, _, _ = fit_model(
                model, fraction[fit], category[fit], mask[fit], time_bin[fit],
                fraction[development], category[development], mask[development], time_bin[development],
                dev_ids, instruments, config, device, int(config["training"]["finetune_epochs"]),
                float(config["training"]["finetune_learning_rate"]), seed + fold * 101 + len(arm))
            models[arm] = model
        for arm, model in models.items():
            destination = checkpoint_root / arm; destination.mkdir(exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "fold": fold, "fields": fields,
                        "dif_items": [fields[i] for i in np.where(model.dif_mask.cpu().numpy() > 0)[0]],
                        "best_dev_nll": model.best_dev_nll}, destination / f"fold_{fold}.pt")
        local_ids = [sample_ids[i] for i in np.flatnonzero(evaluation)]
        evaluation_rows = np.flatnonzero(evaluation)
        for task in ["heldout_item", "heldout_instrument"]:
            input_mask, target_mask = deterministic_targets(mask[evaluation], instruments, local_ids, task,
                                                             float(config["masking"]["item_fraction"]))
            for arm, model in models.items():
                posterior_mean, posterior_logvar = infer_state(
                    model, fraction[evaluation], category[evaluation], input_mask,
                    time_bin[evaluation], config, device)
                with torch.no_grad():
                    expected = model.expected_fraction(
                        torch.as_tensor(posterior_mean, dtype=torch.float32, device=device),
                        torch.as_tensor(time_bin[evaluation], dtype=torch.long, device=device)).cpu().numpy()
                    nll = model.nll_matrix(
                        torch.as_tensor(posterior_mean, dtype=torch.float32, device=device),
                        torch.as_tensor(time_bin[evaluation], dtype=torch.long, device=device),
                        torch.as_tensor(np.nan_to_num(fraction[evaluation]), dtype=torch.float32, device=device),
                        torch.as_tensor(category[evaluation], dtype=torch.long, device=device),
                        torch.as_tensor(target_mask, dtype=torch.bool, device=device)).cpu().numpy()
                prediction_z = severity_fraction_to_z(expected, items, scaling)
                low90, high90, low95, high95 = interval_z(
                    model, posterior_mean, time_bin[evaluation], items, scaling, device)
                for row, item in zip(*np.where(target_mask)):
                    global_row = evaluation_rows[row]; observed = truth_z[global_row, item]
                    predictions.append({"sample_id": sample_ids[global_row],
                                        "subject_id": frame.subject_id.iloc[global_row], "fold": fold,
                                        "task": task, "arm": arm, "item": fields[item],
                                        "instrument": instruments[item], "domain": items[item]["domain"],
                                        "truth_z": float(observed), "prediction_z": float(prediction_z[row, item]),
                                        "absolute_error": float(abs(prediction_z[row, item] - observed)),
                                        "negative_log_likelihood": float(nll[row, item]),
                                        "covered_90": (bool(low90[row, item] <= observed <= high90[row, item])
                                                       if np.isfinite(low90[row, item]) else np.nan),
                                        "covered_95": (bool(low95[row, item] <= observed <= high95[row, item])
                                                       if np.isfinite(low95[row, item]) else np.nan)})
        view_a, view_b = cross_views(mask[evaluation], instruments, local_ids)
        valid = (view_a[:, memory].any(1) & view_b[:, memory].any(1)
                 & (view_a.sum(1) >= 2) & (view_b.sum(1) >= 2))
        for arm, model in models.items():
            mean_a, logvar_a = infer_state(model, fraction[evaluation], category[evaluation], view_a,
                                           time_bin[evaluation], config, device)
            mean_b, logvar_b = infer_state(model, fraction[evaluation], category[evaluation], view_b,
                                           time_bin[evaluation], config, device)
            for row in np.flatnonzero(valid):
                global_row = evaluation_rows[row]
                states.append({"sample_id": sample_ids[global_row], "subject_id": frame.subject_id.iloc[global_row],
                               "fold": fold, "arm": arm, "view_a": float(mean_a[row, 1]),
                               "view_b": float(mean_b[row, 1]),
                               "view_a_sd": float(np.exp(0.5 * logvar_a[row, 1])),
                               "view_b_sd": float(np.exp(0.5 * logvar_b[row, 1])),
                               "absolute_difference": float(abs(mean_a[row, 1] - mean_b[row, 1]))})
        manifests.append({"fold": fold, "fit_subjects": int(frame.loc[fit, "subject_id"].nunique()),
                          "dev_subjects": int(frame.loc[development, "subject_id"].nunique()),
                          "evaluation_subjects": int(frame.loc[evaluation, "subject_id"].nunique()),
                          "fit_rows": int(fit.sum()), "dev_rows": int(development.sum()),
                          "evaluation_rows": int(evaluation.sum()), "time_dif_items": int(time_dif.sum()),
                          "stable_anchor_candidates": int(stable.sum()),
                          "stable_memory_anchor_candidates": int((stable & memory).sum())})
    candidate_predictions = pd.DataFrame(predictions)
    phase5c_predictions = pd.read_csv(ROOT / config["phase5c_output"] / "crossfit_predictions.csv")
    baselines = phase5c_predictions[phase5c_predictions.arm.isin(config["baseline_arms"])].copy()
    baselines["negative_log_likelihood"] = np.nan; baselines["covered_90"] = np.nan; baselines["covered_95"] = np.nan
    predictions_frame = pd.concat([baselines[candidate_predictions.columns], candidate_predictions], ignore_index=True)
    predictions_frame.to_csv(output / "crossfit_predictions.csv", index=False)
    states_frame = pd.DataFrame(states); states_frame.to_csv(output / "cross_view_memory_states.csv", index=False)
    dif_frame = pd.concat(dif_records, ignore_index=True); dif_frame.to_csv(output / "dif_audit.csv", index=False)
    anchors = pd.DataFrame(anchor_records); anchors.to_csv(output / "fold_anchor_candidates.csv", index=False)
    pd.DataFrame(manifests).to_csv(output / "fold_manifest.csv", index=False)
    item_metrics = predictions_frame.groupby(["arm", "task", "item", "domain"], as_index=False).agg(
        decisions=("absolute_error", "size"), subjects=("subject_id", "nunique"),
        mae=("absolute_error", "mean"), mean_nll=("negative_log_likelihood", "mean"),
        coverage_90=("covered_90", "mean"), coverage_95=("covered_95", "mean"))
    item_metrics.to_csv(output / "item_metrics.csv", index=False)
    aggregate = item_metrics.groupby(["arm", "task"], as_index=False).agg(
        macro_item_mae=("mae", "mean"), macro_item_nll=("mean_nll", "mean"),
        macro_coverage_90=("coverage_90", "mean"), macro_coverage_95=("coverage_95", "mean"),
        items=("item", "nunique"), decisions=("decisions", "sum"))
    memory_metrics = item_metrics[item_metrics.domain.eq("Memory")].groupby(["arm", "task"], as_index=False).agg(
        memory_macro_item_mae=("mae", "mean"), memory_items=("item", "nunique"))
    aggregate = aggregate.merge(memory_metrics, on=["arm", "task"]); aggregate.to_csv(output / "aggregate_metrics.csv", index=False)
    state_summary = []
    for arm, group in states_frame.groupby("arm"):
        correlation = spearmanr(group.view_a, group.view_b)
        state_summary.append({"arm": arm, "pairs": len(group), "subjects": group.subject_id.nunique(),
                              "ccc": concordance_correlation(group.view_a, group.view_b),
                              "spearman": float(correlation.statistic),
                              "mean_absolute_difference": group.absolute_difference.mean(),
                              "mean_posterior_sd": pd.concat([group.view_a_sd, group.view_b_sd]).mean()})
    state_summary_frame = pd.DataFrame(state_summary); state_summary_frame.to_csv(output / "cross_view_summary.csv", index=False)
    comparison = cluster_bootstrap(predictions_frame, config); comparison.to_csv(output / "subject_bootstrap.csv", index=False)
    state_ci = ccc_bootstrap(states_frame, config); (output / "cross_view_ccc_bootstrap.json").write_text(json.dumps(state_ci, indent=2) + "\n")
    stable_counts = anchors[anchors.anchor_candidate].groupby(["item", "instrument", "domain"], as_index=False).agg(
        stable_folds=("fold", "nunique"), minimum_fit_observations=("fit_observations", "min"))
    stable_counts["stable_for_gate"] = stable_counts.stable_folds >= int(config["anchor"]["minimum_stable_folds"])
    stable_counts.to_csv(output / "anchor_stability.csv", index=False)
    candidate = config["gate"]["candidate"]; instrument_comparison = comparison[comparison.task.eq("heldout_instrument")].set_index("baseline")
    fixed = config["gate"]["fixed_baseline"]; predictive = config["gate"]["predictive_baseline"]
    fixed_pass = bool(instrument_comparison.loc[fixed, "ci_high"] < 0)
    ae_pass = bool(instrument_comparison.loc[predictive, "ci_high"] <= float(config["gate"]["ae_noninferiority_margin"]))
    ccc_pass = bool(state_ci["ci_low"] > float(config["gate"]["memory_cross_view_ccc_lower_bound"]))
    stable_memory = stable_counts[stable_counts.domain.eq("Memory") & stable_counts.stable_for_gate]
    anchor_pass = bool(len(stable_memory) >= int(config["gate"]["minimum_memory_anchors"])
                       and stable_memory.instrument.nunique() >= int(config["gate"]["minimum_anchor_instruments"]))
    fold_errors = predictions_frame[predictions_frame.task.eq("heldout_instrument")].groupby(
        ["fold", "arm"], as_index=False).absolute_error.mean().pivot(index="fold", columns="arm", values="absolute_error")
    folds_better = int((fold_errors[candidate] < fold_errors[fixed]).sum())
    fold_pass = folds_better >= int(config["gate"]["minimum_folds_better_than_fixed"])
    gate_passed = bool(fixed_pass and ae_pass and ccc_pass and anchor_pass and fold_pass)
    decision = {"run_id": config["run_id"], "development_only": True, "original_train_only": True,
                "validation_loaded": False, "test_loaded": False, "candidate": candidate,
                "fixed_score_improvement": fixed_pass, "autoencoder_noninferiority": ae_pass,
                "memory_cross_view_ccc": state_ci, "cross_view_gate": ccc_pass,
                "stable_memory_anchors": int(len(stable_memory)),
                "stable_memory_anchor_instruments": int(stable_memory.instrument.nunique()),
                "anchor_gate": anchor_pass, "folds_better_than_fixed": folds_better,
                "fold_reproducibility_gate": fold_pass, "phase5d_measurement_gate_passed": gate_passed,
                "continuous_time_dynamics_open": gate_passed,
                "next_step": ("freeze_phase5e_continuous_time_protocol" if gate_passed
                              else "retain_measurement_negative_and_do_not_enter_dynamics")}
    (output / "development_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    metrics = aggregate.set_index(["arm", "task"])
    report = f"""# Phase 5D: Anchored Bifactor MIRT and Measurement Invariance

## Scope

Five-fold nested subject-wise development used only the original ADNI train split. The model used a positive-loading global impairment factor plus a positive-loading Memory-specific factor, typed item likelihoods, and explicit time-DIF parameters. Original validation and test subjects were not loaded. Diagnosis and ADNI phase were audited for residual DIF but were not inputs to state inference.

## Concurrent measurement audit

All five instruments were connected by nominal-visit co-observation. Actual date coverage was available for ADAS, CDR, MMSE, and MoCA; the Logical Memory exam-date columns were empty in the frozen split, so Logical Memory linkage remained visit-code based and is reported separately in `instrument_connectivity.csv`.

## Predictive measurement results

The partial-invariant candidate obtained held-out-item/held-out-instrument macro normalized MAE **{metrics.loc[(candidate, 'heldout_item'), 'macro_item_mae']:.4f}/{metrics.loc[(candidate, 'heldout_instrument'), 'macro_item_mae']:.4f}**. Fixed domain scoring obtained **{metrics.loc[(fixed, 'heldout_item'), 'macro_item_mae']:.4f}/{metrics.loc[(fixed, 'heldout_instrument'), 'macro_item_mae']:.4f}**, and the unconstrained autoencoder obtained **{metrics.loc[(predictive, 'heldout_item'), 'macro_item_mae']:.4f}/{metrics.loc[(predictive, 'heldout_instrument'), 'macro_item_mae']:.4f}**.

The candidate-minus-fixed held-out-instrument participant difference was **{instrument_comparison.loc[fixed, 'difference']:.5f}**, 95% CI **[{instrument_comparison.loc[fixed, 'ci_low']:.5f}, {instrument_comparison.loc[fixed, 'ci_high']:.5f}]**. Candidate-minus-autoencoder was **{instrument_comparison.loc[predictive, 'difference']:.5f}**, 95% CI **[{instrument_comparison.loc[predictive, 'ci_low']:.5f}, {instrument_comparison.loc[predictive, 'ci_high']:.5f}]**.

## State coherence and invariance

The frozen practical rule selected **{int(pd.DataFrame(manifests).time_dif_items.sum())}** fold-specific time-DIF item decisions; when none were selected, the partial-invariant arm was exactly the invariant arm rather than a stochastic refit. Memory cross-view concordance was **{state_ci['ccc']:.4f}**, participant-bootstrap 95% CI **[{state_ci['ci_low']:.4f}, {state_ci['ci_high']:.4f}]**. There were **{len(stable_memory)}** stable Memory anchors from **{stable_memory.instrument.nunique()}** instruments in at least {config['anchor']['minimum_stable_folds']} of {folds} folds. The candidate beat fixed scoring in **{folds_better}/{folds}** outer folds.

## Gate

- improvement over fixed score: `{str(fixed_pass).lower()}`;
- non-inferiority to the unconstrained autoencoder within +{config['gate']['ae_noninferiority_margin']:.2f}: `{str(ae_pass).lower()}`;
- cross-view concordance lower bound above {config['gate']['memory_cross_view_ccc_lower_bound']:.2f}: `{str(ccc_pass).lower()}`;
- stable Memory anchor requirement: `{str(anchor_pass).lower()}`;
- outer-fold reproducibility: `{str(fold_pass).lower()}`.

Therefore `phase5d_measurement_gate_passed={str(gate_passed).lower()}` and `continuous_time_dynamics_open={str(gate_passed).lower()}`. A named and anchored state is accepted only when identification, cross-instrument coherence, and predictive measurement all pass together.
"""
    (output / "report.md").write_text(report)
    manifest = {"config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
                "contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
                "phase5c_predictions_sha256": hashlib.sha256((ROOT / config["phase5c_output"] / "crossfit_predictions.csv").read_bytes()).hexdigest(),
                "train_rows_loaded": len(frame), "train_subjects_loaded": int(frame.subject_id.nunique()),
                "validation_or_test_loaded": False, "device": str(device), "neural_checkpoints": folds * len(config["arms"])}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output


def main() -> None:
    argparse.ArgumentParser().parse_args(); print(run())


if __name__ == "__main__":
    main()
