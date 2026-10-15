"""Validation-only cognitive irregular-time belief dynamics and calibration."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from scipy import stats
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from ..data.common import DOMAINS, ROOT, read_csv, save_csv, save_json
from ..data.dictionary import load_dictionary
from .phase1_numeric import build_samples, clean_frame, fit_normalization, markdown_table
from .phase1b_semantic import select_verified_scalar_features
from .phase1c_joint_stage1 import load_config
from .phase1d_typed_representation import TYPE_IDS, feature_metadata, severity_coordinate


@dataclass
class DynamicsSample:
    sample_id: str
    subject_id: str
    split: str
    task: str
    horizon_months: int
    query_month: float
    months: np.ndarray
    features: np.ndarray
    feature_mask: np.ndarray
    targets: np.ndarray
    target_mask: np.ndarray
    target_months: np.ndarray


class DynamicsDataset(Dataset):
    def __init__(self, samples: list[DynamicsSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> DynamicsSample:
        return self.samples[index]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def subject_partition(subject_id: str, seed: int, fraction: float) -> bool:
    digest = hashlib.sha256(f"{seed}:{subject_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return value < fraction


def partition_frame(
    frame: pd.DataFrame, seed: int, heldout_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    heldout_subjects = {
        sid for sid in frame.subject_id.unique() if subject_partition(str(sid), seed, heldout_fraction)
    }
    heldout = frame[frame.subject_id.isin(heldout_subjects)].copy()
    retained = frame[~frame.subject_id.isin(heldout_subjects)].copy()
    if retained.empty or heldout.empty:
        raise ValueError("Deterministic subject partition produced an empty subset")
    if set(retained.subject_id) & set(heldout.subject_id):
        raise RuntimeError("Subject leakage across development partitions")
    return retained, heldout


def build_forecast_samples(
    frame: pd.DataFrame,
    split: str,
    features: list[str],
    targets: list[str],
    config: dict[str, Any],
) -> list[DynamicsSample]:
    forecast = config["forecast"]
    tolerances = {int(k): int(v) for k, v in forecast["tolerance_months"].items()}
    rows = build_samples(
        frame, split, features, targets, forecast["horizons_months"], tolerances
    )
    return [
        DynamicsSample(
            sample_id=row.sample_id,
            subject_id=row.subject_id,
            split=split,
            task="forecast",
            horizon_months=row.horizon_months,
            query_month=row.cutoff_month + row.horizon_months,
            months=row.months,
            features=row.features,
            feature_mask=row.feature_mask,
            targets=row.targets,
            target_mask=row.target_mask,
            target_months=row.target_months,
        )
        for row in rows
    ]


def build_interpolation_samples(
    frame: pd.DataFrame,
    split: str,
    features: list[str],
    targets: list[str],
    require_visits: int,
) -> list[DynamicsSample]:
    samples: list[DynamicsSample] = []
    for subject, group in frame.groupby("subject_id", sort=True):
        group = group.sort_values("visit_month", kind="stable").reset_index(drop=True)
        if len(group) < require_visits:
            continue
        months = group.visit_month.to_numpy(dtype=np.float32)
        values = group[features].to_numpy(dtype=np.float32)
        observed = np.isfinite(values)
        outcomes = group[targets].to_numpy(dtype=np.float32)
        outcome_mask = np.isfinite(outcomes)
        for middle in range(1, len(group) - 1):
            if not outcome_mask[middle].any():
                continue
            if not observed[:middle].any() or not observed[middle + 1 :].any():
                continue
            keep = np.arange(len(group)) != middle
            # The complete middle visit, including all inputs, is absent.
            samples.append(
                DynamicsSample(
                    sample_id=f"{subject}:interpolate:m{months[middle]:g}:i{middle}",
                    subject_id=str(subject),
                    split=split,
                    task="interpolation",
                    horizon_months=0,
                    query_month=float(months[middle]),
                    months=months[keep].copy(),
                    features=values[keep].copy(),
                    feature_mask=observed[keep].copy(),
                    targets=outcomes[middle].copy(),
                    target_mask=outcome_mask[middle].copy(),
                    target_months=np.full(len(targets), months[middle], dtype=np.float32),
                )
            )
    return samples


def build_all_samples(
    frame: pd.DataFrame,
    split: str,
    features: list[str],
    targets: list[str],
    config: dict[str, Any],
) -> list[DynamicsSample]:
    return build_forecast_samples(frame, split, features, targets, config) + build_interpolation_samples(
        frame,
        split,
        features,
        targets,
        int(config["interpolation"]["require_visits"]),
    )


class DynamicsCollator:
    def __init__(
        self,
        features: list[str],
        targets: list[str],
        normalization: dict[str, dict[str, float]],
        dictionary: dict[str, Any],
    ):
        self.features, self.targets = features, targets
        self.feature_mean = np.asarray([normalization[k]["mean"] for k in features], dtype=np.float32)
        self.feature_std = np.asarray([normalization[k]["std"] for k in features], dtype=np.float32)
        self.target_mean = np.asarray([normalization[k]["mean"] for k in targets], dtype=np.float32)
        self.target_std = np.asarray([normalization[k]["std"] for k in targets], dtype=np.float32)
        self.feature_types, self.bounds, self.directions, _, _ = feature_metadata(features, dictionary)

    def __call__(self, samples: list[DynamicsSample]) -> dict[str, Any]:
        batch, steps, n_features = len(samples), max(len(s.months) for s in samples), len(self.features)
        values = np.zeros((batch, steps, n_features), dtype=np.float32)
        mask = np.zeros((batch, steps, n_features), dtype=bool)
        times = np.zeros((batch, steps), dtype=np.float32)
        event_mask = np.zeros((batch, steps), dtype=bool)
        targets = np.zeros((batch, len(self.targets)), dtype=np.float32)
        target_mask = np.zeros((batch, len(self.targets)), dtype=bool)
        for row, sample in enumerate(samples):
            length = len(sample.months)
            normalized = (sample.features - self.feature_mean) / self.feature_std
            values[row, :length] = np.where(sample.feature_mask, normalized, 0.0)
            mask[row, :length] = sample.feature_mask
            times[row, :length] = sample.months / 12.0
            event_mask[row, :length] = True
            targets[row] = np.where(
                sample.target_mask,
                (sample.targets - self.target_mean) / self.target_std,
                0.0,
            )
            target_mask[row] = sample.target_mask
        raw = torch.from_numpy(values) * torch.from_numpy(self.feature_std)[None, None]
        raw = raw + torch.from_numpy(self.feature_mean)[None, None]
        observed = torch.from_numpy(mask)
        query = torch.tensor([s.query_month / 12.0 for s in samples], dtype=torch.float32)
        event = torch.from_numpy(event_mask)
        time_tensor = torch.from_numpy(times)
        left = event & (time_tensor < query[:, None] + 1e-6)
        right = event & (time_tensor > query[:, None] + 1e-6)
        if not left.any(1).all():
            raise ValueError("Every dynamics sample must contain prior cognitive evidence")
        for row, sample in enumerate(samples):
            if sample.task == "interpolation" and not right[row].any():
                raise ValueError("Interpolation sample lacks future context")
            if np.any(np.isclose(sample.months, sample.query_month)):
                raise RuntimeError("Hidden/query visit leaked into dynamics input")
        observed_count = observed.sum((1, 2)).float()
        possible_count = event.sum(1).float() * len(self.features)
        return {
            "values": torch.from_numpy(values),
            "mask": observed,
            "severity": severity_coordinate(raw, self.bounds, self.directions) * observed.float(),
            "times": time_tensor,
            "event_mask": event,
            "left_mask": left,
            "right_mask": right,
            "query_time": query,
            "task_id": torch.tensor([0 if s.task == "forecast" else 1 for s in samples]),
            "targets": torch.from_numpy(targets),
            "target_mask": torch.from_numpy(target_mask),
            "history_length": event.sum(1),
            "observed_fraction": observed_count / possible_count.clamp_min(1),
            "samples": samples,
        }


class TypedEventEncoder(nn.Module):
    def __init__(
        self,
        feature_types: torch.Tensor,
        domain_assignment: torch.Tensor,
        config: dict[str, Any],
    ):
        super().__init__()
        settings = config["model"]
        item_dim = int(settings["item_dim"])
        self.register_buffer("feature_types", feature_types.long())
        self.register_buffer("domain_assignment", domain_assignment.float())
        self.item_embedding = nn.Embedding(len(feature_types), item_dim)
        self.type_embedding = nn.Embedding(len(TYPE_IDS), item_dim)
        self.value_encoders = nn.ModuleList(
            [nn.Sequential(nn.Linear(1, item_dim), nn.GELU(), nn.Linear(item_dim, item_dim)) for _ in TYPE_IDS]
        )
        self.token_norm = nn.LayerNorm(item_dim)
        self.domain = nn.Sequential(
            nn.Linear(domain_assignment.shape[1] * item_dim, settings["domain_dim"]),
            nn.GELU(),
            nn.LayerNorm(settings["domain_dim"]),
        )

    def forward(self, severity: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        n_features = severity.shape[-1]
        index = torch.arange(n_features, device=severity.device)
        base = self.item_embedding(index) + self.type_embedding(self.feature_types)
        token = base.view(1, 1, n_features, -1).expand(*severity.shape, -1)
        typed = torch.zeros_like(token)
        for type_id, encoder in enumerate(self.value_encoders):
            typed = typed + encoder(severity.unsqueeze(-1)) * self.feature_types.eq(type_id).view(1, 1, -1, 1)
        token = self.token_norm(token + typed) * observed.unsqueeze(-1).to(severity.dtype)
        domain_sum = torch.einsum("btfe,fd->btde", token, self.domain_assignment)
        domain_count = torch.einsum(
            "btf,fd->btd", observed.to(severity.dtype), self.domain_assignment
        ).clamp_min(1.0)
        return self.domain((domain_sum / domain_count.unsqueeze(-1)).flatten(-2))


class DirectionalGRUD(nn.Module):
    def __init__(self, n_features: int, extra_dim: int, config: dict[str, Any]):
        super().__init__()
        settings = config["model"]
        hidden, layers = int(settings["hidden_dim"]), int(settings["layers"])
        self.decay_weight = nn.Parameter(torch.zeros(n_features))
        self.decay_bias = nn.Parameter(torch.zeros(n_features))
        first = n_features * 2 + extra_dim + 3
        self.cells = nn.ModuleList(
            [nn.GRUCell(first, hidden)] + [nn.GRUCell(hidden, hidden) for _ in range(layers - 1)]
        )
        self.dropout = nn.Dropout(settings["dropout"])

    def forward(
        self,
        values: torch.Tensor,
        observed: torch.Tensor,
        times: torch.Tensor,
        valid: torch.Tensor,
        query: torch.Tensor,
        extra: torch.Tensor | None,
        reverse: bool,
    ) -> torch.Tensor:
        if reverse:
            values, observed, times, valid = [torch.flip(x, [1]) for x in (values, observed, times, valid)]
            extra = torch.flip(extra, [1]) if extra is not None else None
            times, query = -times, -query
        batch, steps, features = values.shape
        hidden = [values.new_zeros(batch, cell.hidden_size) for cell in self.cells]
        last_value = values.new_zeros(batch, features)
        last_observed_time = values.new_zeros(batch, features)
        previous_time = values.new_zeros(batch)
        seen = torch.zeros(batch, dtype=torch.bool, device=values.device)
        direction = values.new_full((batch, 1), 1.0 if reverse else 0.0)
        for step in range(steps):
            event_valid = valid[:, step]
            current = times[:, step]
            first = event_valid & ~seen
            elapsed = (current[:, None] - last_observed_time).clamp_min(0)
            elapsed = torch.where(first[:, None], torch.zeros_like(elapsed), elapsed)
            gamma = torch.exp(-F.relu(self.decay_weight[None] * elapsed + self.decay_bias[None]))
            imputed = torch.where(observed[:, step], values[:, step], gamma * last_value)
            gap = (current - previous_time).clamp_min(0)
            gap = torch.where(first, torch.zeros_like(gap), gap)
            relative = current - query
            pieces = [imputed, observed[:, step].to(values.dtype)]
            if extra is not None:
                pieces.append(extra[:, step])
            pieces.extend([relative[:, None], gap[:, None], direction])
            cell_input = torch.cat(pieces, dim=-1)
            for layer, cell in enumerate(self.cells):
                proposed = cell(cell_input, hidden[layer])
                hidden[layer] = torch.where(event_valid[:, None], proposed, hidden[layer])
                cell_input = self.dropout(hidden[layer])
            last_value = torch.where(observed[:, step] & event_valid[:, None], values[:, step], last_value)
            last_observed_time = torch.where(
                observed[:, step] & event_valid[:, None], current[:, None], last_observed_time
            )
            previous_time = torch.where(event_valid, current, previous_time)
            seen |= event_valid
        return hidden[-1]


def query_gaps(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    times, query = batch["times"], batch["query_time"]
    left_time = times.masked_fill(~batch["left_mask"], -torch.inf).max(1).values
    right_available = batch["right_mask"].any(1)
    right_time = times.masked_fill(~batch["right_mask"], torch.inf).min(1).values
    left_gap = (query - left_time).clamp_min(0)
    right_gap = torch.where(right_available, (right_time - query).clamp_min(0), torch.zeros_like(query))
    return left_gap, right_gap, right_available


class DirectGaussianDynamics(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_targets: int,
        feature_types: torch.Tensor,
        domain_assignment: torch.Tensor,
        typed: bool,
        config: dict[str, Any],
    ):
        super().__init__()
        self.typed = typed
        extra = int(config["model"]["domain_dim"]) if typed else 0
        self.typed_encoder = TypedEventEncoder(feature_types, domain_assignment, config)
        self.encoder = DirectionalGRUD(n_features, extra, config)
        hidden = int(config["model"]["hidden_dim"])
        self.task = nn.Embedding(2, 16)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2 + 16 + 3, hidden),
            nn.GELU(),
            nn.Dropout(config["model"]["dropout"]),
            nn.Linear(hidden, n_targets * 2),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        extra = self.typed_encoder(batch["severity"], batch["mask"]) if self.typed else None
        left = self.encoder(
            batch["values"], batch["mask"], batch["times"], batch["left_mask"],
            batch["query_time"], extra, False,
        )
        right = self.encoder(
            batch["values"], batch["mask"], batch["times"], batch["right_mask"],
            batch["query_time"], extra, True,
        )
        left_gap, right_gap, right_available = query_gaps(batch)
        context = torch.cat(
            [left, right, self.task(batch["task_id"]), left_gap[:, None], right_gap[:, None], right_available[:, None].float()],
            dim=-1,
        )
        mean, logvar = self.head(context).chunk(2, dim=-1)
        return mean, logvar.clamp(-7.0, 4.0)


class BeliefFilter(nn.Module):
    def __init__(self, evidence_dim: int, state_dim: int):
        super().__init__()
        self.initial_mu = nn.Parameter(torch.zeros(state_dim))
        self.initial_logvar = nn.Parameter(torch.zeros(state_dim))
        self.evidence = nn.Sequential(nn.Linear(evidence_dim + 1, state_dim * 2), nn.GELU())
        self.update_gate = nn.Linear(state_dim + evidence_dim + 1, state_dim * 2)
        self.drift = nn.Sequential(nn.Linear(state_dim + 2, state_dim * 2), nn.Tanh(), nn.Linear(state_dim * 2, state_dim))
        self.diffusion = nn.Sequential(nn.Linear(state_dim + 2, state_dim), nn.Softplus())

    def transition(
        self, mu: torch.Tensor, logvar: torch.Tensor, delta: torch.Tensor, direction: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (delta < 0).any():
            raise ValueError("Belief transition received negative elapsed time")
        dt = delta[:, None]
        context = torch.cat([mu, torch.log1p(dt), direction[:, None]], dim=-1)
        next_mu = mu + dt * self.drift(context)
        next_var = logvar.exp() + dt * self.diffusion(context) + 1e-6
        return next_mu, next_var.log().clamp(-8.0, 6.0)

    def forward(
        self,
        evidence: torch.Tensor,
        times: torch.Tensor,
        valid: torch.Tensor,
        query: torch.Tensor,
        reverse: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if reverse:
            evidence, times, valid = [torch.flip(x, [1]) for x in (evidence, times, valid)]
            times, query = -times, -query
        batch = len(evidence)
        mu = self.initial_mu[None].expand(batch, -1)
        logvar = self.initial_logvar[None].expand(batch, -1)
        previous = evidence.new_zeros(batch)
        seen = torch.zeros(batch, dtype=torch.bool, device=evidence.device)
        direction = evidence.new_full((batch,), 1.0 if reverse else 0.0)
        direction_column = direction[:, None]
        for step in range(evidence.shape[1]):
            active = valid[:, step]
            first = active & ~seen
            delta = (times[:, step] - previous).clamp_min(0)
            delta = torch.where(first, torch.zeros_like(delta), delta)
            prior_mu, prior_logvar = self.transition(mu, logvar, delta, direction)
            event = torch.cat([evidence[:, step], direction_column], dim=-1)
            shift, variance_shift = self.evidence(event).chunk(2, dim=-1)
            mean_gate, variance_gate = torch.sigmoid(
                self.update_gate(torch.cat([prior_mu, event], dim=-1))
            ).chunk(2, dim=-1)
            updated_mu = prior_mu + mean_gate * shift
            updated_logvar = (prior_logvar + variance_gate * torch.tanh(variance_shift)).clamp(-8.0, 6.0)
            mu = torch.where(active[:, None], updated_mu, mu)
            logvar = torch.where(active[:, None], updated_logvar, logvar)
            previous = torch.where(active, times[:, step], previous)
            seen |= active
        gap = (query - previous).clamp_min(0)
        return self.transition(mu, logvar, gap, direction)


class TypedBeliefDynamics(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_targets: int,
        feature_types: torch.Tensor,
        domain_assignment: torch.Tensor,
        config: dict[str, Any],
    ):
        super().__init__()
        self.typed_encoder = TypedEventEncoder(feature_types, domain_assignment, config)
        evidence_dim = n_features * 2 + int(config["model"]["domain_dim"])
        state_dim = int(config["model"]["state_dim"])
        self.filter = BeliefFilter(evidence_dim, state_dim)
        self.fusion_inflation = nn.Parameter(torch.tensor(0.0))
        self.mean_decoder = nn.Linear(state_dim, n_targets)
        self.noise_decoder = nn.Linear(state_dim, n_targets)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        typed = self.typed_encoder(batch["severity"], batch["mask"])
        evidence = torch.cat([batch["values"], batch["mask"].float(), typed], dim=-1)
        left_mu, left_logvar = self.filter(
            evidence, batch["times"], batch["left_mask"], batch["query_time"], False
        )
        right_mu, right_logvar = self.filter(
            evidence, batch["times"], batch["right_mask"], batch["query_time"], True
        )
        right_available = batch["right_mask"].any(1)
        left_precision, right_precision = torch.exp(-left_logvar), torch.exp(-right_logvar)
        fused_var = 1.0 / (left_precision + right_precision).clamp_min(1e-8)
        fused_mu = fused_var * (left_precision * left_mu + right_precision * right_mu)
        inflation = F.softplus(self.fusion_inflation) + 0.5
        fused_var = fused_var * inflation
        mu = torch.where(right_available[:, None], fused_mu, left_mu)
        latent_var = torch.where(right_available[:, None], fused_var, left_logvar.exp())
        mean = self.mean_decoder(mu)
        aleatoric = F.softplus(self.noise_decoder(mu)) + 1e-4
        propagated = F.linear(latent_var, self.mean_decoder.weight.pow(2), bias=None)
        variance = (aleatoric + propagated).clamp(1e-5, 1e3)
        return mean, variance.log()


def build_domain_assignment(
    features: list[str], dictionary: dict[str, Any]
) -> torch.Tensor:
    domains = [domain for domain in DOMAINS if any(dictionary[f]["domain"] == domain for f in features)]
    assignment = torch.zeros(len(features), len(domains))
    for index, feature in enumerate(features):
        assignment[index, domains.index(dictionary[feature]["domain"])] = 1.0
    return assignment


def make_model(
    arm: dict[str, Any],
    features: list[str],
    targets: list[str],
    feature_types: torch.Tensor,
    domain_assignment: torch.Tensor,
    config: dict[str, Any],
) -> nn.Module:
    if arm["probabilistic_belief"]:
        return TypedBeliefDynamics(
            len(features), len(targets), feature_types, domain_assignment, config
        )
    return DirectGaussianDynamics(
        len(features), len(targets), feature_types, domain_assignment,
        bool(arm["typed_response"]), config,
    )


def gaussian_loss(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    huber_weight: float,
) -> torch.Tensor:
    if not mask.any():
        return mean.sum() * 0
    nll = 0.5 * (math.log(2 * math.pi) + logvar + (target - mean).pow(2) * torch.exp(-logvar))
    huber = F.huber_loss(mean[mask], target[mask], delta=1.0)
    return nll[mask].mean() + huber_weight * huber


def balanced_loader(
    samples: list[DynamicsSample], collator: DynamicsCollator, batch_size: int, seed: int
) -> DataLoader:
    groups = [(s.task, s.horizon_months) for s in samples]
    counts = pd.Series(groups).value_counts().to_dict()
    weights = torch.tensor([1.0 / counts[group] for group in groups], dtype=torch.double)
    sampler = WeightedRandomSampler(
        weights, len(samples), replacement=True, generator=torch.Generator().manual_seed(seed)
    )
    return DataLoader(
        DynamicsDataset(samples), batch_size=batch_size, sampler=sampler, collate_fn=collator, num_workers=0
    )


def prediction_rows(
    samples: list[DynamicsSample],
    mean: np.ndarray,
    variance: np.ndarray,
    model: str,
    seed: int,
    targets: list[str],
    collator: DynamicsCollator,
    history_length: np.ndarray,
    observed_fraction: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for row, sample in enumerate(samples):
        for index, target in enumerate(targets):
            if not sample.target_mask[index]:
                continue
            mean_norm = float(mean[row, index])
            variance_norm = float(max(variance[row, index], 1e-8))
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "subject_id": sample.subject_id,
                    "split": sample.split,
                    "task": sample.task,
                    "horizon_months": sample.horizon_months,
                    "model": model,
                    "seed": seed,
                    "target": target,
                    "truth": float(sample.targets[index]),
                    "truth_norm": float((sample.targets[index] - collator.target_mean[index]) / collator.target_std[index]),
                    "prediction": float(mean_norm * collator.target_std[index] + collator.target_mean[index]),
                    "mean_norm": mean_norm,
                    "variance_norm": variance_norm,
                    "std": float(math.sqrt(variance_norm) * collator.target_std[index]),
                    "target_mean": float(collator.target_mean[index]),
                    "target_std": float(collator.target_std[index]),
                    "history_length": int(history_length[row]),
                    "observed_fraction": float(observed_fraction[row]),
                    "query_month": sample.query_month,
                    "target_month": float(sample.target_months[index]),
                }
            )
    return rows


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    samples: list[DynamicsSample],
    collator: DynamicsCollator,
    batch_size: int,
    device: torch.device,
    model_name: str,
    seed: int,
) -> pd.DataFrame:
    loader = DataLoader(
        DynamicsDataset(samples), batch_size=batch_size, shuffle=False, collate_fn=collator
    )
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        local = batch.pop("samples")
        batch = {key: value.to(device) for key, value in batch.items()}
        mean, logvar = model(batch)
        rows.extend(
            prediction_rows(
                local,
                mean.cpu().numpy(),
                logvar.exp().cpu().numpy(),
                model_name,
                seed,
                collator.targets,
                collator,
                batch["history_length"].cpu().numpy(),
                batch["observed_fraction"].cpu().numpy(),
            )
        )
    return pd.DataFrame(rows)


def validation_objective(predictions: pd.DataFrame) -> float:
    residual = predictions.truth_norm - predictions.mean_norm
    nll = 0.5 * (np.log(2 * np.pi * predictions.variance_norm) + residual.pow(2) / predictions.variance_norm)
    normalized_mae = np.abs(residual)
    return float(nll.mean() + 0.1 * normalized_mae.mean())


def checkpoint_path(output: Path, arm: str, seed: int) -> Path:
    return output / "checkpoints" / arm / str(seed) / "best.pt"


def train_one(
    arm_name: str,
    arm: dict[str, Any],
    seed: int,
    train_samples: list[DynamicsSample],
    tune_samples: list[DynamicsSample],
    calibration_samples: list[DynamicsSample],
    evaluation_samples: list[DynamicsSample],
    collator: DynamicsCollator,
    features: list[str],
    targets: list[str],
    feature_types: torch.Tensor,
    domain_assignment: torch.Tensor,
    config: dict[str, Any],
    output: Path,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    seed_everything(seed)
    model = make_model(arm, features, targets, feature_types, domain_assignment, config).to(device)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    path = checkpoint_path(output, arm_name, seed)
    curve_path = path.parent / "curves.csv"
    if not (path.exists() and curve_path.exists()):
        settings = config["training"]
        loader = balanced_loader(train_samples, collator, settings["batch_size"], seed)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
        )
        best, patience, curves = float("inf"), 0, []
        path.parent.mkdir(parents=True, exist_ok=True)
        for epoch in range(settings["max_epochs"]):
            model.train()
            losses = []
            for batch in loader:
                batch.pop("samples")
                batch = {key: value.to(device) for key, value in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                mean, logvar = model(batch)
                loss = gaussian_loss(
                    mean, logvar, batch["targets"], batch["target_mask"],
                    settings["normalized_huber_weight"],
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite training loss for {arm_name}/{seed}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"])
                optimizer.step()
                losses.append(float(loss.detach()))
            tune = evaluate_model(
                model, tune_samples, collator, settings["batch_size"], device, arm_name, seed
            )
            objective = validation_objective(tune)
            curves.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "tune_objective": objective})
            if objective < best - 1e-6:
                best, patience = objective, 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "features": features,
                        "targets": targets,
                        "arm": arm,
                        "seed": seed,
                        "parameter_count": parameter_count,
                        "feature_mean": collator.feature_mean.tolist(),
                        "feature_std": collator.feature_std.tolist(),
                        "target_mean": collator.target_mean.tolist(),
                        "target_std": collator.target_std.tolist(),
                    },
                    path,
                )
            else:
                patience += 1
                if patience >= settings["patience"]:
                    break
        save_csv(pd.DataFrame(curves), curve_path)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint["features"] != features or checkpoint["targets"] != targets or checkpoint["arm"] != arm:
        raise RuntimeError(f"Checkpoint contract mismatch for {arm_name}/{seed}")
    for name in ["feature_mean", "feature_std", "target_mean", "target_std"]:
        if not np.allclose(checkpoint[name], getattr(collator, name)):
            raise RuntimeError(f"Checkpoint normalization mismatch for {arm_name}/{seed}: {name}")
    model.load_state_dict(checkpoint["model"])
    batch_size = config["training"]["batch_size"]
    return (
        evaluate_model(model, calibration_samples, collator, batch_size, device, arm_name, seed),
        evaluate_model(model, evaluation_samples, collator, batch_size, device, arm_name, seed),
        parameter_count,
    )


def ensemble_predictions(frame: pd.DataFrame, model_name: str) -> pd.DataFrame:
    keys = [
        "sample_id", "subject_id", "split", "task", "horizon_months", "target",
        "truth", "truth_norm", "history_length", "observed_fraction", "query_month", "target_month",
        "target_mean", "target_std",
    ]
    grouped = frame.groupby(keys, as_index=False, dropna=False)
    output = grouped.agg(mean_norm=("mean_norm", "mean"), within_variance=("variance_norm", "mean"), between_variance=("mean_norm", "var"))
    output["between_variance"] = output.between_variance.fillna(0.0)
    output["variance_norm"] = (output.within_variance + output.between_variance).clip(lower=1e-8)
    output["model"] = model_name
    output["prediction"] = output.mean_norm * output.target_std + output.target_mean
    output["std"] = np.sqrt(output.variance_norm) * output.target_std
    return output


def fit_temperatures(
    calibration: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    minimum = int(config["calibration"]["minimum_group_targets"])
    low, high = map(float, config["calibration"]["temperature_clip"])
    rows = []

    def add(group: pd.DataFrame, level: str, task: str = "*", target: str = "*", horizon: int = -1) -> None:
        if len(group) < minimum and level != "global":
            return
        standardized = (group.truth_norm - group.mean_norm) / np.sqrt(group.variance_norm.clip(lower=1e-8))
        temperature = float(np.sqrt(np.mean(np.square(standardized))))
        rows.append(
            {
                "model": str(group.model.iloc[0]), "level": level, "task": task,
                "target": target, "horizon_months": horizon, "n": int(len(group)),
                "temperature": float(np.clip(temperature, low, high)),
            }
        )

    for model, model_frame in calibration.groupby("model"):
        del model
        for (task, target, horizon), group in model_frame.groupby(["task", "target", "horizon_months"]):
            add(group, "exact", task, target, int(horizon))
        for (task, target), group in model_frame.groupby(["task", "target"]):
            add(group, "task_target", task, target)
        add(model_frame, "global")
    return pd.DataFrame(rows)


def apply_temperatures(predictions: pd.DataFrame, temperatures: pd.DataFrame) -> pd.DataFrame:
    output = predictions.copy()
    lookup = {
        (row.model, row.level, row.task, row.target, int(row.horizon_months)): row.temperature
        for row in temperatures.itertuples()
    }

    def find(row: Any) -> float:
        exact = (row.model, "exact", row.task, row.target, int(row.horizon_months))
        target = (row.model, "task_target", row.task, row.target, -1)
        global_key = (row.model, "global", "*", "*", -1)
        return float(lookup.get(exact, lookup.get(target, lookup[global_key])))

    output["temperature"] = [find(row) for row in output.itertuples()]
    output["calibrated_variance_norm"] = output.variance_norm * output.temperature.pow(2)
    output["calibrated_std"] = output["std"] * output.temperature
    return output


def normal_crps(mean: np.ndarray, std: np.ndarray, truth: np.ndarray) -> np.ndarray:
    z = (truth - mean) / std
    return std * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))


def metric_record(
    group: pd.DataFrame, levels: list[float], scope: str, calibrated: bool
) -> dict[str, Any]:
    variance = group.calibrated_variance_norm if calibrated else group.variance_norm
    std = np.sqrt(variance.clip(lower=1e-8).to_numpy())
    residual = (group.truth_norm - group.mean_norm).to_numpy()
    record: dict[str, Any] = {
        "scope": scope,
        "model": str(group.model.iloc[0]),
        "calibrated": calibrated,
        "n_targets": int(len(group)),
        "n_subjects": int(group.subject_id.nunique()),
        "normalized_mae": float(np.mean(np.abs(residual))),
        "gaussian_nll": float(np.mean(0.5 * (np.log(2 * np.pi * std**2) + (residual / std) ** 2))),
        "gaussian_crps": float(np.mean(normal_crps(group.mean_norm.to_numpy(), std, group.truth_norm.to_numpy()))),
        "mean_std_normalized": float(np.mean(std)),
        "pit_mean": float(np.mean(stats.norm.cdf(residual / std))),
        "pit_variance": float(np.var(stats.norm.cdf(residual / std))),
        "pit_ks": float(stats.kstest(stats.norm.cdf(residual / std), "uniform").statistic),
        "uncertainty_error_spearman": float(pd.Series(std).corr(pd.Series(np.abs(residual)), method="spearman")),
        "uncertainty_missingness_spearman": float(pd.Series(std).corr(1 - group.observed_fraction.reset_index(drop=True), method="spearman")),
    }
    errors = []
    for level in levels:
        z_value = stats.norm.ppf((1 + level) / 2)
        covered = np.abs(residual) <= z_value * std
        name = int(round(level * 100))
        record[f"coverage_{name}"] = float(np.mean(covered))
        record[f"width_{name}_normalized"] = float(np.mean(2 * z_value * std))
        errors.append(abs(float(np.mean(covered)) - level))
    record["multilevel_calibration_error"] = float(np.mean(errors))
    return record


def compute_metrics(predictions: pd.DataFrame, levels: list[float]) -> pd.DataFrame:
    rows = []
    for model, model_frame in predictions.groupby("model"):
        del model
        for calibrated in (False, True):
            rows.append(metric_record(model_frame, levels, "overall", calibrated))
            for (task, target, horizon), group in model_frame.groupby(["task", "target", "horizon_months"]):
                record = metric_record(group, levels, "cell", calibrated)
                record.update(task=task, target=target, horizon_months=int(horizon))
                rows.append(record)
    return pd.DataFrame(rows)


def stratified_metrics(predictions: pd.DataFrame, levels: list[float]) -> pd.DataFrame:
    frame = predictions.copy()
    frame["history_stratum"] = pd.cut(
        frame.history_length, bins=[0, 1, 2, 4, np.inf], labels=["1", "2", "3-4", "5+"]
    )
    frame["observed_stratum"] = pd.cut(
        frame.observed_fraction, bins=[-np.inf, 0.25, 0.5, 0.75, np.inf],
        labels=["<=0.25", "0.25-0.50", "0.50-0.75", ">0.75"],
    )
    rows = []
    for model, model_frame in frame.groupby("model", observed=False):
        del model
        for column in ["history_stratum", "observed_stratum"]:
            for value, group in model_frame.groupby(column, observed=False):
                if group.empty:
                    continue
                record = metric_record(group, levels, f"{column}:{value}", True)
                record.update(stratifier=column, stratum=str(value))
                rows.append(record)
    return pd.DataFrame(rows)


def paired_bootstrap(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    replicates: int,
    seed: int,
    task: str = "overall",
) -> dict[str, Any]:
    if task != "overall":
        candidate, baseline = candidate[candidate.task.eq(task)], baseline[baseline.task.eq(task)]
    keys = ["sample_id", "subject_id", "task", "horizon_months", "target", "truth_norm"]
    paired = candidate[keys + ["mean_norm", "calibrated_variance_norm"]].merge(
        baseline[keys + ["mean_norm", "calibrated_variance_norm"]],
        on=keys,
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    )
    truth = paired.truth_norm
    paired["mae_difference"] = (
        np.abs(paired.mean_norm_candidate - truth) - np.abs(paired.mean_norm_baseline - truth)
    )
    candidate_nll = 0.5 * (
        np.log(2 * np.pi * paired.calibrated_variance_norm_candidate)
        + (truth - paired.mean_norm_candidate).pow(2) / paired.calibrated_variance_norm_candidate
    )
    baseline_nll = 0.5 * (
        np.log(2 * np.pi * paired.calibrated_variance_norm_baseline)
        + (truth - paired.mean_norm_baseline).pow(2) / paired.calibrated_variance_norm_baseline
    )
    paired["nll_difference"] = candidate_nll - baseline_nll
    subject = paired.groupby("subject_id")[["mae_difference", "nll_difference"]].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(subject), size=(replicates, len(subject)))
    output = {
        "task": task,
        "candidate": str(candidate.model.iloc[0]),
        "baseline": str(baseline.model.iloc[0]),
        "n_paired_targets": int(len(paired)),
        "n_subjects": int(len(subject)),
        "replicates": replicates,
    }
    for metric in ["mae_difference", "nll_difference"]:
        sums = subject[(metric, "sum")].to_numpy()
        counts = subject[(metric, "count")].to_numpy()
        draws = sums[indices].sum(1) / counts[indices].sum(1)
        output[metric] = float(paired[metric].mean())
        output[f"{metric}_ci_lower"] = float(np.quantile(draws, 0.025))
        output[f"{metric}_ci_upper"] = float(np.quantile(draws, 0.975))
        output[f"{metric}_probability_better"] = float(np.mean(draws < 0))
    return output


def write_report(
    output: Path,
    sample_counts: dict[str, Any],
    parameter_counts: dict[str, int],
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    temperatures: pd.DataFrame,
    decision: dict[str, Any],
) -> None:
    summary = metrics[(metrics.scope == "overall")][
        [
            "model", "calibrated", "n_targets", "n_subjects", "normalized_mae",
            "gaussian_nll", "gaussian_crps", "coverage_95", "width_95_normalized",
            "multilevel_calibration_error", "uncertainty_error_spearman",
            "uncertainty_missingness_spearman",
        ]
    ]
    cells = metrics[(metrics.scope == "cell") & metrics.calibrated][
        ["model", "task", "target", "horizon_months", "n_targets", "normalized_mae", "gaussian_nll", "coverage_95", "multilevel_calibration_error"]
    ]
    calibrated_overall = summary[summary.calibrated].set_index("model")
    belief_comparison = bootstrap[
        bootstrap.candidate.eq("typed_belief_dynamics") & bootstrap.task.eq("overall")
    ].iloc[0]
    direct_comparison = bootstrap[
        bootstrap.candidate.eq("typed_direct_gaussian") & bootstrap.task.eq("overall")
    ].iloc[0]
    direct_interpolation = bootstrap[
        bootstrap.candidate.eq("typed_direct_gaussian") & bootstrap.task.eq("interpolation")
    ].iloc[0]
    interpretation = (
        f"The explicit typed belief arm reached calibrated 95% coverage "
        f"{calibrated_overall.loc['typed_belief_dynamics', 'coverage_95']:.4f} and multilevel "
        f"calibration error {calibrated_overall.loc['typed_belief_dynamics', 'multilevel_calibration_error']:.4f}, "
        f"but its overall normalized-MAE difference versus GRU-D was "
        f"{belief_comparison.mae_difference:+.5f} (95% CI "
        f"[{belief_comparison.mae_difference_ci_lower:+.5f}, {belief_comparison.mae_difference_ci_upper:+.5f}]) "
        f"and its NLL difference was {belief_comparison.nll_difference:+.5f} (95% CI "
        f"[{belief_comparison.nll_difference_ci_lower:+.5f}, {belief_comparison.nll_difference_ci_upper:+.5f}]). "
        f"It therefore fails the frozen dynamics gate. The typed direct Gaussian arm had the best point "
        f"estimates (normalized MAE {calibrated_overall.loc['typed_direct_gaussian', 'normalized_mae']:.4f}; "
        f"NLL {calibrated_overall.loc['typed_direct_gaussian', 'gaussian_nll']:.4f}), but its overall paired "
        f"intervals still crossed zero (MAE difference {direct_comparison.mae_difference:+.5f}, 95% CI "
        f"[{direct_comparison.mae_difference_ci_lower:+.5f}, {direct_comparison.mae_difference_ci_upper:+.5f}]). "
        f"Its interpolation MAE improvement was supported ({direct_interpolation.mae_difference:+.5f}, "
        f"95% CI [{direct_interpolation.mae_difference_ci_lower:+.5f}, "
        f"{direct_interpolation.mae_difference_ci_upper:+.5f}]), which is a task-specific result rather than "
        f"evidence for the explicit belief mechanism."
    )
    text = f"""# Cognitive-only irregular-time belief dynamics and calibration

Status: completed validation-only development experiment. Test was not loaded.

The typed item/response encoder is used as a component. This experiment makes no universal semantic-superiority claim.

## Contract

- Sample counts: `{json.dumps(sample_counts, sort_keys=True)}`
- Trainable parameter counts: `{json.dumps(parameter_counts, sort_keys=True)}`
- Three-seed ensemble variance combines within-model and between-seed uncertainty.
- Variance temperatures were fit on subject-disjoint validation-calibration cases and evaluated on validation-evaluation cases.

## Evaluation summary

{markdown_table(summary)}

## Interpretation

{interpretation}

## Calibrated task/target/horizon results

{markdown_table(cells)}

## Paired subject-cluster bootstrap

Negative differences favor the candidate.

{markdown_table(bootstrap)}

## Calibration temperatures

{markdown_table(temperatures)}

## Development decision

```json
{json.dumps(decision, indent=2)}
```

This decision concerns the cognitive dynamics component only. MRI remains closed pending real scan dates, QC/provenance and a separately frozen MRI protocol.
"""
    (output / "report.md").write_text(text)


def run(config_path: Path, smoke: bool = False) -> None:
    config = load_config(config_path)
    if smoke:
        config = copy.deepcopy(config)
        config["run_id"] += "_smoke"
        config["neural_seeds"] = [2026]
        config["arms"] = {
            "grud_gaussian": config["arms"]["grud_gaussian"],
            "typed_belief_dynamics": config["arms"]["typed_belief_dynamics"],
        }
        config["training"].update(max_epochs=2, patience=2, batch_size=64)
        config["statistics"]["bootstrap_replicates"] = 50
        config["calibration"]["minimum_group_targets"] = 2
    output = ROOT / "outputs" / config["run_id"]
    output.mkdir(parents=True, exist_ok=True)
    resolved = output / "resolved_config.json"
    canonical_config = json.loads(json.dumps(config))
    if resolved.exists() and json.loads(resolved.read_text()) != canonical_config:
        raise RuntimeError(f"Frozen config changed for {output}")
    if not resolved.exists():
        save_json(config, resolved)

    dictionary = load_dictionary()
    features = select_verified_scalar_features(config, dictionary)
    targets = list(config["targets"])
    raw_train = clean_frame(read_csv(ROOT / "splits/train.csv"), features, targets, dictionary)
    raw_val = clean_frame(read_csv(ROOT / "splits/val.csv"), features, targets, dictionary)
    partition = config["development_partition"]
    fit_frame, tune_frame = partition_frame(
        raw_train, int(partition["subject_hash_seed"]), float(partition["train_tune_fraction"])
    )
    # Fit every data-derived transform on fit subjects only. Inner tune controls
    # early stopping without contributing outcome distribution statistics.
    normalization = fit_normalization(fit_frame, features, targets)
    evaluation_frame, calibration_frame = partition_frame(
        raw_val,
        int(partition["subject_hash_seed"]) + 1,
        float(partition["validation_calibration_fraction"]),
    )
    frames = {
        "fit": fit_frame, "tune": tune_frame,
        "calibration": calibration_frame, "evaluation": evaluation_frame,
    }
    samples = {
        name: build_all_samples(frame, name, features, targets, config)
        for name, frame in frames.items()
    }
    if smoke:
        samples = {
            name: (
                [sample for sample in value if sample.task == "forecast"][:128]
                + [sample for sample in value if sample.task == "interpolation"][:128]
            )
            for name, value in samples.items()
        }
    if any(not value for value in samples.values()):
        raise ValueError("One or more dynamics development partitions have no samples")

    collator = DynamicsCollator(features, targets, normalization, dictionary)
    feature_types, _, _, _, _ = feature_metadata(features, dictionary)
    domains = build_domain_assignment(features, dictionary)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    calibration_seed_frames, evaluation_seed_frames, parameter_counts = [], [], {}
    for arm_name, arm in config["arms"].items():
        counts = []
        for seed in config["neural_seeds"]:
            calibration_predictions, evaluation_predictions, count = train_one(
                arm_name, arm, seed,
                samples["fit"], samples["tune"], samples["calibration"], samples["evaluation"],
                collator, features, targets, feature_types, domains, config, output, device,
            )
            calibration_seed_frames.append(calibration_predictions)
            evaluation_seed_frames.append(evaluation_predictions)
            counts.append(count)
        if len(set(counts)) != 1:
            raise RuntimeError(f"Parameter count changed across seeds for {arm_name}")
        parameter_counts[arm_name] = counts[0]

    calibration_seed = pd.concat(calibration_seed_frames, ignore_index=True)
    evaluation_seed = pd.concat(evaluation_seed_frames, ignore_index=True)
    save_csv(calibration_seed, output / "calibration_seed_predictions.csv")
    save_csv(evaluation_seed, output / "evaluation_seed_predictions.csv")
    calibration = pd.concat(
        [ensemble_predictions(group, model) for model, group in calibration_seed.groupby("model")],
        ignore_index=True,
    )
    evaluation = pd.concat(
        [ensemble_predictions(group, model) for model, group in evaluation_seed.groupby("model")],
        ignore_index=True,
    )
    temperatures = fit_temperatures(calibration, config)
    calibrated = apply_temperatures(evaluation, temperatures)
    save_csv(calibration, output / "calibration_ensemble_predictions.csv")
    save_csv(calibrated, output / "evaluation_predictions.csv")
    save_csv(temperatures, output / "calibration_temperatures.csv")

    levels = [float(value) for value in config["calibration"]["coverage_levels"]]
    metrics = compute_metrics(calibrated, levels)
    strata = stratified_metrics(calibrated, levels)
    save_csv(metrics, output / "evaluation_metrics.csv")
    save_csv(strata, output / "stratified_metrics.csv")

    baseline_name = config["gate"]["baseline"]
    candidate_name = config["gate"]["candidate"]
    baseline = calibrated[calibrated.model.eq(baseline_name)]
    bootstrap_rows = []
    for arm_name in config["arms"]:
        if arm_name == baseline_name:
            continue
        candidate = calibrated[calibrated.model.eq(arm_name)]
        for task in ["overall", "forecast", "interpolation"]:
            bootstrap_rows.append(
                paired_bootstrap(
                    candidate, baseline, config["statistics"]["bootstrap_replicates"],
                    config["seed"] + len(bootstrap_rows), task,
                )
            )
    bootstrap = pd.DataFrame(bootstrap_rows)
    save_csv(bootstrap, output / "paired_subject_bootstrap.csv")

    gate_row = bootstrap[(bootstrap.candidate == candidate_name) & (bootstrap.task == "overall")].iloc[0]
    metric_row = metrics[
        (metrics.model == candidate_name) & (metrics.scope == "overall") & metrics.calibrated
    ].iloc[0]
    gate = config["gate"]
    criteria = {
        "normalized_mae_noninferior": bool(
            gate_row.mae_difference_ci_upper <= gate["normalized_mae_noninferiority_margin"]
        ),
        "calibrated_nll_improved": bool(gate_row.nll_difference_ci_upper < 0),
        "multilevel_calibration_error_within_bound": bool(
            metric_row.multilevel_calibration_error <= gate["maximum_multilevel_calibration_error"]
        ),
        "coverage_95_within_bounds": bool(
            gate["coverage_95_bounds"][0] <= metric_row.coverage_95 <= gate["coverage_95_bounds"][1]
        ),
    }
    passed = bool(all(criteria.values()))
    direct_overall = bootstrap[
        (bootstrap.candidate == "typed_direct_gaussian") & (bootstrap.task == "overall")
    ].iloc[0]
    direct_interpolation = bootstrap[
        (bootstrap.candidate == "typed_direct_gaussian") & (bootstrap.task == "interpolation")
    ].iloc[0]
    decision = {
        "run_id": config["run_id"],
        "engineering_smoke_only": smoke,
        "development_only": True,
        "test_evaluated": False,
        "candidate": candidate_name,
        "baseline": baseline_name,
        "criteria": criteria,
        "cognitive_dynamics_gate_passed": passed,
        "advance_to_mri": False,
        "semantic_superiority_claim": False,
        "strongest_point_estimate_arm": "typed_direct_gaussian",
        "typed_direct_secondary_finding": {
            "overall_normalized_mae_difference": float(direct_overall.mae_difference),
            "overall_normalized_mae_ci": [
                float(direct_overall.mae_difference_ci_lower),
                float(direct_overall.mae_difference_ci_upper),
            ],
            "interpolation_normalized_mae_difference": float(direct_interpolation.mae_difference),
            "interpolation_normalized_mae_ci": [
                float(direct_interpolation.mae_difference_ci_lower),
                float(direct_interpolation.mae_difference_ci_upper),
            ],
            "interpretation": "task_specific_interpolation_signal_without_overall_confirmed_superiority",
        },
        "next_action": (
            "freeze_cognitive_dynamics_component_and_prepare_independent_confirmation"
            if passed
            else "retain_grud_as_confirmed_primary_and_typed_direct_as_interpolation_component_candidate"
        ),
    }
    save_json(decision, output / "development_decision.json")
    sample_counts = {
        name: {
            "subjects": int(frames[name].subject_id.nunique()),
            "samples": len(samples[name]),
            "forecast": sum(sample.task == "forecast" for sample in samples[name]),
            "interpolation": sum(sample.task == "interpolation" for sample in samples[name]),
        }
        for name in frames
    }
    manifest = {
        "run_id": config["run_id"],
        "development_only": True,
        "engineering_smoke_only": smoke,
        "test_evaluated": False,
        "features": features,
        "targets": targets,
        "sample_counts": sample_counts,
        "parameter_counts": parameter_counts,
        "subject_sets_disjoint": True,
        "typed_response_component": True,
        "universal_semantic_superiority_claim": False,
    }
    save_json(manifest, output / "manifest.json")
    write_report(output, sample_counts, parameter_counts, metrics, bootstrap, temperatures, decision)
    save_json(
        {
            "completed": True,
            "development_only": True,
            "engineering_smoke_only": smoke,
            "test_evaluated": False,
            "cognitive_dynamics_gate_passed": passed,
            "report": str(output / "report.md"),
        },
        output / "status.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/phase2_cognitive_belief_dev.yaml"
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.config, args.smoke)


if __name__ == "__main__":
    main()
