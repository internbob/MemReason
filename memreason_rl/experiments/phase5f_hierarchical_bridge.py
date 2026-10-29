"""Train-only hierarchical shared-state measurement with instrument bridges.

Phase 5F is deliberately a measurement experiment.  It does not fit temporal
dynamics and cannot open them unless every frozen measurement gate passes.
"""

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
from scipy.stats import spearmanr
from torch import nn
from torch.nn import functional as F

from ..data.common import DOMAINS, ROOT
from .phase5c_memory_measurement import (
    clean_matrix,
    cross_views,
    deterministic_targets,
    fit_scaling,
    random_targets,
    stable_int,
    subject_fold,
    to_severity,
)
from .phase5d_anchored_mirt import ccc_bootstrap, cluster_bootstrap, concordance_correlation


CONFIG = ROOT / "configs/phase5f_hierarchical_bridge_dev.yaml"
DATE_COLUMNS = {
    "ADAS": "adas_visit_date",
    "CDR": "cdr_visit_date",
    "MMSE": "mmse_visit_date",
    "MOCA": "moca_visit_date",
    "LOGMEM": "LogMemIA_ExamDate",
}


def construct_membership(items: list[dict[str, Any]], groups: dict[str, list[str]]) -> tuple[list[str], np.ndarray]:
    """Return frozen construct names and one construct index per item."""
    names = list(groups)
    field_to_group: dict[str, int] = {}
    for index, name in enumerate(names):
        for field in groups[name]:
            if field in field_to_group:
                raise ValueError(f"Item appears in multiple Memory constructs: {field}")
            field_to_group[field] = index
    known = {item["field"] for item in items}
    unknown = sorted(set(field_to_group) - known)
    if unknown:
        raise ValueError(f"Unknown fields in Memory construct map: {unknown}")
    membership = np.asarray([field_to_group.get(item["field"], -1) for item in items], np.int64)
    for name, index in zip(names, range(len(names))):
        if not np.any(membership == index):
            raise ValueError(f"Empty Memory construct: {name}")
    return names, membership


def build_connection_graph(
    frame: pd.DataFrame,
    mask: np.ndarray,
    items: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Build an availability/date-only instrument graph without response values."""
    item_instruments = np.asarray([item["instrument"] for item in items])
    instruments = sorted(set(item_instruments))
    present = {name: mask[:, item_instruments == name].any(1) for name in instruments}
    domains = {name: {item["domain"] for item in items if item["instrument"] == name}
               for name in instruments}
    raw_rows: list[dict[str, Any]] = []
    window = int(config["graph"]["date_window_days"])
    nominal = float(config["graph"]["nominal_visit_weight"])
    for left_index, left in enumerate(instruments):
        for right in instruments[left_index + 1:]:
            both = present[left] & present[right]
            concurrent_rows = int(both.sum())
            concurrent_subjects = int(frame.loc[both, "subject_id"].nunique())
            left_column = DATE_COLUMNS.get(left)
            right_column = DATE_COLUMNS.get(right)
            left_source = frame[left_column] if left_column in frame else pd.Series(pd.NaT, index=frame.index)
            right_source = frame[right_column] if right_column in frame else pd.Series(pd.NaT, index=frame.index)
            left_date = pd.to_datetime(left_source, errors="coerce")
            right_date = pd.to_datetime(right_source, errors="coerce")
            dated = both & left_date.notna().to_numpy() & right_date.notna().to_numpy()
            difference = np.abs((left_date[dated] - right_date[dated]).dt.days.to_numpy())
            within = int((difference <= window).sum())
            date_rows = int(dated.sum())
            timing = within / date_rows if date_rows else (nominal if concurrent_rows else 0.0)
            union = domains[left] | domains[right]
            overlap = len(domains[left] & domains[right]) / max(len(union), 1)
            raw_rows.append({
                "instrument_a": left,
                "instrument_b": right,
                "concurrent_rows": concurrent_rows,
                "concurrent_subjects": concurrent_subjects,
                "both_dates_available": date_rows,
                f"within_{window}_days": within,
                "domain_overlap": overlap,
                "timing_reliability": timing,
            })
    maximum_support = max((row["concurrent_subjects"] for row in raw_rows), default=1)
    matrix = np.zeros((len(instruments), len(instruments)), np.float32)
    index = {name: position for position, name in enumerate(instruments)}
    for row in raw_rows:
        support = math.sqrt(row["concurrent_subjects"] / max(maximum_support, 1))
        weight = float(row["domain_overlap"] * row["timing_reliability"] * support)
        if weight < float(config["graph"]["minimum_edge_weight"]):
            weight = 0.0
        row["support_weight"] = support
        row["edge_weight"] = weight
        a, b = index[row["instrument_a"]], index[row["instrument_b"]]
        matrix[a, b] = matrix[b, a] = weight
    np.fill_diagonal(matrix, 1.0)
    return pd.DataFrame(raw_rows), matrix, instruments


def split_instrument_views(
    observed: np.ndarray,
    item_instrument_index: np.ndarray,
    edge_matrix: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split whole instruments into disjoint graph-supported views."""
    left = np.zeros_like(observed, bool)
    right = np.zeros_like(observed, bool)
    weights = np.zeros(len(observed), np.float32)
    for row in range(len(observed)):
        item_ids = np.flatnonzero(observed[row])
        choices = np.unique(item_instrument_index[item_ids])
        if len(choices) < 2:
            continue
        choices = choices.copy(); rng.shuffle(choices)
        boundary = max(1, len(choices) // 2)
        group_a, group_b = choices[:boundary], choices[boundary:]
        if not len(group_b):
            continue
        left[row] = observed[row] & np.isin(item_instrument_index, group_a)
        right[row] = observed[row] & np.isin(item_instrument_index, group_b)
        weights[row] = float(edge_matrix[np.ix_(group_a, group_b)].max(initial=0.0))
    return left, right, weights


def shared_bridge_weights(
    view_a: np.ndarray,
    view_b: np.ndarray,
    item_instrument_index: np.ndarray,
    domain_index: np.ndarray,
    construct_index: np.ndarray,
    edge_matrix: np.ndarray,
) -> np.ndarray:
    """Return one supported edge weight per row and shared-state dimension.

    A strong edge in an unrelated domain must not authorize alignment of the
    Memory state (or a Memory subconstruct).  Dimension zero is global, the
    next five dimensions are DOMAINS, and the remainder are Memory constructs.
    """
    n_constructs = int(construct_index.max() + 1) if np.any(construct_index >= 0) else 0
    selectors = [np.ones(len(item_instrument_index), bool)]
    selectors.extend(domain_index == domain for domain in range(len(DOMAINS)))
    selectors.extend(construct_index == construct for construct in range(n_constructs))
    weights = np.zeros((len(view_a), len(selectors)), np.float32)
    for row in range(len(view_a)):
        for dimension, selector in enumerate(selectors):
            a = np.unique(item_instrument_index[np.flatnonzero(view_a[row] & selector)])
            b = np.unique(item_instrument_index[np.flatnonzero(view_b[row] & selector)])
            if len(a) and len(b):
                weights[row, dimension] = float(edge_matrix[np.ix_(a, b)].max(initial=0.0))
    return weights


def deterministic_view_weights(
    view_a: np.ndarray,
    view_b: np.ndarray,
    item_instrument_index: np.ndarray,
    edge_matrix: np.ndarray,
) -> np.ndarray:
    weights = np.zeros(len(view_a), np.float32)
    for row in range(len(view_a)):
        a = np.unique(item_instrument_index[np.flatnonzero(view_a[row])])
        b = np.unique(item_instrument_index[np.flatnonzero(view_b[row])])
        if len(a) and len(b):
            weights[row] = float(edge_matrix[np.ix_(a, b)].max(initial=0.0))
    return weights


class ComponentEncoder(nn.Module):
    def __init__(self, item_count: int, hidden: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(item_count * 2, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 2),
        )

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.network(torch.cat([values * mask, mask], dim=1))
        return output[:, 0], output[:, 1]


class HierarchicalBridgeAE(nn.Module):
    """Identified shared hierarchy with optional instrument-only residuals."""

    def __init__(
        self,
        domain_index: np.ndarray,
        construct_index: np.ndarray,
        instrument_index: np.ndarray,
        hidden: int,
        use_residual: bool,
        logvar_bounds: tuple[float, float],
    ):
        super().__init__()
        self.use_residual = bool(use_residual)
        self.logvar_bounds = logvar_bounds
        self.n_items = len(domain_index)
        self.n_constructs = int(construct_index.max() + 1) if np.any(construct_index >= 0) else 0
        self.n_instruments = int(instrument_index.max() + 1)
        self.register_buffer("domain_index", torch.as_tensor(domain_index, dtype=torch.long))
        self.register_buffer("construct_index", torch.as_tensor(construct_index, dtype=torch.long))
        self.register_buffer("instrument_index", torch.as_tensor(instrument_index, dtype=torch.long))
        self.global_encoder = ComponentEncoder(self.n_items, hidden)
        self.domain_encoders = nn.ModuleList([
            ComponentEncoder(int((domain_index == domain).sum()), hidden)
            for domain in range(len(DOMAINS))
        ])
        self.construct_encoders = nn.ModuleList([
            ComponentEncoder(int((construct_index == construct).sum()), hidden)
            for construct in range(self.n_constructs)
        ])
        self.residual_encoders = nn.ModuleList([
            ComponentEncoder(int((instrument_index == instrument).sum()), hidden)
            for instrument in range(self.n_instruments)
        ]) if self.use_residual else nn.ModuleList()
        self.intercept = nn.Parameter(torch.zeros(self.n_items))
        self.log_global_loading = nn.Parameter(torch.full((self.n_items,), -0.25))
        self.log_domain_loading = nn.Parameter(torch.full((self.n_items,), -0.25))
        self.log_construct_loading = nn.Parameter(torch.full((self.n_items,), -0.75))
        self.log_residual_loading = nn.Parameter(torch.full((self.n_items,), -1.50))

    def _component(
        self,
        encoder: ComponentEncoder,
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        available = mask.bool().any(1)
        mean, logvar = encoder(values, mask)
        low, high = self.logvar_bounds
        mean = torch.where(available, mean, torch.zeros_like(mean))
        logvar = torch.where(available, logvar.clamp(low, high), torch.zeros_like(logvar))
        return mean, logvar, available

    def state(self, values: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        shared_mean, shared_logvar, shared_available = [], [], []
        mean, logvar, available = self._component(self.global_encoder, values, mask)
        shared_mean.append(mean); shared_logvar.append(logvar); shared_available.append(available)
        for domain, encoder in enumerate(self.domain_encoders):
            choose = self.domain_index == domain
            mean, logvar, available = self._component(encoder, values[:, choose], mask[:, choose])
            shared_mean.append(mean); shared_logvar.append(logvar); shared_available.append(available)
        for construct, encoder in enumerate(self.construct_encoders):
            choose = self.construct_index == construct
            mean, logvar, available = self._component(encoder, values[:, choose], mask[:, choose])
            shared_mean.append(mean); shared_logvar.append(logvar); shared_available.append(available)
        shared = {
            "shared_mean": torch.stack(shared_mean, 1),
            "shared_logvar": torch.stack(shared_logvar, 1),
            "shared_available": torch.stack(shared_available, 1),
        }
        residual_mean, residual_logvar, residual_available = [], [], []
        if self.use_residual:
            for instrument, encoder in enumerate(self.residual_encoders):
                choose = self.instrument_index == instrument
                mean, logvar, available = self._component(encoder, values[:, choose], mask[:, choose])
                residual_mean.append(mean); residual_logvar.append(logvar); residual_available.append(available)
            shared.update({
                "residual_mean": torch.stack(residual_mean, 1),
                "residual_logvar": torch.stack(residual_logvar, 1),
                "residual_available": torch.stack(residual_available, 1),
            })
        else:
            shape = (len(values), self.n_instruments)
            shared.update({
                "residual_mean": values.new_zeros(shape),
                "residual_logvar": values.new_zeros(shape),
                "residual_available": torch.zeros(shape, dtype=torch.bool, device=values.device),
            })
        return shared

    @staticmethod
    def _sample(mean: torch.Tensor, logvar: torch.Tensor, sample: bool) -> torch.Tensor:
        return mean + torch.randn_like(mean) * torch.exp(0.5 * logvar) if sample else mean

    def loadings(self) -> dict[str, torch.Tensor]:
        construct_mask = (self.construct_index >= 0).float()
        return {
            "global": F.softplus(self.log_global_loading) + 0.02,
            "domain": F.softplus(self.log_domain_loading) + 0.02,
            "construct": (F.softplus(self.log_construct_loading) + 0.02) * construct_mask,
            "residual": ((F.softplus(self.log_residual_loading) + 0.02)
                         if self.use_residual else torch.zeros_like(self.log_residual_loading)),
        }

    def decode(self, state: dict[str, torch.Tensor], sample: bool) -> torch.Tensor:
        shared = self._sample(state["shared_mean"], state["shared_logvar"], sample)
        residual = self._sample(state["residual_mean"], state["residual_logvar"], sample)
        loading = self.loadings()
        output = self.intercept[None] + loading["global"][None] * shared[:, :1]
        output = output + loading["domain"][None] * shared[:, 1 + self.domain_index]
        valid_construct = self.construct_index >= 0
        if valid_construct.any():
            item_ids = torch.where(valid_construct)[0]
            state_ids = 1 + len(DOMAINS) + self.construct_index[item_ids]
            output[:, item_ids] = (output[:, item_ids]
                                   + loading["construct"][item_ids][None] * shared[:, state_ids])
        if self.use_residual:
            output = output + loading["residual"][None] * residual[:, self.instrument_index]
        return output

    def forward(self, values: torch.Tensor, mask: torch.Tensor, sample: bool = True):
        state = self.state(values, mask)
        return self.decode(state, sample), state

    def loading_summary(self) -> dict[str, float]:
        with torch.no_grad():
            loading = {name: float(value.sum().cpu()) for name, value in self.loadings().items()}
        total = sum(loading.values())
        loading["residual_fraction"] = loading["residual"] / total if total else 0.0
        return loading


def available_kl(mean: torch.Tensor, logvar: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
    value = 0.5 * (mean.square() + logvar.exp() - logvar - 1)
    return value[available].mean() if available.any() else value.sum() * 0


def graph_alignment_loss(
    model: HierarchicalBridgeAE,
    values: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    weights: torch.Tensor,
    residual_gate_floor: float | None = None,
    residual_gate_temperature: float = 1.0,
) -> torch.Tensor:
    a, b = model.state(values, left), model.state(values, right)
    if weights.ndim != 2 or weights.shape[1] != a["shared_mean"].shape[1]:
        raise ValueError("Bridge weights must match the shared-state dimensions")
    available = a["shared_available"] & b["shared_available"] & (weights > 0)
    if not available.any():
        return values.sum() * 0
    mean_difference = (a["shared_mean"] - b["shared_mean"]).square()
    sd_difference = (torch.exp(0.5 * a["shared_logvar"])
                     - torch.exp(0.5 * b["shared_logvar"])).square()
    distance = mean_difference + 0.1 * sd_difference
    effective_weights = weights
    if residual_gate_floor is not None:
        gate = residual_evidence_gate(
            a,
            b,
            floor=float(residual_gate_floor),
            temperature=float(residual_gate_temperature),
        )
        effective_weights = weights * gate[:, None]
    weighted = distance * effective_weights
    return weighted[available].sum() / effective_weights[available].sum().clamp_min(1e-6)


def residual_evidence_gate(
    state_a: dict[str, torch.Tensor],
    state_b: dict[str, torch.Tensor],
    floor: float,
    temperature: float,
) -> torch.Tensor:
    """Downweight bridge pressure when instrument-specific evidence is strong.

    The gate is based on the mean absolute standardized residual posterior in
    each instrument view.  It is detached so residual encoders cannot evade the
    bridge objective by increasing their means.  A positive floor prevents the
    graph loss from being switched off entirely.
    """
    if not 0 < floor <= 1:
        raise ValueError("Residual gate floor must lie in (0, 1]")
    if temperature < 0:
        raise ValueError("Residual gate temperature must be non-negative")

    def evidence(state: dict[str, torch.Tensor]) -> torch.Tensor:
        available = state["residual_available"]
        standardized = state["residual_mean"].abs() / torch.exp(
            0.5 * state["residual_logvar"]
        ).clamp_min(1e-4)
        numerator = (standardized * available).sum(1)
        denominator = available.sum(1).clamp_min(1)
        return numerator / denominator

    residual_evidence = 0.5 * (evidence(state_a) + evidence(state_b))
    gate = floor + (1.0 - floor) * torch.exp(-temperature * residual_evidence)
    return gate.detach()


def train_model(
    arm: str,
    fit_z: np.ndarray,
    fit_mask: np.ndarray,
    dev_z: np.ndarray,
    dev_mask: np.ndarray,
    dev_ids: list[str],
    item_instrument_names: np.ndarray,
    item_instrument_index: np.ndarray,
    domain_index: np.ndarray,
    construct_index: np.ndarray,
    edge_matrix: np.ndarray,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
    *,
    graph_weight_override: float | None = None,
    residual_gate: dict[str, float] | None = None,
) -> HierarchicalBridgeAE:
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    settings = config["training"]
    model = HierarchicalBridgeAE(
        domain_index,
        construct_index,
        item_instrument_index,
        int(settings["hidden_dim"]),
        use_residual=arm != "hierarchical_shared",
        logvar_bounds=(float(settings["posterior_logvar_min"]), float(settings["posterior_logvar_max"])),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["learning_rate"]),
                                  weight_decay=float(settings["weight_decay"]))
    fraction = float(config["masking"]["item_fraction"])
    dev_input, dev_target = deterministic_targets(dev_mask, item_instrument_names, dev_ids,
                                                   "heldout_item", fraction)
    best_state, best_loss, stale = copy.deepcopy(model.state_dict()), float("inf"), 0
    batch_size = int(settings["batch_size"])
    for epoch in range(int(settings["max_epochs"])):
        rng = np.random.default_rng(seed * 1000 + epoch)
        input_mask, target_mask = random_targets(fit_mask, item_instrument_names, rng, fraction)
        view_a, view_b, _ = split_instrument_views(
            input_mask, item_instrument_index, edge_matrix, rng)
        bridge_weight = shared_bridge_weights(
            view_a, view_b, item_instrument_index, domain_index, construct_index, edge_matrix)
        order = rng.permutation(len(fit_z)); model.train()
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size]
            values = torch.as_tensor(np.nan_to_num(fit_z[index]), dtype=torch.float32, device=device)
            observed = torch.as_tensor(input_mask[index], dtype=torch.float32, device=device)
            target = torch.as_tensor(target_mask[index], dtype=torch.bool, device=device)
            truth = torch.as_tensor(np.nan_to_num(fit_z[index]), dtype=torch.float32, device=device)
            prediction, state = model(values, observed, sample=True)
            loss = F.smooth_l1_loss(prediction[target], truth[target]) if target.any() else prediction.sum() * 0
            loss = loss + float(settings["kl_weight"]) * available_kl(
                state["shared_mean"], state["shared_logvar"], state["shared_available"])
            if model.use_residual:
                loss = loss + float(settings["residual_kl_weight"]) * available_kl(
                    state["residual_mean"], state["residual_logvar"], state["residual_available"])
                loss = loss + float(settings["residual_loading_weight"]) * model.loadings()["residual"].square().mean()
            if arm == "hierarchical_bridge":
                left = torch.as_tensor(view_a[index], dtype=torch.float32, device=device)
                right = torch.as_tensor(view_b[index], dtype=torch.float32, device=device)
                weights = torch.as_tensor(bridge_weight[index], dtype=torch.float32, device=device)
                graph_weight = (float(settings["graph_weight"])
                                if graph_weight_override is None else float(graph_weight_override))
                gate_floor = None if residual_gate is None else float(residual_gate["floor"])
                gate_temperature = (1.0 if residual_gate is None
                                    else float(residual_gate["temperature"]))
                loss = loss + graph_weight * graph_alignment_loss(
                    model,
                    values,
                    left,
                    right,
                    weights,
                    residual_gate_floor=gate_floor,
                    residual_gate_temperature=gate_temperature,
                )
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings["gradient_clip"]))
            optimizer.step()
        model.eval()
        with torch.no_grad():
            values = torch.as_tensor(np.nan_to_num(dev_z), dtype=torch.float32, device=device)
            observed = torch.as_tensor(dev_input, dtype=torch.float32, device=device)
            target = torch.as_tensor(dev_target, dtype=torch.bool, device=device)
            truth = torch.as_tensor(np.nan_to_num(dev_z), dtype=torch.float32, device=device)
            prediction, _ = model(values, observed, sample=False)
            score = float(F.smooth_l1_loss(prediction[target], truth[target]).item()) if target.any() else float("inf")
        if score < best_loss - 1e-5:
            best_loss, best_state, stale = score, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= int(settings["patience"]):
                break
    model.load_state_dict(best_state); model.eval(); model.best_dev_loss = best_loss
    return model


def predict(model: HierarchicalBridgeAE, z: np.ndarray, mask: np.ndarray, device: torch.device) -> np.ndarray:
    rows = []
    with torch.no_grad():
        for start in range(0, len(z), 1024):
            values = torch.as_tensor(np.nan_to_num(z[start:start + 1024]), dtype=torch.float32, device=device)
            observed = torch.as_tensor(mask[start:start + 1024], dtype=torch.float32, device=device)
            output, _ = model(values, observed, sample=False); rows.append(output.cpu().numpy())
    return np.concatenate(rows)


def shared_state(
    model: HierarchicalBridgeAE,
    z: np.ndarray,
    mask: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means, logvars, available = [], [], []
    with torch.no_grad():
        for start in range(0, len(z), 1024):
            values = torch.as_tensor(np.nan_to_num(z[start:start + 1024]), dtype=torch.float32, device=device)
            observed = torch.as_tensor(mask[start:start + 1024], dtype=torch.float32, device=device)
            state = model.state(values, observed)
            means.append(state["shared_mean"].cpu().numpy())
            logvars.append(state["shared_logvar"].cpu().numpy())
            available.append(state["shared_available"].cpu().numpy())
    return np.concatenate(means), np.concatenate(logvars), np.concatenate(available)


def run(config_path: Path = CONFIG) -> Path:
    config = yaml.safe_load(config_path.read_text())
    contract_path = ROOT / config["item_contract"]
    items = yaml.safe_load(contract_path.read_text())["primary_state_items"]
    fields = [item["field"] for item in items]
    item_instrument_names = np.asarray([item["instrument"] for item in items])
    instrument_names = sorted(set(item_instrument_names))
    instrument_lookup = {name: index for index, name in enumerate(instrument_names)}
    item_instrument_index = np.asarray([instrument_lookup[name] for name in item_instrument_names])
    domain_index = np.asarray([DOMAINS.index(item["domain"]) for item in items])
    construct_names, construct_index = construct_membership(items, config["memory_constructs"])
    frame = pd.read_csv(ROOT / "splits/train.csv", low_memory=False)
    raw = clean_matrix(frame, items); mask = np.isfinite(raw)
    keep = mask.sum(1) >= 2
    frame = frame.loc[keep].reset_index(drop=True); raw = raw[keep]; mask = mask[keep]
    sample_ids = [f"{row.subject_id}:{row.visit_code}:{index}"
                  for index, row in enumerate(frame.itertuples())]
    folds, seed = int(config["folds"]), int(config["partition_seed"])
    assignment = np.asarray([subject_fold(str(subject), seed, folds) for subject in frame.subject_id])
    output = ROOT / "outputs" / config["run_id"]; output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    graph_records: list[pd.DataFrame] = []
    manifests: list[dict[str, Any]] = []
    checkpoint_root = output / "checkpoints"; checkpoint_root.mkdir(exist_ok=True)
    for fold in range(folds):
        evaluation = assignment == fold; outer_fit = ~evaluation
        dev_subjects = {
            subject for subject in frame.loc[outer_fit, "subject_id"].astype(str).unique()
            if stable_int(f"inner:{seed}:{fold}:{subject}") / 2**64 < float(config["inner_dev_fraction"])
        }
        development = outer_fit & frame.subject_id.astype(str).isin(dev_subjects).to_numpy()
        fit = outer_fit & ~development
        scaling = fit_scaling(raw, fit, items); z = to_severity(raw, scaling)
        graph, edge_matrix, graph_instruments = build_connection_graph(frame.loc[fit].reset_index(drop=True),
                                                                        mask[fit], items, config)
        if graph_instruments != instrument_names:
            raise RuntimeError("Instrument order changed while constructing the bridge graph")
        graph.insert(0, "fold", fold); graph_records.append(graph)
        dev_ids = [sample_ids[index] for index in np.flatnonzero(development)]
        models: dict[str, HierarchicalBridgeAE] = {}
        for arm in config["arms"]:
            model = train_model(
                arm, z[fit], mask[fit], z[development], mask[development], dev_ids,
                item_instrument_names, item_instrument_index, domain_index, construct_index,
                edge_matrix, config, seed + fold * 101 + len(arm), device)
            models[arm] = model
            destination = checkpoint_root / arm; destination.mkdir(exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "fold": fold, "fields": fields,
                        "construct_names": construct_names, "instrument_names": instrument_names,
                        "best_dev_loss": model.best_dev_loss}, destination / f"fold_{fold}.pt")
        evaluation_rows = np.flatnonzero(evaluation)
        local_ids = [sample_ids[index] for index in evaluation_rows]
        for task in ["heldout_item", "heldout_instrument"]:
            input_mask, target_mask = deterministic_targets(mask[evaluation], item_instrument_names,
                                                             local_ids, task,
                                                             float(config["masking"]["item_fraction"]))
            for arm, model in models.items():
                estimate = predict(model, z[evaluation], input_mask, device)
                for row, item in zip(*np.where(target_mask)):
                    global_row = evaluation_rows[row]
                    predictions.append({
                        "sample_id": sample_ids[global_row],
                        "subject_id": frame.subject_id.iloc[global_row],
                        "fold": fold,
                        "task": task,
                        "arm": arm,
                        "item": fields[item],
                        "instrument": item_instrument_names[item],
                        "domain": items[item]["domain"],
                        "truth_z": float(z[global_row, item]),
                        "prediction_z": float(estimate[row, item]),
                        "absolute_error": float(abs(estimate[row, item] - z[global_row, item])),
                    })
        view_a, view_b = cross_views(mask[evaluation], item_instrument_names, local_ids)
        memory_dimension = 1 + DOMAINS.index("Memory")
        view_weights = shared_bridge_weights(
            view_a, view_b, item_instrument_index, domain_index, construct_index, edge_matrix)[:, memory_dimension]
        for arm, model in models.items():
            mean_a, logvar_a, available_a = shared_state(model, z[evaluation], view_a, device)
            mean_b, logvar_b, available_b = shared_state(model, z[evaluation], view_b, device)
            valid = ((view_weights > 0) & available_a[:, memory_dimension]
                     & available_b[:, memory_dimension])
            for row in np.flatnonzero(valid):
                global_row = evaluation_rows[row]
                states.append({
                    "sample_id": sample_ids[global_row],
                    "subject_id": frame.subject_id.iloc[global_row],
                    "fold": fold,
                    "arm": arm,
                    "edge_weight": float(view_weights[row]),
                    "view_a": float(mean_a[row, memory_dimension]),
                    "view_b": float(mean_b[row, memory_dimension]),
                    "view_a_sd": float(np.exp(0.5 * logvar_a[row, memory_dimension])),
                    "view_b_sd": float(np.exp(0.5 * logvar_b[row, memory_dimension])),
                    "absolute_difference": float(abs(mean_a[row, memory_dimension]
                                                     - mean_b[row, memory_dimension])),
                })
        candidate_loading = models[config["gate"]["candidate"]].loading_summary()
        manifests.append({
            "fold": fold,
            "fit_subjects": int(frame.loc[fit, "subject_id"].nunique()),
            "dev_subjects": int(frame.loc[development, "subject_id"].nunique()),
            "evaluation_subjects": int(frame.loc[evaluation, "subject_id"].nunique()),
            "fit_rows": int(fit.sum()),
            "dev_rows": int(development.sum()),
            "evaluation_rows": int(evaluation.sum()),
            "supported_graph_edges": int((graph.edge_weight > 0).sum()),
            "candidate_residual_fraction": candidate_loading["residual_fraction"],
        })
    candidate_predictions = pd.DataFrame(predictions)
    phase5c_path = ROOT / config["phase5c_output"] / "crossfit_predictions.csv"
    phase5c = pd.read_csv(phase5c_path, low_memory=False)
    baselines = phase5c[phase5c.arm.isin(config["baseline_arms"])].copy()
    predictions_frame = pd.concat([baselines[candidate_predictions.columns], candidate_predictions],
                                  ignore_index=True)
    predictions_frame.to_csv(output / "crossfit_predictions.csv", index=False)
    states_frame = pd.DataFrame(states); states_frame.to_csv(output / "cross_view_memory_states.csv", index=False)
    graph_frame = pd.concat(graph_records, ignore_index=True); graph_frame.to_csv(output / "connection_graph.csv", index=False)
    manifest_frame = pd.DataFrame(manifests); manifest_frame.to_csv(output / "fold_manifest.csv", index=False)
    item_metrics = predictions_frame.groupby(["arm", "task", "item", "domain"], as_index=False).agg(
        decisions=("absolute_error", "size"), subjects=("subject_id", "nunique"),
        mae=("absolute_error", "mean"))
    item_metrics.to_csv(output / "item_metrics.csv", index=False)
    aggregate = item_metrics.groupby(["arm", "task"], as_index=False).agg(
        macro_item_mae=("mae", "mean"), items=("item", "nunique"), decisions=("decisions", "sum"))
    memory_metrics = item_metrics[item_metrics.domain.eq("Memory")].groupby(["arm", "task"], as_index=False).agg(
        memory_macro_item_mae=("mae", "mean"), memory_items=("item", "nunique"))
    aggregate = aggregate.merge(memory_metrics, on=["arm", "task"])
    aggregate.to_csv(output / "aggregate_metrics.csv", index=False)
    summaries = []
    for arm, group in states_frame.groupby("arm"):
        correlation = spearmanr(group.view_a, group.view_b)
        summaries.append({
            "arm": arm,
            "pairs": len(group),
            "subjects": group.subject_id.nunique(),
            "ccc": concordance_correlation(group.view_a.to_numpy(), group.view_b.to_numpy()),
            "spearman": float(correlation.statistic),
            "mean_absolute_difference": float(group.absolute_difference.mean()),
            "memory_state_std": float(pd.concat([group.view_a, group.view_b]).std()),
            "mean_posterior_sd": float(pd.concat([group.view_a_sd, group.view_b_sd]).mean()),
            "mean_edge_weight": float(group.edge_weight.mean()),
        })
    summary_frame = pd.DataFrame(summaries); summary_frame.to_csv(output / "cross_view_summary.csv", index=False)
    comparisons = cluster_bootstrap(predictions_frame, config); comparisons.to_csv(output / "subject_bootstrap.csv", index=False)
    state_ci = ccc_bootstrap(states_frame, config)
    (output / "cross_view_ccc_bootstrap.json").write_text(json.dumps(state_ci, indent=2) + "\n")
    candidate = config["gate"]["candidate"]
    fixed = config["gate"]["fixed_baseline"]
    predictive = config["gate"]["predictive_baseline"]
    instrument_comparison = comparisons[comparisons.task.eq("heldout_instrument")].set_index("baseline")
    fixed_pass = bool(instrument_comparison.loc[fixed, "ci_high"] < 0)
    ae_pass = bool(instrument_comparison.loc[predictive, "ci_high"]
                   <= float(config["gate"]["ae_noninferiority_margin"]))
    ccc_pass = bool(state_ci["ci_low"] > float(config["gate"]["memory_cross_view_ccc_lower_bound"]))
    candidate_state = summary_frame.set_index("arm").loc[candidate]
    variance_pass = bool(candidate_state.memory_state_std >= float(config["gate"]["minimum_memory_state_std"]))
    residual_fraction = float(manifest_frame.candidate_residual_fraction.mean())
    residual_pass = bool(residual_fraction <= float(config["gate"]["maximum_residual_fraction"]))
    fold_error = predictions_frame[predictions_frame.task.eq("heldout_instrument")].groupby(
        ["fold", "arm"], as_index=False).absolute_error.mean().pivot(index="fold", columns="arm", values="absolute_error")
    folds_better = int((fold_error[candidate] < fold_error[fixed]).sum())
    fold_pass = folds_better >= int(config["gate"]["minimum_folds_better_than_fixed"])
    gate_passed = bool(fixed_pass and ae_pass and ccc_pass and variance_pass and residual_pass and fold_pass)
    decision = {
        "run_id": config["run_id"],
        "development_only": True,
        "original_train_only": True,
        "validation_loaded": False,
        "test_loaded": False,
        "candidate": candidate,
        "fixed_score_improvement": fixed_pass,
        "autoencoder_noninferiority": ae_pass,
        "memory_cross_view_ccc": state_ci,
        "cross_view_gate": ccc_pass,
        "memory_state_std": float(candidate_state.memory_state_std),
        "noncollapse_gate": variance_pass,
        "residual_loading_fraction": residual_fraction,
        "residual_control_gate": residual_pass,
        "folds_better_than_fixed": folds_better,
        "fold_reproducibility_gate": fold_pass,
        "phase5f_measurement_gate_passed": gate_passed,
        "continuous_time_dynamics_open": gate_passed,
        "next_step": ("freeze_phase6_continuous_time_protocol" if gate_passed
                      else "retain_phase5f_measurement_result_and_keep_dynamics_closed"),
    }
    (output / "development_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    metrics = aggregate.set_index(["arm", "task"])
    report = f"""# Phase 5F: Hierarchical Shared State, Instrument Residuals, and Bridge Graph

## Scope

Five-fold nested subject-wise development used only the original ADNI train split. The model separated a global-plus-domain shared state, three prespecified Memory subconstructs, and shrinkage-controlled instrument residuals. Bridge consistency was applied only to availability/date-supported instrument edges. Original validation and test subjects, diagnosis, FAQ, MRI, and future outcomes were not loaded.

## Predictive measurement

The candidate obtained held-out-item/held-out-instrument macro normalized MAE **{metrics.loc[(candidate, 'heldout_item'), 'macro_item_mae']:.4f}/{metrics.loc[(candidate, 'heldout_instrument'), 'macro_item_mae']:.4f}**. Fixed scoring obtained **{metrics.loc[(fixed, 'heldout_item'), 'macro_item_mae']:.4f}/{metrics.loc[(fixed, 'heldout_instrument'), 'macro_item_mae']:.4f}**, and the unconstrained autoencoder obtained **{metrics.loc[(predictive, 'heldout_item'), 'macro_item_mae']:.4f}/{metrics.loc[(predictive, 'heldout_instrument'), 'macro_item_mae']:.4f}**.

The candidate-minus-fixed held-out-instrument participant difference was **{instrument_comparison.loc[fixed, 'difference']:.5f}**, 95% CI **[{instrument_comparison.loc[fixed, 'ci_low']:.5f}, {instrument_comparison.loc[fixed, 'ci_high']:.5f}]**. Candidate-minus-autoencoder was **{instrument_comparison.loc[predictive, 'difference']:.5f}**, 95% CI **[{instrument_comparison.loc[predictive, 'ci_low']:.5f}, {instrument_comparison.loc[predictive, 'ci_high']:.5f}]**.

Relative to the residual-only ablation, the bridge candidate changed held-out-instrument error by **{instrument_comparison.loc['hierarchical_residual', 'difference']:.5f}**, 95% CI **[{instrument_comparison.loc['hierarchical_residual', 'ci_low']:.5f}, {instrument_comparison.loc['hierarchical_residual', 'ci_high']:.5f}]**. Thus the bridge is evaluated as a state-coherence/prediction trade-off rather than assumed to improve both objectives.

## Bridge state

Graph-supported Memory cross-view CCC was **{state_ci['ccc']:.4f}**, participant-bootstrap 95% CI **[{state_ci['ci_low']:.4f}, {state_ci['ci_high']:.4f}]**. Cross-view Memory-state standard deviation was **{candidate_state.memory_state_std:.4f}** and the mean residual share of decoder loading was **{residual_fraction:.4f}**.

Logical Memory had no usable assessment dates and every LOGMEM edge fell below the frozen graph-support threshold. The graph-supported CCC therefore primarily assesses ADAS/CDR/MMSE Memory bridges and does not establish LOGMEM transport validity.

## Frozen development gate

- improvement over fixed score: `{str(fixed_pass).lower()}`;
- non-inferiority to the unconstrained autoencoder: `{str(ae_pass).lower()}`;
- cross-view CCC lower bound: `{str(ccc_pass).lower()}`;
- non-collapse state variance: `{str(variance_pass).lower()}`;
- residual-loading control: `{str(residual_pass).lower()}`;
- outer-fold reproducibility: `{str(fold_pass).lower()}`.

Therefore `phase5f_measurement_gate_passed={str(gate_passed).lower()}` and `continuous_time_dynamics_open={str(gate_passed).lower()}`. Passing would be development evidence only; confirmation still requires a new untouched or external cohort.
"""
    (output / "report.md").write_text(report)
    manifest = {
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        "phase5c_predictions_sha256": hashlib.sha256(phase5c_path.read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256((ROOT / "reports/experiments/phase5f_hierarchical_bridge_protocol.md").read_bytes()).hexdigest(),
        "train_rows_loaded": len(frame),
        "train_subjects_loaded": int(frame.subject_id.nunique()),
        "validation_or_test_loaded": False,
        "device": str(device),
        "neural_checkpoints": folds * len(config["arms"]),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    print(run(args.config))


if __name__ == "__main__":
    main()
