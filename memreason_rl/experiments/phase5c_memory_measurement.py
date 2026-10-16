"""Train-only cross-fitted measurement models for an interpretable cognitive state."""

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


CONFIG = ROOT / "configs/phase5c_memory_measurement_dev.yaml"


def stable_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def subject_fold(subject: str, seed: int, folds: int) -> int:
    return stable_int(f"{seed}:{subject}") % folds


def clean_matrix(frame: pd.DataFrame, items: list[dict[str, Any]]) -> np.ndarray:
    matrix = np.full((len(frame), len(items)), np.nan, dtype=np.float32)
    for j, item in enumerate(items):
        values = pd.to_numeric(frame[item["field"]], errors="coerce").to_numpy(float)
        missing = np.isin(values, np.asarray(item.get("missing_codes") or [-4, -1, 999, 9999], float))
        valid = np.isfinite(values) & ~missing
        if item.get("valid_values") is not None:
            valid &= np.isin(values, np.asarray(item["valid_values"], float))
        elif item.get("range") is not None:
            valid &= (values >= item["range"][0]) & (values <= item["range"][1])
        matrix[valid, j] = values[valid]
    return matrix


def fit_scaling(raw: np.ndarray, fit_rows: np.ndarray, items: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    mean = np.nanmean(raw[fit_rows], axis=0)
    std = np.nanstd(raw[fit_rows], axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)
    direction = np.asarray([1.0 if item["direction"] == "higher=worse" else -1.0 for item in items])
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32),
            "direction": direction.astype(np.float32)}


def to_severity(raw: np.ndarray, scaling: dict[str, np.ndarray]) -> np.ndarray:
    return ((raw - scaling["mean"]) / scaling["std"] * scaling["direction"]).astype(np.float32)


def deterministic_targets(mask: np.ndarray, instruments: np.ndarray, sample_ids: list[str],
                          task: str, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    inputs = mask.copy(); targets = np.zeros_like(mask)
    for row, sample_id in enumerate(sample_ids):
        observed = np.flatnonzero(mask[row])
        if len(observed) < 2:
            continue
        if task == "heldout_item":
            count = min(len(observed) - 1, max(1, int(math.ceil(len(observed) * fraction))))
            ranked = sorted(observed, key=lambda j: stable_int(f"item:{sample_id}:{j}"))[:count]
        else:
            choices = sorted(set(instruments[observed]))
            if len(choices) < 2:
                continue
            held = choices[stable_int(f"instrument:{sample_id}") % len(choices)]
            ranked = observed[instruments[observed] == held].tolist()
            if len(ranked) == len(observed):
                continue
        targets[row, ranked] = True; inputs[row, ranked] = False
    return inputs, targets


def random_targets(mask: np.ndarray, instruments: np.ndarray, rng: np.random.Generator,
                   fraction: float) -> tuple[np.ndarray, np.ndarray]:
    inputs = mask.copy(); targets = np.zeros_like(mask)
    for row in range(len(mask)):
        observed = np.flatnonzero(mask[row])
        if len(observed) < 2:
            continue
        choices = np.unique(instruments[observed])
        if len(choices) >= 2 and rng.random() < 0.5:
            held = rng.choice(choices); selected = observed[instruments[observed] == held]
        else:
            count = min(len(observed) - 1, max(1, int(math.ceil(len(observed) * fraction))))
            selected = rng.choice(observed, count, replace=False)
        targets[row, selected] = True; inputs[row, selected] = False
    return inputs, targets


class FixedDomainScore:
    def __init__(self, domain_index: np.ndarray):
        self.domain_index = domain_index
        self.slope = np.ones(len(domain_index), dtype=np.float32)
        self.intercept = np.zeros(len(domain_index), dtype=np.float32)

    def state(self, values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        states = np.zeros((len(values), len(DOMAINS)), np.float32)
        available = np.zeros_like(states, bool)
        for domain in range(len(DOMAINS)):
            local = mask[:, self.domain_index == domain]
            count = local.sum(1); available[:, domain] = count > 0
            states[:, domain] = (values[:, self.domain_index == domain] * local).sum(1) / np.maximum(count, 1)
        return states, available

    def fit(self, values: np.ndarray, mask: np.ndarray) -> None:
        for item, domain in enumerate(self.domain_index):
            local = mask.copy(); local[:, item] = False
            state, available = self.state(values, local)
            keep = mask[:, item] & available[:, domain]
            if keep.sum() < 10:
                continue
            x, y = state[keep, domain], values[keep, item]
            variance = float(np.var(x))
            self.slope[item] = 0.0 if variance < 1e-8 else float(np.cov(x, y, ddof=0)[0, 1] / variance)
            self.intercept[item] = float(y.mean() - self.slope[item] * x.mean())

    def predict(self, values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        state, _ = self.state(values, mask)
        return self.intercept[None] + self.slope[None] * state[:, self.domain_index]


class UnconstrainedAE(nn.Module):
    def __init__(self, n_items: int, hidden: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n_items * 2, hidden), nn.GELU(), nn.LayerNorm(hidden),
                                     nn.Linear(hidden, len(DOMAINS)))
        self.decoder = nn.Linear(len(DOMAINS), n_items)

    def state(self, values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.encoder(torch.cat([values * mask, mask], 1))
        return state, torch.zeros_like(state)

    def forward(self, values: torch.Tensor, mask: torch.Tensor, sample: bool = True):
        state, logvar = self.state(values, mask)
        return self.decoder(state), state, logvar


class DomainAE(nn.Module):
    def __init__(self, domain_index: np.ndarray, hidden: int, probabilistic: bool):
        super().__init__(); self.probabilistic = probabilistic
        self.register_buffer("domain_index", torch.tensor(domain_index, dtype=torch.long))
        outputs = 2 if probabilistic else 1
        self.encoders = nn.ModuleList()
        for domain in range(len(DOMAINS)):
            count = int((domain_index == domain).sum())
            self.encoders.append(nn.Sequential(nn.Linear(count * 2, hidden), nn.GELU(),
                                               nn.LayerNorm(hidden), nn.Linear(hidden, outputs)))
        self.scale = nn.Parameter(torch.ones(len(domain_index)))
        self.bias = nn.Parameter(torch.zeros(len(domain_index)))

    def state(self, values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        means, logvars = [], []
        for domain, encoder in enumerate(self.encoders):
            choose = self.domain_index == domain
            result = encoder(torch.cat([values[:, choose] * mask[:, choose], mask[:, choose]], 1))
            if self.probabilistic:
                mean, logvar = result[:, 0], result[:, 1].clamp(-6, 3)
            else:
                mean, logvar = result[:, 0], torch.zeros_like(result[:, 0])
            means.append(mean); logvars.append(logvar)
        return torch.stack(means, 1), torch.stack(logvars, 1)

    def forward(self, values: torch.Tensor, mask: torch.Tensor, sample: bool = True):
        mean, logvar = self.state(values, mask)
        latent = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar) if self.probabilistic and sample else mean
        prediction = self.bias[None] + self.scale[None] * latent[:, self.domain_index]
        return prediction, mean, logvar


def typed_loss(prediction: torch.Tensor, target_z: torch.Tensor, target_raw: torch.Tensor,
               target_mask: torch.Tensor, binary: torch.Tensor) -> torch.Tensor:
    losses = []
    scalar_mask = target_mask & ~binary[None]
    if scalar_mask.any():
        losses.append(F.smooth_l1_loss(prediction[scalar_mask], target_z[scalar_mask]))
    binary_mask = target_mask & binary[None]
    if binary_mask.any():
        losses.append(F.binary_cross_entropy_with_logits(prediction[binary_mask], target_raw[binary_mask]))
    return sum(losses) / len(losses) if losses else prediction.sum() * 0


def train_model(arm: str, train_z: np.ndarray, train_raw: np.ndarray, train_mask: np.ndarray,
                dev_z: np.ndarray, dev_raw: np.ndarray, dev_mask: np.ndarray,
                instruments: np.ndarray, domain_index: np.ndarray, binary: np.ndarray,
                config: dict[str, Any], seed: int, device: torch.device) -> nn.Module:
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    n_items = train_z.shape[1]; settings = config["training"]
    if arm == "unconstrained_autoencoder":
        model: nn.Module = UnconstrainedAE(n_items, int(settings["hidden_dim"]))
    else:
        model = DomainAE(domain_index, int(settings["domain_hidden_dim"]), arm == "probabilistic_domain")
    model.to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["learning_rate"]),
                                                     weight_decay=float(settings["weight_decay"]))
    batch_size = int(settings["batch_size"]); fraction = float(config["masking"]["item_fraction"])
    dev_input, dev_target = deterministic_targets(dev_mask, instruments,
                                                   [f"dev:{i}" for i in range(len(dev_mask))],
                                                   "heldout_item", fraction)
    best, best_loss, patience = None, float("inf"), 0
    for epoch in range(int(settings["max_epochs"])):
        rng = np.random.default_rng(seed * 1000 + epoch)
        input_mask, target_mask = random_targets(train_mask, instruments, rng, fraction)
        order = rng.permutation(len(train_z)); model.train()
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size]
            values = torch.tensor(train_z[index], device=device).nan_to_num()
            observed = torch.tensor(input_mask[index], device=device, dtype=torch.float32)
            targets = torch.tensor(target_mask[index], device=device)
            raw = torch.tensor(train_raw[index], device=device).nan_to_num()
            target_z = torch.tensor(train_z[index], device=device).nan_to_num()
            prediction, mean, logvar = model(values, observed, True)
            loss = typed_loss(prediction, target_z, raw, targets, torch.tensor(binary, device=device))
            if arm == "probabilistic_domain":
                loss = loss + float(settings["kl_weight"]) * 0.5 * (
                    mean.pow(2) + logvar.exp() - logvar - 1).mean()
                coin = torch.rand_like(observed) > 0.5
                view_a = observed.bool() & coin; view_b = observed.bool() & ~coin
                state_a, _ = model.state(values, view_a.float()); state_b, _ = model.state(values, view_b.float())
                coverage = []
                domain_tensor = torch.tensor(domain_index, device=device)
                for d in range(len(DOMAINS)):
                    coverage.append(view_a[:, domain_tensor == d].any(1) & view_b[:, domain_tensor == d].any(1))
                coverage = torch.stack(coverage, 1)
                if coverage.any():
                    loss = loss + float(settings["consistency_weight"]) * F.smooth_l1_loss(
                        state_a[coverage], state_b[coverage])
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            values = torch.tensor(dev_z, device=device).nan_to_num()
            observed = torch.tensor(dev_input, device=device, dtype=torch.float32)
            prediction, _, _ = model(values, observed, False)
            dev_loss = float(typed_loss(prediction, torch.tensor(dev_z, device=device).nan_to_num(),
                                        torch.tensor(dev_raw, device=device).nan_to_num(),
                                        torch.tensor(dev_target, device=device),
                                        torch.tensor(binary, device=device)))
        if dev_loss < best_loss - 1e-5:
            best_loss, best, patience = dev_loss, copy.deepcopy(model.state_dict()), 0
        else:
            patience += 1
            if patience >= int(settings["patience"]): break
    model.load_state_dict(best); model.eval()
    model.best_dev_loss = best_loss
    return model


def decode_predictions(output: np.ndarray, binary: np.ndarray, scaling: dict[str, np.ndarray]) -> np.ndarray:
    result = output.copy()
    if binary.any():
        raw = 1.0 / (1.0 + np.exp(-np.clip(output[:, binary], -20, 20)))
        result[:, binary] = ((raw - scaling["mean"][binary]) / scaling["std"][binary]
                             * scaling["direction"][binary])
    return result


def neural_predict(model: nn.Module, z: np.ndarray, input_mask: np.ndarray, binary: np.ndarray,
                   scaling: dict[str, np.ndarray], device: torch.device, batch_size: int = 1024) -> np.ndarray:
    rows = []
    with torch.no_grad():
        for start in range(0, len(z), batch_size):
            values = torch.tensor(z[start:start + batch_size], device=device).nan_to_num()
            mask = torch.tensor(input_mask[start:start + batch_size], device=device, dtype=torch.float32)
            output, _, _ = model(values, mask, False)
            rows.append(output.cpu().numpy())
    return decode_predictions(np.concatenate(rows), binary, scaling)


def neural_state(model: nn.Module, z: np.ndarray, mask: np.ndarray, device: torch.device) -> np.ndarray:
    rows = []
    with torch.no_grad():
        for start in range(0, len(z), 1024):
            values = torch.tensor(z[start:start + 1024], device=device).nan_to_num()
            observed = torch.tensor(mask[start:start + 1024], device=device, dtype=torch.float32)
            mean, _ = model.state(values, observed); rows.append(mean.cpu().numpy())
    return np.concatenate(rows)


def cross_views(mask: np.ndarray, instruments: np.ndarray, sample_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    a = np.zeros_like(mask); b = np.zeros_like(mask)
    for row, sample_id in enumerate(sample_ids):
        observed = np.flatnonzero(mask[row]); choices = sorted(set(instruments[observed]))
        if len(choices) < 2: continue
        choices = sorted(choices, key=lambda x: stable_int(f"view:{sample_id}:{x}"))
        left = set(choices[::2]); a[row, observed[np.isin(instruments[observed], list(left))]] = True
        b[row] = mask[row] & ~a[row]
    return a, b


def bootstrap(predictions: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    candidate = config["stage1_gate"]["candidate"]; rng = np.random.default_rng(config["statistics"]["bootstrap_seed"])
    reps = int(config["statistics"]["bootstrap_replicates"]); rows = []
    keys = ["sample_id", "subject_id", "task", "item"]
    for task in predictions.task.unique():
        local = predictions[predictions.task.eq(task)]
        for baseline in config["stage1_gate"]["baselines"]:
            pair = local[local.arm.isin([candidate, baseline])].pivot_table(index=keys, columns="arm", values="absolute_error").dropna()
            pair["difference"] = pair[candidate] - pair[baseline]
            subject = pair.groupby("subject_id").difference.mean().to_numpy()
            draws = subject[rng.integers(0, len(subject), size=(reps, len(subject)))].mean(1)
            rows.append({"task": task, "candidate": candidate, "baseline": baseline,
                         "difference": float(subject.mean()), "ci_low": float(np.quantile(draws, .025)),
                         "ci_high": float(np.quantile(draws, .975)), "subjects": len(subject),
                         "paired_decisions": len(pair), "replicates": reps})
    return pd.DataFrame(rows)


def run() -> Path:
    config = yaml.safe_load(CONFIG.read_text()); contract_path = ROOT / config["item_contract"]
    contract = yaml.safe_load(contract_path.read_text()); items = contract["primary_state_items"]
    fields = [item["field"] for item in items]; instruments = np.asarray([item["instrument"] for item in items])
    domain_index = np.asarray([DOMAINS.index(item["domain"]) for item in items])
    binary = np.asarray([item["value_type"] == "binary" for item in items])
    frame = pd.read_csv(ROOT / "splits/train.csv", low_memory=False)
    raw = clean_matrix(frame, items); mask = np.isfinite(raw)
    keep = mask.sum(1) >= 2; frame = frame.loc[keep].reset_index(drop=True); raw = raw[keep]; mask = mask[keep]
    sample_ids = [f"{row.subject_id}:{row.visit_code}:{i}" for i, row in enumerate(frame.itertuples())]
    fold_assignment = np.asarray([subject_fold(str(s), int(config["partition_seed"]), int(config["folds"]))
                                  for s in frame.subject_id])
    output = ROOT / "outputs" / config["run_id"]; output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions, state_rows, manifests = [], [], []
    for fold in range(int(config["folds"])):
        evaluation = fold_assignment == fold; outer_fit = ~evaluation
        dev_subjects = {s for s in frame.loc[outer_fit, "subject_id"].astype(str).unique()
                        if stable_int(f"inner:{config['partition_seed']}:{fold}:{s}") / 2**64 < float(config["inner_dev_fraction"])}
        development = outer_fit & frame.subject_id.astype(str).isin(dev_subjects).to_numpy()
        fit = outer_fit & ~development
        scaling = fit_scaling(raw, fit, items); z = to_severity(raw, scaling)
        fixed = FixedDomainScore(domain_index); fixed.fit(np.nan_to_num(z[fit]), mask[fit])
        models: dict[str, Any] = {"fixed_domain_score": fixed}
        for arm in [x for x in config["arms"] if x != "fixed_domain_score"]:
            models[arm] = train_model(arm, z[fit], raw[fit], mask[fit], z[development], raw[development],
                                      mask[development], instruments, domain_index, binary, config,
                                      int(config["partition_seed"]) + fold * 17 + len(arm), device)
            checkpoint = output / "checkpoints" / arm / f"fold_{fold}.pt"; checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": models[arm].state_dict(), "best_dev_loss": models[arm].best_dev_loss,
                        "fold": fold, "fields": fields}, checkpoint)
        local_ids = [sample_ids[i] for i in np.flatnonzero(evaluation)]
        for task in ["heldout_item", "heldout_instrument"]:
            input_mask, target_mask = deterministic_targets(mask[evaluation], instruments, local_ids, task,
                                                             float(config["masking"]["item_fraction"]))
            for arm, model in models.items():
                pred = model.predict(np.nan_to_num(z[evaluation]), input_mask) if arm == "fixed_domain_score" else neural_predict(
                    model, z[evaluation], input_mask, binary, scaling, device)
                for row, item in zip(*np.where(target_mask)):
                    global_row = np.flatnonzero(evaluation)[row]
                    predictions.append({"sample_id": sample_ids[global_row], "subject_id": frame.subject_id.iloc[global_row],
                                        "fold": fold, "task": task, "arm": arm, "item": fields[item],
                                        "instrument": instruments[item], "domain": DOMAINS[domain_index[item]],
                                        "truth_z": float(z[global_row, item]), "prediction_z": float(pred[row, item]),
                                        "absolute_error": float(abs(pred[row, item] - z[global_row, item]))})
        view_a, view_b = cross_views(mask[evaluation], instruments, local_ids)
        valid_rows = view_a.any(1) & view_b.any(1)
        for arm, model in models.items():
            if arm == "fixed_domain_score":
                state_a, avail_a = model.state(np.nan_to_num(z[evaluation]), view_a)
                state_b, avail_b = model.state(np.nan_to_num(z[evaluation]), view_b)
            else:
                state_a = neural_state(model, z[evaluation], view_a, device)
                state_b = neural_state(model, z[evaluation], view_b, device)
                if arm == "unconstrained_autoencoder":
                    avail_a = np.repeat(view_a.any(1)[:, None], len(DOMAINS), 1)
                    avail_b = np.repeat(view_b.any(1)[:, None], len(DOMAINS), 1)
                else:
                    avail_a = np.stack([view_a[:, domain_index == d].any(1) for d in range(len(DOMAINS))], 1)
                    avail_b = np.stack([view_b[:, domain_index == d].any(1) for d in range(len(DOMAINS))], 1)
            for row, dimension in zip(*np.where(valid_rows[:, None] & avail_a & avail_b)):
                global_row = np.flatnonzero(evaluation)[row]
                state_rows.append({"sample_id": sample_ids[global_row], "subject_id": frame.subject_id.iloc[global_row],
                                   "fold": fold, "arm": arm, "state_dimension": DOMAINS[dimension] if arm != "unconstrained_autoencoder" else f"latent_{dimension}",
                                   "view_a": float(state_a[row, dimension]), "view_b": float(state_b[row, dimension]),
                                   "absolute_difference": float(abs(state_a[row, dimension] - state_b[row, dimension]))})
        manifests.append({"fold": fold, "fit_subjects": int(frame.loc[fit, "subject_id"].nunique()),
                          "dev_subjects": int(frame.loc[development, "subject_id"].nunique()),
                          "evaluation_subjects": int(frame.loc[evaluation, "subject_id"].nunique()),
                          "fit_rows": int(fit.sum()), "dev_rows": int(development.sum()),
                          "evaluation_rows": int(evaluation.sum())})
    pred_frame = pd.DataFrame(predictions); state_frame = pd.DataFrame(state_rows)
    pred_frame.to_csv(output / "crossfit_predictions.csv", index=False)
    state_frame.to_csv(output / "cross_view_states.csv", index=False)
    pd.DataFrame(manifests).to_csv(output / "fold_manifest.csv", index=False)
    item_metrics = pred_frame.groupby(["arm", "task", "item", "domain"], as_index=False).agg(
        decisions=("absolute_error", "size"), subjects=("subject_id", "nunique"), mae=("absolute_error", "mean"))
    item_metrics.to_csv(output / "item_metrics.csv", index=False)
    aggregate = item_metrics.groupby(["arm", "task"], as_index=False).agg(
        macro_item_mae=("mae", "mean"), items=("item", "nunique"), decisions=("decisions", "sum"))
    memory = item_metrics[item_metrics.domain.eq("Memory")].groupby(["arm", "task"], as_index=False).agg(
        memory_macro_item_mae=("mae", "mean"), memory_items=("item", "nunique"))
    aggregate = aggregate.merge(memory, on=["arm", "task"])
    aggregate.to_csv(output / "aggregate_metrics.csv", index=False)
    state_summary = []
    for arm, group in state_frame.groupby("arm"):
        corr = spearmanr(group.view_a, group.view_b)
        state_summary.append({"arm": arm, "paired_states": len(group), "subjects": group.subject_id.nunique(),
                              "mean_absolute_difference": group.absolute_difference.mean(),
                              "spearman": corr.statistic, "spearman_pvalue": corr.pvalue,
                              "view_a_std": group.view_a.std(), "view_b_std": group.view_b.std()})
    state_summary = pd.DataFrame(state_summary); state_summary.to_csv(output / "cross_view_summary.csv", index=False)
    comparison = bootstrap(pred_frame, config); comparison.to_csv(output / "subject_bootstrap.csv", index=False)
    candidate = config["stage1_gate"]["candidate"]
    instrument = comparison[comparison.task.eq("heldout_instrument")]
    error_pass = bool((instrument.ci_high < 0).all())
    state_lookup = state_summary.set_index("arm")
    coherence_pass = bool(all(state_lookup.loc[candidate, "spearman"] > state_lookup.loc[b, "spearman"]
                              for b in config["stage1_gate"]["baselines"]))
    decision = {"run_id": config["run_id"], "development_only": True, "original_train_only": True,
                "validation_loaded": False, "test_loaded": False, "primary_items": len(fields),
                "memory_items": int(sum(domain_index == 0)), "folds": int(config["folds"]),
                "heldout_instrument_improvement": error_pass, "cross_view_coherence_improvement": coherence_pass,
                "stage1_measurement_gate_passed": bool(error_pass and coherence_pass),
                "phase5c_complete": False,
                "next_step": "continuous_time_dynamics_if_passed_else_report_measurement_negative"}
    (output / "development_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    candidate_metrics = aggregate[aggregate.arm.eq(candidate)].set_index("task")
    report = f"""# Phase 5C Stage 1: Interpretable Memory-State Measurement

## Scope

Five-fold nested subject-wise development used only the original ADNI train split. The experiment evaluated 70 frozen state items: the prior 68 verified cognitive items plus audited Logical Memory immediate and delayed recall. AVLT intrusion fields were excluded from state input. Original validation and test data were not loaded.

## Results

The probabilistic domain model achieved held-out-item macro normalized MAE **{candidate_metrics.loc['heldout_item','macro_item_mae']:.4f}** and held-out-instrument macro normalized MAE **{candidate_metrics.loc['heldout_instrument','macro_item_mae']:.4f}**. Its corresponding Memory-item values were **{candidate_metrics.loc['heldout_item','memory_macro_item_mae']:.4f}** and **{candidate_metrics.loc['heldout_instrument','memory_macro_item_mae']:.4f}**. Cross-view state Spearman was **{state_lookup.loc[candidate,'spearman']:.4f}** over {int(state_lookup.loc[candidate,'paired_states'])} paired state dimensions.

The prespecified held-out-instrument improvement condition was **{str(error_pass).lower()}** and the cross-view coherence condition was **{str(coherence_pass).lower()}**. Therefore `stage1_measurement_gate_passed={str(decision['stage1_measurement_gate_passed']).lower()}`. This is development evidence only and does not reopen the consumed ADNI validation or test cohorts.

## Interpretation

The gate requires both predictive reconstruction and state coherence; a constrained state is not accepted solely because its axes are named. Full Phase 5C additionally requires continuous-time forecasting, calibration, downstream validity, reasoning faithfulness, and external confirmation, so `phase5c_complete=false` regardless of this Stage 1 outcome.
"""
    (output / "report.md").write_text(report)
    source = {"config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
              "contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
              "train_rows_loaded": len(frame), "train_subjects_loaded": frame.subject_id.nunique(),
              "validation_or_test_loaded": False, "device": str(device)}
    (output / "manifest.json").write_text(json.dumps(source, indent=2) + "\n")
    return output


def main() -> None:
    argparse.ArgumentParser().parse_args(); print(run())


if __name__ == "__main__": main()
