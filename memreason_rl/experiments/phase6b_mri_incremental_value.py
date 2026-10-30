"""MRI incremental-value audit against acquisition and permutation controls.

Phase 6B keeps the Phase 6A latent teacher as auxiliary supervision but makes
the future observed Memory composite the primary endpoint.  It is deliberately
original-train-only exploratory work.
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
from torch import nn
from torch.nn import functional as F

from ..data.common import ROOT
from .phase6a_latent_state_dynamics import (
    LatentSample,
    aggregate_mri_features,
    analytic_transition,
    build_samples,
    fit_reference,
    normalize_mri,
    precision_update,
    reconstruct_state_records,
    tensorize,
)


CONFIG = ROOT / "configs/phase6b_mri_incremental_value_dev.yaml"
PROTOCOL = ROOT / "reports/experiments/phase6b_mri_incremental_value_protocol.md"


ARM_SETTINGS = {
    "cognition_only": {"use_cognition": True, "use_mri": False, "feature_group": "none"},
    "metadata_only": {"use_cognition": True, "use_mri": True, "feature_group": "metadata"},
    "real_mri": {"use_cognition": True, "use_mri": True, "feature_group": "all"},
    "subject_shuffled_mri": {"use_cognition": True, "use_mri": True, "feature_group": "all"},
    "time_shuffled_mri": {"use_cognition": True, "use_mri": True, "feature_group": "all"},
    "mri_only": {"use_cognition": False, "use_mri": True, "feature_group": "all"},
}


def feature_mask(group: str, config: dict[str, Any]) -> torch.Tensor:
    count = len(config["mri"]["features"])
    result = torch.zeros(count, dtype=torch.float32)
    if group == "all":
        result[:] = 1
    elif group == "metadata":
        result[config["mri"]["metadata_feature_indices"]] = 1
    elif group != "none":
        raise ValueError(f"Unknown MRI feature group: {group}")
    return result


def fit_clinical_decoder(records: pd.DataFrame, subjects: set[str]) -> tuple[float, float, float]:
    """Fit a train-fold-only state-to-observed-composite Gaussian decoder."""
    local = records[records.subject_id.astype(str).isin(subjects)].dropna(
        subset=["memory_composite", "state_mean"]
    )
    if not len(local):
        raise RuntimeError("No Memory composites in the outer-fit partition")
    if len(local) < 2 or float(local.state_mean.std()) < 1e-8:
        intercept, slope = float(local.memory_composite.mean()), 0.0
    else:
        slope, intercept = np.polyfit(local.state_mean.to_numpy(float),
                                      local.memory_composite.to_numpy(float), 1)
    residual = local.memory_composite.to_numpy(float) - (
        float(intercept) + float(slope) * local.state_mean.to_numpy(float)
    )
    residual_variance = float(max(np.mean(residual ** 2), 1e-4))
    return float(intercept), float(slope), residual_variance


def subject_shuffle_structural(
    samples: list[LatentSample], structural_indices: list[int], seed: int
) -> tuple[list[LatentSample], float]:
    """Shuffle structural content across subjects while retaining acquisition metadata."""
    result = copy.deepcopy(samples)
    available = [sample for sample in result if sample.mri_available]
    subjects = sorted({sample.subject_id for sample in available})
    if len(subjects) < 2:
        return result, 0.0
    rng = np.random.default_rng(seed)
    shift = int(rng.integers(1, len(subjects)))
    donor_subject = {subject: subjects[(index + shift) % len(subjects)]
                     for index, subject in enumerate(subjects)}
    by_subject: dict[str, list[LatentSample]] = {}
    for sample in available:
        by_subject.setdefault(sample.subject_id, []).append(sample)
    source_values = {
        id(sample): sample.mri_features[structural_indices].copy()
        for sample in available
    }
    changed = 0
    for sample in available:
        donors = by_subject[donor_subject[sample.subject_id]]
        source = min(donors, key=lambda value: abs(value.history_time[-1] - sample.history_time[-1]))
        before = sample.mri_features[structural_indices].copy()
        sample.mri_features[structural_indices] = source_values[id(source)]
        changed += int(not np.allclose(before, sample.mri_features[structural_indices]))
    return result, changed / max(len(available), 1)


def time_shuffle_structural(
    samples: list[LatentSample], structural_indices: list[int], seed: int
) -> tuple[list[LatentSample], float]:
    """Permute structural content among cutoffs of the same subject."""
    result = copy.deepcopy(samples); rng = np.random.default_rng(seed); changed = 0; eligible = 0
    for _, group in pd.DataFrame({
        "row": np.arange(len(result)),
        "subject": [sample.subject_id for sample in result],
        "cutoff": [float(sample.history_time[-1]) for sample in result],
        "available": [sample.mri_available for sample in result],
    }).groupby("subject"):
        cutoff_rows: dict[float, list[int]] = {}
        for row in group[group.available].itertuples(index=False):
            cutoff_rows.setdefault(float(row.cutoff), []).append(int(row.row))
        cutoffs = sorted(cutoff_rows)
        if len(cutoffs) < 2:
            continue
        shift = int(rng.integers(1, len(cutoffs)))
        donor = {cutoff: cutoffs[(index + shift) % len(cutoffs)]
                 for index, cutoff in enumerate(cutoffs)}
        source_values = {
            cutoff: result[cutoff_rows[cutoff][0]].mri_features[structural_indices].copy()
            for cutoff in cutoffs
        }
        for cutoff, rows in cutoff_rows.items():
            for row in rows:
                before = result[row].mri_features[structural_indices].copy()
                result[row].mri_features[structural_indices] = source_values[donor[cutoff]]
                eligible += 1
                changed += int(not np.allclose(before, result[row].mri_features[structural_indices]))
    return result, changed / max(eligible, 1)


class IncrementalValueDynamics(nn.Module):
    """Reference-aware Gaussian filter with controlled cognition/MRI evidence."""

    def __init__(
        self,
        use_cognition: bool,
        use_mri: bool,
        mri_feature_mask: torch.Tensor,
        minimum_variance: float,
    ):
        super().__init__()
        self.use_cognition = bool(use_cognition); self.use_mri = bool(use_mri)
        self.minimum_variance = float(minimum_variance)
        self.register_buffer("mri_feature_mask", mri_feature_mask.float())
        self.raw_decay = nn.Parameter(torch.tensor(-2.0)); self.center = nn.Parameter(torch.tensor(0.0))
        self.drift = nn.Parameter(torch.tensor(0.0)); self.raw_process = nn.Parameter(torch.tensor(-3.0))
        self.raw_evidence_temperature = nn.Parameter(torch.tensor(0.0))
        self.prior_mean = nn.Parameter(torch.tensor(0.0)); self.raw_prior_variance = nn.Parameter(torch.tensor(0.0))
        self.mri_weight = nn.Parameter(torch.zeros(len(mri_feature_mask)))
        self.mri_intercept = nn.Parameter(torch.tensor(0.0)); self.raw_mri_variance = nn.Parameter(torch.tensor(0.0))

    def _transition(self, mean: torch.Tensor, variance: torch.Tensor, delta: torch.Tensor):
        return analytic_transition(
            mean, variance, delta, F.softplus(self.raw_decay) + 1e-4,
            self.center, self.drift, F.softplus(self.raw_process) + 1e-5,
        )

    def forward(
        self, batch: dict[str, torch.Tensor], reference: tuple[float, float]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        times, valid = batch["history_time"], batch["history_mask"]
        lengths = valid.sum(1).long(); rows = torch.arange(len(valid), device=valid.device)
        last = lengths - 1; previous = times[rows, last]
        intercept, slope = reference
        target_reference = intercept + slope * batch["target_time"]
        if self.use_cognition:
            evidence = batch["history_mean"] - (intercept + slope * times)
            scale = F.softplus(self.raw_evidence_temperature) + 0.25
            mean = evidence[:, 0]; variance = batch["history_var"][:, 0] * scale
            previous = times[:, 0]
            for step in range(1, times.shape[1]):
                active = valid[:, step]; delta = (times[:, step] - previous).clamp_min(0)
                prior_mean, prior_var = self._transition(mean, variance, delta)
                update_mean, update_var = precision_update(
                    prior_mean, prior_var, evidence[:, step], batch["history_var"][:, step] * scale
                )
                mean = torch.where(active, update_mean, mean)
                variance = torch.where(active, update_var, variance)
                previous = torch.where(active, times[:, step], previous)
        else:
            mean = self.prior_mean.expand(len(valid))
            variance = (F.softplus(self.raw_prior_variance) + self.minimum_variance).expand(len(valid))
        if self.use_mri:
            weights = self.mri_weight * self.mri_feature_mask
            mri_mean = self.mri_intercept + batch["mri"] @ weights
            mri_var = (F.softplus(self.raw_mri_variance) + self.minimum_variance).expand_as(variance)
            update_mean, update_var = precision_update(mean, variance, mri_mean, mri_var)
            mean = torch.where(batch["mri_available"], update_mean, mean)
            variance = torch.where(batch["mri_available"], update_var, variance)
        mean, variance = self._transition(
            mean, variance, (batch["target_time"] - previous).clamp_min(0)
        )
        return mean + target_reference, variance.clamp_min(self.minimum_variance)


def clinical_distribution(
    state_mean: torch.Tensor,
    state_variance: torch.Tensor,
    decoder: tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    intercept, slope, residual_variance = decoder
    mean = intercept + slope * state_mean
    variance = slope * slope * state_variance + residual_variance
    return mean, variance.clamp_min(1e-6)


def training_batch(samples: list[LatentSample], device: torch.device) -> dict[str, torch.Tensor]:
    batch = tensorize(samples, device)
    target = np.asarray([sample.target_composite for sample in samples], np.float32)
    batch["target_composite"] = torch.as_tensor(np.nan_to_num(target), device=device)
    batch["target_composite_mask"] = torch.as_tensor(np.isfinite(target), device=device)
    return batch


def loss_function(
    state_mean: torch.Tensor,
    state_variance: torch.Tensor,
    batch: dict[str, torch.Tensor],
    decoder: tuple[float, float, float],
    config: dict[str, Any],
) -> torch.Tensor:
    target_var = batch["target_var"].clamp_min(1e-5)
    total_var = state_variance + target_var
    latent_nll = 0.5 * (
        torch.log(2 * torch.pi * total_var)
        + (batch["target_mean"] - state_mean).square() / total_var
    )
    w2 = ((batch["target_mean"] - state_mean).square()
          + (torch.sqrt(target_var) - torch.sqrt(state_variance)).square())
    clinical_mean, clinical_var = clinical_distribution(state_mean, state_variance, decoder)
    clinical_nll = 0.5 * (
        torch.log(2 * torch.pi * clinical_var)
        + (batch["target_composite"] - clinical_mean).square() / clinical_var
    )
    mask = batch["target_composite_mask"]
    clinical = clinical_nll[mask].mean() if mask.any() else clinical_nll.sum() * 0
    return (latent_nll.mean()
            + float(config["training"]["wasserstein_weight"]) * w2.mean()
            + float(config["training"]["clinical_nll_weight"]) * clinical)


def train_model(
    arm: str,
    fit: list[LatentSample],
    development: list[LatentSample],
    reference: tuple[float, float],
    decoder: tuple[float, float, float],
    config: dict[str, Any],
    device: torch.device,
    seed: int,
) -> IncrementalValueDynamics:
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    settings = ARM_SETTINGS[arm]
    model = IncrementalValueDynamics(
        settings["use_cognition"], settings["use_mri"],
        feature_mask(settings["feature_group"], config),
        float(config["training"]["minimum_variance"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    fit_batch, dev_batch = training_batch(fit, device), training_batch(development, device)
    best, best_loss, stale = copy.deepcopy(model.state_dict()), float("inf"), 0
    for _ in range(int(config["training"]["max_epochs"])):
        model.train(); optimizer.zero_grad()
        mean, variance = model(fit_batch, reference)
        loss = loss_function(mean, variance, fit_batch, decoder, config)
        loss.backward(); torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config["training"]["gradient_clip"])
        ); optimizer.step()
        model.eval()
        with torch.no_grad():
            mean, variance = model(dev_batch, reference)
            score = float(loss_function(mean, variance, dev_batch, decoder, config))
        if score < best_loss - 1e-6:
            best, best_loss, stale = copy.deepcopy(model.state_dict()), score, 0
        else:
            stale += 1
            if stale >= int(config["training"]["patience"]):
                break
    model.load_state_dict(best); model.best_dev_loss = best_loss; return model


def prediction_rows(
    arm: str,
    samples: list[LatentSample],
    mean: np.ndarray,
    variance: np.ndarray,
    decoder: tuple[float, float, float],
) -> list[dict[str, Any]]:
    intercept, slope, residual_variance = decoder; rows = []
    for sample, prediction, state_var in zip(samples, mean, variance):
        state_var = max(float(state_var), 1e-8)
        latent_total_var = max(state_var + sample.target_var, 1e-8)
        latent_error = sample.target_mean - float(prediction)
        clinical_mean = float(intercept + slope * prediction)
        clinical_var = max(slope * slope * state_var + residual_variance, 1e-8)
        clinical_error = float(sample.target_composite - clinical_mean)
        rows.append({
            "sample_id": sample.sample_id, "subject_id": sample.subject_id,
            "fold": sample.fold, "arm": arm, "horizon_months": sample.horizon_months,
            "truth_state": sample.target_mean, "prediction_state": float(prediction),
            "prediction_variance": state_var, "teacher_variance": sample.target_var,
            "latent_absolute_error": abs(latent_error),
            "latent_nll": 0.5 * (math.log(2 * math.pi * latent_total_var)
                                  + latent_error * latent_error / latent_total_var),
            "wasserstein2": latent_error * latent_error
            + (math.sqrt(state_var) - math.sqrt(max(sample.target_var, 0))) ** 2,
            "truth_composite": sample.target_composite,
            "prediction_composite": clinical_mean,
            "composite_variance": clinical_var,
            "composite_absolute_error": abs(clinical_error),
            "composite_nll": 0.5 * (math.log(2 * math.pi * clinical_var)
                                     + clinical_error * clinical_error / clinical_var),
            "composite_covered_95": abs(clinical_error) <= 1.96 * math.sqrt(clinical_var),
            "composite_interval_width": 2 * 1.96 * math.sqrt(clinical_var),
            "mri_available": sample.mri_available,
        })
    return rows


def summarize(predictions: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    return predictions.groupby(groups, as_index=False).agg(
        samples=("sample_id", "size"), subjects=("subject_id", "nunique"),
        composite_mae=("composite_absolute_error", "mean"),
        composite_nll=("composite_nll", "mean"),
        coverage_95=("composite_covered_95", "mean"),
        interval_width=("composite_interval_width", "mean"),
        latent_mae=("latent_absolute_error", "mean"),
        latent_nll=("latent_nll", "mean"), wasserstein2=("wasserstein2", "mean"),
    )


def paired_bootstrap(predictions: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    candidate = config["selection"]["candidate"]
    rng = np.random.default_rng(int(config["statistics"]["bootstrap_seed"]))
    repetitions = int(config["statistics"]["bootstrap_replicates"]); rows = []
    for comparator in config["arms"]:
        if comparator == candidate:
            continue
        for metric in ["composite_absolute_error", "latent_absolute_error"]:
            pair = predictions[predictions.arm.isin([candidate, comparator])].pivot_table(
                index=["sample_id", "subject_id"], columns="arm", values=metric
            ).dropna()
            subject_difference = (pair[candidate] - pair[comparator]).groupby("subject_id").mean()
            values = subject_difference.to_numpy()
            draws = values[rng.integers(0, len(values), size=(repetitions, len(values)))].mean(1)
            rows.append({
                "candidate": candidate, "comparator": comparator, "metric": metric,
                "difference": float(values.mean()), "ci_low": float(np.quantile(draws, 0.025)),
                "ci_high": float(np.quantile(draws, 0.975)), "subjects": len(values),
                "replicates": repetitions,
            })
    return pd.DataFrame(rows)


def arm_samples(
    arm: str,
    samples: list[LatentSample],
    structural_indices: list[int],
    seed: int,
) -> tuple[list[LatentSample], float]:
    if arm == "subject_shuffled_mri":
        return subject_shuffle_structural(samples, structural_indices, seed)
    if arm == "time_shuffled_mri":
        return time_shuffle_structural(samples, structural_indices, seed)
    return copy.deepcopy(samples), 0.0


def run(config_path: Path = CONFIG) -> Path:
    config = yaml.safe_load(config_path.read_text())
    if set(config["arms"]) != set(ARM_SETTINGS):
        raise ValueError("Config arms must match the frozen Phase 6B arm contract")
    contract_path = ROOT / config["item_contract"]
    items = yaml.safe_load(contract_path.read_text())["primary_state_items"]
    records = reconstruct_state_records(config, items)
    mri = aggregate_mri_features(records, config)
    samples = build_samples(records, mri, config)
    if not samples:
        raise RuntimeError("No Phase 6B forecast samples were constructed")
    raw_mri_metadata = {
        sample.sample_id: (
            float(sample.mri_features[3]) if sample.mri_available else np.nan,
            float(np.expm1(sample.mri_features[4])) if sample.mri_available else np.nan,
        )
        for sample in samples
    }
    output = ROOT / "outputs" / config["run_id"]; output.mkdir(parents=True, exist_ok=True)
    checkpoint_root = output / "checkpoints"; checkpoint_root.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_rows: list[dict[str, Any]] = []; manifests = []; control_rows = []
    structural = list(map(int, config["mri"]["structural_feature_indices"]))
    for fold in range(int(config["folds"])):
        evaluation = copy.deepcopy([sample for sample in samples if sample.fold == fold])
        outer_fit = copy.deepcopy([sample for sample in samples if sample.fold != fold])
        dev_subjects = {
            sample.subject_id for sample in outer_fit
            if int.from_bytes(hashlib.sha256(f"inner:{fold}:{sample.subject_id}".encode()).digest()[:8], "big") / 2**64
            < float(config["inner_dev_fraction"])
        }
        development = [sample for sample in outer_fit if sample.subject_id in dev_subjects]
        fit = [sample for sample in outer_fit if sample.subject_id not in dev_subjects]
        fit_subjects = {sample.subject_id for sample in fit}
        reference = fit_reference(records, fit_subjects, config)
        decoder = fit_clinical_decoder(records, fit_subjects)
        normalize_mri(fit, [fit, development, evaluation])
        for arm in config["arms"]:
            local_fit, fit_changed = arm_samples(arm, fit, structural, 10000 + fold)
            local_dev, dev_changed = arm_samples(arm, development, structural, 20000 + fold)
            local_eval, eval_changed = arm_samples(arm, evaluation, structural, 30000 + fold)
            model = train_model(
                arm, local_fit, local_dev, reference, decoder, config, device,
                20261029 + fold * 101,
            )
            model.eval()
            with torch.no_grad():
                batch = training_batch(local_eval, device)
                predicted, variance = model(batch, reference)
            all_rows.extend(prediction_rows(
                arm, local_eval, predicted.cpu().numpy(), variance.cpu().numpy(), decoder
            ))
            destination = checkpoint_root / arm; destination.mkdir(exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(), "fold": fold, "arm": arm,
                "reference": reference, "clinical_decoder": decoder,
                "best_dev_loss": model.best_dev_loss,
            }, destination / f"fold_{fold}.pt")
            control_rows.append({
                "fold": fold, "arm": arm, "fit_changed_fraction": fit_changed,
                "dev_changed_fraction": dev_changed, "evaluation_changed_fraction": eval_changed,
            })
        manifests.append({
            "fold": fold, "fit_subjects": len(fit_subjects), "dev_subjects": len(dev_subjects),
            "evaluation_subjects": len({sample.subject_id for sample in evaluation}),
            "fit_samples": len(fit), "dev_samples": len(development),
            "evaluation_samples": len(evaluation), "reference_intercept": reference[0],
            "reference_slope_per_year": reference[1], "decoder_intercept": decoder[0],
            "decoder_slope": decoder[1], "decoder_residual_variance": decoder[2],
        })
    predictions = pd.DataFrame(all_rows)
    predictions["mri_recency_years"] = predictions.sample_id.map(
        lambda sample_id: raw_mri_metadata[sample_id][0]
    )
    predictions["mri_scan_count"] = predictions.sample_id.map(
        lambda sample_id: raw_mri_metadata[sample_id][1]
    )
    predictions.to_csv(output / "crossfit_predictions.csv", index=False)
    pd.DataFrame(manifests).to_csv(output / "fold_manifest.csv", index=False)
    pd.DataFrame(control_rows).to_csv(output / "control_diagnostics.csv", index=False)
    aggregate = summarize(predictions, ["arm"])
    horizon = summarize(predictions, ["arm", "horizon_months"])
    folds = summarize(predictions, ["arm", "fold"])
    available = summarize(predictions[predictions.mri_available], ["arm"])
    aggregate.to_csv(output / "aggregate_metrics.csv", index=False)
    horizon.to_csv(output / "metrics_by_horizon.csv", index=False)
    folds.to_csv(output / "metrics_by_fold.csv", index=False)
    available.to_csv(output / "mri_available_metrics.csv", index=False)
    bootstrap = paired_bootstrap(predictions, config)
    bootstrap.to_csv(output / "paired_subject_bootstrap.csv", index=False)

    candidate = config["selection"]["candidate"]
    comparators = list(config["selection"]["comparators"])
    aggregate_table = aggregate.set_index("arm")
    fold_table = folds.pivot(index="fold", columns="arm", values="composite_mae")
    horizon_table = horizon.pivot(index="horizon_months", columns="arm", values="composite_mae")
    comparisons: dict[str, Any] = {}
    for comparator in comparators:
        comparisons[comparator] = {
            "overall_difference": float(aggregate_table.loc[candidate, "composite_mae"]
                                        - aggregate_table.loc[comparator, "composite_mae"]),
            "folds_better": int((fold_table[candidate] < fold_table[comparator]).sum()),
            "horizons_noninferior": int((horizon_table[candidate] - horizon_table[comparator]
                                         <= float(config["selection"]["maximum_horizon_degradation"])).sum()),
        }
    coverage_low, coverage_high = map(float, config["selection"]["coverage_95_bounds"])
    coverage = float(aggregate_table.loc[candidate, "coverage_95"])
    overall_pass = all(
        value["overall_difference"] < 0
        and value["folds_better"] >= int(config["selection"]["minimum_folds_better"])
        and value["horizons_noninferior"] >= int(config["selection"]["minimum_horizons_noninferior"])
        for value in comparisons.values()
    )
    coverage_pass = coverage_low <= coverage <= coverage_high
    decision = {
        "run_id": config["run_id"], "exploratory_only": True,
        "original_train_only": True, "validation_or_test_loaded": False,
        "candidate": candidate, "primary_endpoint": "future_observed_memory_composite_mae",
        "comparisons": comparisons, "candidate_coverage_95": coverage,
        "coverage_gate": coverage_pass, "mri_content_signal_supported": bool(overall_pass and coverage_pass),
        "disagreement_filter_open": bool(overall_pass and coverage_pass),
        "measurement_gate_reopened": False, "confirmatory_multimodal_claim_open": False,
    }
    (output / "development_decision.json").write_text(json.dumps(decision, indent=2) + "\n")

    bootstrap_primary = bootstrap[bootstrap.metric.eq("composite_absolute_error")].set_index("comparator")
    lines = [
        "# Phase 6B: MRI incremental-value and negative-control audit", "",
        "Original-train-only five-fold subject cross-fitting; validation and test were not loaded.", "",
        "## Observable clinical endpoint", "",
        "| Arm | Memory composite MAE | Clinical NLL | 95% coverage | Interval width | Latent MAE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate.itertuples(index=False):
        lines.append(f"| {row.arm} | {row.composite_mae:.5f} | {row.composite_nll:.4f} | "
                     f"{row.coverage_95:.4f} | {row.interval_width:.4f} | {row.latent_mae:.5f} |")
    lines.extend(["", "## Matched real-MRI effects", ""])
    for comparator in comparators:
        result = comparisons[comparator]; boot = bootstrap_primary.loc[comparator]
        lines.append(
            f"- real MRI minus `{comparator}` participant-mean composite error: **{boot.difference:+.6f}**; "
            f"participant-bootstrap 95% CI **[{boot.ci_low:+.6f}, {boot.ci_high:+.6f}]**; "
            f"folds better **{result['folds_better']}/5**; horizons non-inferior "
            f"**{result['horizons_noninferior']}/4**."
        )
    lines.extend([
        "", "## Decision", "",
        f"The real-MRI candidate clinical coverage was **{coverage:.4f}**. MRI-content support was "
        f"`{str(decision['mri_content_signal_supported']).lower()}` and the disagreement-aware "
        f"filter remains `{'open' if decision['disagreement_filter_open'] else 'closed'}`.", "",
        "This development audit does not reopen the Phase 5F measurement gate and cannot establish "
        "a confirmatory multimodal or continuous-time claim.",
    ])
    (output / "report.md").write_text("\n".join(lines) + "\n")
    manifest = {
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "phase5g_states_sha256": hashlib.sha256((ROOT / config["phase5g_states"]).read_bytes()).hexdigest(),
        "mri_source_sha256": hashlib.sha256((ROOT / config["mri_burden"]).read_bytes()).hexdigest(),
        "device": str(device), "state_records": len(records), "forecast_samples": len(samples),
        "arms": len(config["arms"]), "neural_checkpoints": len(config["arms"]) * int(config["folds"]),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args(); print(run(args.config))


if __name__ == "__main__":
    main()
