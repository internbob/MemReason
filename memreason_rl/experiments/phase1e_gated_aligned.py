"""Validation-only Phase 1E gated semantic and metric-aligned loss study."""

from __future__ import annotations

import argparse
import copy
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from ..data.common import ROOT, read_csv, save_csv, save_json
from ..data.dictionary import load_dictionary
from .phase1_numeric import clean_frame, fit_normalization, markdown_table
from .phase1b_semantic import seed_everything, select_verified_scalar_features
from .phase1c_joint_stage1 import (
    JointDataset,
    JointSample,
    bootstrap_difference,
    build_domain_assignment,
    build_joint_samples,
    equal_task_macro_score,
    flatten_tasks,
    load_config,
    metrics,
)
from .phase1d_typed_representation import (
    TASK_GROUPS,
    TYPE_IDS,
    TypedCollator,
    TypedQuestionStage1,
    encode_task_descriptors,
    evaluate,
    feature_metadata,
)


class GatedTypedQuestionStage1(TypedQuestionStage1):
    """Semantic residual starts at zero while preserving matched parameters."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.semantic_gate = nn.Parameter(torch.zeros(len(self.feature_types)))

    def _static_query(self) -> torch.Tensor:
        indices = torch.arange(len(self.feature_types), device=self.feature_types.device)
        query = self.item_embedding(indices) + self.type_embedding(self.feature_types)
        if self.task_semantics_active:
            semantic = self.semantic_projection(self.task_descriptors)
            query = query + torch.tanh(self.semantic_gate)[:, None] * semantic
        return self.token_norm(query)


def metric_aligned_loss(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    batch: dict[str, torch.Tensor],
    model: GatedTypedQuestionStage1,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    ce_weight: float,
    huber_weight: float,
) -> torch.Tensor:
    total, continuous, category = outputs
    losses = []
    for group_id in batch["task_group_id"].unique():
        rows = batch["task_group_id"].eq(group_id)
        if int(group_id) < 3:
            mask = batch["total_mask"] & rows[:, None]
            if mask.any():
                losses.append(F.huber_loss(total[mask], batch["total_targets"][mask]))
            continue
        target_mask = batch["reconstruction_mask"] & rows[:, None]
        n_features = target_mask.shape[1]
        item_sum = total.new_zeros(n_features)
        item_count = total.new_zeros(n_features)
        continuous_mask = target_mask & model.feature_types.eq(TYPE_IDS["continuous"])[None]
        if continuous_mask.any():
            positions = continuous_mask.nonzero(as_tuple=False)
            feature_indices = positions[:, 1]
            per_target = F.huber_loss(
                continuous[continuous_mask],
                batch["reconstruction_targets"][continuous_mask],
                reduction="none",
            )
            item_sum.scatter_add_(0, feature_indices, per_target)
            item_count.scatter_add_(0, feature_indices, torch.ones_like(per_target))
        category_mask = target_mask & model.feature_types.ne(TYPE_IDS["continuous"])[None]
        if category_mask.any():
            positions = category_mask.nonzero(as_tuple=False)
            feature_indices = positions[:, 1]
            logits = category[category_mask]
            labels = batch["reconstruction_category"][category_mask]
            ce = F.cross_entropy(logits, labels, reduction="none")
            expected_raw = (
                logits.softmax(-1) * model.category_values[feature_indices]
            ).sum(-1)
            expected_normalized = (
                expected_raw - feature_mean[feature_indices]
            ) / feature_std[feature_indices].clamp_min(1e-6)
            aligned = F.huber_loss(
                expected_normalized,
                batch["reconstruction_targets"][category_mask],
                reduction="none",
            )
            per_target = ce_weight * ce + huber_weight * aligned
            item_sum.scatter_add_(0, feature_indices, per_target)
            item_count.scatter_add_(0, feature_indices, torch.ones_like(per_target))
        available = item_count > 0
        if available.any():
            losses.append((item_sum[available] / item_count[available]).mean())
    return torch.stack(losses).mean() if losses else total.sum() * 0


def make_loader(
    tasks: dict[str, list[JointSample]],
    collator: TypedCollator,
    batch_size: int,
    seed: int,
) -> DataLoader:
    samples = flatten_tasks(tasks)
    counts = Counter(sample.task for sample in samples)
    weights = torch.tensor([1.0 / counts[sample.task] for sample in samples], dtype=torch.double)
    sampler = WeightedRandomSampler(
        weights,
        len(samples),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    return DataLoader(
        JointDataset(samples),
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=0,
    )


def checkpoint_path(output: Path, arm: str, seed: int) -> Path:
    return output / "checkpoints" / arm / str(seed) / "best.pt"


def train_one(
    arm_name: str,
    arm: dict[str, Any],
    seed: int,
    task_descriptors: torch.Tensor,
    domain_assignment: torch.Tensor,
    feature_types: torch.Tensor,
    category_values: torch.Tensor,
    category_counts: torch.Tensor,
    train_tasks: dict[str, list[JointSample]],
    val_samples: list[JointSample],
    collator: TypedCollator,
    features: list[str],
    targets: list[str],
    normalization: dict[str, dict[str, float]],
    config: dict[str, Any],
    output: Path,
    device: torch.device,
    model_class: type[GatedTypedQuestionStage1] = GatedTypedQuestionStage1,
) -> tuple[pd.DataFrame, int, dict[str, float]]:
    seed_everything(seed)
    settings = config["training"]
    model = model_class(
        task_descriptors,
        domain_assignment,
        feature_types,
        category_values,
        category_counts,
        len(targets),
        arm,
        config,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    path = checkpoint_path(output, arm_name, seed)
    if path.exists() and (path.parent / "curves.csv").exists():
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["features"] != features or checkpoint["targets"] != targets:
            raise RuntimeError(f"Checkpoint contract mismatch for {arm_name}/{seed}")
        model.load_state_dict(checkpoint["model"])
        gate = torch.tanh(model.semantic_gate.detach()).cpu().numpy()
        return (
            evaluate(
                model,
                val_samples,
                collator,
                settings["batch_size"],
                device,
                arm_name,
                seed,
                features,
                targets,
            ),
            parameter_count,
            {"mean_abs": float(np.abs(gate).mean()), "max_abs": float(np.abs(gate).max())},
        )

    loader = make_loader(train_tasks, collator, settings["batch_size"], seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )
    feature_mean = torch.tensor(collator.feature_mean, device=device)
    feature_std = torch.tensor(collator.feature_std, device=device)
    best_score, best_state, patience, curves = float("inf"), None, 0, []
    for epoch in range(settings["max_epochs"]):
        model.train()
        losses = []
        for batch in loader:
            batch.pop("samples")
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            loss = metric_aligned_loss(
                outputs,
                batch,
                model,
                feature_mean,
                feature_std,
                settings["categorical_ce_weight"],
                settings["categorical_normalized_huber_weight"],
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss for {arm_name}/{seed}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"])
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = evaluate(
            model,
            val_samples,
            collator,
            settings["batch_size"],
            device,
            arm_name,
            seed,
            features,
            targets,
        )
        score = equal_task_macro_score(validation, normalization)
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_equal_task_macro_normalized_mae": score,
        }
        curves.append(record)
        print(json.dumps({"arm": arm_name, "seed": seed, **record}), flush=True)
        if score < best_score - 1e-5:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= settings["patience"]:
                break
    if best_state is None:
        raise RuntimeError(f"No checkpoint for {arm_name}/{seed}")
    model.load_state_dict(best_state)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": best_state,
            "arm": arm_name,
            "seed": seed,
            "features": features,
            "targets": targets,
            "parameter_count": parameter_count,
            "validation_score": best_score,
        },
        path,
    )
    save_csv(pd.DataFrame(curves), path.parent / "curves.csv")
    gate = torch.tanh(model.semantic_gate.detach()).cpu().numpy()
    gate_summary = {
        "mean_abs": float(np.abs(gate).mean()),
        "max_abs": float(np.abs(gate).max()),
    }
    return (
        evaluate(
            model,
            val_samples,
            collator,
            settings["batch_size"],
            device,
            arm_name,
            seed,
            features,
            targets,
        ),
        parameter_count,
        gate_summary,
    )


def write_report(
    output: Path,
    macro: pd.DataFrame,
    bootstrap: pd.DataFrame,
    selection: dict[str, float],
    decision: dict[str, Any],
    parameter_counts: dict[str, int],
    gate_summaries: dict[str, dict[str, float]],
) -> None:
    text = f"""# Phase 1E gated aligned representation development results

Status: completed validation-only development study. The test split was not loaded.

## Contract

- Trainable parameter counts: {parameter_counts}
- Equal-task validation scores: {selection}
- Learned tanh semantic-gate summaries: {gate_summaries}

## Validation macro normalized MAE

{markdown_table(macro)}

## Subject-cluster bootstrap against numeric-query aligned baseline

Negative normalized-MAE differences favor the candidate.

{markdown_table(bootstrap)}

## Provisional decision

{json.dumps(decision, indent=2)}

The run is development-only. A pass permits a new three-seed confirmation protocol, not MRI Stage II or a clean external-confirmation claim.
"""
    (output / "report.md").write_text(text)


def run(config_path: Path, smoke: bool = False) -> None:
    config = load_config(config_path)
    if smoke:
        config = copy.deepcopy(config)
        config["arms"] = {
            "numeric_query_aligned": config["arms"]["numeric_query_aligned"],
            "typed_semantic_gated_aligned": config["arms"]["typed_semantic_gated_aligned"],
        }
        config["training"].update(max_epochs=2, patience=2, batch_size=64)
        config["statistics"]["bootstrap_replicates"] = 50
    output = ROOT / "outputs" / (config["run_id"] + ("_smoke" if smoke else ""))
    output.mkdir(parents=True, exist_ok=True)
    resolved = output / "resolved_config.json"
    if resolved.exists() and json.loads(resolved.read_text()) != config:
        raise RuntimeError(f"Frozen config changed for {output}")
    if not resolved.exists():
        save_json(config, resolved)

    dictionary = load_dictionary()
    features = select_verified_scalar_features(config, dictionary)
    targets = list(config["targets"])
    frames = {
        split: clean_frame(read_csv(ROOT / f"splits/{split}.csv"), features, targets, dictionary)
        for split in ["train", "val"]
    }
    normalization = fit_normalization(frames["train"], features, targets)
    tasks = {
        split: build_joint_samples(frame, split, features, targets, dictionary, config)
        for split, frame in frames.items()
    }
    if smoke:
        for split_tasks in tasks.values():
            for task in split_tasks:
                split_tasks[task] = split_tasks[task][: min(96, len(split_tasks[task]))]
    val_samples = flatten_tasks(tasks["val"])
    collator = TypedCollator(features, targets, normalization, dictionary)
    domains, domain_assignment = build_domain_assignment(features, dictionary)
    feature_types, _, _, category_values, category_counts = feature_metadata(features, dictionary)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_descriptors = encode_task_descriptors(features, dictionary, config, output, device)

    validation_frames, parameter_counts, gate_summaries = [], {}, {}
    for arm_name, arm in config["arms"].items():
        arm_counts, arm_gates = [], []
        for seed in config["neural_seeds"]:
            frame, count, gate = train_one(
                arm_name,
                arm,
                seed,
                task_descriptors,
                domain_assignment,
                feature_types,
                category_values,
                category_counts,
                tasks["train"],
                val_samples,
                collator,
                features,
                targets,
                normalization,
                config,
                output,
                device,
            )
            validation_frames.append(frame)
            arm_counts.append(count)
            arm_gates.append(gate)
        parameter_counts[arm_name] = arm_counts[0]
        gate_summaries[arm_name] = {
            key: float(np.mean([record[key] for record in arm_gates])) for key in arm_gates[0]
        }
    if len(set(parameter_counts.values())) != 1:
        raise RuntimeError(f"Arms are not parameter matched: {parameter_counts}")

    validation = pd.concat(validation_frames, ignore_index=True)
    save_csv(validation, output / "validation_predictions.csv")
    validation_metrics = metrics(validation, normalization)
    save_csv(validation_metrics, output / "validation_metrics.csv")
    macro = validation_metrics.groupby(["task", "model"], as_index=False).normalized_mae.mean()
    selection = {
        arm: equal_task_macro_score(validation[validation.model.eq(arm)], normalization)
        for arm in config["arms"]
    }
    save_json(selection, output / "validation_selection.json")

    baseline_name = "numeric_query_aligned"
    baseline = validation[validation.model.eq(baseline_name)]
    bootstrap_rows = []
    for arm in config["arms"]:
        if arm == baseline_name:
            continue
        for task in TASK_GROUPS:
            bootstrap_rows.append(
                bootstrap_difference(
                    validation[validation.model.eq(arm) & validation.task.eq(task)],
                    baseline[baseline.task.eq(task)],
                    normalization,
                    config["statistics"]["bootstrap_replicates"],
                    config["seed"],
                )
            )
    bootstrap = pd.DataFrame(bootstrap_rows)
    save_csv(bootstrap, output / "paired_subject_bootstrap.csv")
    selected = "typed_semantic_gated_aligned"
    rows = bootstrap[bootstrap.candidate.eq(selected)].set_index("task")
    criteria = {
        "masked_item_improved": bool(rows.loc["masked_item", "ci_upper"] < 0),
        "masked_instrument_improved": bool(rows.loc["masked_instrument", "ci_upper"] < 0),
        "cross_scale_two_pairs_improved": bool(
            rows.loc[
                ["cross_adas_cdr_to_moca", "cross_mmse_cdr_to_adas"], "ci_upper"
            ].lt(0).all()
        ),
        "future_noninferior": bool(
            rows.loc["future_forecast", "ci_upper"]
            <= config["gate"]["future_noninferiority_margin_normalized_mae"]
        ),
    }
    decision = {
        "run_id": config["run_id"],
        "engineering_smoke_only": smoke,
        "selected_semantic_candidate": selected,
        "criteria": criteria,
        "provisional_validation_passed": bool(all(criteria.values())),
        "test_evaluated": False,
        "advance_to_mri_stage2": False,
        "next_action": (
            "freeze_three_seed_confirmation_protocol"
            if all(criteria.values())
            else "revise_or_reframe_on_train_validation_only"
        ),
    }
    save_json(decision, output / "development_decision.json")
    save_json(
        {
            "run_id": config["run_id"],
            "development_only": True,
            "engineering_smoke_only": smoke,
            "test_evaluated": False,
            "features": features,
            "targets": targets,
            "domains": domains,
            "parameter_counts": parameter_counts,
            "gate_summaries": gate_summaries,
            "item_macro_reconstruction_loss": True,
            "categorical_metric_alignment": True,
            "zero_initialized_semantic_gate": True,
        },
        output / "manifest.json",
    )
    write_report(
        output, macro, bootstrap, selection, decision, parameter_counts, gate_summaries
    )
    save_json(
        {
            "completed": True,
            "development_only": True,
            "engineering_smoke_only": smoke,
            "test_evaluated": False,
            "provisional_validation_passed": decision["provisional_validation_passed"],
            "report": str(output / "report.md"),
        },
        output / "status.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/phase1e_gated_aligned_dev.yaml",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.config, args.smoke)


if __name__ == "__main__":
    main()
