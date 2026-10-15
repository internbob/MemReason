"""Terminal validation-only Stage I development with task-specific gates."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn
from torch.nn import functional as F

from ..data.common import ROOT, read_csv, save_csv, save_json
from ..data.dictionary import load_dictionary
from .phase1_numeric import clean_frame, fit_normalization, markdown_table
from .phase1b_semantic import select_verified_scalar_features
from .phase1c_joint_stage1 import (
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
    encode_task_descriptors,
    feature_metadata,
)
from .phase1e_gated_aligned import GatedTypedQuestionStage1, train_one


class TaskGatedTypedQuestionStage1(GatedTypedQuestionStage1):
    """Per-task zero-initialized value and semantic residual gates."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        n_features = len(self.feature_types)
        self.semantic_gate = nn.Parameter(torch.zeros(len(TASK_GROUPS), n_features))
        self.value_gate = nn.Parameter(torch.zeros(len(TASK_GROUPS), n_features))

    def _dynamic_query(self, task_group_id: torch.Tensor) -> torch.Tensor:
        indices = torch.arange(len(self.feature_types), device=self.feature_types.device)
        base = self.item_embedding(indices) + self.type_embedding(self.feature_types)
        query = base[None].expand(len(task_group_id), -1, -1)
        if self.task_semantics_active:
            semantic = self.semantic_projection(self.task_descriptors)
            gate = torch.tanh(self.semantic_gate[task_group_id])
            query = query + gate.unsqueeze(-1) * semantic[None]
        return self.token_norm(query)

    def forward(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values, observed = batch["values"], batch["mask"]
        times, valid = batch["times"], batch["event_mask"]
        batch_size, steps, n_features = values.shape
        query = self._dynamic_query(batch["task_group_id"])
        token = query[:, None].expand(-1, steps, -1, -1)
        if self.typed_value_active:
            typed = torch.zeros_like(token)
            for type_id, encoder in enumerate(self.value_encoders):
                encoded = encoder(batch["severity"].unsqueeze(-1))
                typed = typed + encoded * self.feature_types.eq(type_id).view(1, 1, -1, 1)
            gate = torch.tanh(self.value_gate[batch["task_group_id"]])
            token = self.token_norm(token + gate[:, None, :, None] * typed)
        token = token * observed.unsqueeze(-1).to(values.dtype)
        domain_sum = torch.einsum("btfe,fd->btde", token, self.domain_assignment)
        domain_count = torch.einsum(
            "btf,fd->btd", observed.to(values.dtype), self.domain_assignment
        ).clamp_min(1.0)
        domain_summary = self.domain_aggregation(
            (domain_sum / domain_count.unsqueeze(-1)).flatten(-2)
        )

        hidden = [values.new_zeros(batch_size, cell.hidden_size) for cell in self.cells]
        last_value = values.new_zeros(batch_size, n_features)
        last_observed_time = times[:, :1].expand(-1, n_features).clone()
        previous_time = times[:, 0]
        for step in range(steps):
            event_valid = valid[:, step]
            current_time = times[:, step]
            elapsed = (current_time[:, None] - last_observed_time).clamp_min(0)
            gamma = torch.exp(
                -F.relu(self.decay_weight[None] * elapsed + self.decay_bias[None])
            )
            imputed = torch.where(observed[:, step], values[:, step], gamma * last_value)
            gap = (current_time - previous_time).clamp_min(0)
            cell_input = torch.cat(
                [
                    imputed,
                    observed[:, step].to(values.dtype),
                    domain_summary[:, step],
                    current_time[:, None],
                    gap[:, None],
                ],
                dim=-1,
            )
            for layer, cell in enumerate(self.cells):
                proposed = cell(cell_input, hidden[layer])
                hidden[layer] = torch.where(event_valid[:, None], proposed, hidden[layer])
                cell_input = self.dropout(hidden[layer])
            last_value = torch.where(
                observed[:, step] & event_valid[:, None], values[:, step], last_value
            )
            last_observed_time = torch.where(
                observed[:, step] & event_valid[:, None],
                current_time[:, None],
                last_observed_time,
            )
            previous_time = torch.where(event_valid, current_time, previous_time)

        task = self.task_embedding(batch["task_group_id"])
        horizon = self.horizon(batch["horizon"][:, None])
        total = self.total_head(torch.cat([hidden[-1], horizon, task], dim=-1))
        state = self.decoder_state(torch.cat([hidden[-1], task], dim=-1))[:, None]
        interaction = F.gelu(state + self.decoder_query(query))
        continuous = self.continuous_head(interaction).squeeze(-1)
        category = self.category_head(interaction)
        positions = torch.arange(category.shape[-1], device=category.device)
        invalid = positions.view(1, 1, -1) >= self.category_counts.view(1, -1, 1)
        category = category.masked_fill(invalid, -1e4)
        return total, continuous, category


def checkpoint_path(output: Path, arm: str, seed: int) -> Path:
    return output / "checkpoints" / arm / str(seed) / "best.pt"


def gate_by_task(
    output: Path,
    arm_name: str,
    arm: dict[str, Any],
    seed: int,
    task_descriptors: torch.Tensor,
    domain_assignment: torch.Tensor,
    feature_types: torch.Tensor,
    category_values: torch.Tensor,
    category_counts: torch.Tensor,
    targets: list[str],
    config: dict[str, Any],
) -> dict[str, dict[str, float]]:
    model = TaskGatedTypedQuestionStage1(
        task_descriptors,
        domain_assignment,
        feature_types,
        category_values,
        category_counts,
        len(targets),
        arm,
        config,
    )
    checkpoint = torch.load(checkpoint_path(output, arm_name, seed), map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    semantic = torch.tanh(model.semantic_gate.detach()).abs()
    value = torch.tanh(model.value_gate.detach()).abs()
    return {
        task: {
            "semantic_mean_abs": float(semantic[index].mean()),
            "semantic_max_abs": float(semantic[index].max()),
            "value_mean_abs": float(value[index].mean()),
            "value_max_abs": float(value[index].max()),
        }
        for index, task in enumerate(TASK_GROUPS)
    }


def write_report(
    output: Path,
    macro: pd.DataFrame,
    bootstrap: pd.DataFrame,
    selection: dict[str, float],
    decision: dict[str, Any],
    parameter_counts: dict[str, int],
    selected_gates: dict[str, dict[str, float]],
) -> None:
    text = f"""# Phase 1F task-gated representation development results

Status: completed terminal validation-only representation development. Test was not loaded.

## Contract

- Trainable parameter counts: {parameter_counts}
- Equal-task validation scores: {selection}
- Selected semantic arm gate summaries by task: {selected_gates}

## Validation macro normalized MAE

{markdown_table(macro)}

## Subject-cluster bootstrap against numeric task-gated baseline

Negative normalized-MAE differences favor the candidate.

{markdown_table(bootstrap)}

## Provisional decision

{json.dumps(decision, indent=2)}

This was the terminal architecture revision on the current validation set. Failure stops further semantic tuning on this split.
"""
    (output / "report.md").write_text(text)


def run(config_path: Path, smoke: bool = False) -> None:
    config = load_config(config_path)
    if smoke:
        config = copy.deepcopy(config)
        config["arms"] = {
            "numeric_task_gated": config["arms"]["numeric_task_gated"],
            "typed_semantic_task_gated": config["arms"]["typed_semantic_task_gated"],
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

    validation_frames, parameter_counts = [], {}
    for arm_name, arm in config["arms"].items():
        counts = []
        for seed in config["neural_seeds"]:
            frame, count, _ = train_one(
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
                model_class=TaskGatedTypedQuestionStage1,
            )
            validation_frames.append(frame)
            counts.append(count)
        parameter_counts[arm_name] = counts[0]
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

    baseline_name = "numeric_task_gated"
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
    selected = "typed_semantic_task_gated"
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
    passed = bool(all(criteria.values()))
    decision = {
        "run_id": config["run_id"],
        "engineering_smoke_only": smoke,
        "selected_semantic_candidate": selected,
        "criteria": criteria,
        "provisional_validation_passed": passed,
        "test_evaluated": False,
        "advance_to_mri_stage2": False,
        "terminal_current_validation_revision": True,
        "next_action": (
            "freeze_three_seed_confirmation_protocol"
            if passed
            else "stop_semantic_tuning_and_reframe_or_obtain_new_development_cohort"
        ),
    }
    save_json(decision, output / "development_decision.json")
    selected_gates = gate_by_task(
        output,
        selected,
        config["arms"][selected],
        config["neural_seeds"][0],
        task_descriptors,
        domain_assignment,
        feature_types,
        category_values,
        category_counts,
        targets,
        config,
    )
    save_json(selected_gates, output / "selected_gate_summary.json")
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
            "task_specific_value_gate": True,
            "task_specific_semantic_gate": True,
            "terminal_current_validation_revision": True,
        },
        output / "manifest.json",
    )
    write_report(
        output,
        macro,
        bootstrap,
        selection,
        decision,
        parameter_counts,
        selected_gates,
    )
    save_json(
        {
            "completed": True,
            "development_only": True,
            "engineering_smoke_only": smoke,
            "test_evaluated": False,
            "provisional_validation_passed": passed,
            "report": str(output / "report.md"),
        },
        output / "status.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/phase1f_task_gated_dev.yaml"
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.config, args.smoke)


if __name__ == "__main__":
    main()
