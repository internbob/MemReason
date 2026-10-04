"""Formal classical and deep cognitive forecasting baseline suite."""
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
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
import yaml

from ..data.common import ROOT, read_csv, save_csv, save_json
from ..data.dictionary import load_dictionary
from .phase1_numeric import (
    GRUDForecaster,
    MixedEffectsForecaster,
    NumericCollator,
    NumericForecastDataset,
    TimeAwareTransformerForecaster,
    build_samples,
    clean_frame,
    compute_metrics,
    ensemble_predictions,
    evaluate_model,
    fit_lme_models,
    fit_normalization,
    lme_predictions,
    locf_predictions,
    macro_score,
    markdown_table,
    masked_huber,
)
from .phase1b_semantic import seed_everything, select_verified_scalar_features


class RecurrentForecaster(nn.Module):
    """GRU/LSTM baseline over value and observation-mask event vectors."""

    def __init__(
        self,
        n_features: int,
        n_targets: int,
        hidden: int,
        layers: int,
        dropout: float,
        cell: str,
    ):
        super().__init__()
        recurrent = nn.GRU if cell == "GRU" else nn.LSTM
        self.recurrent = recurrent(
            n_features * 2,
            hidden,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.horizon = nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.output = nn.Linear(hidden * 2, n_targets)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = torch.cat([batch["values"], batch["mask"].to(batch["values"].dtype)], dim=-1)
        lengths = batch["event_mask"].sum(1).to("cpu")
        packed = pack_padded_sequence(tokens, lengths, batch_first=True, enforce_sorted=False)
        _, state = self.recurrent(packed)
        hidden = state[0] if isinstance(state, tuple) else state
        summary = hidden[-1]
        horizon = self.horizon(batch["horizon"][:, None])
        return self.output(torch.cat([summary, horizon], dim=-1))


class TimeAwareLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.short_term = nn.Linear(hidden, hidden)
        self.gates = nn.Linear(input_dim + hidden, hidden * 4)

    def forward(
        self,
        inputs: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
        elapsed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        short = torch.tanh(self.short_term(cell))
        adjusted = cell - short + short * torch.exp(-elapsed[:, None].clamp_min(0))
        input_gate, forget_gate, output_gate, candidate = self.gates(
            torch.cat([inputs, hidden], dim=-1)
        ).chunk(4, dim=-1)
        cell_new = torch.sigmoid(forget_gate) * adjusted + torch.sigmoid(input_gate) * torch.tanh(candidate)
        hidden_new = torch.sigmoid(output_gate) * torch.tanh(cell_new)
        return hidden_new, cell_new


class TimeAwareLSTMForecaster(nn.Module):
    """T-LSTM with elapsed-time decay of the short-term cell component."""

    def __init__(self, n_features: int, n_targets: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.cells = nn.ModuleList(
            [TimeAwareLSTMCell(n_features * 2, hidden)]
            + [TimeAwareLSTMCell(hidden, hidden) for _ in range(layers - 1)]
        )
        self.dropout = nn.Dropout(dropout)
        self.horizon = nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.output = nn.Linear(hidden * 2, n_targets)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        values, observed = batch["values"], batch["mask"]
        times, valid = batch["times"], batch["event_mask"]
        batch_size, steps, _ = values.shape
        hidden = [values.new_zeros(batch_size, cell.hidden) for cell in self.cells]
        memory = [values.new_zeros(batch_size, cell.hidden) for cell in self.cells]
        previous_time = times[:, 0]
        for step in range(steps):
            event_valid = valid[:, step]
            current_time = times[:, step]
            elapsed = (current_time - previous_time).clamp_min(0)
            layer_input = torch.cat([values[:, step], observed[:, step].to(values.dtype)], dim=-1)
            for layer, cell in enumerate(self.cells):
                proposed_h, proposed_c = cell(layer_input, hidden[layer], memory[layer], elapsed)
                hidden[layer] = torch.where(event_valid[:, None], proposed_h, hidden[layer])
                memory[layer] = torch.where(event_valid[:, None], proposed_c, memory[layer])
                layer_input = self.dropout(hidden[layer])
            previous_time = torch.where(event_valid, current_time, previous_time)
        horizon = self.horizon(batch["horizon"][:, None])
        return self.output(torch.cat([hidden[-1], horizon], dim=-1))


def build_model(
    architecture: str,
    n_features: int,
    n_targets: int,
    hidden: int,
    layers: int,
    dropout: float,
) -> nn.Module:
    if architecture in {"GRU", "LSTM"}:
        return RecurrentForecaster(n_features, n_targets, hidden, layers, dropout, architecture)
    if architecture == "T-LSTM":
        return TimeAwareLSTMForecaster(n_features, n_targets, hidden, layers, dropout)
    if architecture == "GRU-D":
        return GRUDForecaster(n_features, n_targets, hidden, layers, dropout)
    if architecture == "Transformer":
        return TimeAwareTransformerForecaster(n_features, n_targets, hidden, layers, dropout)
    raise ValueError(f"Unknown architecture: {architecture}")


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def checkpoint_path(output: Path, architecture: str, seed: int) -> Path:
    name = architecture.lower().replace("-", "_")
    return output / "checkpoints" / name / str(seed) / "best.pt"


def train_one(
    architecture: str,
    seed: int,
    train_samples: list,
    val_samples: list,
    features: list[str],
    targets: list[str],
    normalization: dict[str, dict[str, float]],
    config: dict[str, Any],
    output: Path,
    device: torch.device,
) -> tuple[pd.DataFrame, int]:
    seed_everything(seed)
    settings = config["training"]
    collator = NumericCollator(features, targets, normalization)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        NumericForecastDataset(train_samples),
        batch_size=settings["batch_size"],
        shuffle=True,
        collate_fn=collator,
        generator=generator,
        num_workers=0,
    )
    model = build_model(
        architecture,
        len(features),
        len(targets),
        settings["hidden_dim"],
        settings["layers"],
        settings["dropout"],
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    best_score, best_state, patience = float("inf"), None, 0
    curves = []
    for epoch in range(settings["max_epochs"]):
        model.train()
        losses = []
        for batch in loader:
            batch.pop("samples")
            batch = {key: value.to(device) for key, value in batch.items()}
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
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_macro_normalized_mae": score,
        }
        curves.append(record)
        print(json.dumps({"model": architecture, "seed": seed, **record}), flush=True)
        if score < best_score - 1e-5:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= settings["patience"]:
                break
    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {architecture}/{seed}")
    model.load_state_dict(best_state)
    path = checkpoint_path(output, architecture, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": best_state,
            "architecture": architecture,
            "seed": seed,
            "features": features,
            "targets": targets,
            "trainable_parameters": parameter_count,
            "validation_macro_normalized_mae": best_score,
        },
        path,
    )
    save_csv(pd.DataFrame(curves), path.parent / "curves.csv")
    validation = evaluate_model(
        model, val_samples, collator, settings["batch_size"], device, targets, architecture, seed
    )
    return validation, parameter_count


def evaluate_checkpoint(
    architecture: str,
    seed: int,
    samples: list,
    features: list[str],
    targets: list[str],
    normalization: dict[str, dict[str, float]],
    config: dict[str, Any],
    output: Path,
    device: torch.device,
) -> pd.DataFrame:
    settings = config["training"]
    record = torch.load(checkpoint_path(output, architecture, seed), map_location="cpu", weights_only=True)
    if record["features"] != features or record["targets"] != targets:
        raise RuntimeError("Checkpoint data contract mismatch")
    model = build_model(
        architecture,
        len(features),
        len(targets),
        settings["hidden_dim"],
        settings["layers"],
        settings["dropout"],
    ).to(device)
    model.load_state_dict(record["model"])
    return evaluate_model(
        model,
        samples,
        NumericCollator(features, targets, normalization),
        settings["batch_size"],
        device,
        targets,
        architecture,
        seed,
    )


def bootstrap_normalized_difference(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    normalization: dict[str, dict[str, float]],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    keys = ["sample_id", "subject_id", "target", "horizon_months", "truth"]
    paired = candidate[keys + ["prediction"]].merge(
        baseline[keys + ["prediction"]],
        on=keys,
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    ).dropna()
    scale = paired.target.map(lambda target: normalization[target]["std"]).to_numpy()
    paired["difference"] = (
        np.abs(paired.prediction_candidate - paired.truth)
        - np.abs(paired.prediction_baseline - paired.truth)
    ) / scale
    subject = paired.groupby("subject_id").difference.agg(["sum", "count"])
    sums, counts = subject["sum"].to_numpy(), subject["count"].to_numpy()
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(subject), size=(replicates, len(subject)))
    draws = sums[indices].sum(1) / counts[indices].sum(1)
    return {
        "candidate": str(candidate.model.iloc[0]),
        "baseline": str(baseline.model.iloc[0]),
        "n_paired_targets": int(len(paired)),
        "n_subjects": int(len(subject)),
        "normalized_mae_difference": float(paired.difference.mean()),
        "ci_lower": float(np.quantile(draws, 0.025)),
        "ci_upper": float(np.quantile(draws, 0.975)),
        "bootstrap_probability_candidate_better": float(np.mean(draws < 0)),
        "replicates": replicates,
    }


def write_report(
    output: Path,
    config: dict[str, Any],
    features: list[str],
    sample_counts: dict[str, int],
    selection: dict[str, Any],
    test_metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    parameter_counts: dict[str, int],
) -> None:
    reported = ["LOCF", "LME"] + [name + " ensemble" for name in config["architectures"]]
    macro = (
        test_metrics[test_metrics.model.isin(reported)]
        .groupby("model", as_index=False)
        .agg(normalized_mae=("normalized_mae", "mean"), coverage=("coverage", "mean"))
        .sort_values("normalized_mae")
    )
    text = f"""# Formal cognitive forecasting baseline suite

Status: completed under the frozen baseline protocol.

## Data and training

- Dictionary-verified scalar inputs: {len(features)}
- Forecast samples: {sample_counts}
- Neural seeds: {config['neural_seeds']}
- Trainable parameter counts: {parameter_counts}
- Validation-selected strongest neural architecture: **{selection['selected_architecture']}**
- Test data were loaded only after all validation-trained checkpoints completed.

## Test macro normalized MAE

{markdown_table(macro)}

## Validation seed scores

{markdown_table(pd.DataFrame(selection['summary']))}

## Paired subject bootstrap

Negative normalized-MAE differences favor the neural ensemble.

{markdown_table(bootstrap)}

## Interpretation limits

- Time is the nominal ADNI visit month, not a verified MRI acquisition date.
- LOCF coverage can be below one when no prior observation of a target scale exists.
- MoCA 6-month targets are sparse and must not be interpreted alone.
- This suite contains no semantic, MRI, probabilistic-belief, SFT or RL input.
- Test results cannot be used to alter this run ID or tune these baselines.
"""
    (output / "report.md").write_text(text)


def run(config_path: Path, smoke: bool = False) -> None:
    config = load_config(config_path)
    if smoke:
        config = copy.deepcopy(config)
        config["neural_seeds"] = [config["neural_seeds"][0]]
        config["training"].update(max_epochs=2, patience=2, batch_size=64)
    output = ROOT / "outputs" / (config["run_id"] + ("_smoke" if smoke else ""))
    output.mkdir(parents=True, exist_ok=True)
    dictionary = load_dictionary()
    features = select_verified_scalar_features(config, dictionary)
    targets = list(config["targets"])
    train_val = {
        split: clean_frame(read_csv(ROOT / f"splits/{split}.csv"), features, targets, dictionary)
        for split in ["train", "val"]
    }
    normalization = fit_normalization(train_val["train"], features, targets)
    horizons = config["future_forecast"]["horizons_months"]
    tolerances = {
        int(key): int(value) for key, value in config["future_forecast"]["tolerance_months"].items()
    }
    samples = {
        split: build_samples(frame, split, features, targets, horizons, tolerances)
        for split, frame in train_val.items()
    }
    if smoke:
        samples = {key: value[: min(256, len(value))] for key, value in samples.items()}

    lme_models: list[MixedEffectsForecaster] = fit_lme_models(train_val["train"], targets)
    classical_validation = [locf_predictions(samples["val"], targets), lme_predictions(lme_models, samples["val"], targets)]
    save_json({"models": [model.diagnostics for model in lme_models]}, output / "lme_diagnostics.json")

    device = torch.device("cuda" if torch.cuda.is_available() and not smoke else "cpu")
    validation_frames = []
    parameter_counts = {}
    for architecture in config["architectures"]:
        counts = []
        for seed in config["neural_seeds"]:
            frame, count = train_one(
                architecture,
                seed,
                samples["train"],
                samples["val"],
                features,
                targets,
                normalization,
                config,
                output,
                device,
            )
            validation_frames.append(frame)
            counts.append(count)
        if len(set(counts)) != 1:
            raise RuntimeError(f"Parameter count changed across seeds for {architecture}: {counts}")
        parameter_counts[architecture] = counts[0]

    validation = pd.concat(classical_validation + validation_frames, ignore_index=True)
    for architecture in config["architectures"]:
        validation = pd.concat([validation, ensemble_predictions(validation, architecture)], ignore_index=True)
    save_csv(validation, output / "validation_predictions.csv")
    validation_metrics = compute_metrics(validation, normalization)
    save_csv(validation_metrics, output / "validation_metrics.csv")

    summary = []
    architecture_scores = {}
    for architecture in config["architectures"]:
        scores = {
            str(int(seed)): macro_score(group, normalization)
            for seed, group in validation[validation.model.eq(architecture)].groupby("seed")
        }
        values = np.asarray(list(scores.values()))
        architecture_scores[architecture] = scores
        summary.append(
            {
                "architecture": architecture,
                "validation_mean": float(values.mean()),
                "validation_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
        )
    selected = min(summary, key=lambda row: row["validation_mean"])["architecture"]
    selection = {
        "metric": config["selection"]["metric"],
        "seed_scores": architecture_scores,
        "summary": summary,
        "selected_architecture": selected,
    }
    save_json(selection, output / "selection.json")
    manifest = {
        "run_id": config["run_id"],
        "engineering_smoke_only": smoke,
        "features": features,
        "targets": targets,
        "architectures": config["architectures"],
        "parameter_counts": parameter_counts,
        "sample_counts_before_test": {key: len(value) for key, value in samples.items()},
        "dictionary_verified": all(dictionary[feature]["verified_from_dictionary"] for feature in features),
        "diagnosis_input": False,
        "test_evaluated": False,
        "deterministic_algorithms": True,
        "cuda_attention_backend": "math_sdp",
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }
    save_json(manifest, output / "manifest.json")
    save_json(config, output / "resolved_config.json")
    if smoke:
        save_json(
            {"completed": True, "engineering_smoke_only": True, "test_evaluated": False},
            output / "status.json",
        )
        return

    test_frame = clean_frame(read_csv(ROOT / "splits/test.csv"), features, targets, dictionary)
    test_samples = build_samples(test_frame, "test", features, targets, horizons, tolerances)
    test_frames = [locf_predictions(test_samples, targets), lme_predictions(lme_models, test_samples, targets)]
    for architecture in config["architectures"]:
        for seed in config["neural_seeds"]:
            test_frames.append(
                evaluate_checkpoint(
                    architecture,
                    seed,
                    test_samples,
                    features,
                    targets,
                    normalization,
                    config,
                    output,
                    device,
                )
            )
    test = pd.concat(test_frames, ignore_index=True)
    for architecture in config["architectures"]:
        test = pd.concat([test, ensemble_predictions(test, architecture)], ignore_index=True)
    save_csv(test, output / "test_predictions.csv")
    test_metrics = compute_metrics(test, normalization)
    save_csv(test_metrics, output / "test_metrics.csv")

    bootstrap_rows = []
    for architecture in config["architectures"]:
        candidate = test[test.model.eq(architecture + " ensemble")]
        for baseline in ["LOCF", "LME"]:
            bootstrap_rows.append(
                bootstrap_normalized_difference(
                    candidate,
                    test[test.model.eq(baseline)],
                    normalization,
                    config["statistics"]["bootstrap_replicates"],
                    config["seed"],
                )
            )
    bootstrap = pd.DataFrame(bootstrap_rows)
    save_csv(bootstrap, output / "paired_subject_bootstrap.csv")
    sample_counts = {"train": len(samples["train"]), "val": len(samples["val"]), "test": len(test_samples)}
    manifest.update(sample_counts=sample_counts, test_evaluated=True)
    save_json(manifest, output / "manifest.json")
    write_report(output, config, features, sample_counts, selection, test_metrics, bootstrap, parameter_counts)
    save_json(
        {
            "run_id": config["run_id"],
            "completed": True,
            "engineering_smoke_only": False,
            "test_evaluated": True,
            "selected_architecture": selected,
            "report": str(output / "report.md"),
        },
        output / "status.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/phase1_baseline_suite_formal.yaml"
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.config, args.smoke)


if __name__ == "__main__":
    main()
