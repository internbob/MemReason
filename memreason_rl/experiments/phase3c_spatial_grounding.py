"""Validation-only concurrent target-to-spatial MRI grounding experiment."""

from __future__ import annotations

import argparse
import copy
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import rankdata
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..data.common import ROOT, save_csv, save_json
from .phase1_numeric import markdown_table, seed_everything
from .phase3b_mri_representation import (
    ConcurrentRegressor,
    bootstrap_difference,
    build_concurrent_frame,
    ensemble,
    macro_huber,
    make_shuffle_indices,
    metrics,
    normalized_macro,
)


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def prepare_arrays(frame: pd.DataFrame, target_names: list[str], phase3b: Path):
    normalization = json.loads((phase3b / "normalization.json").read_text())
    target_mean = np.asarray([normalization["target_mean"][name] for name in target_names], dtype=np.float32)
    target_std = np.asarray([normalization["target_std"][name] for name in target_names], dtype=np.float32)
    age_mean = float(normalization["age_mean"])
    age_std = float(normalization["age_std"])
    demos = np.stack(
        [(frame.age.to_numpy() - age_mean) / age_std, frame.sex_numeric.to_numpy()], axis=1
    ).astype(np.float32)
    raw = frame[target_names].to_numpy(dtype=np.float32)
    masks = np.isfinite(raw)
    targets = np.where(masks, (raw - target_mean) / target_std, 0).astype(np.float32)
    return demos, targets, masks, target_mean, target_std


def extract_spatial_features(
    frame: pd.DataFrame,
    seed: int,
    config: dict[str, Any],
    phase3b: Path,
    output: Path,
    device: torch.device,
) -> Path:
    destination = output / f"spatial_features_seed{seed}.npy"
    expected = (
        len(frame),
        int(np.prod(config["backbone"]["feature_shape"])),
        int(config["backbone"]["feature_channels"]),
    )
    if destination.exists() and tuple(np.load(destination, mmap_mode="r").shape) == expected:
        return destination
    volume_index = pd.read_csv(phase3b / "volume_index.csv")
    volume_lookup = {
        str(scan_id): int(index)
        for index, scan_id in enumerate(volume_index.scan_id.astype(str))
    }
    if not set(frame.scan_id.astype(str)).issubset(volume_lookup):
        raise RuntimeError("Phase 3C contains scans absent from the Phase 3B volume cache")
    source_positions = np.asarray([volume_lookup[str(scan_id)] for scan_id in frame.scan_id])
    volumes = np.load(phase3b / "volumes_48x56x48.npy", mmap_mode="r")
    pretrained = ConcurrentRegressor(
        "sfcn_3d",
        global_dim=518,
        n_targets=len(config["targets"]),
        dropout=0.15,
    ).to(device)
    checkpoint = torch.load(
        phase3b / "checkpoints/sfcn_3d" / str(seed) / "best.pt",
        map_location=device,
        weights_only=False,
    )
    pretrained.load_state_dict(checkpoint["model"])
    blocks = int(config["backbone"]["frozen_blocks"])
    backbone = pretrained.encoder.network[: blocks * 3].eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    temporary = output / f"spatial_features_seed{seed}.tmp.npy"
    features = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16, shape=expected)
    batch_size = 32
    with torch.no_grad():
        for start in range(0, len(frame), batch_size):
            stop = min(start + batch_size, len(frame))
            batch = torch.from_numpy(
                np.asarray(volumes[source_positions[start:stop]], dtype=np.float32)
            )[:, None].to(device)
            spatial = backbone(batch)
            actual_shape = tuple(spatial.shape[2:])
            if actual_shape != tuple(config["backbone"]["feature_shape"]):
                raise RuntimeError(f"Unexpected spatial feature shape: {actual_shape}")
            flattened = spatial.flatten(2).transpose(1, 2).cpu().numpy().astype(np.float16)
            features[start:stop] = flattened
            if stop % 1000 < batch_size or stop == len(frame):
                print(json.dumps({"seed": seed, "spatial_features": stop, "total": len(frame)}), flush=True)
    features.flush()
    del features
    temporary.replace(destination)
    return destination


def coordinate_grid(shape: tuple[int, int, int]) -> np.ndarray:
    axes = [np.linspace(-1, 1, size, dtype=np.float32) for size in shape]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


class SpatialDataset(Dataset):
    def __init__(
        self,
        indices: np.ndarray,
        frame: pd.DataFrame,
        demos: np.ndarray,
        targets: np.ndarray,
        masks: np.ndarray,
        feature_path: Path,
        source_indices: np.ndarray | None = None,
    ):
        self.indices = np.asarray(indices)
        self.frame = frame
        self.demos = demos
        self.targets = targets
        self.masks = masks
        self.feature_path = feature_path
        self.source_indices = source_indices
        self._features = None

    @property
    def features(self):
        if self._features is None:
            self._features = np.load(self.feature_path, mmap_mode="r")
        return self._features

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int):
        row = int(self.indices[item])
        source = int(self.source_indices[row]) if self.source_indices is not None else row
        return {
            "row": row,
            "source": source,
            "demo": torch.from_numpy(self.demos[row]),
            "tokens": torch.from_numpy(np.asarray(self.features[source], dtype=np.float32)),
            "target": torch.from_numpy(self.targets[row]),
            "mask": torch.from_numpy(self.masks[row]),
        }


class SpatialGrounder(nn.Module):
    """Parameter-matched global and target-conditioned spatial heads."""

    def __init__(
        self,
        arm: str,
        input_dim: int,
        hidden: int,
        targets: int,
        heads: int,
        dropout: float,
        coordinates: np.ndarray,
    ):
        super().__init__()
        self.arm = arm
        self.targets = targets
        self.token = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.position = nn.Linear(3, hidden, bias=False)
        self.demo = nn.Sequential(nn.Linear(2, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.queries = nn.Parameter(torch.randn(targets, hidden) * 0.02)
        self.attention = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.decoder = nn.Sequential(
            nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )
        self.register_buffer("coordinates", torch.from_numpy(coordinates.astype(np.float32)))

    def forward(
        self,
        tokens: torch.Tensor,
        demo: torch.Tensor,
        spatial_permutation: torch.Tensor | None = None,
        query_permutation: torch.Tensor | None = None,
        need_weights: bool = False,
    ):
        if spatial_permutation is not None:
            tokens = tokens[:, spatial_permutation]
        values = self.token(tokens)
        keys = values + self.position(self.coordinates)[None]
        queries = self.queries
        if query_permutation is not None:
            queries = queries[query_permutation]
        query = queries[None].expand(len(tokens), -1, -1) + self.demo(demo)[:, None]
        if self.arm == "target_spatial_attention":
            context, weights = self.attention(
                query, keys, values, need_weights=need_weights, average_attn_weights=True
            )
            context = context + query
        elif self.arm == "global_pool":
            context = values.mean(1)[:, None].expand(-1, self.targets, -1) + query
            weights = None
        else:
            raise ValueError(self.arm)
        return self.decoder(context).squeeze(-1), weights


def evaluate(
    model: SpatialGrounder,
    dataset: SpatialDataset,
    batch_size: int,
    device: torch.device,
    target_names: list[str],
    target_mean: np.ndarray,
    target_std: np.ndarray,
    seed: int,
    pairing: str,
    spatial_permutation: np.ndarray | None = None,
    query_permutation: np.ndarray | None = None,
    return_attention: bool = False,
) -> tuple[pd.DataFrame, np.ndarray | None, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    rows, attention, ordered_rows = [], [], []
    spatial_tensor = None if spatial_permutation is None else torch.from_numpy(spatial_permutation).to(device)
    query_tensor = None if query_permutation is None else torch.from_numpy(query_permutation).to(device)
    with torch.no_grad():
        for batch in loader:
            prediction, weights = model(
                batch["tokens"].to(device),
                batch["demo"].to(device),
                spatial_tensor,
                query_tensor,
                return_attention,
            )
            prediction = prediction.cpu().numpy() * target_std[None] + target_mean[None]
            if weights is not None:
                attention.append(weights.cpu().numpy())
            for local, row_index in enumerate(batch["row"].numpy()):
                record = dataset.frame.iloc[row_index]
                source = int(batch["source"][local])
                ordered_rows.append(row_index)
                for target_index, target in enumerate(target_names):
                    if dataset.masks[row_index, target_index]:
                        rows.append(
                            {
                                "scan_id": record.scan_id,
                                "source_scan_id": dataset.frame.iloc[source].scan_id,
                                "subject_id": record.subject_id,
                                "split": record.split,
                                "model": model.arm,
                                "seed": seed,
                                "pairing": pairing,
                                "target": target,
                                "truth": float(dataset.targets[row_index, target_index] * target_std[target_index] + target_mean[target_index]),
                                "prediction": float(prediction[local, target_index]),
                            }
                        )
    weights = np.concatenate(attention) if attention else None
    return pd.DataFrame(rows), weights, np.asarray(ordered_rows)


def train_one(
    arm: str,
    seed: int,
    feature_path: Path,
    frame: pd.DataFrame,
    demos: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    coordinates: np.ndarray,
    val_shuffle: np.ndarray,
    config: dict[str, Any],
    output: Path,
    device: torch.device,
) -> tuple[pd.DataFrame, np.ndarray | None, np.ndarray, int]:
    seed_everything(seed)
    settings = config["training"]
    train_indices = np.flatnonzero(frame.split.eq("train").to_numpy())
    val_indices = np.flatnonzero(frame.split.eq("val").to_numpy())
    train = SpatialDataset(train_indices, frame, demos, targets, masks, feature_path)
    validation = SpatialDataset(val_indices, frame, demos, targets, masks, feature_path)
    shuffled = SpatialDataset(
        val_indices, frame, demos, targets, masks, feature_path, source_indices=val_shuffle
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train,
        batch_size=int(settings["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    model = SpatialGrounder(
        arm,
        int(config["backbone"]["feature_channels"]),
        int(settings["hidden_dim"]),
        len(config["targets"]),
        int(settings["attention_heads"]),
        float(settings["dropout"]),
        coordinates,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    target_names = list(config["targets"])
    scales = dict(zip(target_names, target_std))
    best_score, best_state, patience = float("inf"), None, 0
    curves = []
    for epoch in range(int(settings["max_epochs"])):
        model.train()
        losses = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = model(batch["tokens"].to(device), batch["demo"].to(device))
            loss = macro_huber(prediction, batch["target"].to(device), batch["mask"].to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss for {arm}/{seed}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(settings["gradient_clip"]))
            optimizer.step()
            losses.append(float(loss.detach()))
        prediction, _, _ = evaluate(
            model, validation, int(settings["batch_size"]), device, target_names,
            target_mean, target_std, seed, "observed"
        )
        score = normalized_macro(prediction, scales)
        curve = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_macro_normalized_mae": score}
        curves.append(curve)
        print(json.dumps({"arm": arm, "seed": seed, **curve}), flush=True)
        if score < best_score - 1e-5:
            best_score, best_state, patience = score, copy.deepcopy(model.state_dict()), 0
        else:
            patience += 1
            if patience >= int(settings["patience"]):
                break
    if best_state is None:
        raise RuntimeError("No checkpoint selected")
    model.load_state_dict(best_state)
    directory = output / "checkpoints" / arm / str(seed)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": best_state, "arm": arm, "seed": seed, "trainable_parameters": parameter_count, "best_score": best_score},
        directory / "best.pt",
    )
    save_csv(pd.DataFrame(curves), directory / "curves.csv")
    observed, weights, order = evaluate(
        model, validation, int(settings["batch_size"]), device, target_names,
        target_mean, target_std, seed, "observed", return_attention=arm == "target_spatial_attention"
    )
    predictions = [observed]
    shuffled_prediction, _, _ = evaluate(
        model, shuffled, int(settings["batch_size"]), device, target_names,
        target_mean, target_std, seed, "shuffled_subject"
    )
    predictions.append(shuffled_prediction)
    if arm == "target_spatial_attention":
        spatial = np.roll(np.arange(len(coordinates)), 1).astype(np.int64)
        query = np.roll(np.arange(len(target_names)), 1).astype(np.int64)
        spatial_prediction, _, _ = evaluate(
            model, validation, int(settings["batch_size"]), device, target_names,
            target_mean, target_std, seed, "spatial_permutation", spatial_permutation=spatial
        )
        query_prediction, _, _ = evaluate(
            model, validation, int(settings["batch_size"]), device, target_names,
            target_mean, target_std, seed, "query_permutation", query_permutation=query
        )
        predictions.extend([spatial_prediction, query_prediction])
    return pd.concat(predictions, ignore_index=True), weights, order, parameter_count


def stability_table(
    weight_by_seed: dict[int, np.ndarray], target_names: list[str], top_fraction: float
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    token_count = next(iter(weight_by_seed.values())).shape[-1]
    top_k = max(1, int(np.ceil(token_count * top_fraction)))
    for left, right in combinations(sorted(weight_by_seed), 2):
        a, b = weight_by_seed[left], weight_by_seed[right]
        for target_index, target in enumerate(target_names):
            for sample in range(len(a)):
                rank_a, rank_b = rankdata(a[sample, target_index]), rankdata(b[sample, target_index])
                correlation = (
                    float(np.corrcoef(rank_a, rank_b)[0, 1])
                    if rank_a.std() > 0 and rank_b.std() > 0
                    else 0.0
                )
                top_a = set(np.argpartition(a[sample, target_index], -top_k)[-top_k:])
                top_b = set(np.argpartition(b[sample, target_index], -top_k)[-top_k:])
                rows.append(
                    {
                        "seed_left": left,
                        "seed_right": right,
                        "target": target,
                        "sample_index": sample,
                        "spearman": correlation,
                        "top_token_overlap": len(top_a & top_b) / top_k,
                    }
                )
    frame = pd.DataFrame(rows)
    summary = {
        "median_spearman": float(frame.spearman.median()),
        "median_top_token_overlap": float(frame.top_token_overlap.median()),
    }
    return frame, summary


def world_coordinates(frame: pd.DataFrame, feature_shape: tuple[int, int, int]) -> np.ndarray:
    image = nib.as_closest_canonical(nib.load(str(frame.image_path.iloc[0])))
    axes = [
        (np.arange(size, dtype=float) + 0.5) * image.shape[index] / size - 0.5
        for index, size in enumerate(feature_shape)
    ]
    voxels = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    homogeneous = np.concatenate([voxels, np.ones((len(voxels), 1))], axis=1)
    return (homogeneous @ image.affine.T)[:, :3]


def localization_table(
    weight_by_seed: dict[int, np.ndarray], target_names: list[str], world: np.ndarray
) -> pd.DataFrame:
    rows = []
    for seed, weights in weight_by_seed.items():
        centers = weights @ world
        for target_index, target in enumerate(target_names):
            for axis, name in enumerate(["x_mm", "y_mm", "z_mm"]):
                rows.append(
                    {
                        "seed": seed,
                        "target": target,
                        "coordinate": name,
                        "mean": float(centers[:, target_index, axis].mean()),
                        "std": float(centers[:, target_index, axis].std()),
                    }
                )
    return pd.DataFrame(rows)


def main(config_path: Path, smoke: bool = False) -> None:
    config = load_config(config_path)
    if smoke:
        config = copy.deepcopy(config)
        config["run_id"] += "_smoke"
        config["seeds"] = config["seeds"][:1]
        config["training"]["max_epochs"] = 2
        config["training"]["patience"] = 2
        config["statistics"]["bootstrap_replicates"] = 100
    output = ROOT / "outputs" / config["run_id"]
    output.mkdir(parents=True, exist_ok=True)
    phase3b = ROOT / "outputs" / config["backbone"]["source_run"]
    frame = build_concurrent_frame(config)
    if smoke:
        frame = pd.concat(
            [frame[frame.split.eq("train")].head(256), frame[frame.split.eq("val")].head(96)],
            ignore_index=True,
        )
    target_names = list(config["targets"])
    demos, targets, masks, target_mean, target_std = prepare_arrays(frame, target_names, phase3b)
    save_csv(
        frame[["anchor_key", "scan_id", "subject_id", "split", "cutoff_month", "scan_date"] + target_names],
        output / "sample_manifest.csv",
    )
    coordinates = coordinate_grid(tuple(config["backbone"]["feature_shape"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_paths = {
        int(seed): extract_spatial_features(frame, int(seed), config, phase3b, output, device)
        for seed in config["seeds"]
    }
    train_mask = frame.split.eq("train").to_numpy()
    val_rows = np.flatnonzero(~train_mask)
    local_shuffle = make_shuffle_indices(
        frame.iloc[val_rows].reset_index(drop=True), int(config["statistics"]["bootstrap_seed"])
    )
    val_shuffle = np.arange(len(frame))
    val_shuffle[val_rows] = val_rows[local_shuffle]
    predictions, weights, parameter_counts = [], {}, {}
    weight_order = None
    for arm in config["arms"]:
        for seed in config["seeds"]:
            result, attention, order, count = train_one(
                arm, int(seed), feature_paths[int(seed)], frame, demos, targets, masks,
                target_mean, target_std, coordinates, val_shuffle, config, output, device
            )
            predictions.append(result)
            parameter_counts[arm] = count
            if attention is not None:
                weights[int(seed)] = attention
                if weight_order is None:
                    weight_order = order
                elif not np.array_equal(weight_order, order):
                    raise RuntimeError("Attention row order differs across seeds")
    seed_predictions = pd.concat(predictions, ignore_index=True)
    ensembles = ensemble(seed_predictions)
    all_predictions = pd.concat([seed_predictions, ensembles], ignore_index=True)
    save_csv(all_predictions, output / "validation_predictions.csv")
    scales = dict(zip(target_names, target_std))
    metric = metrics(ensembles, scales)
    save_csv(metric, output / "validation_metrics.csv")
    comparisons = []
    candidate = ensembles[
        ensembles.model.eq("target_spatial_attention") & ensembles.pairing.eq("observed")
    ]
    baselines = [
        ensembles[ensembles.model.eq("global_pool") & ensembles.pairing.eq("observed")],
        ensembles[ensembles.model.eq("target_spatial_attention") & ensembles.pairing.eq("shuffled_subject")],
        ensembles[ensembles.model.eq("target_spatial_attention") & ensembles.pairing.eq("spatial_permutation")],
        ensembles[ensembles.model.eq("target_spatial_attention") & ensembles.pairing.eq("query_permutation")],
    ]
    for baseline in baselines:
        for target in [None] + target_names:
            comparisons.append(
                bootstrap_difference(
                    candidate,
                    baseline,
                    scales,
                    int(config["statistics"]["bootstrap_replicates"]),
                    int(config["statistics"]["bootstrap_seed"]) + len(comparisons),
                    target=target,
                )
            )
    bootstrap = pd.DataFrame(comparisons)
    save_csv(bootstrap, output / "bootstrap_comparisons.csv")
    if len(weights) >= 2:
        stability, stability_summary = stability_table(
            weights, target_names, float(config["statistics"]["attention_top_fraction"])
        )
    else:
        stability = pd.DataFrame()
        stability_summary = {"median_spearman": None, "median_top_token_overlap": None}
    save_csv(stability, output / "attention_stability.csv")
    np.savez_compressed(
        output / "attention_weights.npz",
        seeds=np.asarray(sorted(weights)),
        scan_ids=frame.iloc[weight_order].scan_id.astype(str).to_numpy() if weight_order is not None else np.asarray([]),
        weights=np.stack([weights[seed] for seed in sorted(weights)]) if weights else np.empty(0),
    )
    world = world_coordinates(frame, tuple(config["backbone"]["feature_shape"]))
    localization = localization_table(weights, target_names, world) if weights else pd.DataFrame()
    save_csv(localization, output / "attention_centers_template_mm.csv")
    overall = bootstrap[bootstrap.target.eq("overall")].set_index(["baseline", "baseline_pairing"])
    global_key = ("global_pool", "observed")
    spatial_key = ("target_spatial_attention", "spatial_permutation")
    query_key = ("target_spatial_attention", "query_permutation")
    versus_global = bootstrap[
        bootstrap.baseline.eq("global_pool") & bootstrap.baseline_pairing.eq("observed")
    ]
    significant_endpoints = int(
        ((versus_global.target != "overall") & (versus_global.ci_upper < 0)).sum()
    )
    criteria = {
        "attention_better_than_global_pool": bool(overall.loc[global_key, "ci_upper"] < 0),
        "attention_better_than_spatial_permutation": bool(overall.loc[spatial_key, "ci_upper"] < 0),
        "attention_better_than_query_permutation": bool(overall.loc[query_key, "ci_upper"] < 0),
        "minimum_significant_endpoints": significant_endpoints >= int(config["gate"]["minimum_significant_endpoints"]),
        "attention_spearman_stable": bool(
            stability_summary["median_spearman"] >= float(config["gate"]["minimum_attention_spearman"])
        ) if not smoke else False,
        "top_token_overlap_stable": bool(
            stability_summary["median_top_token_overlap"] >= float(config["gate"]["minimum_top_token_overlap"])
        ) if not smoke else False,
    }
    gate = bool(all(criteria.values())) and not smoke
    decision = {
        "run_id": config["run_id"],
        "engineering_smoke_only": smoke,
        "development_only": True,
        "test_evaluated": False,
        "criteria": criteria,
        "significant_endpoints_vs_global": significant_endpoints,
        "attention_stability": stability_summary,
        "concurrent_spatial_grounding_passed": gate,
        "next_action": "freeze_multimodal_concurrent_grounding_protocol" if gate else "retain_sfcn_global_encoder_without_anatomical_attention_claim",
    }
    save_json(decision, output / "development_decision.json")
    summary = (
        metric.groupby(["model", "pairing"], as_index=False)
        .agg(normalized_mae=("normalized_mae", "mean"), mae=("mae", "mean"))
        .sort_values("normalized_mae")
    )
    report = f"""# Phase 3C concurrent question-to-spatial grounding

Status: completed validation-only experiment. Test was not loaded.

## Contract

- Train/validation MRI anchors: {int(train_mask.sum())} / {int((~train_mask).sum())}
- Validation targets: {int(masks[~train_mask].sum())}
- Frozen SFCN feature grid: {config['backbone']['feature_shape']} × {config['backbone']['feature_channels']} channels
- Seeds: {config['seeds']}
- Trainable parameters: {parameter_counts}

## Validation summary

{markdown_table(summary)}

## Paired subject-cluster bootstrap

Negative differences favor observed target-conditioned spatial attention.

{markdown_table(bootstrap)}

## Cross-seed attention stability

```json
{json.dumps(stability_summary, indent=2)}
```

## Decision

```json
{json.dumps(decision, indent=2)}
```

Attention centers are reported in aligned-template millimeter coordinates without atlas-region labels. This experiment does not establish clinical anatomical localization or longitudinal MRI benefit.
"""
    (output / "report.md").write_text(report)
    save_json(
        {
            "run_id": config["run_id"],
            "device": str(device),
            "sample_counts": {"train": int(train_mask.sum()), "val": int((~train_mask).sum())},
            "subjects": {
                "train": int(frame.loc[train_mask, "subject_id"].nunique()),
                "val": int(frame.loc[~train_mask, "subject_id"].nunique()),
            },
            "parameter_counts": parameter_counts,
            "frozen_backbone_checkpoints": [str(phase3b / "checkpoints/sfcn_3d" / str(seed) / "best.pt") for seed in config["seeds"]],
            "test_evaluated": False,
        },
        output / "run_manifest.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/phase3c_concurrent_spatial_grounding_dev.yaml"
    )
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    main(arguments.config, arguments.smoke)
