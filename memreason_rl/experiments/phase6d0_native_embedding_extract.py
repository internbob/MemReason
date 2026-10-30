"""Train fold-aligned native-T1 encoders and cache their 128-D embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader

from ..data.common import ROOT
from .phase3i_native_resolution_adcn import NativeSFCNClassifier, train_native_sfcn
from .phase3j_native_longitudinal_fusion import NativeScanDataset


CONFIG = ROOT / "configs/phase6d0_native_embedding_extract.yaml"
PROTOCOL = ROOT / "reports/experiments/phase6d0_native_embedding_extract_protocol.md"


def state_fold_map(config: dict[str, Any]) -> dict[str, int]:
    states = pd.read_csv(ROOT / config["phase5g_states"], usecols=["subject_id", "fold", "arm"])
    states = states[states.arm.eq(config["phase5g_selected_arm"])].copy()
    counts = states.groupby(states.subject_id.astype(str)).fold.nunique()
    if (counts > 1).any():
        raise RuntimeError("A Phase 5G subject occurs in multiple outer folds")
    return states.assign(subject_id=states.subject_id.astype(str)).drop_duplicates("subject_id").set_index("subject_id").fold.astype(int).to_dict()


def training_frame(config: dict[str, Any]) -> pd.DataFrame:
    frame = pd.read_csv(ROOT / config["phase3i_cohort"], low_memory=False)
    original = pd.read_csv(ROOT / config["phase3i_subject_folds"], usecols=["subject_id", "fold"])
    original = original.assign(subject_id=original.subject_id.astype(str)).set_index("subject_id").fold.astype(int).to_dict()
    phase6 = state_fold_map(config)
    frame["subject_id"] = frame.subject_id.astype(str)
    frame["encoder_fold"] = [phase6.get(subject, original[subject]) for subject in frame.subject_id]
    if not frame.encoder_fold.between(0, int(config["folds"]) - 1).all():
        raise RuntimeError("Invalid encoder fold assignment")
    return frame


def scan_frame(config: dict[str, Any]) -> pd.DataFrame:
    burden = pd.read_csv(ROOT / config["longitudinal_mri_burden"], low_memory=False)
    burden = burden[burden.split.eq("train")].copy()
    registry = pd.read_csv(ROOT / config["train_mri_registry"], low_memory=False)
    registry = registry.drop_duplicates("scan_id", keep="first")
    paths = registry[["scan_id", "image_path", "mask_path"]]
    frame = burden.drop(columns=[column for column in ["image_path"] if column in burden]).merge(
        paths, on="scan_id", how="left", validate="one_to_one"
    )
    if frame[["image_path", "mask_path"]].isna().any().any():
        raise RuntimeError("Longitudinal scan could not be mapped to image/mask paths")
    return frame.sort_values(["subject_id", "scan_date", "scan_id"], kind="stable").reset_index(drop=True)


def train_folds(config: dict[str, Any], output: Path, folds: list[int], device: torch.device) -> None:
    frame = training_frame(config)
    cache = ROOT / config["phase3i_volume_cache"]
    phase6 = state_fold_map(config)
    for fold in folds:
        evaluation_subjects = {subject for subject, value in phase6.items() if value == fold}
        outer = np.flatnonzero(frame.encoder_fold.ne(fold).to_numpy())
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=float(config["inner_tune_fraction"]),
            random_state=int(config["inner_seed"]) + fold,
        )
        fit_local, tune_local = next(splitter.split(outer, frame.iloc[outer].label.to_numpy()))
        fit_indices, tune_indices = outer[fit_local], outer[tune_local]
        fit_subjects = set(frame.iloc[fit_indices].subject_id)
        tune_subjects = set(frame.iloc[tune_indices].subject_id)
        leaked = sorted(evaluation_subjects & (fit_subjects | tune_subjects))
        if leaked:
            raise RuntimeError(f"Encoder fold {fold} leaked {len(leaked)} Phase 6 evaluation subjects")
        embedding, curve = train_native_sfcn(
            fold, int(config["encoder_seed"]), frame, fit_indices, tune_indices,
            cache, config, output, device,
        )
        if embedding.shape != (len(frame), int(config["encoding"]["embedding_dimension"])):
            raise RuntimeError(f"Unexpected training-cohort embedding shape: {embedding.shape}")
        curve.to_csv(output / f"training_curve_fold{fold}.csv", index=False)
        pd.DataFrame({"subject_id": sorted(fit_subjects), "partition": "fit"})._append(
            pd.DataFrame({"subject_id": sorted(tune_subjects), "partition": "tune"}), ignore_index=True
        ).to_csv(output / f"encoder_subjects_fold{fold}.csv", index=False)
        record = {
            "fold": fold, "fit_subjects": len(fit_subjects), "tune_subjects": len(tune_subjects),
            "phase6_evaluation_subjects": len(evaluation_subjects), "leaked_subjects": len(leaked),
            "fit_cn": int((frame.iloc[fit_indices].label == 0).sum()),
            "fit_ad": int((frame.iloc[fit_indices].label == 1).sum()),
            "best_tune_auc": float(curve.tune_auc.max()), "device": str(device),
        }
        (output / f"training_manifest_fold{fold}.json").write_text(json.dumps(record, indent=2) + "\n")


def encode_folds(config: dict[str, Any], output: Path, folds: list[int], device: torch.device) -> None:
    frame = scan_frame(config)
    index_path = output / "scan_index.csv"
    index_columns = ["scan_id", "subject_id", "scan_date", "visit_month", "split"]
    if index_path.exists():
        existing = pd.read_csv(index_path, low_memory=False)
        if existing.scan_id.astype(str).tolist() != frame.scan_id.astype(str).tolist():
            raise RuntimeError("Existing scan index does not match current registry")
    else:
        frame[index_columns].to_csv(index_path, index=False)
    settings = config["encoding"]
    dataset = NativeScanDataset(frame, tuple(map(int, config["native_volume"]["expected_shape"])))
    workers = int(settings["num_workers"])
    loader = DataLoader(
        dataset, batch_size=int(settings["batch_size"]), shuffle=False,
        num_workers=workers, pin_memory=device.type == "cuda", persistent_workers=workers > 0,
    )
    models: dict[int, NativeSFCNClassifier] = {}
    for fold in folds:
        checkpoint = output / "checkpoints" / f"fold{fold}" / f"seed{int(config['encoder_seed'])}.pt"
        record = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = NativeSFCNClassifier(float(config["sfcn_training"]["dropout"])).to(device)
        model.load_state_dict(record["model"]); model.eval(); models[fold] = model
    dimension = int(settings["embedding_dimension"])
    embeddings = {
        fold: np.lib.format.open_memmap(output / f"embeddings_fold{fold}_float16.npy", mode="w+", dtype=np.float16,
                                        shape=(len(frame), dimension)) for fold in folds
    }
    probabilities = {fold: np.empty(len(frame), np.float32) for fold in folds}
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            volume = batch["volume"].to(device, non_blocking=True)
            indices = batch["index"].numpy()
            for fold, model in models.items():
                with torch.amp.autocast(device_type=device.type,
                                        enabled=bool(settings["amp"]) and device.type == "cuda"):
                    logits, embedding = model(volume)
                embeddings[fold][indices] = embedding.float().cpu().numpy().astype(np.float16)
                probabilities[fold][indices] = torch.softmax(logits.float(), 1)[:, 1].cpu().numpy()
            if (batch_index + 1) % 100 == 0 or int(indices[-1]) == len(frame) - 1:
                print(json.dumps({"encoded": int(indices[-1]) + 1, "total": len(frame), "folds": folds}), flush=True)
    for fold in folds:
        embeddings[fold].flush()
        np.save(output / f"probability_fold{fold}_float32.npy", probabilities[fold])


def finalize(config: dict[str, Any], output: Path) -> None:
    phase6 = state_fold_map(config); scans = pd.read_csv(output / "scan_index.csv")
    dimension = int(config["encoding"]["embedding_dimension"]); audit = []; manifests = []
    for fold in range(int(config["folds"])):
        subject_file = output / f"encoder_subjects_fold{fold}.csv"
        subjects = set(pd.read_csv(subject_file).subject_id.astype(str))
        evaluation = {subject for subject, value in phase6.items() if value == fold}
        leaked = subjects & evaluation
        embedding_path = output / f"embeddings_fold{fold}_float16.npy"
        probability_path = output / f"probability_fold{fold}_float32.npy"
        matrix = np.load(embedding_path, mmap_mode="r"); probability = np.load(probability_path, mmap_mode="r")
        valid = matrix.shape == (len(scans), dimension) and np.isfinite(matrix).all()
        probability_valid = probability.shape == (len(scans),) and np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
        audit.append({"fold": fold, "leaked_subjects": len(leaked), "embedding_valid": bool(valid),
                      "probability_valid": bool(probability_valid), "scans": len(scans), "dimension": dimension})
        manifests.append(json.loads((output / f"training_manifest_fold{fold}.json").read_text()))
    audit_frame = pd.DataFrame(audit); audit_frame.to_csv(output / "leakage_and_embedding_audit.csv", index=False)
    passed = bool((audit_frame.leaked_subjects == 0).all() and audit_frame.embedding_valid.all() and audit_frame.probability_valid.all())
    decision = {"run_id": config["run_id"], "original_train_only": True, "validation_or_test_loaded": False,
                "fold_aligned_embeddings_ready": passed, "folds": int(config["folds"]),
                "scans": len(scans), "embedding_dimension": dimension, "coordinatewise_fold_averaging": False}
    (output / "development_decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    pd.DataFrame(manifests).to_csv(output / "encoder_fold_manifest.csv", index=False)
    manifest = {"config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
                "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
                "scan_index_sha256": hashlib.sha256((output / "scan_index.csv").read_bytes()).hexdigest(),
                "folds": int(config["folds"]), "scans": len(scans), "embedding_dimension": dimension,
                "checkpoint_files": len(list((output / "checkpoints").glob("fold*/seed*.pt")))}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    lines = ["# Phase 6D0: fold-aligned native-T1 embeddings", "",
             f"Cached **{len(scans):,}** original-train longitudinal scans under **{config['folds']}** fold-specific encoders.", "",
             f"Every embedding has **{dimension}** dimensions. Leakage/QC status: **{'passed' if passed else 'failed'}**.", "",
             "Each downstream outer fold uses only its corresponding encoder; independently trained embedding coordinates are not averaged."]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    if not passed:
        raise RuntimeError("Phase 6D0 leakage/embedding audit failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--mode", choices=["train", "encode", "finalize", "all"], default="all")
    parser.add_argument("--folds", nargs="*", type=int)
    args = parser.parse_args(); config = yaml.safe_load(args.config.read_text())
    folds = args.folds if args.folds else list(range(int(config["folds"])))
    output = ROOT / "outputs" / config["run_id"]; output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mode in {"train", "all"}: train_folds(config, output, folds, device)
    if args.mode in {"encode", "all"}: encode_folds(config, output, folds, device)
    if args.mode in {"finalize", "all"}: finalize(config, output)
    print(output)


if __name__ == "__main__":
    main()
