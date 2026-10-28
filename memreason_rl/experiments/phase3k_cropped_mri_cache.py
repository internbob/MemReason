"""Build a no-interpolation 160x192x160 T1 cache for classification reruns."""

from __future__ import annotations

import argparse
import atexit
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..data.common import ROOT
from ..data.mri import STANDARD_CROP_SHAPE, load_mri


CONFIG = ROOT / "configs/phase3k_cropped_mri_diagnosis_crossfit.yaml"
_CACHE: np.memmap | None = None


def _flush_worker_cache() -> None:
    if _CACHE is not None:
        _CACHE.flush()


def _audit_all_rows(cache: np.ndarray) -> dict[str, int | bool]:
    """Check every cached scan; boundary-only checks can miss sparse corruption."""
    nonempty = 0
    finite = 0
    for index in range(len(cache)):
        volume = np.asarray(cache[index])
        finite += int(bool(np.isfinite(volume).all()))
        nonempty += int(int(np.count_nonzero(volume)) > 100)
    return {
        "rows_checked": int(len(cache)),
        "finite_rows": int(finite),
        "nonempty_rows": int(nonempty),
        "all_rows_finite": bool(finite == len(cache)),
        "all_rows_nonempty": bool(nonempty == len(cache)),
    }


def _initialize_cache(path: str) -> None:
    global _CACHE
    _CACHE = np.load(path, mmap_mode="r+")
    atexit.register(_flush_worker_cache)


def _write_volume(task: tuple[int, str, str]) -> tuple[int, str | None]:
    index, image_path, mask_path = task
    try:
        volume = load_mri(image_path, mask_path).numpy()[0].astype(np.float16)
        if tuple(volume.shape) != STANDARD_CROP_SHAPE:
            raise ValueError(f"cropped_shape={volume.shape}")
        if _CACHE is None:
            raise RuntimeError("cache worker was not initialized")
        _CACHE[index] = volume
        return index, None
    except Exception as error:
        return index, f"{type(error).__name__}:{error}"


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def build_index(config: dict[str, Any]) -> pd.DataFrame:
    diagnostic = pd.read_csv(
        ROOT / "outputs" / config["reuse_diagnostic_run_id"] / "diagnostic_manifest.csv",
        usecols=["scan_id", "subject_id"],
    ).drop_duplicates("scan_id")
    source = pd.read_csv(
        ROOT / "outputs" / config["source_mri_run_id"] / "volume_index.csv",
        usecols=["scan_id", "image_path", "mask_path"],
    )
    frame = diagnostic.merge(source, on="scan_id", how="left", validate="one_to_one")
    if frame[["image_path", "mask_path"]].isna().any().any():
        raise RuntimeError("Cropped cache index contains missing MRI paths")
    frame = frame.sort_values("scan_id").reset_index(drop=True)
    frame["volume_index"] = np.arange(len(frame), dtype=int)
    return frame


def run(config_path: Path = CONFIG) -> None:
    config = load_config(config_path)
    settings = config["volume_cache"]
    output = ROOT / "outputs" / settings["run_id"]
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / "volume_index.csv"
    cache_path = ROOT / settings["path"]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path = output / "cache_audit.json"
    frame = build_index(config)

    if cache_path.exists():
        cache = np.load(cache_path, mmap_mode="r")
        valid_shape = tuple(cache.shape) == (len(frame),) + STANDARD_CROP_SHAPE
        full_audit = _audit_all_rows(cache) if valid_shape else {
            "rows_checked": 0, "finite_rows": 0, "nonempty_rows": 0,
            "all_rows_finite": False, "all_rows_nonempty": False,
        }
        if valid_shape and full_audit["all_rows_finite"] and full_audit["all_rows_nonempty"]:
            recovered_index = not index_path.exists() or index_path.stat().st_size == 0
            frame.to_csv(index_path, index=False)
            payload = {
                "n_scans": int(len(frame)),
                "subjects": int(frame.subject_id.nunique()),
                "source_shape": list(settings["source_shape"]),
                "target_shape": list(STANDARD_CROP_SHAPE),
                "voxel_size_mm": list(settings["voxel_size_mm"]),
                "spatial_interpolation": False,
                "padding": False,
                "storage_dtype": str(cache.dtype),
                "cache_shape": list(cache.shape),
                **full_audit,
                "failures": [],
                "recovered_index": recovered_index,
            }
            audit_path.write_text(json.dumps(payload, indent=2) + "\n")
            print(json.dumps({"cache_reused": True, "scans": len(frame)}), flush=True)
            return

    temporary = cache_path.with_suffix(".tmp.npy")
    temporary.unlink(missing_ok=True)
    cache = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float16,
        shape=(len(frame),) + STANDARD_CROP_SHAPE,
    )
    cache.flush()
    del cache

    tasks = [
        (int(row.volume_index), str(row.image_path), str(row.mask_path))
        for row in frame.itertuples(index=False)
    ]
    failures = []
    with ProcessPoolExecutor(
        max_workers=int(settings["extraction_workers"]),
        initializer=_initialize_cache,
        initargs=(str(temporary),),
    ) as pool:
        futures = [pool.submit(_write_volume, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), 1):
            index, error = future.result()
            if error is not None:
                failures.append({"volume_index": index, "error": error})
            if completed % 50 == 0 or completed == len(tasks):
                print(json.dumps({"cached": completed, "total": len(tasks)}), flush=True)
    if failures:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Cropped cache failures ({len(failures)}): {failures[:10]}")

    temporary.replace(cache_path)
    frame.to_csv(index_path, index=False)
    cache = np.load(cache_path, mmap_mode="r")
    full_audit = _audit_all_rows(cache)
    if not full_audit["all_rows_finite"] or not full_audit["all_rows_nonempty"]:
        raise RuntimeError(f"Completed crop cache failed full-row audit: {full_audit}")
    payload = {
        "n_scans": int(len(frame)),
        "subjects": int(frame.subject_id.nunique()),
        "source_shape": list(settings["source_shape"]),
        "target_shape": list(STANDARD_CROP_SHAPE),
        "voxel_size_mm": list(settings["voxel_size_mm"]),
        "spatial_interpolation": False,
        "padding": False,
        "storage_dtype": str(cache.dtype),
        "cache_shape": list(cache.shape),
        **full_audit,
        "failures": [],
    }
    audit_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
