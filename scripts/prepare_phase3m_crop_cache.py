#!/usr/bin/env python3
"""Create and fully audit the matched 896-scan Phase 3M crop cache."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from memreason_rl.data.common import ROOT
from memreason_rl.data.mri import STANDARD_CROP_SHAPE, center_crop_slices


SOURCE_RUN = ROOT / "outputs/phase3i_native_resolution_adcn_20261017"
OUTPUT = ROOT / "outputs/phase3m_mri_preprocessing_sensitivity_20261018"
CACHE = Path("/public_bme/data/zhouwl2023/MemReason_cache/phase3m_mri_preprocessing_sensitivity_20261018/volumes_center_crop_160x192x160_float16.npy")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(SOURCE_RUN / "cohort_manifest.csv", low_memory=False)
    index = cohort[["scan_id", "subject_id", "native_volume_index"]].copy()
    index["volume_index"] = np.arange(len(index), dtype=int)
    source = np.load(SOURCE_RUN / "volumes_native_193x229x193_float16.npy", mmap_mode="r")
    expected = (len(index),) + STANDARD_CROP_SHAPE
    temporary = CACHE.with_suffix(".tmp.npy")
    temporary.unlink(missing_ok=True)
    target = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16, shape=expected)
    slices = center_crop_slices(source.shape[1:], STANDARD_CROP_SHAPE)
    counts = []
    for row in index.itertuples(index=False):
        volume = np.asarray(source[int(row.native_volume_index)][slices], dtype=np.float16)
        count = int(np.count_nonzero(volume))
        if tuple(volume.shape) != STANDARD_CROP_SHAPE or count <= 100:
            raise RuntimeError(f"Invalid crop for {row.scan_id}: shape={volume.shape}, nonzero={count}")
        target[int(row.volume_index)] = volume
        counts.append(count)
        if (int(row.volume_index) + 1) % 100 == 0:
            target.flush()
            print(json.dumps({"written": int(row.volume_index) + 1, "total": len(index)}), flush=True)
    target.flush()
    del target
    check = np.load(temporary, mmap_mode="r")
    verified_counts = [int(np.count_nonzero(check[i])) for i in range(len(check))]
    if tuple(check.shape) != expected or min(verified_counts) <= 100:
        raise RuntimeError("Full post-write cache audit failed")
    del check
    temporary.replace(CACHE)
    index.to_csv(OUTPUT / "center_crop_volume_index.csv", index=False)
    audit = {
        "passed": True,
        "n_scans": len(index),
        "n_nonempty": sum(x > 100 for x in verified_counts),
        "source_shape": list(source.shape[1:]),
        "target_shape": list(STANDARD_CROP_SHAPE),
        "center_crop_only": True,
        "spatial_interpolation": False,
        "normalization_reused_from_fullres": True,
        "storage_dtype": "float16",
        "minimum_nonzero_voxels": min(verified_counts),
        "maximum_nonzero_voxels": max(verified_counts),
        "cache_path": str(CACHE),
    }
    (OUTPUT / "center_crop_cache_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
