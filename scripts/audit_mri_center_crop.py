#!/usr/bin/env python3
"""Audit brain-mask coverage under the standard 160x192x160 center crop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memreason_rl.data.mri import (
    STANDARD_CROP_SHAPE,
    STANDARD_SOURCE_SHAPE,
    center_crop_slices,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-scans", type=int, default=None)
    args = parser.parse_args()

    frame = pd.read_csv(args.index, usecols=["scan_id", "mask_path"])
    if args.maximum_scans is not None:
        frame = frame.iloc[: args.maximum_scans].copy()
    crop = center_crop_slices(STANDARD_SOURCE_SHAPE, STANDARD_CROP_SHAPE)

    rows = []
    failures = []
    for row in frame.itertuples(index=False):
        try:
            mask_image = nib.as_closest_canonical(nib.load(str(row.mask_path)))
            if tuple(mask_image.shape) != STANDARD_SOURCE_SHAPE:
                raise ValueError(
                    f"shape={mask_image.shape}, expected={STANDARD_SOURCE_SHAPE}"
                )
            mask = np.asarray(mask_image.dataobj, dtype=np.float32) > 0.5
            total = int(mask.sum())
            retained = int(mask[crop].sum())
            rows.append(
                {
                    "scan_id": str(row.scan_id),
                    "mask_voxels": total,
                    "retained_voxels": retained,
                    "cropped_mask_voxels": total - retained,
                    "retained_fraction": retained / total if total else np.nan,
                }
            )
        except Exception as error:
            failures.append(
                {
                    "scan_id": str(row.scan_id),
                    "error": f"{type(error).__name__}:{error}",
                }
            )

    detail = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output.with_suffix(".csv"), index=False)
    payload = {
        "n_requested": int(len(frame)),
        "n_audited": int(len(detail)),
        "n_failures": int(len(failures)),
        "source_shape": list(STANDARD_SOURCE_SHAPE),
        "target_shape": list(STANDARD_CROP_SHAPE),
        "spatial_interpolation": False,
        "voxel_reduction_fraction": 1.0
        - (np.prod(STANDARD_CROP_SHAPE) / np.prod(STANDARD_SOURCE_SHAPE)),
        "scans_with_cropped_mask_voxels": int(
            (detail.cropped_mask_voxels > 0).sum()
        )
        if len(detail)
        else 0,
        "total_cropped_mask_voxels": int(detail.cropped_mask_voxels.sum())
        if len(detail)
        else 0,
        "minimum_retained_fraction": float(detail.retained_fraction.min())
        if len(detail)
        else None,
        "median_retained_fraction": float(detail.retained_fraction.median())
        if len(detail)
        else None,
        "failures": failures,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
