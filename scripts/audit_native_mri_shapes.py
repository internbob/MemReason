#!/usr/bin/env python3
"""Audit native MRI and brain-mask grids without loading voxel arrays."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import nibabel as nib
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.index)
    image_shapes: Counter[tuple[int, ...]] = Counter()
    mask_shapes: Counter[tuple[int, ...]] = Counter()
    mismatched = []
    affines = Counter()
    zooms = Counter()
    failures = []
    for row in frame.itertuples(index=False):
        try:
            image = nib.load(str(row.image_path))
            mask = nib.load(str(row.mask_path))
            image_shape = tuple(int(value) for value in image.shape)
            mask_shape = tuple(int(value) for value in mask.shape)
            image_shapes[image_shape] += 1
            mask_shapes[mask_shape] += 1
            zooms[tuple(round(float(value), 5) for value in image.header.get_zooms()[:3])] += 1
            affines[tuple(round(float(value), 4) for value in image.affine.ravel())] += 1
            if image_shape != mask_shape:
                mismatched.append(str(row.scan_id))
        except Exception as error:  # audit records every failure
            failures.append({"scan_id": str(row.scan_id), "error": f"{type(error).__name__}:{error}"})

    payload = {
        "n_scans": int(len(frame)),
        "image_shapes": {"x".join(map(str, key)): value for key, value in image_shapes.most_common()},
        "mask_shapes": {"x".join(map(str, key)): value for key, value in mask_shapes.most_common()},
        "voxel_sizes": {"x".join(map(str, key)): value for key, value in zooms.most_common()},
        "n_unique_affines": len(affines),
        "n_shape_mismatches": len(mismatched),
        "shape_mismatch_scan_ids": mismatched,
        "n_failures": len(failures),
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
