"""Shared T1 preprocessing for new MemReason MRI experiments.

The ADNI_Linear images are already registered to a common 1-mm grid. New
experiments retain that voxel spacing and remove only fixed background margins
with a deterministic center crop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np
import torch


STANDARD_SOURCE_SHAPE = (193, 229, 193)
STANDARD_CROP_SHAPE = (160, 192, 160)
STANDARD_VOXEL_SIZE_MM = (1.0, 1.0, 1.0)
MINIMUM_MASK_RETAINED_FRACTION = 0.97


def center_crop_slices(
    source_shape: Sequence[int],
    target_shape: Sequence[int] = STANDARD_CROP_SHAPE,
) -> tuple[slice, slice, slice]:
    """Return deterministic centered slices without interpolation or padding."""
    source = tuple(int(value) for value in source_shape)
    target = tuple(int(value) for value in target_shape)
    if len(source) != 3 or len(target) != 3:
        raise ValueError(f"Expected 3D shapes, got source={source}, target={target}")
    if any(
        target_dim <= 0 or target_dim > source_dim
        for source_dim, target_dim in zip(source, target)
    ):
        raise ValueError(f"Invalid center crop: source={source}, target={target}")
    starts = tuple(
        (source_dim - target_dim) // 2
        for source_dim, target_dim in zip(source, target)
    )
    return tuple(
        slice(start, start + target_dim)
        for start, target_dim in zip(starts, target)
    )


def center_crop_3d(
    array: np.ndarray,
    target_shape: Sequence[int] = STANDARD_CROP_SHAPE,
) -> np.ndarray:
    """Center-crop a 3D array while preserving its voxel spacing."""
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D array, got shape={array.shape}")
    cropped = array[center_crop_slices(array.shape, target_shape)]
    expected = tuple(int(value) for value in target_shape)
    if cropped.shape != expected:
        raise RuntimeError(f"Center crop produced {cropped.shape}, expected {expected}")
    return cropped


def load_mri(
    path: str | Path,
    mask_path: str | Path | None = None,
    *,
    source_shape: Sequence[int] = STANDARD_SOURCE_SHAPE,
    target_shape: Sequence[int] = STANDARD_CROP_SHAPE,
    voxel_size_mm: Sequence[float] = STANDARD_VOXEL_SIZE_MM,
    minimum_mask_retained_fraction: float = MINIMUM_MASK_RETAINED_FRACTION,
) -> torch.Tensor:
    """Load, validate, center-crop, and foreground-normalize a registered T1.

    This default contract performs no spatial resampling. Under the standard
    policy the returned tensor has shape ``[1, 160, 192, 160]``.
    """
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    image = nib.as_closest_canonical(nib.load(str(image_path)))
    expected_source = tuple(int(value) for value in source_shape)
    if tuple(image.shape) != expected_source:
        raise ValueError(f"T1 grid mismatch: {image.shape}, expected {expected_source}")
    observed_voxels = tuple(
        round(float(value), 5) for value in image.header.get_zooms()[:3]
    )
    expected_voxels = tuple(round(float(value), 5) for value in voxel_size_mm)
    if observed_voxels != expected_voxels:
        raise ValueError(
            f"T1 voxel size mismatch: {observed_voxels}, expected {expected_voxels}"
        )

    array = np.asarray(image.dataobj, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite MRI voxels; requires QC")

    if mask_path is None:
        mask = np.abs(array) > 1e-6
    else:
        resolved_mask_path = Path(mask_path)
        if not resolved_mask_path.is_file():
            raise FileNotFoundError(resolved_mask_path)
        mask_image = nib.as_closest_canonical(nib.load(str(resolved_mask_path)))
        if tuple(mask_image.shape) != expected_source:
            raise ValueError(
                f"Mask grid mismatch: {mask_image.shape}, expected {expected_source}"
            )
        if not np.allclose(image.affine, mask_image.affine, atol=1e-4):
            raise ValueError("T1 and mask affines differ")
        mask = np.asarray(mask_image.dataobj, dtype=np.float32) > 0.5

    mask_voxels = int(mask.sum())
    cropped_mask = center_crop_3d(mask, target_shape).astype(bool, copy=False)
    retained_fraction = float(cropped_mask.sum()) / max(mask_voxels, 1)
    if retained_fraction < float(minimum_mask_retained_fraction):
        raise ValueError(
            "Center crop retained only "
            f"{retained_fraction:.6f} of foreground; minimum is "
            f"{float(minimum_mask_retained_fraction):.6f}"
        )
    array = center_crop_3d(array, target_shape)
    mask = cropped_mask
    foreground = array[mask]
    if foreground.size < 100 or not np.isfinite(foreground).all():
        raise ValueError("MRI foreground is empty or nonfinite after center crop")
    std = float(foreground.std())
    if std < 1e-6:
        raise ValueError("MRI foreground is constant after center crop")

    normalized = np.zeros(array.shape, dtype=np.float32)
    normalized[mask] = np.clip(
        (foreground - float(foreground.mean())) / std,
        -5.0,
        5.0,
    )
    return torch.from_numpy(normalized[None])
