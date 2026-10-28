import numpy as np

from memreason_rl.data.mri import center_crop_slices
from memreason_rl.experiments.phase3k_cropped_mri_cache import _audit_all_rows


def test_full_row_audit_detects_zero_volume_between_valid_boundaries():
    cache = np.ones((3, 5, 5, 5), dtype=np.float16)
    cache[1] = 0
    audit = _audit_all_rows(cache)
    assert audit["rows_checked"] == 3
    assert audit["finite_rows"] == 3
    assert audit["nonempty_rows"] == 2
    assert audit["all_rows_finite"] is True
    assert audit["all_rows_nonempty"] is False


def test_standard_crop_is_exact_center_subarray():
    source = np.arange(193 * 229 * 193, dtype=np.int32).reshape(193, 229, 193)
    cropped = source[center_crop_slices(source.shape)]
    assert cropped.shape == (160, 192, 160)
    assert cropped[0, 0, 0] == source[16, 18, 16]
    assert cropped[-1, -1, -1] == source[175, 209, 175]
