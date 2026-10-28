#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 {center_crop|lowres}" >&2
  exit 2
fi

branch="$1"
case "$branch" in
  center_crop)
    source_cache=/public_bme/data/zhouwl2023/MemReason_cache/phase3m_mri_preprocessing_sensitivity_20261018/volumes_center_crop_160x192x160_float16.npy
    ;;
  lowres)
    source_cache=/home_data/home/zhouwl2023/project/MemReason/outputs/phase3b_mri_representation_dev_20261014/volumes_48x56x48.npy
    ;;
  *)
    echo "unknown branch: $branch" >&2
    exit 2
    ;;
esac

local_cache="/dev/shm/memreason_phase3m_${branch}_${USER}.npy"
cleanup() {
  rm -f "$local_cache"
}
trap cleanup EXIT
cp "$source_cache" "$local_cache"
cd /home_data/home/zhouwl2023/project/MemReason
bash scripts/run_phase3m_branch.sh "$branch" "$local_cache"
