#!/usr/bin/env bash
set -euo pipefail

ROOT=/home_data/home/zhouwl2023/project/MemReason
PYTHON_BIN=/home_data/home/zhouwl2023/miniconda3/bin/python
LOG_DIR="$ROOT/outputs/phase3k_cropped_classification_logs_20261017"
DURABLE_CACHE=/public_bme/data/zhouwl2023/MemReason_cache/phase3k_cropped_t1_cache_20261017/volumes_center_crop_160x192x160_float16.npy
RUNTIME_DIR=/dev/shm/zhouwl2023/phase3k_cropped_t1_cache_20261017
RUNTIME_CACHE="$RUNTIME_DIR/volumes_center_crop_160x192x160_float16.npy"

mkdir -p "$LOG_DIR" "$RUNTIME_DIR"
cd "$ROOT"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if [[ ! -f "$RUNTIME_CACHE" ]] || [[ $(stat -c%s "$RUNTIME_CACHE") -ne $(stat -c%s "$DURABLE_CACHE") ]]; then
  cp "$DURABLE_CACHE" "$RUNTIME_CACHE.tmp"
  mv "$RUNTIME_CACHE.tmp" "$RUNTIME_CACHE"
fi

"$PYTHON_BIN" -m memreason_rl.experiments.phase3h_mci_conversion \
  --config configs/phase3l_cropped_mci_conversion_crossfit.yaml \
  2>&1 | tee "$LOG_DIR/03_conversion.log"
