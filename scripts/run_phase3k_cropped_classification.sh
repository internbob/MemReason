#!/usr/bin/env bash
set -euo pipefail

ROOT=/home_data/home/zhouwl2023/project/MemReason
PYTHON_BIN=/home_data/home/zhouwl2023/miniconda3/bin/python
LOG_DIR="$ROOT/outputs/phase3k_cropped_classification_logs_20261017"
mkdir -p "$LOG_DIR"
cd "$ROOT"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"$PYTHON_BIN" -m memreason_rl.experiments.phase3k_cropped_mri_cache \
  --config configs/phase3k_cropped_mri_diagnosis_crossfit.yaml \
  2>&1 | tee "$LOG_DIR/01_cache.log"

DURABLE_CACHE=/public_bme/data/zhouwl2023/MemReason_cache/phase3k_cropped_t1_cache_20261017/volumes_center_crop_160x192x160_float16.npy
RUNTIME_DIR=/dev/shm/zhouwl2023/phase3k_cropped_t1_cache_20261017
RUNTIME_CACHE="$RUNTIME_DIR/volumes_center_crop_160x192x160_float16.npy"
mkdir -p "$RUNTIME_DIR"
if [[ ! -f "$RUNTIME_CACHE" ]] || [[ $(stat -c%s "$RUNTIME_CACHE") -ne $(stat -c%s "$DURABLE_CACHE") ]]; then
  cp "$DURABLE_CACHE" "$RUNTIME_CACHE.tmp"
  mv "$RUNTIME_CACHE.tmp" "$RUNTIME_CACHE"
fi
echo "runtime_cache=$RUNTIME_CACHE bytes=$(stat -c%s "$RUNTIME_CACHE")" | tee "$LOG_DIR/01_runtime_cache.log"

"$PYTHON_BIN" -m memreason_rl.experiments.phase3g_mri_diagnosis \
  --config configs/phase3k_cropped_mri_diagnosis_crossfit.yaml \
  2>&1 | tee "$LOG_DIR/02_diagnosis.log"

CONVERSION_OUTPUT="$ROOT/outputs/phase3l_cropped_mci_conversion_crossfit_20261017"
mkdir -p "$CONVERSION_OUTPUT/checkpoints"
CONVERSION_CHECKPOINTS=$(find "$CONVERSION_OUTPUT/checkpoints" -name '*.pt' 2>/dev/null | wc -l)
if [[ -f "$CONVERSION_OUTPUT/conversion_metrics.csv" && "$CONVERSION_CHECKPOINTS" -eq 30 ]]; then
  echo "conversion_complete=true checkpoints=$CONVERSION_CHECKPOINTS" | tee "$LOG_DIR/03_conversion.log"
else
  "$PYTHON_BIN" -m memreason_rl.experiments.phase3h_mci_conversion \
    --config configs/phase3l_cropped_mci_conversion_crossfit.yaml \
    2>&1 | tee "$LOG_DIR/03_conversion.log"
fi

"$PYTHON_BIN" scripts/verify_phase3k_cropped_classification.py \
  2>&1 | tee "$LOG_DIR/04_verification.log"
