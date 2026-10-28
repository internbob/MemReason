#!/usr/bin/env bash
set -euo pipefail

ROOT=/home_data/home/zhouwl2023/project/MemReason
PYTHON_BIN=/home_data/home/zhouwl2023/miniconda3/bin/python
LOG_DIR="$ROOT/outputs/phase3k_cropped_classification_logs_20261017"
cd "$ROOT"
mkdir -p "$LOG_DIR"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"$PYTHON_BIN" -m memreason_rl.experiments.phase3g_mri_diagnosis \
  --config configs/phase3k_cropped_mri_diagnosis_crossfit_shard_b.yaml \
  2>&1 | tee "$LOG_DIR/02_diagnosis_shard_b.log"
