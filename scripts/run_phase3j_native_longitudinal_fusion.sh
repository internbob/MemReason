#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home_data/home/zhouwl2023/project/MemReason
PYTHON_BIN=/home_data/home/zhouwl2023/miniconda3/bin/python
CONFIG_PATH=${1:-$PROJECT_ROOT/configs/phase3j_native_longitudinal_gated_fusion.yaml}

cd "$PROJECT_ROOT"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

"$PYTHON_BIN" -m memreason_rl.experiments.phase3j_native_longitudinal_fusion \
  --config "$CONFIG_PATH"
