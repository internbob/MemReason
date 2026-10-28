#!/usr/bin/env bash
set -euo pipefail

ROOT=/home_data/home/zhouwl2023/project/MemReason
LOG_DIR="$ROOT/outputs/phase3k_cropped_classification_logs_20261017"
mkdir -p "$LOG_DIR"

while pgrep -f '/python -m memreason_rl.experiments.phase3h_mci_conversion' >/dev/null; do
  sleep 30
done

cd "$ROOT"
exec bash scripts/run_phase3k_diagnostic_shard_b.sh
