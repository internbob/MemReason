#!/usr/bin/env bash
set -euo pipefail

ROOT=/home_data/home/zhouwl2023/project/MemReason
MAIN="$ROOT/outputs/phase3k_cropped_mri_diagnosis_crossfit_20261017"
SHARD="$ROOT/outputs/phase3k_cropped_mri_diagnosis_crossfit_shard_b_20261017"
CONVERSION="$ROOT/outputs/phase3l_cropped_mci_conversion_crossfit_20261017"
LOG_DIR="$ROOT/outputs/phase3k_cropped_classification_logs_20261017"
PYTHON_BIN=/home_data/home/zhouwl2023/miniconda3/bin/python
mkdir -p "$MAIN/checkpoints" "$SHARD/checkpoints" "$CONVERSION/checkpoints" "$LOG_DIR"

count_checkpoints() {
  local directory=$1
  find "$directory/checkpoints" -name '*.pt' 2>/dev/null | wc -l
}

while true; do
  main_count=$(count_checkpoints "$MAIN")
  shard_count=$(count_checkpoints "$SHARD")
  conversion_count=$(count_checkpoints "$CONVERSION")
  if [[ "$main_count" -eq 9 && "$shard_count" -eq 6 && "$conversion_count" -eq 30 \
        && -f "$MAIN/crossfit_predictions.csv" \
        && -f "$SHARD/crossfit_predictions.csv" \
        && -f "$CONVERSION/conversion_metrics.csv" ]]; then
    break
  fi
  sleep 30
done

cd "$ROOT"
"$PYTHON_BIN" scripts/merge_phase3k_diagnostic_shards.py \
  2>&1 | tee "$LOG_DIR/05_merge_diagnostic_shards.log"
"$PYTHON_BIN" scripts/verify_phase3k_cropped_classification.py \
  2>&1 | tee "$LOG_DIR/06_final_verification.log"
