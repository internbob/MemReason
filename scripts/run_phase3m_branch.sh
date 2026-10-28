#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 {center_crop|lowres} [local-cache-path]" >&2
  exit 2
fi

branch="$1"
cache_override="${2:-}"
cd /home_data/home/zhouwl2023/project/MemReason
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

args=(--config configs/phase3m_mri_preprocessing_sensitivity.yaml --representation "$branch")
if [[ -n "$cache_override" ]]; then
  args+=(--cache-override "$cache_override")
fi
python -m memreason_rl.experiments.phase3m_mri_preprocessing_sensitivity "${args[@]}"
