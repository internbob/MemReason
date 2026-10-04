#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=
"${PYTHON_BIN:-python}" -u -m memreason_rl.data.prepare > reports_prepare.log 2>&1
bash scripts/check_project.sh
"${PYTHON_BIN:-python}" scripts/finalize_manifest.py
