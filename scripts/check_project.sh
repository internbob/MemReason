#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=
"${PYTHON_BIN:-python}" -m compileall -q memreason_rl scripts
"${PYTHON_BIN:-python}" -m pytest -q > reports_tests.log 2>&1
"${PYTHON_BIN:-python}" -u -m memreason_rl.train.stage1_cognition --smoke --exploratory > reports_stage1_smoke.log 2>&1
"${PYTHON_BIN:-python}" scripts/verify_build.py > reports_integration.log 2>&1
"${PYTHON_BIN:-python}" -m memreason_rl.eval.baselines --split val --exploratory > reports_baseline.log 2>&1
"${PYTHON_BIN:-python}" -m memreason_rl.eval.leakage > reports_leakage.log 2>&1
