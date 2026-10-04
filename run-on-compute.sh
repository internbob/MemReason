#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="${TASK_ROOT}"
exec /home_data/home/zhouwl2023/project/Cortexbridge/cortexweave_roiclip/run-on-gpu-node.sh "$@"
