#!/usr/bin/env bash
# Prepare, train, summarize, and verify Phase 7A on allocated idle GPU nodes.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home_data/home/zhouwl2023/miniconda3/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-configs/phase7a_joint_multitask_native_t1.yaml}"
SHARED_CACHE_ROOT="${PHASE7A_SHARED_CACHE_ROOT:-/public_bme/data/zhouwl2023/MemReason_cache/phase7a_joint_multitask_native_t1_dev_20261030}"
LOCAL_CACHE_ROOT="${PHASE7A_LOCAL_CACHE_ROOT:-/tmp/$USER/MemReason_cache/phase7a_joint_multitask_native_t1_dev_20261030}"
CACHE_FILENAME="volumes_native_193x229x193_float16.npy"

mapfile -t ALLOCATED_NODES < <(
  squeue -h -u "$USER" -t R -o "%N" \
    | while read -r spec; do scontrol show hostnames "$spec"; done \
    | awk 'NF && !seen[$0]++'
)
if (( ${#ALLOCATED_NODES[@]} == 0 )); then
  echo "No running allocated compute node was found." >&2
  exit 1
fi

IDLE_NODES=()
for node in "${ALLOCATED_NODES[@]}"; do
  used="$(ssh "$node" "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1")"
  if [[ "$used" =~ ^[0-9]+$ ]] && (( used < 1024 )); then
    IDLE_NODES+=("$node")
  fi
done
if (( ${#IDLE_NODES[@]} == 0 )); then
  echo "Allocated GPUs are busy; refusing to oversubscribe." >&2
  exit 1
fi

NODE_A="${GPU_NODE_A:-${IDLE_NODES[0]}}"
NODE_B="${GPU_NODE_B:-${IDLE_NODES[1]:-${IDLE_NODES[0]}}}"
REMOTE_PREFIX="cd $REPO_ROOT && export PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

echo "[1/4] Preparing native cache on $NODE_A"
ssh "$NODE_A" "$REMOTE_PREFIX && $PYTHON_BIN -m memreason_rl.experiments.phase7a_joint_multitask_native_t1 --config $CONFIG_PATH --mode prepare"

mkdir -p "$REPO_ROOT/outputs/phase7a_joint_multitask_native_t1_dev_20261030/logs"
echo "[1b/4] Staging native cache on node-local disks"
STAGE_PIDS=()
for node in "$NODE_A" "$NODE_B"; do
  if [[ " ${STAGED_NODES[*]:-} " == *" $node "* ]]; then
    continue
  fi
  STAGED_NODES+=("$node")
  ssh "$node" "mkdir -p '$LOCAL_CACHE_ROOT' && \
    if [[ ! -f '$LOCAL_CACHE_ROOT/$CACHE_FILENAME' || \
      \$(stat -c %s '$LOCAL_CACHE_ROOT/$CACHE_FILENAME' 2>/dev/null || echo 0) -ne \
      \$(stat -c %s '$SHARED_CACHE_ROOT/$CACHE_FILENAME') ]]; then \
      cp '$SHARED_CACHE_ROOT/$CACHE_FILENAME' '$LOCAL_CACHE_ROOT/$CACHE_FILENAME.copying' && \
      mv '$LOCAL_CACHE_ROOT/$CACHE_FILENAME.copying' '$LOCAL_CACHE_ROOT/$CACHE_FILENAME'; \
    fi" &
  STAGE_PIDS+=("$!")
done
for stage_pid in "${STAGE_PIDS[@]}"; do wait "$stage_pid"; done
TRAIN_PREFIX="$REMOTE_PREFIX && export MEMREASON_PHASE7A_CACHE_ROOT='$LOCAL_CACHE_ROOT'"

echo "[2/4] Training folds 0,1,2 on $NODE_A and folds 3,4 on $NODE_B"
TRAIN_PIDS=()
for fold in 0 1 2; do
  ssh "$NODE_A" "$TRAIN_PREFIX && $PYTHON_BIN -m memreason_rl.experiments.phase7a_joint_multitask_native_t1 --config $CONFIG_PATH --mode train --folds $fold" \
    > "$REPO_ROOT/outputs/phase7a_joint_multitask_native_t1_dev_20261030/logs/fold_${fold}.log" 2>&1 &
  TRAIN_PIDS+=("$!")
done
if [[ "$NODE_A" == "$NODE_B" ]]; then
  for train_pid in "${TRAIN_PIDS[@]}"; do wait "$train_pid"; done
  TRAIN_PIDS=()
fi
for fold in 3 4; do
  ssh "$NODE_B" "$TRAIN_PREFIX && $PYTHON_BIN -m memreason_rl.experiments.phase7a_joint_multitask_native_t1 --config $CONFIG_PATH --mode train --folds $fold" \
    > "$REPO_ROOT/outputs/phase7a_joint_multitask_native_t1_dev_20261030/logs/fold_${fold}.log" 2>&1 &
  TRAIN_PIDS+=("$!")
done
for train_pid in "${TRAIN_PIDS[@]}"; do wait "$train_pid"; done

echo "[3/4] Summarizing on $NODE_A"
ssh "$NODE_A" "$TRAIN_PREFIX && $PYTHON_BIN -m memreason_rl.experiments.phase7a_joint_multitask_native_t1 --config $CONFIG_PATH --mode summarize"

echo "[4/4] Verifying on $NODE_A"
ssh "$NODE_A" "$TRAIN_PREFIX && $PYTHON_BIN scripts/verify_phase7a_joint_multitask_native_t1.py --config $CONFIG_PATH"
echo "Phase 7A completed."
