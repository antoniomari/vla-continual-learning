#!/bin/bash
# Rerun only the two Hybrid OPD jobs that crashed on the previous launch.
#
# Preview:
#   DRY_RUN=1 bash scripts/rerun_failed_hybrid_opd_two.sh
#
# Submit:
#   bash scripts/rerun_failed_hybrid_opd_two.sh

set -euo pipefail

COMMON_ENV=(
  "MAX_EPOCH=${MAX_EPOCH:-200}"
  "SAVE_INTERVAL=${SAVE_INTERVAL:-25}"
  "GROUP_SIZE=${GROUP_SIZE:-8}"
  "NUM_GROUP_ENVS=${NUM_GROUP_ENVS:-4}"
  "ROLLOUT_EPOCH=${ROLLOUT_EPOCH:-1}"
  "NORMALIZE_ADVANTAGES=0"
  "REWARD_NORMALIZATIONS=__empty__"
)

run_hybrid() {
  env "${COMMON_ENV[@]}" "$@" bash scripts/run_sft_opd_success_gate.sh
}

echo "============================================================"
echo "Rerunning failed no-norm Hybrid OPD jobs only"
echo "  1. task=1 seed=1 lambda=0.1"
echo "  2. task=4 seed=3 lambda=1.0"
echo "============================================================"

echo ""
echo "[1/2] Task 1, seed 1, lambda=0.1, no reward normalization"
run_hybrid \
  "TASKS=1" \
  "HYBRID_SEEDS=1" \
  "LAMBDAS=0.1"

echo ""
echo "[2/2] Task 4, seed 3, lambda=1.0, no reward normalization"
run_hybrid \
  "TASKS=4" \
  "HYBRID_SEEDS=3" \
  "LAMBDAS=1.0"
