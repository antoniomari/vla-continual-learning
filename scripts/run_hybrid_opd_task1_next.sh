#!/bin/bash
# Next Hybrid OPD sweep.
#
# Active delta runs:
#   1. Task 1, no reward normalization, lambda=0.1, extra seeds 1 and 3
#   2. Task 4, no reward normalization, lambdas=0.1 and 1.0, seeds 1, 2, and 3
#
# Already launched in the previous script version, kept here as comments:
#   - group z-score, lambda=1.0, extra seeds 1 and 3
#   - group z-score, lambda=2.0 and 5.0, one seed
#   - no reward normalization, lambda=0.1 and 1.0, one seed
#   - no reward normalization, lambda=10.0, one seed
#   - no reward normalization, lambda=1.0, extra seeds 1 and 3
#
# Preview:
#   DRY_RUN=1 bash scripts/run_hybrid_opd_task1_next.sh
#
# Submit:
#   bash scripts/run_hybrid_opd_task1_next.sh

set -euo pipefail

TASKS="1"
SINGLE_SEED="${SINGLE_SEED:-2}"
EXTRA_SEEDS="${EXTRA_SEEDS:-1 3}"
ALL_SEEDS="${ALL_SEEDS:-1 2 3}"

COMMON_ENV=(
  "TASKS=${TASKS}"
  "MAX_EPOCH=${MAX_EPOCH:-200}"
  "SAVE_INTERVAL=${SAVE_INTERVAL:-25}"
  "GROUP_SIZE=${GROUP_SIZE:-8}"
  "NUM_GROUP_ENVS=${NUM_GROUP_ENVS:-4}"
  "ROLLOUT_EPOCH=${ROLLOUT_EPOCH:-1}"
)

run_hybrid() {
  env "${COMMON_ENV[@]}" "$@" bash scripts/run_sft_opd_success_gate.sh
}

echo "============================================================"
echo "Hybrid OPD next sweep"
echo "  extra_seeds=${EXTRA_SEEDS}"
echo "  single_seed=${SINGLE_SEED}"
echo "  all_seeds=${ALL_SEEDS}"
echo "============================================================"

echo ""
# Already launched:
# echo ""
# echo "[launched] Group z-score, lambda=1.0, extra seeds"
# run_hybrid \
#   "HYBRID_SEEDS=${EXTRA_SEEDS}" \
#   "NORMALIZE_ADVANTAGES=1" \
#   "REWARD_NORMALIZATIONS=group_zscore" \
#   "LAMBDAS=1.0"
#
# echo ""
# echo "[launched] Group z-score, lambda=2.0 and 5.0, one seed"
# run_hybrid \
#   "HYBRID_SEEDS=${SINGLE_SEED}" \
#   "NORMALIZE_ADVANTAGES=1" \
#   "REWARD_NORMALIZATIONS=group_zscore" \
#   "LAMBDAS=2.0 5.0"
#
# echo ""
# echo "[launched] No reward normalization, lambda=0.1 and 1.0, one seed"
# run_hybrid \
#   "HYBRID_SEEDS=${SINGLE_SEED}" \
#   "NORMALIZE_ADVANTAGES=0" \
#   "REWARD_NORMALIZATIONS=__empty__" \
#   "LAMBDAS=0.1 1.0"

echo "[1/2] Task 1 no reward normalization, lambda=0.1, extra seeds"
run_hybrid \
  "HYBRID_SEEDS=${EXTRA_SEEDS}" \
  "NORMALIZE_ADVANTAGES=0" \
  "REWARD_NORMALIZATIONS=__empty__" \
  "LAMBDAS=0.1"

echo ""
echo "[2/2] Task 4 no reward normalization, lambdas=0.1 and 1.0, three seeds"
run_hybrid \
  "TASKS=4" \
  "HYBRID_SEEDS=${ALL_SEEDS}" \
  "NORMALIZE_ADVANTAGES=0" \
  "REWARD_NORMALIZATIONS=__empty__" \
  "LAMBDAS=0.1 1.0"
