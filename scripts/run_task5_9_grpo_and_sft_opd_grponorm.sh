#!/bin/bash
# Submit/preview task 5 and 9 runs for:
#   1. GRPO baseline.
#   2. Current SFT-teacher Hybrid OPD:
#      normalized GRPO/env advantage + failure-gated raw SFT-OPD advantage.
#
# The older hybrid variant with non-normalized environment reward is deprecated
# and is intentionally not launched here.
#
# Preview:
#   DRY_RUN=1 bash scripts/run_task5_9_grpo_and_sft_opd_grponorm.sh
#
# Submit:
#   bash scripts/run_task5_9_grpo_and_sft_opd_grponorm.sh
#
# Useful overrides:
#   RUN_GRPO=0 bash scripts/run_task5_9_grpo_and_sft_opd_grponorm.sh
#   RUN_HYBRID=0 bash scripts/run_task5_9_grpo_and_sft_opd_grponorm.sh
#   HYBRID_LAMBDAS="1.0" SEEDS="1 2 3" bash scripts/run_task5_9_grpo_and_sft_opd_grponorm.sh

set -euo pipefail

TASKS="${TASKS:-5 9}"
SEEDS="${SEEDS:-1 2 3}"
HYBRID_LAMBDAS="${HYBRID_LAMBDAS:-0.1 1.0}"
MAX_EPOCH="${MAX_EPOCH:-200}"
SAVE_INTERVAL="${SAVE_INTERVAL:-25}"
RPS_MODE="${RPS_MODE:-rps32}"
GROUP_SIZE="${GROUP_SIZE:-8}"
NUM_GROUP_ENVS="${NUM_GROUP_ENVS:-4}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-a143}"
RUN_GRPO="${RUN_GRPO:-1}"
RUN_HYBRID="${RUN_HYBRID:-1}"

case "${RPS_MODE}" in
  rps32)
    ROLLOUT_EPOCH="${ROLLOUT_EPOCH:-1}"
    ;;
  rps128)
    ROLLOUT_EPOCH="${ROLLOUT_EPOCH:-4}"
    ;;
  *)
    echo "ERROR: RPS_MODE must be 'rps32' or 'rps128', got '${RPS_MODE}'"
    exit 1
    ;;
esac

echo "============================================================"
echo "Task 5/9 GRPO + current SFT-teacher Hybrid OPD launcher"
echo "  tasks=${TASKS}"
echo "  seeds=${SEEDS}"
echo "  max_epoch=${MAX_EPOCH}"
echo "  save_interval=${SAVE_INTERVAL}"
echo "  rps_mode=${RPS_MODE}"
echo "  group_size=${GROUP_SIZE}"
echo "  num_group_envs=${NUM_GROUP_ENVS}"
echo "  rollout_epoch=${ROLLOUT_EPOCH}"
echo "  hybrid_lambdas=${HYBRID_LAMBDAS}"
echo "  run_grpo=${RUN_GRPO}"
echo "  run_hybrid=${RUN_HYBRID}"
echo "============================================================"

if [[ "${RUN_GRPO}" == "1" ]]; then
  echo "Launching GRPO baseline on tasks ${TASKS}"
  TASKS="${TASKS}" \
  GRPO_SEEDS="${SEEDS}" \
  MAX_EPOCH="${MAX_EPOCH}" \
  SAVE_INTERVAL="${SAVE_INTERVAL}" \
  RPS_MODE="${RPS_MODE}" \
  GROUP_SIZE="${GROUP_SIZE}" \
  NUM_GROUP_ENVS="${NUM_GROUP_ENVS}" \
  ROLLOUT_EPOCH="${ROLLOUT_EPOCH}" \
  SLURM_ACCOUNT="${SLURM_ACCOUNT}" \
  bash scripts/run_grpo_rps32_new_seeds.sh
fi

if [[ "${RUN_HYBRID}" == "1" ]]; then
  echo "Launching current Hybrid OPD on tasks ${TASKS}"
  echo "  loss=embodied_opd_grpo_plus_success_gate"
  echo "  env_grpo_advantage=normalized"
  echo "  opd_advantage=raw, SFT-teacher, failure-gated"
  TASKS="${TASKS}" \
  HYBRID_SEEDS="${SEEDS}" \
  LAMBDAS="${HYBRID_LAMBDAS}" \
  MAX_EPOCH="${MAX_EPOCH}" \
  SAVE_INTERVAL="${SAVE_INTERVAL}" \
  GROUP_SIZE="${GROUP_SIZE}" \
  NUM_GROUP_ENVS="${NUM_GROUP_ENVS}" \
  ROLLOUT_EPOCH="${ROLLOUT_EPOCH}" \
  SLURM_ACCOUNT="${SLURM_ACCOUNT}" \
  bash scripts/run_sft_opd_rawopd_grponorm_gate.sh
fi
