#!/bin/bash
# Submit/preview GRPO-only runs at rps32 or rps128 for 200 steps.
#
# Uses:
#   examples/crl_experiment/jobs/embodiment_slurm_sweep.sh
#
# Default geometry:
#   group_size=8, num_group_envs=4
#   RPS_MODE=rps32  -> rollout_epoch=1
#   RPS_MODE=rps128 -> rollout_epoch=4
#
# Default targets:
#   tasks 1 and 4
#   seeds "2" (override with GRPO_SEEDS)
#
# Preview:
#   DRY_RUN=1 bash scripts/run_grpo_rps32_new_seeds.sh
#
# Submit:
#   bash scripts/run_grpo_rps32_new_seeds.sh
#
# Override examples:
#   GRPO_SEEDS="2 3 5" bash scripts/run_grpo_rps32_new_seeds.sh
#   TASKS="1" MAX_EPOCH=200 bash scripts/run_grpo_rps32_new_seeds.sh
#   RPS_MODE=rps128 bash scripts/run_grpo_rps32_new_seeds.sh

set -euo pipefail

TASKS="${TASKS:-1 4}"
GRPO_SEEDS="${GRPO_SEEDS:-2}"
MAX_EPOCH="${MAX_EPOCH:-200}"
SAVE_INTERVAL="${SAVE_INTERVAL:-25}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-a143}"
GROUP_SIZE="${GROUP_SIZE:-8}"
NUM_GROUP_ENVS="${NUM_GROUP_ENVS:-4}"
RPS_MODE="${RPS_MODE:-rps32}" # rps32 | rps128

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

GRPO_SWEEP="examples/crl_experiment/jobs/embodiment_slurm_sweep.sh"

ROLLOUTS_PER_STEP=$((GROUP_SIZE * NUM_GROUP_ENVS * ROLLOUT_EPOCH))
WANDB_TAG_DEFAULT="steps${MAX_EPOCH}_si${SAVE_INTERVAL}"
WANDB_TAG="${WANDB_TAG:-${WANDB_TAG_DEFAULT}}"

echo "============================================================"
echo "GRPO rps${ROLLOUTS_PER_STEP} launcher"
echo "  tasks=${TASKS}"
echo "  seeds=${GRPO_SEEDS}"
echo "  max_epoch=${MAX_EPOCH}"
echo "  save_interval=${SAVE_INTERVAL}"
echo "  rps_mode=${RPS_MODE}"
echo "  group_size=${GROUP_SIZE}"
echo "  num_group_envs=${NUM_GROUP_ENVS}"
echo "  rollout_epoch=${ROLLOUT_EPOCH}"
echo "  slurm_account=${SLURM_ACCOUNT}"
echo "============================================================"

env \
  "RUN_MODE=train" \
  "BASE_MODEL=1" \
  "GRPO_HP_FROM_SWEEP=1" \
  "TRAIN_TASK_INPUTS_OVERRIDE=${TASKS}" \
  "TRAIN_MAX_EPOCHS_OVERRIDE=${MAX_EPOCH}" \
  "TRAIN_SEEDS_OVERRIDE=${GRPO_SEEDS}" \
  "TRAIN_GROUP_SIZES_OVERRIDE=${GROUP_SIZE}" \
  "TRAIN_NUM_GROUP_ENVS_OVERRIDE=${NUM_GROUP_ENVS}" \
  "TRAIN_ROLLOUT_EPOCHS_OVERRIDE=${ROLLOUT_EPOCH}" \
  "SWEEP_SAVE_INTERVAL=${SAVE_INTERVAL}" \
  "SWEEP_WANDB_EXTRA_TAG=${WANDB_TAG}" \
  "SLURM_ACCOUNT=${SLURM_ACCOUNT}" \
  bash "${GRPO_SWEEP}"
