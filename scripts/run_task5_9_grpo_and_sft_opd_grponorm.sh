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
#   RUN_TEACHER_PREP=0 bash scripts/run_task5_9_grpo_and_sft_opd_grponorm.sh
#   RUN_GRPO=0 bash scripts/run_task5_9_grpo_and_sft_opd_grponorm.sh
#   RUN_GRPO=0 RUN_TEACHER_PREP=0 RUN_HYBRID=1 bash scripts/run_task5_9_grpo_and_sft_opd_grponorm.sh
#   HYBRID_LAMBDAS="1.0" SEEDS="1 2 3" bash scripts/run_task5_9_grpo_and_sft_opd_grponorm.sh

set -euo pipefail

TASKS="${TASKS:-5 9}"
SEEDS="${SEEDS:-1 2 3}"
TEACHER_SEED="${TEACHER_SEED:-0}"
TEACHER_BC_STEPS="${TEACHER_BC_STEPS:-1000}"
TEACHER_BC_SAVE_STEPS="${TEACHER_BC_SAVE_STEPS:-[250,500,750,1000]}"
TRAIN_OPD_SFT_MATCH_IMAGE_ROTATION="${TRAIN_OPD_SFT_MATCH_IMAGE_ROTATION:-1}"
TRAIN_OPD_SFT_GRIPPER_FROM_NEG1_0_TO_0_1="${TRAIN_OPD_SFT_GRIPPER_FROM_NEG1_0_TO_0_1:-1}"
TRAIN_OPD_FORCE_RETRAIN_TEACHER="${TRAIN_OPD_FORCE_RETRAIN_TEACHER:-1}"
TEACHER_EXTRA_TAG="${TEACHER_EXTRA_TAG:-teacherprep_seed${TEACHER_SEED}}"
if [[ "${TRAIN_OPD_SFT_GRIPPER_FROM_NEG1_0_TO_0_1}" == "1" && "${TEACHER_EXTRA_TAG}" != *"grip01"* ]]; then
  TEACHER_EXTRA_TAG="${TEACHER_EXTRA_TAG}_grip01"
fi
HYBRID_LAMBDAS="${HYBRID_LAMBDAS:-0.1 1.0}"
MAX_EPOCH="${MAX_EPOCH:-200}"
SAVE_INTERVAL="${SAVE_INTERVAL:-25}"
RPS_MODE="${RPS_MODE:-rps32}"
GROUP_SIZE="${GROUP_SIZE:-8}"
NUM_GROUP_ENVS="${NUM_GROUP_ENVS:-4}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-a143}"
RUN_GRPO="${RUN_GRPO:-1}"
RUN_TEACHER_PREP="${RUN_TEACHER_PREP:-1}"
RUN_HYBRID="${RUN_HYBRID:-0}"

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
echo "  teacher_seed=${TEACHER_SEED}"
echo "  teacher_bc_steps=${TEACHER_BC_STEPS}"
echo "  teacher_bc_save_steps=${TEACHER_BC_SAVE_STEPS}"
echo "  train_opd_sft_match_image_rotation=${TRAIN_OPD_SFT_MATCH_IMAGE_ROTATION}"
echo "  train_opd_sft_gripper_from_neg1_0_to_0_1=${TRAIN_OPD_SFT_GRIPPER_FROM_NEG1_0_TO_0_1}"
echo "  train_opd_force_retrain_teacher=${TRAIN_OPD_FORCE_RETRAIN_TEACHER}"
echo "  teacher_extra_tag=${TEACHER_EXTRA_TAG}"
echo "  max_epoch=${MAX_EPOCH}"
echo "  save_interval=${SAVE_INTERVAL}"
echo "  rps_mode=${RPS_MODE}"
echo "  group_size=${GROUP_SIZE}"
echo "  num_group_envs=${NUM_GROUP_ENVS}"
echo "  rollout_epoch=${ROLLOUT_EPOCH}"
echo "  hybrid_lambdas=${HYBRID_LAMBDAS}"
echo "  run_grpo=${RUN_GRPO}"
echo "  run_teacher_prep=${RUN_TEACHER_PREP}"
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

if [[ "${RUN_TEACHER_PREP}" == "1" ]]; then
  echo "Launching seed-${TEACHER_SEED} SFT-teacher BC prep on tasks ${TASKS}"
  echo "  checkpoints: ${TEACHER_BC_SAVE_STEPS}"
  RUN_MODE=train \
  BASE_MODEL=1 \
  GRPO_HP_FROM_SWEEP=1 \
  OPD_USE_TEACHER_MAPPING=0 \
  TRAIN_TASK_INPUTS_OVERRIDE="${TASKS}" \
  TRAIN_MAX_EPOCHS_OVERRIDE=0 \
  TRAIN_SEEDS_OVERRIDE="${TEACHER_SEED}" \
  TRAIN_GROUP_SIZES_OVERRIDE="${GROUP_SIZE}" \
  TRAIN_NUM_GROUP_ENVS_OVERRIDE="${NUM_GROUP_ENVS}" \
  TRAIN_ROLLOUT_EPOCHS_OVERRIDE="${ROLLOUT_EPOCH}" \
  TRAIN_OPD_BC_STEPS_OVERRIDE="${TEACHER_BC_STEPS}" \
  TRAIN_OPD_BC_SAVE_STEPS="${TEACHER_BC_SAVE_STEPS}" \
  TRAIN_OPD_FORCE_RETRAIN_TEACHER="${TRAIN_OPD_FORCE_RETRAIN_TEACHER}" \
  TRAIN_OPD_SFT_MATCH_IMAGE_ROTATION="${TRAIN_OPD_SFT_MATCH_IMAGE_ROTATION}" \
  TRAIN_OPD_SFT_GRIPPER_FROM_NEG1_0_TO_0_1="${TRAIN_OPD_SFT_GRIPPER_FROM_NEG1_0_TO_0_1}" \
  TRAIN_OPD_NORMALIZE_ADVANTAGES_OVERRIDE=1 \
  TRAIN_OPD_REWARD_NORMALIZATIONS_OVERRIDE=group_zscore \
  TRAIN_OPD_LOSS_TYPES_OVERRIDE=embodied_opd_reinforce \
  SWEEP_WANDB_EXTRA_TAG="${TEACHER_EXTRA_TAG}" \
  SWEEP_SAVE_INTERVAL="${SAVE_INTERVAL}" \
  SLURM_ACCOUNT="${SLURM_ACCOUNT}" \
  USE_MINIMAL_SBATCH_RESOURCES=1 \
  bash examples/crl_experiment/jobs/embodiment_slurm_opd_sweep.sh
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
