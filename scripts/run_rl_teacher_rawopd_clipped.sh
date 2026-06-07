#!/bin/bash
# Launch RL-teacher OPD with raw OPD reward and GRPO/PPO-style clipped OPD loss.
#
# This is the RL-teacher counterpart to the current SFT-teacher hybrid loss style:
#   - teacher source: mapped RL teacher checkpoints from teacher_rl_by_task
#   - OPD reward/advantage branch: raw, no normalization
#   - actor objective: embodied_opd (GRPO/PPO-style clipped ratio loss)
#   - no GRPO environment reward is mixed in here
#
# Preview on cluster:
#   DRY_RUN=1 bash scripts/run_rl_teacher_rawopd_clipped.sh
#
# Submit on cluster:
#   bash scripts/run_rl_teacher_rawopd_clipped.sh

set -euo pipefail

TASKS="${TASKS:-1 4 9}"
SEED="${SEED:-2}"
MAX_EPOCH="${MAX_EPOCH:-200}"
SAVE_INTERVAL="${SAVE_INTERVAL:-25}"
GROUP_SIZE="${GROUP_SIZE:-8}"
NUM_GROUP_ENVS="${NUM_GROUP_ENVS:-4}"
ROLLOUT_EPOCH="${ROLLOUT_EPOCH:-1}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-a143}"
WANDB_EXTRA_TAG="${WANDB_EXTRA_TAG:-rawopd_clipped}"

echo "============================================================"
echo "RL-teacher OPD: raw OPD reward + clipped OPD loss"
echo "  tasks=${TASKS}"
echo "  seed=${SEED}"
echo "  max_epoch=${MAX_EPOCH}"
echo "  save_interval=${SAVE_INTERVAL}"
echo "  group_size=${GROUP_SIZE}"
echo "  num_group_envs=${NUM_GROUP_ENVS}"
echo "  rollout_epoch=${ROLLOUT_EPOCH}"
echo "  rollouts_per_step=$((GROUP_SIZE * NUM_GROUP_ENVS * ROLLOUT_EPOCH))"
echo "  teacher_mapping_group=teacher_rl_by_task"
echo "  loss_type=embodied_opd"
echo "  normalize_advantages=0"
echo "  reward_normalization=none"
echo "  wandb_extra_tag=${WANDB_EXTRA_TAG}"
echo "============================================================"

RUN_MODE=train \
BASE_MODEL=1 \
GRPO_HP_FROM_SWEEP=1 \
OPD_TEACHER_MAPPING_GROUP=teacher_rl_by_task \
OPD_USE_TEACHER_MAPPING=1 \
OPD_REQUIRE_MAPPED_TEACHER=1 \
TRAIN_TASK_INPUTS_OVERRIDE="${TASKS}" \
TRAIN_MAX_EPOCHS_OVERRIDE="${MAX_EPOCH}" \
TRAIN_SEEDS_OVERRIDE="${SEED}" \
TRAIN_GROUP_SIZES_OVERRIDE="${GROUP_SIZE}" \
TRAIN_NUM_GROUP_ENVS_OVERRIDE="${NUM_GROUP_ENVS}" \
TRAIN_ROLLOUT_EPOCHS_OVERRIDE="${ROLLOUT_EPOCH}" \
TRAIN_OPD_BC_STEPS_OVERRIDE=0 \
TRAIN_OPD_RL_TEACHER=0 \
TRAIN_OPD_NORMALIZE_ADVANTAGES_OVERRIDE=0 \
TRAIN_OPD_REWARD_NORMALIZATIONS_OVERRIDE=__empty__ \
TRAIN_OPD_LOSS_TYPES_OVERRIDE=embodied_opd \
TRAIN_OPD_PRECOMPUTE_TEACHER_IN_ROLLOUT=1 \
TRAIN_OPD_TEACHER_STASH_LOGPROBS_ON_CPU=0 \
TRAIN_OPD_TEACHER_MICRO_BATCH_SIZES_OVERRIDE=32 \
SWEEP_WANDB_EXTRA_TAG="${WANDB_EXTRA_TAG}" \
SWEEP_SAVE_INTERVAL="${SAVE_INTERVAL}" \
SLURM_ACCOUNT="${SLURM_ACCOUNT}" \
USE_MINIMAL_SBATCH_RESOURCES=1 \
bash examples/crl_experiment/jobs/embodiment_slurm_opd_sweep.sh
