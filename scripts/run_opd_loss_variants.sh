#!/bin/bash
# Submit/preview the OPD loss-normalization variants through the canonical SLURM sweep scripts.
#
# This wrapper intentionally calls:
#   - examples/crl_experiment/jobs/embodiment_slurm_sweep.sh for the GRPO baseline
#   - examples/crl_experiment/jobs/embodiment_slurm_opd_sweep.sh for OPD variants
#
# Defaults mirror the current sweep settings:
#   - tasks: 1 and 4
#   - OPD seed: 184
#   - GRPO seed: 4096
#   - 128 rollouts/step: group_size 8 * num_group_envs 4 * rollout_epoch 4
#
# Useful preview:
#   DRY_RUN=1 bash scripts/run_opd_loss_variants.sh
#
# Useful narrowing:
#   TASKS="1" DRY_RUN=1 bash scripts/run_opd_loss_variants.sh

set -euo pipefail

TASKS="${TASKS:-1 4}"
OPD_SEED="${OPD_SEED:-184}"
GRPO_SEED="${GRPO_SEED:-4096}"
MAX_EPOCH="${MAX_EPOCH:-60}"
GROUP_SIZE="${GROUP_SIZE:-8}"
NUM_GROUP_ENVS="${NUM_GROUP_ENVS:-4}"
ROLLOUT_EPOCH="${ROLLOUT_EPOCH:-4}"

GRPO_SWEEP="examples/crl_experiment/jobs/embodiment_slurm_sweep.sh"
OPD_SWEEP="examples/crl_experiment/jobs/embodiment_slurm_opd_sweep.sh"

COMMON_ENV=(
  "RUN_MODE=train"
  "BASE_MODEL=1"
  "GRPO_HP_FROM_SWEEP=1"
  "TRAIN_TASK_INPUTS_OVERRIDE=${TASKS}"
  "TRAIN_MAX_EPOCHS_OVERRIDE=${MAX_EPOCH}"
  "TRAIN_GROUP_SIZES_OVERRIDE=${GROUP_SIZE}"
  "TRAIN_NUM_GROUP_ENVS_OVERRIDE=${NUM_GROUP_ENVS}"
  "TRAIN_ROLLOUT_EPOCHS_OVERRIDE=${ROLLOUT_EPOCH}"
)

run_grpo_baseline() {
  echo "============================================================"
  echo "GRPO baseline through ${GRPO_SWEEP}"
  echo "  loss_type comes from the GRPO config: embodied_grpo"
  echo "============================================================"

  env "${COMMON_ENV[@]}" \
    "TRAIN_SEEDS_OVERRIDE=${GRPO_SEED}" \
    "TRAIN_CONFIG_NAMES_OVERRIDE=crl_experiment/libero_spatial_grpo_openvlaoft_spatial" \
    bash "${GRPO_SWEEP}"
}

run_opd_variant() {
  local label="$1"
  local normalize_advantages="$2"
  local reward_normalizations="$3"
  local loss_types="$4"

  echo "============================================================"
  echo "${label} through ${OPD_SWEEP}"
  echo "  TRAIN_OPD_NORMALIZE_ADVANTAGES=${normalize_advantages}"
  echo "  TRAIN_OPD_REWARD_NORMALIZATIONS=${reward_normalizations}"
  echo "  TRAIN_OPD_LOSS_TYPES=${loss_types}"
  echo "============================================================"

  env "${COMMON_ENV[@]}" \
    "TRAIN_SEEDS_OVERRIDE=${OPD_SEED}" \
    "TRAIN_CONFIG_NAMES_OVERRIDE=crl_experiment/libero_spatial_opd_openvlaoft_spatial" \
    "TRAIN_OPD_NORMALIZE_ADVANTAGES_OVERRIDE=${normalize_advantages}" \
    "TRAIN_OPD_REWARD_NORMALIZATIONS_OVERRIDE=${reward_normalizations}" \
    "TRAIN_OPD_LOSS_TYPES_OVERRIDE=${loss_types}" \
    bash "${OPD_SWEEP}"
}

# Reference GRPO run using the existing GRPO SLURM sweep.
run_grpo_baseline

# Current OPD setting: REINFORCE-style OPD objective with group-zscore-normalized OPD rewards.
run_opd_variant "Current OPD group-normalized REINFORCE" "1" "group_zscore" "embodied_opd_reinforce"

# OPD ablation: same REINFORCE-style objective, but no reward/advantage normalization.
# "__empty__" tells the OPD sweep to leave SWEEP_OPD_REWARD_NORMALIZATION empty, so the
# W&B name does not append a misleading norm suffix; the sweep prefix includes "nonorm".
run_opd_variant "OPD REINFORCE without normalization" "0" "__empty__" "embodied_opd_reinforce"

# OPD clipped-loss variant: keep group-zscore-normalized OPD rewards, but use embodied_opd,
# which dispatches to the GRPO-style clipped ratio loss. The sweep prefix includes "grpo_loss".
run_opd_variant "OPD with GRPO clipping loss" "1" "group_zscore" "embodied_opd"
