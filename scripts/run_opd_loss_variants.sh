#!/bin/bash
# Submit/preview the OPD loss-normalization variants through the canonical SLURM sweep scripts.
#
# This wrapper intentionally calls:
#   - examples/crl_experiment/jobs/embodiment_slurm_sweep.sh for the GRPO baseline
#   - examples/crl_experiment/jobs/embodiment_slurm_opd_sweep.sh for OPD variants
#
# By default this submits/previews the new rps8_200 suite:
#   - 8 rollouts/step for 200 steps
#
# The same loss/normalization variants are submitted for each enabled suite:
#   - tasks: 1 and 4
#   - OPD seed: 184
#   - GRPO seed: 4096
#
# Useful preview:
#   DRY_RUN=1 bash scripts/run_opd_loss_variants.sh
#
# Useful narrowing:
#   TASKS="1" DRY_RUN=1 bash scripts/run_opd_loss_variants.sh
#
# Suite selection:
#   RUN_CURRENT=1 RUN_RPS8_200=0 DRY_RUN=1 bash scripts/run_opd_loss_variants.sh
#   RUN_CURRENT=1 RUN_RPS8_200=1 DRY_RUN=1 bash scripts/run_opd_loss_variants.sh

set -euo pipefail

TASKS="${TASKS:-1 4}"
OPD_SEED="${OPD_SEED:-184}"
GRPO_SEED="${GRPO_SEED:-4096}"
RUN_CURRENT="${RUN_CURRENT:-0}"
RUN_RPS8_200="${RUN_RPS8_200:-1}"
# OPD OOM-safety knobs (override per run if needed).
OPD_PRECOMPUTE_TEACHER_IN_ROLLOUT="${OPD_PRECOMPUTE_TEACHER_IN_ROLLOUT:-1}"
OPD_TEACHER_STASH_LOGPROBS_ON_CPU="${OPD_TEACHER_STASH_LOGPROBS_ON_CPU:-1}"
OPD_TEACHER_MICRO_BATCH="${OPD_TEACHER_MICRO_BATCH:-8}"
# Helps reduce fragmentation-related OOMs.
PYTORCH_CUDA_ALLOC_CONF_VALUE="${PYTORCH_CUDA_ALLOC_CONF_VALUE:-expandable_segments:True}"

GRPO_SWEEP="examples/crl_experiment/jobs/embodiment_slurm_sweep.sh"
OPD_SWEEP="examples/crl_experiment/jobs/embodiment_slurm_opd_sweep.sh"

COMMON_ENV=()

run_grpo_baseline() {
  local suite_label="$1"

  echo "============================================================"
  echo "[${suite_label}] GRPO baseline through ${GRPO_SWEEP}"
  echo "  loss_type comes from the GRPO config: embodied_grpo"
  echo "============================================================"

  env "${COMMON_ENV[@]}" \
    "TRAIN_SEEDS_OVERRIDE=${GRPO_SEED}" \
    "TRAIN_CONFIG_NAMES_OVERRIDE=crl_experiment/libero_spatial_grpo_openvlaoft_spatial" \
    bash "${GRPO_SWEEP}"
}

run_opd_variant() {
  local suite_label="$1"
  local label="$2"
  local normalize_advantages="$3"
  local reward_normalizations="$4"
  local loss_types="$5"

  echo "============================================================"
  echo "[${suite_label}] ${label} through ${OPD_SWEEP}"
  echo "  TRAIN_OPD_NORMALIZE_ADVANTAGES=${normalize_advantages}"
  echo "  TRAIN_OPD_REWARD_NORMALIZATIONS=${reward_normalizations}"
  echo "  TRAIN_OPD_LOSS_TYPES=${loss_types}"
  echo "============================================================"

  env "${COMMON_ENV[@]}" \
    "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF_VALUE}" \
    "TRAIN_SEEDS_OVERRIDE=${OPD_SEED}" \
    "TRAIN_CONFIG_NAMES_OVERRIDE=crl_experiment/libero_spatial_opd_openvlaoft_spatial" \
    "TRAIN_OPD_NORMALIZE_ADVANTAGES_OVERRIDE=${normalize_advantages}" \
    "TRAIN_OPD_REWARD_NORMALIZATIONS_OVERRIDE=${reward_normalizations}" \
    "TRAIN_OPD_LOSS_TYPES_OVERRIDE=${loss_types}" \
    "TRAIN_OPD_PRECOMPUTE_TEACHER_IN_ROLLOUT=${OPD_PRECOMPUTE_TEACHER_IN_ROLLOUT}" \
    "TRAIN_OPD_TEACHER_STASH_LOGPROBS_ON_CPU=${OPD_TEACHER_STASH_LOGPROBS_ON_CPU}" \
    "TRAIN_OPD_TEACHER_MICRO_BATCH_SIZES_OVERRIDE=${OPD_TEACHER_MICRO_BATCH}" \
    bash "${OPD_SWEEP}"
}

run_suite() {
  local suite_label="$1"
  local max_epoch="$2"
  local group_size="$3"
  local num_group_envs="$4"
  local rollout_epoch="$5"
  local wandb_extra_tag="${6:-}"
  local rollouts_per_step=$((group_size * num_group_envs * rollout_epoch))

  COMMON_ENV=(
    "RUN_MODE=train"
    "BASE_MODEL=1"
    "GRPO_HP_FROM_SWEEP=1"
    "TRAIN_TASK_INPUTS_OVERRIDE=${TASKS}"
    "TRAIN_MAX_EPOCHS_OVERRIDE=${max_epoch}"
    "TRAIN_GROUP_SIZES_OVERRIDE=${group_size}"
    "TRAIN_NUM_GROUP_ENVS_OVERRIDE=${num_group_envs}"
    "TRAIN_ROLLOUT_EPOCHS_OVERRIDE=${rollout_epoch}"
    "SWEEP_WANDB_EXTRA_TAG=${wandb_extra_tag}"
  )

  echo ""
  echo "############################################################"
  echo "Suite: ${suite_label}"
  echo "  max_epoch=${max_epoch}"
  echo "  group_size=${group_size}"
  echo "  num_group_envs=${num_group_envs}"
  echo "  rollout_epoch=${rollout_epoch}"
  echo "  rollouts_per_step=${rollouts_per_step}"
  echo "  wandb_extra_tag=${wandb_extra_tag:-none}"
  echo "############################################################"

  # Reference GRPO run using the existing GRPO SLURM sweep.
  run_grpo_baseline "${suite_label}"

  # Current OPD setting: REINFORCE-style OPD objective with group-zscore-normalized OPD rewards.
  run_opd_variant "${suite_label}" "Current OPD group-normalized REINFORCE" "1" "group_zscore" "embodied_opd_reinforce"

  # OPD ablation: same REINFORCE-style objective, but no reward/advantage normalization.
  # "__empty__" tells the OPD sweep to leave SWEEP_OPD_REWARD_NORMALIZATION empty, so the
  # W&B name does not append a misleading norm suffix; the sweep prefix includes "nonorm".
  run_opd_variant "${suite_label}" "OPD REINFORCE without normalization" "0" "__empty__" "embodied_opd_reinforce"

  # OPD clipped-loss variant: keep group-zscore-normalized OPD rewards, but use embodied_opd,
  # which dispatches to the GRPO-style clipped ratio loss. The sweep prefix includes "grpo_loss".
  run_opd_variant "${suite_label}" "OPD with GRPO clipping loss" "1" "group_zscore" "embodied_opd"
}

# Existing sweep settings: 8 * 4 * 4 = 128 rollouts/step for 60 training steps.
if [[ "${RUN_CURRENT}" == "1" ]]; then
  run_suite "current_rps128_steps60" 60 8 4 4 ""
fi

# Same loss/normalization settings with only 8 rollouts/step and 200 training steps.
if [[ "${RUN_RPS8_200}" == "1" ]]; then
  run_suite "rps8_steps200" 200 8 1 1 "steps200"
fi
