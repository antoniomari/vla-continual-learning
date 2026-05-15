#!/bin/bash
# Run OPD loss/normalization variants with the current sweep settings.
#
# Defaults mirror examples/crl_experiment/jobs/embodiment_slurm_opd_sweep.sh:
#   - tasks: 1 and 4
#   - seed: 184
#   - SFT teacher BC warmup
#   - 128 rollouts/step: group_size 8 * num_group_envs 4 * rollout_epoch 4
#   - embodied_opd_reinforce unless the variant explicitly changes it
#
# Override examples:
#   TASKS="1" bash examples/crl_experiment/jobs/run_opd_loss_variants.sh
#   SEED=185 MAX_EPOCH=50 bash examples/crl_experiment/jobs/run_opd_loss_variants.sh

set -euo pipefail

TASKS="${TASKS:-1 4}"
SEED="${SEED:-184}"
MAX_EPOCH="${MAX_EPOCH:-60}"
CONFIG_NAME="${CONFIG_NAME:-crl_experiment/libero_spatial_opd_openvlaoft_spatial}"
CHECKPOINT="${CHECKPOINT:-base}"

# Current sweep rollout geometry: 8 * 4 * 4 = 128 rollouts/step.
SWEEP_GROUP_SIZE="${SWEEP_GROUP_SIZE:-8}"
SWEEP_NUM_GROUP_ENVS="${SWEEP_NUM_GROUP_ENVS:-4}"
SWEEP_ROLLOUT_EPOCH="${SWEEP_ROLLOUT_EPOCH:-4}"
SWEEP_GLOBAL_BATCH_SIZE="${SWEEP_GLOBAL_BATCH_SIZE:-2048}"
SWEEP_SAVE_INTERVAL="${SWEEP_SAVE_INTERVAL:-20}"

# Current sweep OPD teacher/SFT settings.
SWEEP_OPD_BC_GLOBAL_BATCH_SIZE="${SWEEP_OPD_BC_GLOBAL_BATCH_SIZE:-32}"
SWEEP_OPD_BC_BATCH_SIZE="${SWEEP_OPD_BC_BATCH_SIZE:-8}"
SWEEP_OPD_BC_STEPS="${SWEEP_OPD_BC_STEPS:-1000}"
SWEEP_OPD_TEACHER_LR="${SWEEP_OPD_TEACHER_LR:-1e-04}"
SWEEP_OPD_SFT_FILTER_FIXED_TASK_IDS="${SWEEP_OPD_SFT_FILTER_FIXED_TASK_IDS:-1}"
SWEEP_OPD_SFT_MATCH_TASK_LANGUAGE="${SWEEP_OPD_SFT_MATCH_TASK_LANGUAGE:-1}"
SWEEP_OPD_SFT_MATCH_OBS_ACTION_ALIGNMENT="${SWEEP_OPD_SFT_MATCH_OBS_ACTION_ALIGNMENT:-0}"
SWEEP_OPD_RL_TEACHER="${SWEEP_OPD_RL_TEACHER:-0}"

run_opd_variant() {
  local task="$1"
  local name_prefix="$2"
  local normalize_advantages="$3"
  local loss_type="$4"
  local reward_normalization="${5:-}"

  echo "============================================================"
  echo "Task ${task}: ${name_prefix}"
  echo "  normalize_advantages=${normalize_advantages}"
  echo "  loss_type=${loss_type}"
  echo "  reward_normalization=${reward_normalization:-disabled/not set}"
  echo "============================================================"

  local env_args=(
    "EXPERIMENT_NAME_PREFIX=${name_prefix}"
    "SKIP_POST_TRAIN_EVAL=1"
    "SWEEP_GROUP_SIZE=${SWEEP_GROUP_SIZE}"
    "SWEEP_NUM_GROUP_ENVS=${SWEEP_NUM_GROUP_ENVS}"
    "SWEEP_ROLLOUT_EPOCH=${SWEEP_ROLLOUT_EPOCH}"
    "SWEEP_GLOBAL_BATCH_SIZE=${SWEEP_GLOBAL_BATCH_SIZE}"
    "SWEEP_SAVE_INTERVAL=${SWEEP_SAVE_INTERVAL}"
    "SWEEP_OPD_BC_GLOBAL_BATCH_SIZE=${SWEEP_OPD_BC_GLOBAL_BATCH_SIZE}"
    "SWEEP_OPD_BC_BATCH_SIZE=${SWEEP_OPD_BC_BATCH_SIZE}"
    "SWEEP_OPD_BC_STEPS=${SWEEP_OPD_BC_STEPS}"
    "SWEEP_OPD_TEACHER_LR=${SWEEP_OPD_TEACHER_LR}"
    "SWEEP_OPD_SFT_FILTER_FIXED_TASK_IDS=${SWEEP_OPD_SFT_FILTER_FIXED_TASK_IDS}"
    "SWEEP_OPD_SFT_MATCH_TASK_LANGUAGE=${SWEEP_OPD_SFT_MATCH_TASK_LANGUAGE}"
    "SWEEP_OPD_SFT_MATCH_OBS_ACTION_ALIGNMENT=${SWEEP_OPD_SFT_MATCH_OBS_ACTION_ALIGNMENT}"
    "SWEEP_OPD_NORMALIZE_ADVANTAGES=${normalize_advantages}"
    "SWEEP_OPD_RL_TEACHER=${SWEEP_OPD_RL_TEACHER}"
    "SWEEP_OPD_LOSS_TYPE=${loss_type}"
  )

  if [[ -n "${reward_normalization}" ]]; then
    env_args+=("SWEEP_OPD_REWARD_NORMALIZATION=${reward_normalization}")
  fi

  env "${env_args[@]}" \
    bash examples/crl_experiment/run_embodiment_opd_sequential.sh \
      "${task}" \
      "${CHECKPOINT}" \
      "${MAX_EPOCH}" \
      "${CONFIG_NAME}" \
      "${SEED}"
}

for task in ${TASKS}; do
  # Current setting: OPD REINFORCE objective with group-zscore-normalized OPD rewards.
  run_opd_variant \
    "${task}" \
    "opd_sftteacher_adv1_rps128_" \
    "1" \
    "embodied_opd_reinforce" \
    "group_zscore"

  # Ablation: same OPD REINFORCE objective, but no OPD reward/advantage normalization.
  # We intentionally do not set SWEEP_OPD_REWARD_NORMALIZATION here, so W&B does not
  # append a misleading "_norm_group_zscore" suffix.
  run_opd_variant \
    "${task}" \
    "opd_sftteacher_adv0_nonorm_rps128_" \
    "0" \
    "embodied_opd_reinforce"

  # Clipped-loss variant: OPD rewards are still group-zscore-normalized, but the actor
  # objective is embodied_opd, which delegates to the GRPO-style clipped ratio loss.
  run_opd_variant \
    "${task}" \
    "opd_sftteacher_adv1_grpo_loss_rps128_" \
    "1" \
    "embodied_opd" \
    "group_zscore"
done
