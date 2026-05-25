#!/bin/bash
# Launch/preview SFT-teacher OPD with success-gated GRPO/teacher advantages.
#
# Successful rollouts use the environment/GRPO advantage.
# Failed rollouts use lambda_teacher * SFT-teacher OPD advantage.
# Set LOSS_TYPE=embodied_opd_grpo_plus_success_gate for additive Hybrid OPD:
# normalized GRPO is always used and raw OPD is added only on failures.
#
# W&B names include:
#   opd_sftteacher_adv1_group_zscore_success_gate_lam{...}_thr0p0_rps32_
#   opd_sftteacher_adv0_nonorm_success_gate_lam{...}_thr0p0_rps32_
#
# Preview:
#   DRY_RUN=1 bash scripts/run_sft_opd_success_gate.sh
#
# More seeds for a single best setting:
#   TASKS="1" HYBRID_SEEDS="1 3" LAMBDAS="1.0" bash scripts/run_sft_opd_success_gate.sh
#   TASKS="4" HYBRID_SEEDS="1 3" LAMBDAS="0.1" bash scripts/run_sft_opd_success_gate.sh
#
# No-normalization Hybrid OPD:
#   NORMALIZE_ADVANTAGES=0 REWARD_NORMALIZATIONS=__empty__ LAMBDAS="1.0" TASKS="1" bash scripts/run_sft_opd_success_gate.sh
#
# Raw OPD + normalized GRPO additive Hybrid OPD:
#   LOSS_TYPE=embodied_opd_grpo_plus_success_gate NORMALIZE_ADVANTAGES=0 REWARD_NORMALIZATIONS=__empty__ SUCCESS_GATE_ENV_NORMALIZE_ADVANTAGES=1 LAMBDAS="1.0" TASKS="1" bash scripts/run_sft_opd_success_gate.sh

set -euo pipefail

TASKS="${TASKS:-1 4}"
HYBRID_SEEDS="${HYBRID_SEEDS:-2}"
MAX_EPOCH="${MAX_EPOCH:-200}"
SAVE_INTERVAL="${SAVE_INTERVAL:-25}"
GROUP_SIZE="${GROUP_SIZE:-8}"
NUM_GROUP_ENVS="${NUM_GROUP_ENVS:-4}"
ROLLOUT_EPOCH="${ROLLOUT_EPOCH:-1}"
NORMALIZE_ADVANTAGES="${NORMALIZE_ADVANTAGES:-1}"
REWARD_NORMALIZATIONS="${REWARD_NORMALIZATIONS:-group_zscore}"
LOSS_TYPE="${LOSS_TYPE:-embodied_opd_success_gate}"
SUCCESS_GATE_ENV_NORMALIZE_ADVANTAGES="${SUCCESS_GATE_ENV_NORMALIZE_ADVANTAGES:-}"
LAMBDAS="${LAMBDAS:-0.1 0.3 1.0}"
REWARD_THRESHOLDS="${REWARD_THRESHOLDS:-0.0}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-a143}"
USE_MINIMAL_SBATCH_RESOURCES="${USE_MINIMAL_SBATCH_RESOURCES:-1}"
WANDB_EXTRA_TAG="${WANDB_EXTRA_TAG:-}"
if [[ "${LOSS_TYPE}" == "embodied_opd_grpo_plus_success_gate" && "${SUCCESS_GATE_ENV_NORMALIZE_ADVANTAGES}" == "1" && -z "${WANDB_EXTRA_TAG}" ]]; then
  WANDB_EXTRA_TAG="rawopd_grponorm"
fi

ROLLOUTS_PER_STEP=$((GROUP_SIZE * NUM_GROUP_ENVS * ROLLOUT_EPOCH))

echo "============================================================"
echo "SFT-teacher Hybrid OPD success-gate launcher"
echo "  tasks=${TASKS}"
echo "  seeds=${HYBRID_SEEDS}"
echo "  max_epoch=${MAX_EPOCH}"
echo "  save_interval=${SAVE_INTERVAL}"
echo "  group_size=${GROUP_SIZE}"
echo "  num_group_envs=${NUM_GROUP_ENVS}"
echo "  rollout_epoch=${ROLLOUT_EPOCH}"
echo "  rollouts_per_step=${ROLLOUTS_PER_STEP}"
echo "  normalize_advantages=${NORMALIZE_ADVANTAGES}"
echo "  reward_normalizations=${REWARD_NORMALIZATIONS}"
echo "  loss_type=${LOSS_TYPE}"
echo "  success_gate_env_normalize_advantages=${SUCCESS_GATE_ENV_NORMALIZE_ADVANTAGES:-default}"
echo "  lambdas=${LAMBDAS}"
echo "  reward_thresholds=${REWARD_THRESHOLDS}"
echo "  wandb_extra_tag=${WANDB_EXTRA_TAG:-none}"
echo "  slurm_account=${SLURM_ACCOUNT}"
echo "  use_minimal_sbatch_resources=${USE_MINIMAL_SBATCH_RESOURCES}"
echo "============================================================"

# Never train/warmup teacher in this launcher: use mapped SFT teacher only.
RUN_MODE=train \
BASE_MODEL=1 \
GRPO_HP_FROM_SWEEP=1 \
TRAIN_TASK_INPUTS_OVERRIDE="${TASKS}" \
TRAIN_MAX_EPOCHS_OVERRIDE="${MAX_EPOCH}" \
TRAIN_SEEDS_OVERRIDE="${HYBRID_SEEDS}" \
TRAIN_GROUP_SIZES_OVERRIDE="${GROUP_SIZE}" \
TRAIN_NUM_GROUP_ENVS_OVERRIDE="${NUM_GROUP_ENVS}" \
TRAIN_ROLLOUT_EPOCHS_OVERRIDE="${ROLLOUT_EPOCH}" \
TRAIN_OPD_BC_STEPS_OVERRIDE=0 \
TRAIN_OPD_RL_TEACHER=0 \
OPD_TEACHER_MAPPING_GROUP=teacher_sft_by_task \
TRAIN_OPD_LOSS_TYPES_OVERRIDE="${LOSS_TYPE}" \
TRAIN_OPD_NORMALIZE_ADVANTAGES_OVERRIDE="${NORMALIZE_ADVANTAGES}" \
TRAIN_OPD_REWARD_NORMALIZATIONS_OVERRIDE="${REWARD_NORMALIZATIONS}" \
TRAIN_OPD_SUCCESS_GATE_ENV_NORMALIZE_ADVANTAGES="${SUCCESS_GATE_ENV_NORMALIZE_ADVANTAGES}" \
TRAIN_OPD_SUCCESS_GATE_TEACHER_LAMBDAS_OVERRIDE="${LAMBDAS}" \
TRAIN_OPD_SUCCESS_GATE_REWARD_THRESHOLDS_OVERRIDE="${REWARD_THRESHOLDS}" \
SWEEP_WANDB_EXTRA_TAG="${WANDB_EXTRA_TAG}" \
SWEEP_SAVE_INTERVAL="${SAVE_INTERVAL}" \
SLURM_ACCOUNT="${SLURM_ACCOUNT}" \
USE_MINIMAL_SBATCH_RESOURCES="${USE_MINIMAL_SBATCH_RESOURCES}" \
bash examples/crl_experiment/jobs/embodiment_slurm_opd_sweep.sh
