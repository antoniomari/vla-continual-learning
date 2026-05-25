#!/bin/bash
# Launch SFT-teacher Hybrid OPD with:
#   A = normalized GRPO advantage + (failure gate) * lambda * raw OPD advantage
#
# This keeps GRPO credit assignment normalized for all rollouts while using the
# SFT teacher only as an extra failure-recovery signal.
#
# Preview:
#   DRY_RUN=1 bash scripts/run_sft_opd_rawopd_grponorm_gate.sh
#
# Submit:
#   bash scripts/run_sft_opd_rawopd_grponorm_gate.sh

set -euo pipefail

TASKS="${TASKS:-1 4}"
HYBRID_SEEDS="${HYBRID_SEEDS:-2}"
LAMBDAS="${LAMBDAS:-0.1 1.0}"
MAX_EPOCH="${MAX_EPOCH:-200}"
SAVE_INTERVAL="${SAVE_INTERVAL:-25}"
GROUP_SIZE="${GROUP_SIZE:-8}"
NUM_GROUP_ENVS="${NUM_GROUP_ENVS:-4}"
ROLLOUT_EPOCH="${ROLLOUT_EPOCH:-1}"

LOSS_TYPE=embodied_opd_grpo_plus_success_gate \
NORMALIZE_ADVANTAGES=0 \
REWARD_NORMALIZATIONS=__empty__ \
SUCCESS_GATE_ENV_NORMALIZE_ADVANTAGES=1 \
WANDB_EXTRA_TAG="${WANDB_EXTRA_TAG:-rawopd_grponorm}" \
TASKS="${TASKS}" \
HYBRID_SEEDS="${HYBRID_SEEDS}" \
LAMBDAS="${LAMBDAS}" \
MAX_EPOCH="${MAX_EPOCH}" \
SAVE_INTERVAL="${SAVE_INTERVAL}" \
GROUP_SIZE="${GROUP_SIZE}" \
NUM_GROUP_ENVS="${NUM_GROUP_ENVS}" \
ROLLOUT_EPOCH="${ROLLOUT_EPOCH}" \
bash scripts/run_sft_opd_success_gate.sh
