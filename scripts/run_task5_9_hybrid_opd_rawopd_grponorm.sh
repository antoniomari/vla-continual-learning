#!/bin/bash
# Train task 5/9 Hybrid OPD using the current version:
#   normalized GRPO/env advantage + failure-gated raw SFT-OPD advantage.
#
# Uses seed-0 SFT teacher checkpoints from:
#   examples/crl_experiment/jobs/opd_teacher_mapping.json
#
# Preview:
#   DRY_RUN=1 bash scripts/run_task5_9_hybrid_opd_rawopd_grponorm.sh
#
# Submit:
#   bash scripts/run_task5_9_hybrid_opd_rawopd_grponorm.sh

set -euo pipefail

TASKS="${TASKS:-5 9}"
SEEDS="${SEEDS:-1 2 3}"
LAMBDAS="${LAMBDAS:-0.1 1.0}"
MAX_EPOCH="${MAX_EPOCH:-200}"
SAVE_INTERVAL="${SAVE_INTERVAL:-25}"
GROUP_SIZE="${GROUP_SIZE:-8}"
NUM_GROUP_ENVS="${NUM_GROUP_ENVS:-4}"
ROLLOUT_EPOCH="${ROLLOUT_EPOCH:-1}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-a143}"

echo "Task 5/9 Hybrid OPD raw-OPD + normalized-GRPO launcher"
echo "  tasks=${TASKS}"
echo "  seeds=${SEEDS}"
echo "  lambdas=${LAMBDAS}"
echo "  max_epoch=${MAX_EPOCH}"
echo "  save_interval=${SAVE_INTERVAL}"
echo "  group_size=${GROUP_SIZE}"
echo "  num_group_envs=${NUM_GROUP_ENVS}"
echo "  rollout_epoch=${ROLLOUT_EPOCH}"
echo "=================================="

TASKS="${TASKS}" \
HYBRID_SEEDS="${SEEDS}" \
LAMBDAS="${LAMBDAS}" \
MAX_EPOCH="${MAX_EPOCH}" \
SAVE_INTERVAL="${SAVE_INTERVAL}" \
GROUP_SIZE="${GROUP_SIZE}" \
NUM_GROUP_ENVS="${NUM_GROUP_ENVS}" \
ROLLOUT_EPOCH="${ROLLOUT_EPOCH}" \
SLURM_ACCOUNT="${SLURM_ACCOUNT}" \
bash scripts/run_sft_opd_rawopd_grponorm_gate.sh
