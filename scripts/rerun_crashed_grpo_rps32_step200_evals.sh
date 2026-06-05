#!/bin/bash
# Rerun only the crashed GRPO rps32 step-200 eval points.
#
# Preview:
#   DRY_RUN=1 bash scripts/rerun_crashed_grpo_rps32_step200_evals.sh
#
# Submit:
#   bash scripts/rerun_crashed_grpo_rps32_step200_evals.sh

set -euo pipefail

FULL_EVAL="examples/crl_experiment/jobs/embodiment_slurm_full_eval.sh"
EVAL_CONFIG_NAME="${EVAL_CONFIG_NAME:-crl_experiment/libero_spatial_grpo_openvlaoft_eval_spatial}"
EVAL_SEED="${EVAL_SEED:-184}"
EVAL_ROLLOUTS_PER_TASK="${EVAL_ROLLOUTS_PER_TASK:-320}"

submit_eval() {
  local task="$1"
  local seed="$2"
  local step="$3"
  local target="logs_spatial/sequential/grpo_rps32_gb2048_gs8_steps200_si25_task_${task}_seed${seed}_spatial"

  echo "Submit crashed GRPO eval target=${target} step=${step}"
  env \
    "EVAL_ROLLOUTS_PER_TASK=${EVAL_ROLLOUTS_PER_TASK}" \
    bash "${FULL_EVAL}" "${target}" "${step}" "${EVAL_CONFIG_NAME}" "${EVAL_SEED}"
}

echo "Crashed GRPO rps32 step-200 eval rerun"
echo "  eval helper: ${FULL_EVAL}"
echo "  config: ${EVAL_CONFIG_NAME}"
echo "  eval seed: ${EVAL_SEED}"
echo "  rollouts/task: ${EVAL_ROLLOUTS_PER_TASK}"
echo "  points:"
echo "    task 1 seed 2 step 200"
echo "    task 5 seed 1 step 200"
echo "    task 9 seed 3 step 200"
echo "=================================="

submit_eval 1 2 200
submit_eval 5 1 200
submit_eval 9 3 200

echo "=================================="
echo "Submitted/previewed 3 crashed GRPO eval point(s)."
