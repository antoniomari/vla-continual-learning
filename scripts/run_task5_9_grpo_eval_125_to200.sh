#!/bin/bash
# Eval task 5/9 GRPO rps32 runs from checkpoint 125 through 200.
#
# Preview:
#   DRY_RUN=1 bash scripts/run_task5_9_grpo_eval_125_to200.sh
#
# Submit:
#   bash scripts/run_task5_9_grpo_eval_125_to200.sh

set -euo pipefail

FULL_EVAL="examples/crl_experiment/jobs/embodiment_slurm_full_eval.sh"

TASKS="${TASKS:-5 9}"
SEEDS="${SEEDS:-1 2 3}"
EVAL_STEPS="${EVAL_STEPS:-125,150,175,200}"
EVAL_CONFIG_NAME="${EVAL_CONFIG_NAME:-crl_experiment/libero_spatial_grpo_openvlaoft_eval_spatial}"
EVAL_SEED="${EVAL_SEED:-184}"
EVAL_ROLLOUTS_PER_TASK="${EVAL_ROLLOUTS_PER_TASK:-320}"

echo "Task 5/9 GRPO eval wrapper, steps 125-200"
echo "  eval helper: ${FULL_EVAL}"
echo "  tasks: ${TASKS}"
echo "  train seeds: ${SEEDS}"
echo "  eval seed: ${EVAL_SEED}"
echo "  steps: ${EVAL_STEPS}"
echo "  rollouts/task: ${EVAL_ROLLOUTS_PER_TASK}"
echo "=================================="

job_group_count=0
for TASK in ${TASKS}; do
  for SEED in ${SEEDS}; do
    TARGET="logs_spatial/sequential/grpo_rps32_gb2048_gs8_steps200_si25_task_${TASK}_seed${SEED}_spatial"
    echo "Submit full eval target=${TARGET} steps=${EVAL_STEPS}"
    env \
      "EVAL_ROLLOUTS_PER_TASK=${EVAL_ROLLOUTS_PER_TASK}" \
      bash "${FULL_EVAL}" "${TARGET}" "${EVAL_STEPS}" "${EVAL_CONFIG_NAME}" "${EVAL_SEED}"
    job_group_count=$((job_group_count + 1))
  done
done

echo "=================================="
echo "Submitted/previewed ${job_group_count} target group(s)."
