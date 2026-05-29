#!/bin/bash
# Eval seed-0 SFT teachers for tasks 5 and 9 at BC checkpoints 250/500/750/1000.
#
# Preview:
#   DRY_RUN=1 bash scripts/run_task5_9_sft_teacher_eval_checkpoints.sh
#
# Submit:
#   bash scripts/run_task5_9_sft_teacher_eval_checkpoints.sh

set -euo pipefail

FULL_EVAL="examples/crl_experiment/jobs/embodiment_slurm_full_eval.sh"

TASKS="${TASKS:-5 9}"
TEACHER_STEPS="${TEACHER_STEPS:-250 500 750 1000}"
TEACHER_SEED="${TEACHER_SEED:-0}"
TEACHER_EXTRA_TAG="${TEACHER_EXTRA_TAG:-teacherprep_seed${TEACHER_SEED}}"
EVAL_CONFIG_NAME="${EVAL_CONFIG_NAME:-crl_experiment/libero_spatial_grpo_openvlaoft_eval_spatial}"
EVAL_SEED="${EVAL_SEED:-184}"
EVAL_ROLLOUTS_PER_TASK="${EVAL_ROLLOUTS_PER_TASK:-320}"

echo "Task 5/9 SFT teacher checkpoint eval wrapper"
echo "  eval helper: ${FULL_EVAL}"
echo "  tasks: ${TASKS}"
echo "  teacher steps: ${TEACHER_STEPS}"
echo "  teacher extra tag: ${TEACHER_EXTRA_TAG}"
echo "  eval seed: ${EVAL_SEED}"
echo "  rollouts/task: ${EVAL_ROLLOUTS_PER_TASK}"
echo "=================================="

job_group_count=0
for TASK in ${TASKS}; do
  RUN_DIR="logs_spatial/sequential/opd_sftteacher_adv1_group_zscore_rps32_${TEACHER_EXTRA_TAG}_task_${TASK}_seed${TEACHER_SEED}_spatial_norm_group_zscore"
  for STEP in ${TEACHER_STEPS}; do
    TEACHER_PATH="${RUN_DIR}/opd_bc_teacher/checkpoints/step_${STEP}/actor"
    TEACHER_NAME="sft_teacher_task_${TASK}_bc_step_${STEP}"
    echo "Submit SFT teacher eval task=${TASK} bc_step=${STEP} path=${TEACHER_PATH}"
    env \
      "EVAL_ROLLOUTS_PER_TASK=${EVAL_ROLLOUTS_PER_TASK}" \
      "SFT_TEACHER_PATH=${TEACHER_PATH}" \
      "SFT_TEACHER_NAME=${TEACHER_NAME}" \
      bash "${FULL_EVAL}" base 0 "${EVAL_CONFIG_NAME}" "${EVAL_SEED}"
    job_group_count=$((job_group_count + 1))
  done
done

echo "=================================="
echo "Submitted/previewed ${job_group_count} SFT teacher eval group(s)."
