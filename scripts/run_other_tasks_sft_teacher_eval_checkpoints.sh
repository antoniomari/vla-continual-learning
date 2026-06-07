#!/bin/bash
# Eval seed-0 SFT teachers for the non-core LIBERO spatial tasks.
#
# Defaults cover the teachers trained for tasks 0, 2, 3, 6, 7, and 8.
# They were trained as:
#   opd_sftteacher_adv1_group_zscore_rps32_teacherprep_seed0_task_<TASK>_seed0_spatial_norm_group_zscore
#
# Preview:
#   DRY_RUN=1 bash scripts/run_other_tasks_sft_teacher_eval_checkpoints.sh
#
# Submit:
#   bash scripts/run_other_tasks_sft_teacher_eval_checkpoints.sh

set -euo pipefail

FULL_EVAL="examples/crl_experiment/jobs/embodiment_slurm_full_eval.sh"

TASKS="${TASKS:-0 2 3 6 7 8}"
TEACHER_STEPS="${TEACHER_STEPS:-1000}"
TEACHER_SEED="${TEACHER_SEED:-0}"
TEACHER_EXTRA_TAG="${TEACHER_EXTRA_TAG:-teacherprep_seed${TEACHER_SEED}}"
EVAL_CONFIG_NAME="${EVAL_CONFIG_NAME:-crl_experiment/libero_spatial_grpo_openvlaoft_eval_spatial}"
EVAL_SEED="${EVAL_SEED:-184}"
EVAL_ROLLOUTS_PER_TASK="${EVAL_ROLLOUTS_PER_TASK:-320}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-a143}"

echo "Other-task SFT teacher checkpoint eval wrapper"
echo "  eval helper: ${FULL_EVAL}"
echo "  tasks: ${TASKS}"
echo "  teacher steps: ${TEACHER_STEPS}"
echo "  teacher seed: ${TEACHER_SEED}"
echo "  teacher extra tag: ${TEACHER_EXTRA_TAG}"
echo "  eval seed: ${EVAL_SEED}"
echo "  rollouts/task: ${EVAL_ROLLOUTS_PER_TASK}"
echo "  slurm account: ${SLURM_ACCOUNT}"
echo "=================================="

job_group_count=0
for TASK in ${TASKS}; do
  RUN_DIR="logs_spatial/sequential/opd_sftteacher_adv1_group_zscore_rps32_${TEACHER_EXTRA_TAG}_task_${TASK}_seed${TEACHER_SEED}_spatial_norm_group_zscore"
  for STEP in ${TEACHER_STEPS}; do
    CHECKPOINT_TEACHER_PATH="${RUN_DIR}/opd_bc_teacher/checkpoints/step_${STEP}/actor"
    FINAL_TEACHER_PATH="${RUN_DIR}/opd_bc_teacher/actor"
    TEACHER_PATH="${CHECKPOINT_TEACHER_PATH}"

    if [[ "${DRY_RUN:-0}" != "1" && ! -d "${TEACHER_PATH}" ]]; then
      if [[ -d "${FINAL_TEACHER_PATH}" ]]; then
        echo "WARN: step checkpoint missing for task=${TASK} step=${STEP}; using final actor: ${FINAL_TEACHER_PATH}"
        TEACHER_PATH="${FINAL_TEACHER_PATH}"
      else
        echo "ERROR: SFT teacher path does not exist for task=${TASK} step=${STEP}"
        echo "       checked: ${CHECKPOINT_TEACHER_PATH}"
        echo "       checked: ${FINAL_TEACHER_PATH}"
        exit 1
      fi
    fi

    TEACHER_NAME="sft_teacher_task_${TASK}_bc_step_${STEP}"
    echo "Submit SFT teacher eval task=${TASK} bc_step=${STEP} path=${TEACHER_PATH}"
    env \
      "EVAL_ROLLOUTS_PER_TASK=${EVAL_ROLLOUTS_PER_TASK}" \
      "SLURM_ACCOUNT=${SLURM_ACCOUNT}" \
      "SFT_TEACHER_PATH=${TEACHER_PATH}" \
      "SFT_TEACHER_NAME=${TEACHER_NAME}" \
      bash "${FULL_EVAL}" base 0 "${EVAL_CONFIG_NAME}" "${EVAL_SEED}"
    job_group_count=$((job_group_count + 1))
  done
done

echo "=================================="
echo "Submitted/previewed ${job_group_count} SFT teacher eval group(s)."
