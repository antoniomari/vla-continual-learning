#!/usr/bin/env bash
# Submit a small video-focused eval for the task-5 SFT teacher.
#
# Default target is the longer task-5 teacher run at BC step 2500, because that
# was the best full-eval checkpoint so far. Override TEACHER_STEP or TEACHER_PATH
# to inspect another checkpoint.
#
# Example:
#   bash scripts/run_task5_sft_teacher_video_eval.sh
#   TEACHER_STEP=5000 bash scripts/run_task5_sft_teacher_video_eval.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
VENV_PATH="${VENV_PATH:-${PROJECT_ROOT}/.venv}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-${PROJECT_ROOT}/logs/slurm_task5_teacher_video_eval}"

mkdir -p "${SLURM_LOG_DIR}"

TASK_ID="${TASK_ID:-5}"
TEACHER_STEP="${TEACHER_STEP:-2500}"
TEACHER_RUN="${TEACHER_RUN:-opd_sftteacher_adv1_group_zscore_rps32_teacherprep_seed0_longer_task_${TASK_ID}_seed0_spatial_norm_group_zscore}"
TEACHER_PATH="${TEACHER_PATH:-logs_spatial/sequential/${TEACHER_RUN}/opd_bc_teacher/checkpoints/step_${TEACHER_STEP}/actor}"
CONFIG_NAME="${CONFIG_NAME:-crl_experiment/libero_spatial_grpo_openvlaoft_eval_spatial}"
EVAL_SEED="${EVAL_SEED:-184}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-8}"
EVAL_ROLLOUT_EPOCH="${EVAL_ROLLOUT_EPOCH:-4}"
USE_GREEDY="${USE_GREEDY:-0}"
TIME="${TIME:-04:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM_PER_CPU="${MEM_PER_CPU:-16G}"
GPU="${GPU:-pro_6000:1}"
SLURM_PARTITION="${SLURM_PARTITION:-}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"
SBATCH_EXTRA="${SBATCH_EXTRA:-}"

if [[ "${DRY_RUN:-0}" != "1" && ! -d "${PROJECT_ROOT}/${TEACHER_PATH}" && ! -d "${TEACHER_PATH}" ]]; then
  echo "ERROR: teacher path does not exist:"
  echo "  ${TEACHER_PATH}"
  echo "Checked relative to PROJECT_ROOT=${PROJECT_ROOT} and as an absolute path."
  exit 1
fi

W_NAME="video_sft_teacher_task_${TASK_ID}_bc_step_${TEACHER_STEP}_seed_${EVAL_SEED}"
JOB_NAME="${W_NAME}"
if ((${#JOB_NAME} > 40)); then
  JOB_NAME="${JOB_NAME:0:40}"
fi

HYDRA_OVERRIDES=(
  "runner.logger.experiment_name=${W_NAME}"
  "actor.seed=${EVAL_SEED}"
  "env.fixed_task_ids=[${TASK_ID}]"
  "env.eval.fixed_task_ids=[${TASK_ID}]"
  "env.eval.num_envs=${EVAL_NUM_ENVS}"
  "env.eval.eval_per_task=${EVAL_NUM_ENVS}"
  "algorithm.eval_rollout_epoch=${EVAL_ROLLOUT_EPOCH}"
  "env.eval.video_cfg.save_video=True"
  "env.eval.video_cfg.save_frequency=1"
  "env.eval.video_cfg.save_rank=0"
  "env.eval.video_cfg.info_on_video=True"
  "+actor.model.lora_path=${TEACHER_PATH}"
)

if [[ "${USE_GREEDY}" == "1" ]]; then
  HYDRA_OVERRIDES+=(
    "algorithm.sampling_params.use_greedy=True"
    "algorithm.sampling_params.temperature_eval=1.0"
  )
fi

batch="$(mktemp)"
{
  echo "#!/bin/bash"
  echo "#SBATCH --job-name=${JOB_NAME}"
  echo "#SBATCH --time=${TIME}"
  if [[ "${HOME}" != "/users/anmari" ]]; then
    echo "#SBATCH --cpus-per-task=${CPUS_PER_TASK}"
    echo "#SBATCH --mem-per-cpu=${MEM_PER_CPU}"
    echo "#SBATCH --gpus=${GPU}"
  fi
  echo "#SBATCH --output=${SLURM_LOG_DIR}/video_eval_%j.out"
  echo "#SBATCH --error=${SLURM_LOG_DIR}/video_eval_%j.err"
  if [[ -n "${SLURM_PARTITION}" ]]; then
    echo "#SBATCH --partition=${SLURM_PARTITION}"
  fi
  if [[ "${HOME}" == "/users/anmari" ]]; then
    echo "#SBATCH --account=${SLURM_ACCOUNT:-a143}"
  elif [[ -n "${SLURM_ACCOUNT}" ]]; then
    echo "#SBATCH --account=${SLURM_ACCOUNT}"
  fi
  echo "set -euo pipefail"
  echo "cd \"${PROJECT_ROOT}\""
  echo "source \"${VENV_PATH}/bin/activate\""
  printf 'bash examples/embodiment/eval_embodiment.sh %q ' "${CONFIG_NAME}"
  for override in "${HYDRA_OVERRIDES[@]}"; do
    printf '%q ' "${override}"
  done
  echo
} >"${batch}"

echo "Submitting task-${TASK_ID} SFT teacher video eval"
echo "  teacher path: ${TEACHER_PATH}"
echo "  teacher step: ${TEACHER_STEP}"
echo "  eval envs: ${EVAL_NUM_ENVS}"
echo "  eval rollout epoch: ${EVAL_ROLLOUT_EPOCH}"
echo "  greedy: ${USE_GREEDY}"
echo "  videos will be under logs_spatial/evals/.../video/train/rank_0/"
echo "=================================="

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[DRY_RUN] would sbatch:"
  cat "${batch}"
  rm -f "${batch}"
  exit 0
fi

# shellcheck disable=SC2086
sbatch ${SBATCH_EXTRA} "${batch}"
rm -f "${batch}"
