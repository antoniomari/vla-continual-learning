#!/bin/bash
# Submit ONE SLURM eval job for Libero spatial full evaluation.
# Supports:
#   - base model eval (no LoRA): target "base"
#   - checkpoint eval: target is run directory containing checkpoints/
#
# It calls:
#   examples/crl_experiment/eval_embodiment.sh
#
# Usage:
#   bash examples/crl_experiment/jobs/embodiment_slurm_full_eval.sh [TARGET] [STEP] [CONFIG_NAME]
#
# Examples:
#   # Base model, default config, 320 rollouts/task
#   bash examples/crl_experiment/jobs/embodiment_slurm_full_eval.sh base 0
#
#   # Specific checkpoint (relative path), global_step_50
#   bash examples/crl_experiment/jobs/embodiment_slurm_full_eval.sh logs/sequential/task_4_seed184 50
#
# Optional env:
#   DRY_RUN=1                -> print batch script, do not submit
#   PROJECT_ROOT, VENV_PATH, SLURM_LOG_DIR
#   SLURM_PARTITION, SLURM_ACCOUNT, SBATCH_EXTRA
#   TIME, CPUS_PER_TASK, MEM_PER_CPU, GPU
#   LIBERO_REPO_PATH, LIBERO_CONFIG_PATH
#   EVAL_NUM_ENVS_TOTAL      -> total eval envs across tasks used by OPD sweep/config (default: 80)
#   EVAL_TASK_COUNT          -> number of tasks in suite (default: 10)
#   EVAL_ROLLOUTS_PER_TASK   -> desired rollouts per task (default: 320)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-${PROJECT_ROOT}/logs/slurm_embodiment_full_eval}"
VENV_PATH="${VENV_PATH:-${PROJECT_ROOT}/.venv}"

mkdir -p "${SLURM_LOG_DIR}"

TIME="${TIME:-12:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM_PER_CPU="${MEM_PER_CPU:-16G}"
GPU="${GPU:-pro_6000:1}"

SLURM_PARTITION="${SLURM_PARTITION:-}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"
SBATCH_EXTRA="${SBATCH_EXTRA:-}"

if [[ "${HOME}" == "/users/anmari" ]]; then
  NEW_CLUSTER_ACCOUNT="${SLURM_ACCOUNT:-a143}"
else
  NEW_CLUSTER_ACCOUNT=""
fi

LIBERO_REPO_PATH="${LIBERO_REPO_PATH:-}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-}"

TARGET="${1:-${EVAL_TARGET:-base}}"
STEP="${2:-${EVAL_STEP:-0}}"
CONFIG_NAME="${3:-${EVAL_CONFIG_NAME:-crl_experiment/libero_spatial_grpo_openvlaoft_eval_spatial}}"

if ! [[ "${STEP}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: STEP must be a non-negative integer, got: ${STEP}"
  exit 1
fi

EVAL_NUM_ENVS_TOTAL="${EVAL_NUM_ENVS_TOTAL:-80}"
EVAL_TASK_COUNT="${EVAL_TASK_COUNT:-10}"
EVAL_ROLLOUTS_PER_TASK="${EVAL_ROLLOUTS_PER_TASK:-320}"

if ! [[ "${EVAL_NUM_ENVS_TOTAL}" =~ ^[0-9]+$ ]] || ! [[ "${EVAL_TASK_COUNT}" =~ ^[0-9]+$ ]] || ! [[ "${EVAL_ROLLOUTS_PER_TASK}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: EVAL_NUM_ENVS_TOTAL, EVAL_TASK_COUNT, EVAL_ROLLOUTS_PER_TASK must be integers."
  exit 1
fi
if (( EVAL_TASK_COUNT <= 0 )); then
  echo "ERROR: EVAL_TASK_COUNT must be > 0"
  exit 1
fi
if (( EVAL_NUM_ENVS_TOTAL % EVAL_TASK_COUNT != 0 )); then
  echo "ERROR: EVAL_NUM_ENVS_TOTAL (${EVAL_NUM_ENVS_TOTAL}) must be divisible by EVAL_TASK_COUNT (${EVAL_TASK_COUNT})."
  exit 1
fi

PER_EPOCH_PER_TASK=$((EVAL_NUM_ENVS_TOTAL / EVAL_TASK_COUNT))
EVAL_ROLLOUT_EPOCH=$(((EVAL_ROLLOUTS_PER_TASK + PER_EPOCH_PER_TASK - 1) / PER_EPOCH_PER_TASK))
ACTUAL_ROLLOUTS_PER_TASK=$((PER_EPOCH_PER_TASK * EVAL_ROLLOUT_EPOCH))

submit_job() {
  local job_name="$1"
  local cmd="$2"
  local batch
  batch="$(mktemp)"

  {
    echo "#!/bin/bash"
    echo "#SBATCH --job-name=${job_name}"
    echo "#SBATCH --time=${TIME}"
    if [[ "${HOME}" != "/users/anmari" ]]; then
      echo "#SBATCH --cpus-per-task=${CPUS_PER_TASK}"
      echo "#SBATCH --mem-per-cpu=${MEM_PER_CPU}"
      echo "#SBATCH --gpus=${GPU}"
    fi
    echo "#SBATCH --output=${SLURM_LOG_DIR}/eval_%j.out"
    echo "#SBATCH --error=${SLURM_LOG_DIR}/eval_%j.err"
    if [[ -n "${SLURM_PARTITION}" ]]; then
      echo "#SBATCH --partition=${SLURM_PARTITION}"
    fi
    if [[ "${HOME}" == "/users/anmari" ]]; then
      echo "#SBATCH --account=${NEW_CLUSTER_ACCOUNT}"
    elif [[ -n "${SLURM_ACCOUNT}" ]]; then
      echo "#SBATCH --account=${SLURM_ACCOUNT}"
    fi
    echo "set -euo pipefail"
    echo "if [[ \"\${RLINF_ALLOW_CORE_DUMPS:-0}\" != \"1\" ]]; then ulimit -c 0 2>/dev/null || true; fi"
    echo "cd \"${PROJECT_ROOT}\""
    echo "source \"${VENV_PATH}/bin/activate\""
    if [[ -n "${LIBERO_REPO_PATH}" ]]; then
      echo "export LIBERO_REPO_PATH=$(printf '%q' "${LIBERO_REPO_PATH}")"
      echo "export LIBERO_CONFIG_PATH=$(printf '%q' "${LIBERO_CONFIG_PATH:-${LIBERO_REPO_PATH}}")"
    fi
    printf '%s\n' "${cmd}"
  } >"${batch}"

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[DRY_RUN] would sbatch ${batch}:"
    cat "${batch}"
    rm -f "${batch}"
    return 0
  fi

  # shellcheck disable=SC2086
  sbatch ${SBATCH_EXTRA} "${batch}"
  rm -f "${batch}"
}

TARGET_STEM="$(basename "${TARGET%/}")"
W_NAME="eval_full_${TARGET_STEM}_step_${STEP}_rpt_${ACTUAL_ROLLOUTS_PER_TASK}"
W_NAME="${W_NAME//[^a-zA-Z0-9._-]/_}"
JOB_NAME="${W_NAME}"
if ((${#JOB_NAME} > 40)); then
  JOB_NAME="${JOB_NAME:0:40}"
fi

# Keep eval env count aligned with OPD sweep/config; only override eval_rollout_epoch.
EVAL_HYDRA_OVERRIDES="runner.logger.experiment_name=${W_NAME} algorithm.eval_rollout_epoch=${EVAL_ROLLOUT_EPOCH}"
CMD=$(printf '%q ' env EVAL_HYDRA_OVERRIDES="${EVAL_HYDRA_OVERRIDES}" bash examples/crl_experiment/eval_embodiment.sh "${TARGET}" "${STEP}" "${CONFIG_NAME}")

echo "Full eval submit helper"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "TARGET=${TARGET}"
echo "STEP=${STEP}"
echo "CONFIG=${CONFIG_NAME}"
echo "EVAL_NUM_ENVS_TOTAL(assumed from OPD eval config)=${EVAL_NUM_ENVS_TOTAL}"
echo "EVAL_TASK_COUNT=${EVAL_TASK_COUNT}"
echo "EVAL_ROLLOUTS_PER_TASK(requested)=${EVAL_ROLLOUTS_PER_TASK}"
echo "PER_EPOCH_PER_TASK=${PER_EPOCH_PER_TASK}"
echo "algorithm.eval_rollout_epoch(derived)=${EVAL_ROLLOUT_EPOCH}"
echo "ACTUAL_ROLLOUTS_PER_TASK=${ACTUAL_ROLLOUTS_PER_TASK}"
echo "wandb_name=${W_NAME}"
echo "Submitting job: ${JOB_NAME}"

submit_job "${JOB_NAME}" "${CMD}"

echo "Submitted 1 job. Logs: ${SLURM_LOG_DIR}/eval_*.out"
echo "Check: squeue -u \$USER"
