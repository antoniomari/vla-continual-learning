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
#   bash examples/crl_experiment/jobs/embodiment_slurm_full_eval.sh [TARGET] [STEP_OR_STEP_SET] [CONFIG_NAME] [SEED] [SFT_MODEL_EVAL_TASK]
#
# Examples:
#   # Base model, default config, 320 rollouts/task
#   bash examples/crl_experiment/jobs/embodiment_slurm_full_eval.sh base 0
#
#   # Base model with explicit eval seed
#   bash examples/crl_experiment/jobs/embodiment_slurm_full_eval.sh base 0 crl_experiment/libero_spatial_grpo_openvlaoft_eval_spatial 2
#
#   # Specific checkpoint (relative path), global_step_50
#   bash examples/crl_experiment/jobs/embodiment_slurm_full_eval.sh logs/sequential/task_4_seed184 50
#
#   # Specific checkpoint (absolute path under PROJECT_ROOT), multiple steps
#   bash examples/crl_experiment/jobs/embodiment_slurm_full_eval.sh ${PROJECT_ROOT}/logs_spatial/sequential/task_1_seed184 10,20,30
#
#   # Evaluate mapped SFT teacher for task 1 (uses JSON map path directly)
#   bash examples/crl_experiment/jobs/embodiment_slurm_full_eval.sh base 0 crl_experiment/libero_spatial_grpo_openvlaoft_eval_spatial 184 1
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
#   EVAL_STEPS               -> optional step set, e.g. "10,20,30" (same as STEP_OR_STEP_SET arg)
#   OPD_TEACHER_MAPPING_JSON -> task->teacher adapter map JSON (default: jobs/opd_teacher_mapping.json)
#   SFT_MODEL_EVAL_TASK      -> task id to evaluate mapped SFT teacher adapter (same as positional arg 5)
#   SFT_TEACHER_PATH         -> explicit SFT teacher adapter path to evaluate instead of JSON lookup
#   SFT_TEACHER_NAME         -> optional W&B target stem for SFT_TEACHER_PATH eval jobs
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

TARGET_INPUT="${1:-${EVAL_TARGET:-base}}"
STEP_SET_RAW="${2:-${EVAL_STEPS:-${EVAL_STEP:-0}}}"
CONFIG_NAME="${3:-${EVAL_CONFIG_NAME:-crl_experiment/libero_spatial_grpo_openvlaoft_eval_spatial}}"
SEED="${4:-${EVAL_SEED:-1234}}"
SFT_MODEL_EVAL_TASK="${5:-${SFT_MODEL_EVAL_TASK:-}}"
OPD_TEACHER_MAPPING_JSON="${OPD_TEACHER_MAPPING_JSON:-${SCRIPT_DIR}/opd_teacher_mapping.json}"
SFT_TEACHER_PATH="${SFT_TEACHER_PATH:-}"
SFT_TEACHER_NAME="${SFT_TEACHER_NAME:-}"

if ! [[ "${SEED}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: SEED must be a non-negative integer, got: ${SEED}"
  exit 1
fi

normalize_target_for_eval() {
  local target="$1"
  target="${target%/}"
  if [[ "${target}" == "base" ]]; then
    printf '%s\n' "base"
    return 0
  fi
  if [[ "${target}" = /* ]]; then
    case "${target}" in
      "${PROJECT_ROOT}"/*)
        printf '%s\n' "${target#${PROJECT_ROOT}/}"
        ;;
      *)
        echo "ERROR: Absolute TARGET must be under PROJECT_ROOT (${PROJECT_ROOT}), got: ${target}"
        exit 1
        ;;
    esac
  else
    printf '%s\n' "${target}"
  fi
}

TARGET="$(normalize_target_for_eval "${TARGET_INPUT}")"

lookup_mapped_teacher_path() {
  local task_id="$1"
  if [[ ! -f "${OPD_TEACHER_MAPPING_JSON}" ]]; then
    echo "ERROR: OPD_TEACHER_MAPPING_JSON not found: ${OPD_TEACHER_MAPPING_JSON}"
    exit 1
  fi
  python3 - "${OPD_TEACHER_MAPPING_JSON}" "${task_id}" <<'PY'
import json
import sys

mapping_path = sys.argv[1]
task_id = str(sys.argv[2])
with open(mapping_path, "r", encoding="utf-8") as f:
    data = json.load(f)
path = (
    data.get("teacher_sft_by_task", {}).get(task_id)
    or data.get(task_id)
    or ""
)
print(path)
PY
}

SFT_TEACHER_MODE=0
if [[ -n "${SFT_TEACHER_PATH}" ]]; then
  SFT_TEACHER_MODE=1
elif [[ -n "${SFT_MODEL_EVAL_TASK}" ]]; then
  if ! [[ "${SFT_MODEL_EVAL_TASK}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: SFT_MODEL_EVAL_TASK must be a non-negative integer, got: ${SFT_MODEL_EVAL_TASK}"
    exit 1
  fi
  SFT_TEACHER_PATH="$(lookup_mapped_teacher_path "${SFT_MODEL_EVAL_TASK}")"
  if [[ -z "${SFT_TEACHER_PATH}" ]]; then
    echo "ERROR: No teacher mapping found for task ${SFT_MODEL_EVAL_TASK} in ${OPD_TEACHER_MAPPING_JSON}"
    exit 1
  fi
  if [[ ! -d "${SFT_TEACHER_PATH}" ]]; then
    echo "ERROR: Mapped SFT teacher path does not exist: ${SFT_TEACHER_PATH}"
    exit 1
  fi
  SFT_TEACHER_MODE=1
fi
if [[ "${SFT_TEACHER_MODE}" == "1" && ! -d "${SFT_TEACHER_PATH}" && "${DRY_RUN:-0}" != "1" ]]; then
  echo "ERROR: SFT teacher path does not exist: ${SFT_TEACHER_PATH}"
  exit 1
fi

STEP_SET_NORMALIZED="${STEP_SET_RAW//,/ }"
read -r -a STEP_LIST <<< "${STEP_SET_NORMALIZED}"
if ((${#STEP_LIST[@]} == 0)); then
  echo "ERROR: STEP_OR_STEP_SET is empty. Provide e.g. 50 or 10,20,30"
  exit 1
fi
for STEP in "${STEP_LIST[@]}"; do
  if ! [[ "${STEP}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: STEP values must be non-negative integers, got: ${STEP}"
    exit 1
  fi
done

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

echo "Full eval submit helper"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "TARGET(input)=${TARGET_INPUT}"
echo "TARGET(resolved_for_eval)=${TARGET}"
echo "STEPS=${STEP_SET_RAW}"
echo "SEED=${SEED}"
echo "CONFIG=${CONFIG_NAME}"
echo "OPD_TEACHER_MAPPING_JSON=${OPD_TEACHER_MAPPING_JSON}"
if [[ "${SFT_TEACHER_MODE}" == "1" ]]; then
  echo "SFT teacher eval mode: task=${SFT_MODEL_EVAL_TASK:-explicit}"
  echo "SFT teacher path=${SFT_TEACHER_PATH}"
fi
echo "EVAL_NUM_ENVS_TOTAL(assumed from OPD eval config)=${EVAL_NUM_ENVS_TOTAL}"
echo "EVAL_TASK_COUNT=${EVAL_TASK_COUNT}"
echo "EVAL_ROLLOUTS_PER_TASK(requested)=${EVAL_ROLLOUTS_PER_TASK}"
echo "PER_EPOCH_PER_TASK=${PER_EPOCH_PER_TASK}"
echo "algorithm.eval_rollout_epoch(derived)=${EVAL_ROLLOUT_EPOCH}"
echo "ACTUAL_ROLLOUTS_PER_TASK=${ACTUAL_ROLLOUTS_PER_TASK}"
echo "=================================="

job_count=0
for STEP in "${STEP_LIST[@]}"; do
  if [[ "${SFT_TEACHER_MODE}" == "1" ]]; then
    if [[ -n "${SFT_TEACHER_NAME}" ]]; then
      TARGET_STEM="${SFT_TEACHER_NAME}"
    else
      TARGET_STEM="sft_teacher_task_${SFT_MODEL_EVAL_TASK:-explicit}"
    fi
  else
    TARGET_STEM="$(basename "${TARGET%/}")"
  fi
  W_NAME="eval_full_${TARGET_STEM}_step_${STEP}_seed_${SEED}_rpt_${ACTUAL_ROLLOUTS_PER_TASK}"
  W_NAME="${W_NAME//[^a-zA-Z0-9._-]/_}"
  JOB_NAME="${W_NAME}"
  if ((${#JOB_NAME} > 40)); then
    JOB_NAME="${JOB_NAME:0:40}"
  fi

  # Keep eval env count aligned with OPD sweep/config; only override eval_rollout_epoch.
  if [[ "${SFT_TEACHER_MODE}" == "1" ]]; then
    CMD=$(printf '%q ' bash examples/embodiment/eval_embodiment.sh "${CONFIG_NAME}" "runner.logger.experiment_name=${W_NAME}" "actor.seed=${SEED}" "algorithm.eval_rollout_epoch=${EVAL_ROLLOUT_EPOCH}" "+actor.model.lora_path=${SFT_TEACHER_PATH}")
  else
    EVAL_HYDRA_OVERRIDES="runner.logger.experiment_name=${W_NAME} actor.seed=${SEED} algorithm.eval_rollout_epoch=${EVAL_ROLLOUT_EPOCH}"
    CMD=$(printf '%q ' env EVAL_HYDRA_OVERRIDES="${EVAL_HYDRA_OVERRIDES}" bash examples/crl_experiment/eval_embodiment.sh "${TARGET}" "${STEP}" "${CONFIG_NAME}" "${SEED}")
  fi

  echo "Submitting job: ${JOB_NAME} (step=${STEP}, wandb_name=${W_NAME})"
  submit_job "${JOB_NAME}" "${CMD}"
  job_count=$((job_count + 1))
done

echo "Submitted ${job_count} job(s). Logs: ${SLURM_LOG_DIR}/eval_*.out"
echo "Check: squeue -u \$USER"
