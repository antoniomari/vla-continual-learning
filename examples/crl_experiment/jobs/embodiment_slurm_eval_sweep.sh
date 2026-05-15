#!/bin/bash
# Submit SLURM jobs that run checkpoint evaluation only:
#   examples/crl_experiment/eval_embodiment.sh
#
# Companion to embodiment_slurm_sweep.sh (train + optional eval); this file is eval-only.
#
# Usage:
#   bash examples/crl_experiment/jobs/embodiment_slurm_eval_sweep.sh
#
# Base model (no LoRA) on Slurm: set
#   EVAL_CHECKPOINT_LOCS=("base")
#   EVAL_STEPS=(0)
#   (step is only for logs; path is not used). Same works with RUN_MODE=eval in embodiment_slurm_sweep.sh.
#
# Optional:
#   DRY_RUN=1 bash ... # print batch scripts, no sbatch
#   EVAL_TARGET=/abs/or/relative/path  # override EVAL_CHECKPOINT_LOCS with one target
#   EVAL_STEPS="10,20,30"              # override EVAL_STEPS array
#   EVAL_SEED=1234                     # seed passed to eval_embodiment.sh
#   PROJECT_ROOT, VENV_PATH, SLURM_LOG_DIR — overrides
#   SLURM_PARTITION — optional on any cluster
#   SLURM_ACCOUNT — on other clusters; on $HOME=/users/anmari defaults to a143 (override if needed)
#   SBATCH_EXTRA — extra sbatch flags
#
# Libero: export LIBERO_REPO_PATH=/absolute/path/to/LIBERO if not at ${REPO_PATH}/LIBERO.
#
# Cluster: if HOME is /users/anmari, emit only #SBATCH --time (no cpus/mem/gpus); --account defaults to a143.
# Else emit full resource lines. Default TIME=12h (override with TIME=...; match embodiment_slurm_sweep.sh).
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-${PROJECT_ROOT}/logs/slurm_embodiment_eval}"
VENV_PATH="${VENV_PATH:-${PROJECT_ROOT}/.venv}"

mkdir -p "${SLURM_LOG_DIR}"

# ============== SLURM RESOURCES (edit for your cluster) ==============
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

# ============== EVAL SWEEP ==============
# eval_embodiment.sh args: CHECKPOINT_LOCATION [STEP_NUMBER] [CONFIG_NAME]
# CHECKPOINT_LOCATION: relative to repo root (directory that contains checkpoints/)
#
# LoRA checkpoint: path is run dir that contains checkpoints/ (no trailing checkpoints/...).
# Base SFT only: use literal "base" and any numeric step (e.g. 0).
EVAL_CHECKPOINT_LOCS=(
 # "logs_spatial/sequential/task_0_seed1234"
 "base"
)
EVAL_STEPS=(0)
# For base-only eval use: EVAL_STEPS=(0) and only "base" in EVAL_CHECKPOINT_LOCS.
EVAL_CONFIG_NAMES=("crl_experiment/libero_spatial_grpo_openvlaoft_eval_spatial")
EVAL_SEED="${EVAL_SEED:-1234}"

if [[ -n "${EVAL_TARGET:-}" ]]; then
  EVAL_CHECKPOINT_LOCS=("${EVAL_TARGET}")
fi
if [[ -n "${EVAL_STEPS:-}" ]]; then
  STEP_SET_NORMALIZED="${EVAL_STEPS//,/ }"
  read -r -a EVAL_STEPS <<< "${STEP_SET_NORMALIZED}"
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
        echo "ERROR: Absolute checkpoint location must be under PROJECT_ROOT (${PROJECT_ROOT}), got: ${target}"
        exit 1
        ;;
    esac
  else
    printf '%s\n' "${target}"
  fi
}

# ============== SUBMIT HELPER ==============
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

# ============== MAIN ==============
echo "Embodied eval SLURM sweep"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "SLURM_LOG_DIR=${SLURM_LOG_DIR}"
echo "EVAL_SEED=${EVAL_SEED}"
if [[ -n "${LIBERO_REPO_PATH}" ]]; then
  echo "LIBERO_REPO_PATH (in jobs)=${LIBERO_REPO_PATH}"
else
  echo "LIBERO_REPO_PATH: (unset — jobs use default in eval_embodiment.sh: \${REPO_PATH}/LIBERO)"
fi
if [[ "${HOME}" == "/users/anmari" ]]; then
  echo "Resources (HOME=/users/anmari): TIME=${TIME} — no #SBATCH cpus/mem/gpus; account=${NEW_CLUSTER_ACCOUNT}"
else
  echo "Resources: TIME=${TIME} CPUS_PER_TASK=${CPUS_PER_TASK} MEM_PER_CPU=${MEM_PER_CPU} GPU=${GPU}"
fi
echo "=================================="

job_count=0
for LOC in "${EVAL_CHECKPOINT_LOCS[@]}"; do
  RESOLVED_LOC="$(normalize_target_for_eval "${LOC}")"
  for STEP in "${EVAL_STEPS[@]}"; do
    if ! [[ "${STEP}" =~ ^[0-9]+$ ]]; then
      echo "ERROR: EVAL_STEPS must contain only non-negative integers, got: ${STEP}"
      exit 1
    fi
    for ECFG in "${EVAL_CONFIG_NAMES[@]}"; do
      EVAL_LOC_STEM="$(basename "${RESOLVED_LOC%/}")"
      EVAL_WANDB_NAME="eval_${EVAL_LOC_STEM}_step_${STEP}_seed_${EVAL_SEED}"
      EVAL_WANDB_NAME="${EVAL_WANDB_NAME//[^a-zA-Z0-9._-]/_}"
      JOB_NAME="${EVAL_WANDB_NAME}"
      JOB_NAME="${JOB_NAME//[^a-zA-Z0-9._-]/_}"
      if ((${#JOB_NAME} > 40)); then
        JOB_NAME="${JOB_NAME:0:40}"
      fi

      EVAL_HYDRA_OVERRIDES="runner.logger.experiment_name=${EVAL_WANDB_NAME} actor.seed=${EVAL_SEED}"
      CMD=$(printf '%q ' env EVAL_HYDRA_OVERRIDES="${EVAL_HYDRA_OVERRIDES}" bash examples/crl_experiment/eval_embodiment.sh "${RESOLVED_LOC}" "${STEP}" "${ECFG}" "${EVAL_SEED}")
      echo "Submit eval: loc=${LOC} resolved_loc=${RESOLVED_LOC} step=${STEP} seed=${EVAL_SEED} cfg=${ECFG} wandb_name=${EVAL_WANDB_NAME}"

      submit_job "${JOB_NAME}" "${CMD}"
      job_count=$((job_count + 1))
    done
  done
done

echo "=================================="
echo "Submitted ${job_count} job(s). Logs: ${SLURM_LOG_DIR}/eval_*.out"
echo "Check: squeue -u \$USER"
