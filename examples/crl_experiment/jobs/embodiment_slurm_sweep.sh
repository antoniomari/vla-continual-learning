#!/bin/bash
# Submit SLURM jobs that run:
#   - sequential embodied training:  examples/crl_experiment/run_embodiment_sequential.sh
#   - or checkpoint evaluation:       examples/crl_experiment/eval_embodiment.sh
#
# Style aligned with meta_vlas/meta_libero/jobs/on_policy_distillation_sweep.sh
# (PROJECT_ROOT, venv, log dir, nested loops, sbatch).
#
# Usage:
#   export RUN_MODE=train   # or eval
#   bash examples/crl_experiment/jobs/embodiment_slurm_sweep.sh
#
# Optional:
#   DRY_RUN=1 bash ... # print jobs without sbatch
#   PROJECT_ROOT, VENV_PATH, SLURM_LOG_DIR — overrides
#
set -euo pipefail

# Repo root: this file lives in examples/crl_experiment/jobs/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-${PROJECT_ROOT}/logs/slurm_embodiment}"
VENV_PATH="${VENV_PATH:-${PROJECT_ROOT}/.venv}"

RUN_MODE="${RUN_MODE:-train}" # train | eval

mkdir -p "${SLURM_LOG_DIR}"

# ============== SLURM RESOURCES (edit for your cluster) ==============
# Defaults aligned with an interactive allocation like:
#   srun --time=24:0:0 --cpus-per-task=8 --mem-per-cpu=16G --gpus=pro_6000:1 --pty bash -l
TIME="${TIME:-24:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM_PER_CPU="${MEM_PER_CPU:-16G}"
GPU="${GPU:-pro_6000:1}"

# Optional: export SLURM_PARTITION / SLURM_ACCOUNT before running, or set here.
SLURM_PARTITION="${SLURM_PARTITION:-}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"

# Extra flags passed to sbatch (quoted string), e.g. '--qos=normal'
SBATCH_EXTRA="${SBATCH_EXTRA:-}"

# ============== TRAIN SWEEP (run_embodiment_sequential.sh) ==============
# Args to run_embodiment_sequential.sh:
#   TASK_ID_OR_RANGE [CHECKPOINT_PATH] [MAX_EPOCH] [CONFIG_NAME] [SEED]
TRAIN_TASK_INPUTS=("0,1")
TRAIN_MANUAL_CHECKPOINT=("") # e.g. ("" "/abs/path") — usually ""
TRAIN_MAX_EPOCHS=("") # e.g. ("" "15") — empty = use config default
# GRPO: no LiberoSFT replay buffer. For OPD (`libero_spatial_opd_*`) set opd_bc_steps>0 and
# LIBERO_REPO_PATH + datasets_with_logits/... or use_experience_replay needs the same data.
TRAIN_CONFIG_NAMES=("crl_experiment/libero_spatial_grpo_openvlaoft_spatial")
TRAIN_SEEDS=(1234)

# ============== EVAL SWEEP (eval_embodiment.sh) ==============
# Usage of eval_embodiment.sh:
#   CHECKPOINT_LOCATION [STEP_NUMBER] [CONFIG_NAME]
# CHECKPOINT_LOCATION is relative to repo root (see eval_embodiment.sh header).
EVAL_CHECKPOINT_LOCS=("logs/sequential/task_0_seed1234")
EVAL_STEPS=(10)
EVAL_CONFIG_NAMES=("crl_experiment/libero_spatial_grpo_openvlaoft_eval_spatial")

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
    echo "#SBATCH --cpus-per-task=${CPUS_PER_TASK}"
    echo "#SBATCH --mem-per-cpu=${MEM_PER_CPU}"
    echo "#SBATCH --gpus=${GPU}"
    echo "#SBATCH --output=${SLURM_LOG_DIR}/${RUN_MODE}_%j.out"
    echo "#SBATCH --error=${SLURM_LOG_DIR}/${RUN_MODE}_%j.err"
    if [[ -n "${SLURM_PARTITION}" ]]; then
      echo "#SBATCH --partition=${SLURM_PARTITION}"
    fi
    if [[ -n "${SLURM_ACCOUNT}" ]]; then
      echo "#SBATCH --account=${SLURM_ACCOUNT}"
    fi
    echo "set -euo pipefail"
    echo "cd \"${PROJECT_ROOT}\""
    echo "source \"${VENV_PATH}/bin/activate\""
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
echo "Embodied SLURM sweep — RUN_MODE=${RUN_MODE}"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "SLURM_LOG_DIR=${SLURM_LOG_DIR}"
echo "Resources: TIME=${TIME} CPUS_PER_TASK=${CPUS_PER_TASK} MEM_PER_CPU=${MEM_PER_CPU} GPU=${GPU}"
echo "=================================="

job_count=0

if [[ "${RUN_MODE}" == "train" ]]; then
  for TASK in "${TRAIN_TASK_INPUTS[@]}"; do
    for CKPT in "${TRAIN_MANUAL_CHECKPOINT[@]}"; do
      for MAX_EP in "${TRAIN_MAX_EPOCHS[@]}"; do
        for CFG in "${TRAIN_CONFIG_NAMES[@]}"; do
          for SEED in "${TRAIN_SEEDS[@]}"; do
            # Short job name for squeue
            JOB_NAME="emb_t${TASK}_s${SEED}"
            JOB_NAME="${JOB_NAME//[^a-zA-Z0-9._-]/_}"
            if ((${#JOB_NAME} > 40)); then
              JOB_NAME="${JOB_NAME:0:40}"
            fi

            ARGS=(bash examples/crl_experiment/run_embodiment_sequential.sh "${TASK}")
            [[ -n "${CKPT}" ]] && ARGS+=("${CKPT}") || ARGS+=("")
            [[ -n "${MAX_EP}" ]] && ARGS+=("${MAX_EP}") || ARGS+=("")
            ARGS+=("${CFG}" "${SEED}")

            CMD=$(printf '%q ' "${ARGS[@]}")
            echo "Submit train: task=${TASK} seed=${SEED} cfg=${CFG} max_epoch=${MAX_EP:-default} ckpt=${CKPT:-none}"

            submit_job "${JOB_NAME}" "${CMD}"
            job_count=$((job_count + 1))
          done
        done
      done
    done
  done
elif [[ "${RUN_MODE}" == "eval" ]]; then
  for LOC in "${EVAL_CHECKPOINT_LOCS[@]}"; do
    for STEP in "${EVAL_STEPS[@]}"; do
      for ECFG in "${EVAL_CONFIG_NAMES[@]}"; do
        JOB_NAME="emb_ev_s${STEP}"
        JOB_NAME="${JOB_NAME//[^a-zA-Z0-9._-]/_}"
        if ((${#JOB_NAME} > 40)); then
          JOB_NAME="${JOB_NAME:0:40}"
        fi

        CMD=$(printf '%q ' bash examples/crl_experiment/eval_embodiment.sh "${LOC}" "${STEP}" "${ECFG}")
        echo "Submit eval: loc=${LOC} step=${STEP} cfg=${ECFG}"

        submit_job "${JOB_NAME}" "${CMD}"
        job_count=$((job_count + 1))
      done
    done
  done
else
  echo "ERROR: RUN_MODE must be 'train' or 'eval', got: ${RUN_MODE}"
  exit 1
fi

echo "=================================="
echo "Submitted ${job_count} job(s). Logs: ${SLURM_LOG_DIR}/${RUN_MODE}_*.out"
echo "Check: squeue -u \$USER"
