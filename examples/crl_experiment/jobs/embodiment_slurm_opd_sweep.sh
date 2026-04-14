#!/bin/bash
# Submit SLURM jobs for on-policy distillation (OPD) embodied training / eval.
# Same layout as embodiment_slurm_sweep.sh but defaults to:
#   - train: crl_experiment/libero_spatial_opd_openvlaoft_spatial
#   - eval:  crl_experiment/libero_spatial_grpo_openvlaoft_eval_spatial
#     (no separate opd_*_eval yaml in-repo; student checkpoint uses same eval setup as GRPO.)
#
# Train entrypoint: examples/crl_experiment/run_embodiment_opd_sequential.sh (wraps run_embodiment_sequential.sh)
# Eval entrypoint:  examples/crl_experiment/eval_embodiment.sh
# GRPO / generic sweep: jobs/embodiment_slurm_sweep.sh
#
# OPD notes (see libero_spatial_opd_openvlaoft_spatial.yaml):
#   - opd_bc_steps, LiberoSFT / teacher paths — tune in config or Hydra overrides.
#   - BC teacher under logger.log_path/opd_bc_teacher/actor; same snapshot at opd_bc_student/actor.
#
# Usage:
#   export RUN_MODE=train   # or eval
#   bash examples/crl_experiment/jobs/embodiment_slurm_opd_sweep.sh
#
# Optional:
#   DRY_RUN=1 bash ... # print jobs without sbatch
#   GRPO_HP_FROM_SWEEP=0 — do not override algorithm.group_size / num_group_envs / rollout_epoch /
#       actor.global_batch_size (yaml only). Default 1 uses TRAIN_GROUP_SIZES × TRAIN_NUM_GROUP_ENVS ×
#       TRAIN_ROLLOUT_EPOCHS × TRAIN_SEEDS with SWEEP_GLOBAL_BATCH_SIZE = group_size × num_group_envs × rollout_epoch × 64
#       (same formula as jobs/embodiment_slurm_sweep.sh; passed via env vars read by run_embodiment_sequential.sh).
#   PROJECT_ROOT, VENV_PATH, SLURM_LOG_DIR — overrides (default logs: logs/slurm_embodiment_opd)
#   SLURM_PARTITION, SLURM_ACCOUNT, SBATCH_EXTRA
#
# Libero: training/eval scripts default to ${REPO_PATH}/LIBERO. If your clone lives elsewhere,
#   export LIBERO_REPO_PATH=/absolute/path/to/LIBERO
# before running this script (values are baked into each sbatch script). Optional:
#   export LIBERO_CONFIG_PATH=...   # defaults to LIBERO_REPO_PATH when exporting.
#
# Cluster: if HOME is /users/anmari, emit only #SBATCH --time (no cpus/mem/gpus); --account defaults to a143.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-${PROJECT_ROOT}/logs/slurm_embodiment_opd}"
VENV_PATH="${VENV_PATH:-${PROJECT_ROOT}/.venv}"

RUN_MODE="${RUN_MODE:-train}" # train | eval

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

# Baked into each job if non-empty (see header). Otherwise run_embodiment.sh uses ${REPO_PATH}/LIBERO.
LIBERO_REPO_PATH="${LIBERO_REPO_PATH:-}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-}"

# ============== TRAIN SWEEP (run_embodiment_opd_sequential.sh → run_embodiment_sequential.sh) ==============
# Args: TASK_ID_OR_RANGE [CHECKPOINT_PATH] [MAX_EPOCH] [CONFIG_NAME] [SEED]
TRAIN_TASK_INPUTS=("0,1")
TRAIN_MANUAL_CHECKPOINT=("")
# Passed as MAX_EPOCH (runner.max_epochs + checkpoint index). Empty = yaml max_epochs; post-train eval
# step still uses get_default_global_step for checkpoint folder unless you align EVAL_STEPS.
TRAIN_MAX_EPOCHS=(50)
# LiberoSFT / opd_bc_steps / teacher paths — tune in libero_spatial_opd_openvlaoft_spatial.yaml or Hydra.
TRAIN_CONFIG_NAMES=("crl_experiment/libero_spatial_opd_openvlaoft_spatial")
TRAIN_SEEDS=(4096)

# Rollout geometry overrides (Hydra), same env vars as jobs/embodiment_slurm_sweep.sh.
# OPD: algorithm.normalize_advantages is false — group_size / num_group_envs / rollout_epoch do not
# reshape rewards via group normalization; they only scale how many parallel rollouts you collect
# (and, with filter_rewards, which prompts share a group). SWEEP_GLOBAL_BATCH_SIZE should still
# match the rollout product ×64 so actor.global_batch_size stays consistent with the batch builder.
# Cartesian product of the three lists × TRAIN_SEEDS when GRPO_HP_FROM_SWEEP=1:
#   SWEEP_GLOBAL_BATCH_SIZE = group_size × num_group_envs × rollout_epoch × 64
GRPO_HP_FROM_SWEEP="${GRPO_HP_FROM_SWEEP:-1}"
TRAIN_GROUP_SIZES=(8)
TRAIN_NUM_GROUP_ENVS=(4)
TRAIN_ROLLOUT_EPOCHS=(2)

# ============== EVAL SWEEP ==============
# Same OpenVLA-OFT Libero spatial eval config as GRPO (student LoRA after OPD).
# Keep EVAL_* in sync with TRAIN_* above: checkpoint dir is logs/sequential/task_<id>_seed<SEED>/;
# EVAL_STEPS should match TRAIN_MAX_EPOCHS (global_step_<N> folder). Edit if you change seeds or log roots.
EVAL_CHECKPOINT_LOCS=("logs/sequential/task_0_seed4096")
EVAL_STEPS=(10)
EVAL_CONFIG_NAMES=("crl_experiment/libero_spatial_grpo_openvlaoft_eval_spatial")

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
    echo "#SBATCH --output=${SLURM_LOG_DIR}/${RUN_MODE}_%j.out"
    echo "#SBATCH --error=${SLURM_LOG_DIR}/${RUN_MODE}_%j.err"
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

echo "OPD embodied SLURM sweep — RUN_MODE=${RUN_MODE}"
echo "GRPO_HP_FROM_SWEEP=${GRPO_HP_FROM_SWEEP} (1 = override group_size, num_group_envs, rollout_epoch, global_batch_size)"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "SLURM_LOG_DIR=${SLURM_LOG_DIR}"
if [[ -n "${LIBERO_REPO_PATH}" ]]; then
  echo "LIBERO_REPO_PATH (in jobs)=${LIBERO_REPO_PATH}"
else
  echo "LIBERO_REPO_PATH: (unset — jobs use default in run_embodiment.sh: \${REPO_PATH}/LIBERO)"
fi
if [[ "${HOME}" == "/users/anmari" ]]; then
  echo "Resources (HOME=/users/anmari): TIME=${TIME} — no #SBATCH cpus/mem/gpus; account=${NEW_CLUSTER_ACCOUNT}"
else
  echo "Resources: TIME=${TIME} CPUS_PER_TASK=${CPUS_PER_TASK} MEM_PER_CPU=${MEM_PER_CPU} GPU=${GPU}"
fi
echo "=================================="

job_count=0

if [[ "${RUN_MODE}" == "train" ]]; then
  for TASK in "${TRAIN_TASK_INPUTS[@]}"; do
    for CKPT in "${TRAIN_MANUAL_CHECKPOINT[@]}"; do
      for MAX_EP in "${TRAIN_MAX_EPOCHS[@]}"; do
        for CFG in "${TRAIN_CONFIG_NAMES[@]}"; do
          if [[ "${GRPO_HP_FROM_SWEEP}" == "1" ]]; then
            for GS in "${TRAIN_GROUP_SIZES[@]}"; do
              for NGE in "${TRAIN_NUM_GROUP_ENVS[@]}"; do
                for RE in "${TRAIN_ROLLOUT_EPOCHS[@]}"; do
                  G_BATCH=$((GS * NGE * RE * 64))
                  for SEED in "${TRAIN_SEEDS[@]}"; do
                    JOB_NAME="opd_g${GS}n${NGE}r${RE}s${SEED}"
                    JOB_NAME="${JOB_NAME//[^a-zA-Z0-9._-]/_}"
                    if ((${#JOB_NAME} > 40)); then
                      JOB_NAME="${JOB_NAME:0:40}"
                    fi

                    ARGS=(bash examples/crl_experiment/run_embodiment_opd_sequential.sh "${TASK}")
                    [[ -n "${CKPT}" ]] && ARGS+=("${CKPT}") || ARGS+=("")
                    [[ -n "${MAX_EP}" ]] && ARGS+=("${MAX_EP}") || ARGS+=("")
                    ARGS+=("${CFG}" "${SEED}")
                    CMD="SWEEP_GROUP_SIZE=${GS} SWEEP_NUM_GROUP_ENVS=${NGE} SWEEP_ROLLOUT_EPOCH=${RE} SWEEP_GLOBAL_BATCH_SIZE=${G_BATCH} $(printf '%q ' "${ARGS[@]}")"
                    echo "Submit OPD train: task=${TASK} seed=${SEED} cfg=${CFG} max_epoch=${MAX_EP:-default} ckpt=${CKPT:-none} group_size=${GS} num_group_envs=${NGE} rollout_epoch=${RE} global_batch_size=${G_BATCH}"

                    submit_job "${JOB_NAME}" "${CMD}"
                    job_count=$((job_count + 1))
                  done
                done
              done
            done
          else
            for SEED in "${TRAIN_SEEDS[@]}"; do
              JOB_NAME="opd_t${TASK}_s${SEED}"
              JOB_NAME="${JOB_NAME//[^a-zA-Z0-9._-]/_}"
              if ((${#JOB_NAME} > 40)); then
                JOB_NAME="${JOB_NAME:0:40}"
              fi

              ARGS=(bash examples/crl_experiment/run_embodiment_opd_sequential.sh "${TASK}")
              [[ -n "${CKPT}" ]] && ARGS+=("${CKPT}") || ARGS+=("")
              [[ -n "${MAX_EP}" ]] && ARGS+=("${MAX_EP}") || ARGS+=("")
              ARGS+=("${CFG}" "${SEED}")

              CMD=$(printf '%q ' "${ARGS[@]}")
              echo "Submit OPD train: task=${TASK} seed=${SEED} cfg=${CFG} max_epoch=${MAX_EP:-default} ckpt=${CKPT:-none}"

              submit_job "${JOB_NAME}" "${CMD}"
              job_count=$((job_count + 1))
            done
          fi
        done
      done
    done
  done
elif [[ "${RUN_MODE}" == "eval" ]]; then
  for LOC in "${EVAL_CHECKPOINT_LOCS[@]}"; do
    for STEP in "${EVAL_STEPS[@]}"; do
      for ECFG in "${EVAL_CONFIG_NAMES[@]}"; do
        JOB_NAME="opd_ev_s${STEP}"
        JOB_NAME="${JOB_NAME//[^a-zA-Z0-9._-]/_}"
        if ((${#JOB_NAME} > 40)); then
          JOB_NAME="${JOB_NAME:0:40}"
        fi

        CMD=$(printf '%q ' bash examples/crl_experiment/eval_embodiment.sh "${LOC}" "${STEP}" "${ECFG}")
        echo "Submit OPD eval: loc=${LOC} step=${STEP} cfg=${ECFG}"

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
