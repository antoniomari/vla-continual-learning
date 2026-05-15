#!/bin/bash
# Submit SLURM jobs that run:
#   - sequential embodied training:  examples/crl_experiment/run_embodiment_sequential.sh
#   - or checkpoint evaluation:       examples/crl_experiment/eval_embodiment.sh
# Eval-only helper: jobs/embodiment_slurm_eval_sweep.sh
# OPD (on-policy distillation): jobs/embodiment_slurm_opd_sweep.sh
#
# Style aligned with meta_vlas/meta_libero/jobs/on_policy_distillation_sweep.sh
# (PROJECT_ROOT, venv, log dir, nested loops, sbatch).
#
# Usage:
#   export RUN_MODE=train   # or eval
#   bash examples/crl_experiment/jobs/embodiment_slurm_sweep.sh
#
# Eval base model (no LoRA) on Slurm: RUN_MODE=eval and EVAL_CHECKPOINT_LOCS=("base"), EVAL_STEPS=(0).
#
# Optional:
#   DRY_RUN=1 bash ... # print jobs without sbatch
#   BASE_MODEL=1 — force the first train task in each submitted job to start from base SFT
#       (passes CHECKPOINT_PATH=base to run_embodiment_sequential.sh), even if TASK > first-task-id.
#   SKIP_POST_TRAIN_EVAL=1 is always set for train jobs from this sweep; run eval separately.
#   GRPO_HP_FROM_SWEEP=0 — do not override algorithm.group_size / num_group_envs / rollout_epoch /
#       actor.global_batch_size (yaml only). Default1 uses TRAIN_GROUP_SIZES × TRAIN_NUM_GROUP_ENVS ×
#       TRAIN_ROLLOUT_EPOCHS × TRAIN_SEEDS with global_batch_size = product of the first three.
#   TRAIN_MAX_EPOCHS — passed as 3rd arg to run_embodiment_sequential (runner.max_epochs + checkpoint index).
#       Post-train eval uses global_step_<TRAIN_MAX_EPOCHS>; keep EVAL_STEPS in sync for RUN_MODE=eval.
#   Eval-only jobs: W&B run name eval_<basename(LOG_DIR)>_step_<STEP> via EVAL_HYDRA_OVERRIDES.
#   PROJECT_ROOT, VENV_PATH, SLURM_LOG_DIR — overrides
#   SLURM_PARTITION — optional on any cluster
#   SLURM_ACCOUNT — on other clusters; on $HOME=/users/anmari defaults to a143 (override if needed)
#   SBATCH_EXTRA — extra sbatch flags
#
# Libero: export LIBERO_REPO_PATH=/absolute/path/to/LIBERO before running if not at ${REPO_PATH}/LIBERO.
#   Optional: LIBERO_CONFIG_PATH (defaults to LIBERO_REPO_PATH in the job).
#
# Cluster: if HOME is /users/anmari, emit only #SBATCH --time (no cpus/mem/gpus); --account defaults to a143.
# Else emit --time, --cpus-per-task, --mem-per-cpu, --gpus. Override TIME, etc. via env.
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

# ============== TRAIN SWEEP (run_embodiment_sequential.sh) ==============
# Args to run_embodiment_sequential.sh:
#   TASK_ID_OR_RANGE [CHECKPOINT_PATH] [MAX_EPOCH] [CONFIG_NAME] [SEED]
TRAIN_TASK_INPUTS=("1" "4")
TRAIN_MANUAL_CHECKPOINT=("") # e.g. ("" "/abs/path") — usually ""
# Passed as MAX_EPOCH to run_embodiment_sequential (sets runner.max_epochs and checkpoint global_step).
# Empty = training uses yaml max_epochs but inter-task / post-train eval still use get_default_global_step (50 for spatial).
TRAIN_MAX_EPOCHS=(60)
# GRPO: no LiberoSFT replay buffer. For OPD (`libero_spatial_opd_*`) set opd_bc_steps>0 and
# LIBERO_REPO_PATH + datasets_with_logits/... or use_experience_replay needs the same data.
TRAIN_CONFIG_NAMES=("crl_experiment/libero_spatial_grpo_openvlaoft_spatial")
TRAIN_SEEDS=(4096)
BASE_MODEL="${BASE_MODEL:-1}"
if [[ "${BASE_MODEL}" == "1" ]]; then
  TRAIN_MANUAL_CHECKPOINT=("base")
fi

# GRPO algorithm / batch overrides (Hydra), passed via env vars read by run_embodiment_sequential.sh.
# Cartesian product of the three lists × TRAIN_SEEDS. For each combo:
#   SWEEP_GLOBAL_BATCH_SIZE = group_size × num_group_envs × rollout_epoch
# GRPO_HP_FROM_SWEEP=0 disables (yaml defaults for these fields only).
GRPO_HP_FROM_SWEEP="${GRPO_HP_FROM_SWEEP:-1}"
TRAIN_GROUP_SIZES=(8)
TRAIN_NUM_GROUP_ENVS=(4)
TRAIN_ROLLOUT_EPOCHS=(4)
DEFAULT_ROLLOUTS_PER_STEP=$((TRAIN_GROUP_SIZES[0] * TRAIN_NUM_GROUP_ENVS[0] * TRAIN_ROLLOUT_EPOCHS[0]))

apply_array_override() {
  local array_name="$1"
  local env_name="$2"
  if [[ -z "${!env_name+x}" ]]; then
    return 0
  fi

  local raw="${!env_name}"
  local values=()
  local assignment
  read -r -a values <<< "${raw}"
  assignment="${array_name}=("
  for value in "${values[@]}"; do
    assignment+="$(printf '%q' "${value}") "
  done
  assignment+=")"
  eval "${assignment}"
}

apply_array_override TRAIN_TASK_INPUTS TRAIN_TASK_INPUTS_OVERRIDE
apply_array_override TRAIN_MANUAL_CHECKPOINT TRAIN_MANUAL_CHECKPOINT_OVERRIDE
apply_array_override TRAIN_MAX_EPOCHS TRAIN_MAX_EPOCHS_OVERRIDE
apply_array_override TRAIN_CONFIG_NAMES TRAIN_CONFIG_NAMES_OVERRIDE
apply_array_override TRAIN_SEEDS TRAIN_SEEDS_OVERRIDE
apply_array_override TRAIN_GROUP_SIZES TRAIN_GROUP_SIZES_OVERRIDE
apply_array_override TRAIN_NUM_GROUP_ENVS TRAIN_NUM_GROUP_ENVS_OVERRIDE
apply_array_override TRAIN_ROLLOUT_EPOCHS TRAIN_ROLLOUT_EPOCHS_OVERRIDE

# ============== EVAL SWEEP (eval_embodiment.sh) ==============
# Usage of eval_embodiment.sh:
#   CHECKPOINT_LOCATION [STEP_NUMBER] [CONFIG_NAME]
# CHECKPOINT_LOCATION is relative to repo root (see eval_embodiment.sh header).
EVAL_CHECKPOINT_LOCS=("logs/sequential/task_0_seed1234")
# Must match the global_step_* folder produced by training (typically == runner.max_epochs).
EVAL_STEPS=(50)
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

# ============== MAIN ==============
echo "Embodied SLURM sweep — RUN_MODE=${RUN_MODE}"
echo "BASE_MODEL=${BASE_MODEL} (1 = force first task in each job to use base SFT checkpoint)"
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
                  ROLLOUTS_PER_STEP=$((G_BATCH / 64))
                  for SEED in "${TRAIN_SEEDS[@]}"; do
                    JOB_NAME="grpo_rps${ROLLOUTS_PER_STEP}_gb${G_BATCH}_gs${GS}_t${TASK}_s${SEED}"
                    JOB_NAME="${JOB_NAME//[^a-zA-Z0-9._-]/_}"
                    if ((${#JOB_NAME} > 40)); then
                      JOB_NAME="${JOB_NAME:0:40}"
                    fi

                    ARGS=(bash examples/crl_experiment/run_embodiment_sequential.sh "${TASK}")
                    [[ -n "${CKPT}" ]] && ARGS+=("${CKPT}") || ARGS+=("")
                    [[ -n "${MAX_EP}" ]] && ARGS+=("${MAX_EP}") || ARGS+=("")
                    ARGS+=("${CFG}" "${SEED}")
                    SAVE_INTERVAL_OVERRIDE="${SWEEP_SAVE_INTERVAL:-20}"
                    WANDB_PREFIX="grpo_rps${ROLLOUTS_PER_STEP}_gb${G_BATCH}_gs${GS}_"
                    CMD="EXPERIMENT_NAME_PREFIX=${WANDB_PREFIX} SKIP_POST_TRAIN_EVAL=1 SWEEP_GROUP_SIZE=${GS} SWEEP_NUM_GROUP_ENVS=${NGE} SWEEP_ROLLOUT_EPOCH=${RE} SWEEP_GLOBAL_BATCH_SIZE=${G_BATCH} SWEEP_SAVE_INTERVAL=${SAVE_INTERVAL_OVERRIDE} $(printf '%q ' "${ARGS[@]}")"
                    echo "Submit train: task=${TASK} seed=${SEED} cfg=${CFG} max_epoch=${MAX_EP:-default} ckpt=${CKPT:-none} group_size=${GS} num_group_envs=${NGE} rollout_epoch=${RE} global_batch_size=${G_BATCH} rollouts_per_step=${ROLLOUTS_PER_STEP}"

                    submit_job "${JOB_NAME}" "${CMD}"
                    job_count=$((job_count + 1))
                  done
                done
              done
            done
          else
            for SEED in "${TRAIN_SEEDS[@]}"; do
              JOB_NAME="grpo_t${TASK}_s${SEED}"
              JOB_NAME="${JOB_NAME//[^a-zA-Z0-9._-]/_}"
              if ((${#JOB_NAME} > 40)); then
                JOB_NAME="${JOB_NAME:0:40}"
              fi

              ARGS=(bash examples/crl_experiment/run_embodiment_sequential.sh "${TASK}")
              [[ -n "${CKPT}" ]] && ARGS+=("${CKPT}") || ARGS+=("")
              [[ -n "${MAX_EP}" ]] && ARGS+=("${MAX_EP}") || ARGS+=("")
              ARGS+=("${CFG}" "${SEED}")

              SAVE_INTERVAL_OVERRIDE="${MAX_EP}"
              WANDB_PREFIX="grpo_rps${DEFAULT_ROLLOUTS_PER_STEP}_gbna_gsna_"
              CMD="EXPERIMENT_NAME_PREFIX=${WANDB_PREFIX} SKIP_POST_TRAIN_EVAL=1 SWEEP_SAVE_INTERVAL=${SAVE_INTERVAL_OVERRIDE} $(printf '%q ' "${ARGS[@]}")"
              echo "Submit train: task=${TASK} seed=${SEED} cfg=${CFG} max_epoch=${MAX_EP:-default} ckpt=${CKPT:-none}"

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
        EVAL_LOC_STEM="$(basename "${LOC%/}")"
        EVAL_WANDB_NAME="eval_${EVAL_LOC_STEM}_step_${STEP}"
        EVAL_WANDB_NAME="${EVAL_WANDB_NAME//[^a-zA-Z0-9._-]/_}"
        JOB_NAME="${EVAL_WANDB_NAME}"
        if ((${#JOB_NAME} > 40)); then
          JOB_NAME="${JOB_NAME:0:40}"
        fi

        EVAL_HYDRA_OVERRIDES="runner.logger.experiment_name=${EVAL_WANDB_NAME}"
        CMD=$(printf '%q ' env EVAL_HYDRA_OVERRIDES="${EVAL_HYDRA_OVERRIDES}" bash examples/crl_experiment/eval_embodiment.sh "${LOC}" "${STEP}" "${ECFG}")
        echo "Submit eval: loc=${LOC} step=${STEP} cfg=${ECFG} wandb_name=${EVAL_WANDB_NAME}"

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
