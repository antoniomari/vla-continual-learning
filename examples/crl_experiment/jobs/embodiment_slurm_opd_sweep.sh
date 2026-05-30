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
#   - BC teacher under logger.log_path/opd_bc_teacher/actor (student RL reloads from base via runner).
#
# Usage:
#   export RUN_MODE=train   # or eval
#   bash examples/crl_experiment/jobs/embodiment_slurm_opd_sweep.sh
#
# Optional:
#   DRY_RUN=1 bash ... # print jobs without sbatch
#   OPD_TEACHER_MAPPING_JSON=/path/to/json
#       JSON map for per-task SFT teacher adapters; if task key exists, the train job passes
#       algorithm.opd_teacher_model_path from this map (fallback: current behavior).
#   BASE_MODEL=1 — force the first train task in each submitted job to start from base SFT
#       (passes CHECKPOINT_PATH=base to run_embodiment_sequential.sh), even if TASK > first-task-id.
#   SKIP_POST_TRAIN_EVAL=1 is always set for train jobs from this sweep; run eval separately.
#   GRPO_HP_FROM_SWEEP=0 — do not override algorithm.group_size / num_group_envs / rollout_epoch /
#       actor.global_batch_size (yaml only). Default 1 uses TRAIN_GROUP_SIZES × TRAIN_NUM_GROUP_ENVS ×
#       TRAIN_ROLLOUT_EPOCHS × TRAIN_SEEDS with SWEEP_GLOBAL_BATCH_SIZE = group_size × num_group_envs × rollout_epoch × 64
#       (same formula as jobs/embodiment_slurm_sweep.sh; passed via env vars read by run_embodiment_sequential.sh).
#   OPD BC warmup (always forwarded when training): TRAIN_OPD_BC_GLOBAL_BATCH_SIZES, TRAIN_OPD_BC_BATCH_SIZES,
#       TRAIN_OPD_BC_STEPS, TRAIN_OPD_TEACHER_LRS — Cartesian product with the train grid; set as Hydra overrides via
#       SWEEP_OPD_* env vars in run_embodiment_sequential.sh (algorithm.opd_bc_* and actor.optim.opd_teacher_lr).
#       TRAIN_OPD_BC_SAVE_STEPS is optional and forwards algorithm.opd_bc_save_steps for intermediate
#       teacher checkpoints, e.g. "[250,500,750,1000]".
#       TRAIN_OPD_FORCE_RETRAIN_TEACHER=1 ignores an existing opd_bc_teacher checkpoint in the same
#       log dir and reruns BC teacher warmup.
#   OPD SFT preprocessing toggles (Cartesian product too): TRAIN_OPD_SFT_FILTER_FIXED_TASK_IDS,
#       TRAIN_OPD_SFT_MATCH_TASK_LANGUAGE,
#       TRAIN_OPD_SFT_MATCH_IMAGE_ROTATION, TRAIN_OPD_SFT_MATCH_OBS_ACTION_ALIGNMENT,
#       TRAIN_OPD_SFT_GRIPPER_FROM_NEG1_0_TO_0_1
#       (each 0/1; wired to algorithm.sft_* overrides).
#   OPD advantage normalization toggle: TRAIN_OPD_NORMALIZE_ADVANTAGES (0/1), wired to
#       algorithm.normalize_advantages.
#   OPD reward normalization mode: TRAIN_OPD_REWARD_NORMALIZATIONS
#       (e.g., group_zscore|token_zscore|action_dim_zscore|positive_clip|teacher_prob|mad_abs|batch_zscore|tanh_squash|clip),
#       wired to algorithm.opd_reward_normalization.
#   OPD reward tanh temperature: TRAIN_OPD_REWARD_TANH_TAU
#       wired to algorithm.opd_reward_tanh_tau (used by tanh_squash).
#   OPD reward clip bound: TRAIN_OPD_REWARD_CLIP_C
#       wired to algorithm.opd_reward_clip_c (used by clip mode).
#   OPD success-gated hybrid knobs:
#       TRAIN_OPD_SUCCESS_GATE_TEACHER_LAMBDAS -> algorithm.opd_success_gate_teacher_lambda
#       TRAIN_OPD_SUCCESS_GATE_REWARD_THRESHOLDS -> algorithm.opd_success_gate_reward_threshold
#   OPD rollout teacher memory knobs:
#       TRAIN_OPD_TEACHER_MICRO_BATCH_SIZES -> algorithm.opd_teacher_micro_batch_size
#       TRAIN_OPD_PRECOMPUTE_TEACHER_IN_ROLLOUT -> algorithm.opd_precompute_teacher_in_rollout (0/1)
#       TRAIN_OPD_TEACHER_STASH_LOGPROBS_ON_CPU -> algorithm.opd_teacher_stash_logprobs_on_cpu (0/1; slower)
#   OPD teacher warmup mode toggle: TRAIN_OPD_RL_TEACHER (0/1), wired to algorithm.rl_teacher.
#   SWEEP_WANDB_EXTRA_TAG — optional sanitized token appended to the generated W&B prefix.
#   PROJECT_ROOT, VENV_PATH, SLURM_LOG_DIR — overrides (default logs: logs/slurm_embodiment_opd)
#   SLURM_PARTITION, SLURM_ACCOUNT, SBATCH_EXTRA
#   PYTORCH_CUDA_ALLOC_CONF — CUDA allocator tuning (default: expandable_segments:True)
#
# Libero: training/eval scripts default to ${REPO_PATH}/LIBERO. If your clone lives elsewhere,
#   export LIBERO_REPO_PATH=/absolute/path/to/LIBERO
# before running this script (values are baked into each sbatch script). Optional:
#   export LIBERO_CONFIG_PATH=...   # defaults to LIBERO_REPO_PATH when exporting.
#
# Cluster override: if USE_MINIMAL_SBATCH_RESOURCES=1, emit only #SBATCH --time
# (no cpus/mem/gpus). Useful on clusters where those are injected externally.
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
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

USE_MINIMAL_SBATCH_RESOURCES="1"
NEW_CLUSTER_ACCOUNT="${SLURM_ACCOUNT:-a143}"

# Baked into each job if non-empty (see header). Otherwise run_embodiment.sh uses ${REPO_PATH}/LIBERO.
LIBERO_REPO_PATH="${LIBERO_REPO_PATH:-}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-}"
if [[ -z "${OPD_TEACHER_MAPPING_JSON:-}" ]]; then
  if [[ -n "${SCRATCH:-}" ]]; then
    OPD_TEACHER_MAPPING_JSON="${SCRATCH%/}/vla-continual-learning/examples/crl_experiment/jobs/opd_teacher_mapping.json"
  else
    OPD_TEACHER_MAPPING_JSON="${SCRIPT_DIR}/opd_teacher_mapping.json"
  fi
fi
OPD_TEACHER_MAPPING_GROUP="${OPD_TEACHER_MAPPING_GROUP:-teacher_sft_by_task}"
OPD_USE_TEACHER_MAPPING="${OPD_USE_TEACHER_MAPPING:-1}"
OPD_REQUIRE_MAPPED_TEACHER="${OPD_REQUIRE_MAPPED_TEACHER:-0}"

lookup_mapped_teacher_path() {
  local task_id="$1"
  if [[ ! -f "${OPD_TEACHER_MAPPING_JSON}" ]]; then
    return 0
  fi
  python3 - "${OPD_TEACHER_MAPPING_JSON}" "${task_id}" "${PROJECT_ROOT}" "${SCRATCH:-}" "${OPD_TEACHER_MAPPING_GROUP}" <<'PY'
import json
import sys
from pathlib import Path

mapping_path = sys.argv[1]
task_id = str(sys.argv[2])
project_root = sys.argv[3]
scratch_base = sys.argv[4]
mapping_group = sys.argv[5]
try:
    with open(mapping_path, "r", encoding="utf-8") as f:
        data = json.load(f)
except Exception:
    print("")
    raise SystemExit(0)

path = (
    data.get(mapping_group, {}).get(task_id)
    or data.get(task_id)
    or ""
)
if not path:
    print("")
    raise SystemExit(0)

p = Path(path)
if p.is_absolute():
    print(str(p))
    raise SystemExit(0)

# Relative mapping entries are resolved from SCRATCH repo mirror first, then PROJECT_ROOT.
if scratch_base:
    scratch_repo = Path(scratch_base) / "vla-continual-learning"
    scratch_candidate = (scratch_repo / p).resolve()
    if scratch_candidate.exists():
        print(str(scratch_candidate))
        raise SystemExit(0)

project_candidate = (Path(project_root) / p).resolve()
print(str(project_candidate))
PY
}

# ============== TRAIN SWEEP (run_embodiment_opd_sequential.sh → run_embodiment_sequential.sh) ==============
# Args: TASK_ID_OR_RANGE [CHECKPOINT_PATH] [MAX_EPOCH] [CONFIG_NAME] [SEED]
TRAIN_TASK_INPUTS=("1" "4")
TRAIN_MANUAL_CHECKPOINT=("")
# Passed as MAX_EPOCH (runner.max_epochs + checkpoint index). Empty = yaml max_epochs; post-train eval
# step still uses get_default_global_step for checkpoint folder unless you align EVAL_STEPS.
TRAIN_MAX_EPOCHS=(60)
# LiberoSFT / opd_bc_steps / teacher paths — tune in libero_spatial_opd_openvlaoft_spatial.yaml or Hydra.
TRAIN_CONFIG_NAMES=("crl_experiment/libero_spatial_opd_openvlaoft_spatial")
TRAIN_SEEDS=(184)
BASE_MODEL="${BASE_MODEL:-1}"
if [[ "${BASE_MODEL}" == "1" ]]; then
  TRAIN_MANUAL_CHECKPOINT=("base")
fi

# 256 -> lr 1e-04  batch size 32
# 255 -> lr 2e-05  batch size 256
# 254 -> lr 2e-06  batch size 32
# 253 -> mirrored data, lr 2e-05 batch size 32
# 252 -> mirrored data, lr 1e-04 batch size 32
# 250 -> single task, lr 2e-05 batch size 32
# 251 -> single task, lr 1e-04 batch size 32
# 230 -> single task, lr 2e-04 batch size 32
# 229 -> single task, lr 1e-03 batch size 32

# 199 -> look like the teacher was fixed
# 198 -> GRPO style loss for OPD, lr 1e-04 teacher
# 197 -> GRPO style loss for OPD, lr 2e-05 teacher
# 196 -> GRPO style loss for OPD, lr 2e-05 teacher 10 training steps
# 195 -> GRPO style loss for OPD, lr 1e-04 teacher 10 training steps
# 194 -> OPD after fixing, lr1e-04 teacher, 50 training steps student.
# 193 after restoring files
# 192 -> 1000 trainin steps teacher (lr 1e-04), 20 train step student
# 191 -> 1000 training steps teacher (lr 1e-04), 50 training steps student
# 190 -> same as 191 but with advantage normalization
# 189 -> REINFORCE-style OPD objective (no ratio clipping), lr 1e-04 teacher, 50 training steps student
# 188 -> same as 189 but with no advantage normalization
# 187 -> same as 188 but with RL teacher
# 186 -> same but with RL teacher and 128 rollouts per step
# 185 -> SFT teacher (1000 steps, lr 1e-04), 128 rollouts per step
# 184 -> same as 185 but with advantage normalization (trying both mad_abs and batch_zscore)

# 200 -> with fixed preprocessing, lr 2e-05 batch size 32

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
TRAIN_ROLLOUT_EPOCHS=(4)
DEFAULT_ROLLOUTS_PER_STEP=$((TRAIN_GROUP_SIZES[0] * TRAIN_NUM_GROUP_ENVS[0] * TRAIN_ROLLOUT_EPOCHS[0]))

# OPD teacher BC warmup (libero_spatial_opd_openvlaoft_spatial.yaml defaults). Expand any list to sweep.
TRAIN_OPD_BC_GLOBAL_BATCH_SIZES=(32)
TRAIN_OPD_BC_BATCH_SIZES=(8)
TRAIN_OPD_BC_STEPS=(1000)
TRAIN_OPD_BC_SAVE_STEPS="${TRAIN_OPD_BC_SAVE_STEPS:-}"
TRAIN_OPD_BC_LOG_INTERVAL_STEPS="${TRAIN_OPD_BC_LOG_INTERVAL_STEPS:-}"
TRAIN_OPD_FORCE_RETRAIN_TEACHER="${TRAIN_OPD_FORCE_RETRAIN_TEACHER:-0}"
TRAIN_OPD_TEACHER_LRS=('1e-04')
# SFT preprocessing toggles (0/1) for OPD BC dataset path.
TRAIN_OPD_SFT_FILTER_FIXED_TASK_IDS=(1)
TRAIN_OPD_SFT_MATCH_TASK_LANGUAGE=(1)
TRAIN_OPD_SFT_MATCH_IMAGE_ROTATION="${TRAIN_OPD_SFT_MATCH_IMAGE_ROTATION:-1}"
TRAIN_OPD_SFT_MATCH_OBS_ACTION_ALIGNMENT=(0)
TRAIN_OPD_SFT_GRIPPER_FROM_NEG1_0_TO_0_1="${TRAIN_OPD_SFT_GRIPPER_FROM_NEG1_0_TO_0_1:-0}"
TRAIN_OPD_NORMALIZE_ADVANTAGES=(1)
# Select one or more OPD normalization modes. Override with e.g.
#   TRAIN_OPD_REWARD_NORMALIZATIONS_OVERRIDE="group_zscore token_zscore action_dim_zscore positive_clip teacher_prob"
# For no reward normalization, also set TRAIN_OPD_NORMALIZE_ADVANTAGES_OVERRIDE=0 and use:
#   TRAIN_OPD_REWARD_NORMALIZATIONS_OVERRIDE="__empty__"
TRAIN_OPD_REWARD_NORMALIZATIONS=("group_zscore")
TRAIN_OPD_REWARD_TANH_TAU="${TRAIN_OPD_REWARD_TANH_TAU:-5.0}"
TRAIN_OPD_REWARD_CLIP_C="${TRAIN_OPD_REWARD_CLIP_C:-1.0}"
TRAIN_OPD_TEACHER_MICRO_BATCH_SIZES=(32)
TRAIN_OPD_PRECOMPUTE_TEACHER_IN_ROLLOUT="${TRAIN_OPD_PRECOMPUTE_TEACHER_IN_ROLLOUT:-1}"
TRAIN_OPD_TEACHER_STASH_LOGPROBS_ON_CPU="${TRAIN_OPD_TEACHER_STASH_LOGPROBS_ON_CPU:-0}"
TRAIN_OPD_RL_TEACHER="${TRAIN_OPD_RL_TEACHER:-0}"
# OPD actor loss type:
#   - embodied_opd                  : full GRPO-style clipped objective
#   - embodied_opd_reinforce        : plain REINFORCE-style objective (no ratio clipping)
#   - embodied_opd_success_gate     : success-gated GRPO/OPD hybrid; successful rollouts use
#                                     env/GRPO advantages, failed rollouts use teacher OPD
#   - embodied_opd_grpo_plus_success_gate: additive hybrid; normalized env/GRPO advantages are
#                                     always used, failed rollouts additionally get teacher OPD
TRAIN_OPD_LOSS_TYPES=("embodied_opd_reinforce")
TRAIN_OPD_SUCCESS_GATE_TEACHER_LAMBDAS=(1.0)
TRAIN_OPD_SUCCESS_GATE_REWARD_THRESHOLDS=(0.0)
TRAIN_OPD_SUCCESS_GATE_ENV_NORMALIZE_ADVANTAGES="${TRAIN_OPD_SUCCESS_GATE_ENV_NORMALIZE_ADVANTAGES:-}"
# OPD mode:
#   - opd_bc: original behavior (BC warmup then RL)
#   - rl_opd: skip BC warmup and use a provided HF teacher checkpoint.
TRAIN_OPD_MODE="${TRAIN_OPD_MODE:-opd_bc}"
TRAIN_OPD_TEACHER_HF_REPO="${TRAIN_OPD_TEACHER_HF_REPO:-Haozhan72/Openvla-oft-SFT-libero-spatial-trajall}"

apply_array_override() {
  local array_name="$1"
  local env_name="$2"
  if [[ -z "${!env_name+x}" ]]; then
    return 0
  fi

  local raw="${!env_name}"
  if [[ "${raw}" == "__empty__" ]]; then
    eval "${array_name}=(\"\")"
    return 0
  fi

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
apply_array_override TRAIN_OPD_BC_GLOBAL_BATCH_SIZES TRAIN_OPD_BC_GLOBAL_BATCH_SIZES_OVERRIDE
apply_array_override TRAIN_OPD_BC_BATCH_SIZES TRAIN_OPD_BC_BATCH_SIZES_OVERRIDE
apply_array_override TRAIN_OPD_BC_STEPS TRAIN_OPD_BC_STEPS_OVERRIDE
apply_array_override TRAIN_OPD_TEACHER_LRS TRAIN_OPD_TEACHER_LRS_OVERRIDE
apply_array_override TRAIN_OPD_SFT_FILTER_FIXED_TASK_IDS TRAIN_OPD_SFT_FILTER_FIXED_TASK_IDS_OVERRIDE
apply_array_override TRAIN_OPD_SFT_MATCH_TASK_LANGUAGE TRAIN_OPD_SFT_MATCH_TASK_LANGUAGE_OVERRIDE
apply_array_override TRAIN_OPD_SFT_MATCH_OBS_ACTION_ALIGNMENT TRAIN_OPD_SFT_MATCH_OBS_ACTION_ALIGNMENT_OVERRIDE
apply_array_override TRAIN_OPD_NORMALIZE_ADVANTAGES TRAIN_OPD_NORMALIZE_ADVANTAGES_OVERRIDE
apply_array_override TRAIN_OPD_REWARD_NORMALIZATIONS TRAIN_OPD_REWARD_NORMALIZATIONS_OVERRIDE
apply_array_override TRAIN_OPD_TEACHER_MICRO_BATCH_SIZES TRAIN_OPD_TEACHER_MICRO_BATCH_SIZES_OVERRIDE
apply_array_override TRAIN_OPD_LOSS_TYPES TRAIN_OPD_LOSS_TYPES_OVERRIDE
apply_array_override TRAIN_OPD_SUCCESS_GATE_TEACHER_LAMBDAS TRAIN_OPD_SUCCESS_GATE_TEACHER_LAMBDAS_OVERRIDE
apply_array_override TRAIN_OPD_SUCCESS_GATE_REWARD_THRESHOLDS TRAIN_OPD_SUCCESS_GATE_REWARD_THRESHOLDS_OVERRIDE

append_wandb_extra_tag() {
  local prefix="$1"
  local extra="${SWEEP_WANDB_EXTRA_TAG:-}"
  if [[ -n "${extra}" ]]; then
    extra="${extra//[^a-zA-Z0-9._-]/_}"
    prefix="${prefix}${extra}_"
  fi
  echo "${prefix}"
}

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
    if [[ "${USE_MINIMAL_SBATCH_RESOURCES}" != "1" ]]; then
      echo "#SBATCH --cpus-per-task=${CPUS_PER_TASK}"
      echo "#SBATCH --mem-per-cpu=${MEM_PER_CPU}"
      echo "#SBATCH --gpus=${GPU}"
    fi
    echo "#SBATCH --output=${SLURM_LOG_DIR}/${RUN_MODE}_%j.out"
    echo "#SBATCH --error=${SLURM_LOG_DIR}/${RUN_MODE}_%j.err"
    if [[ -n "${SLURM_PARTITION}" ]]; then
      echo "#SBATCH --partition=${SLURM_PARTITION}"
    fi
    if [[ "${USE_MINIMAL_SBATCH_RESOURCES}" == "1" ]]; then
      echo "#SBATCH --account=${NEW_CLUSTER_ACCOUNT}"
    elif [[ -n "${SLURM_ACCOUNT}" ]]; then
      echo "#SBATCH --account=${SLURM_ACCOUNT}"
    fi
    echo "set -euo pipefail"
    echo "if [[ \"\${RLINF_ALLOW_CORE_DUMPS:-0}\" != \"1\" ]]; then ulimit -c 0 2>/dev/null || true; fi"
    echo "cd \"${PROJECT_ROOT}\""
    echo "source \"${VENV_PATH}/bin/activate\""
    echo "export PYTORCH_CUDA_ALLOC_CONF=$(printf '%q' "${PYTORCH_CUDA_ALLOC_CONF}")"
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

# Emit SWEEP_OPD_* exports for the sbatch wrapper (safe quoting for scientific lr strings).
build_opd_sweep_exports() {
  # shellcheck disable=SC2312
  printf 'SWEEP_OPD_BC_GLOBAL_BATCH_SIZE=%q SWEEP_OPD_BC_BATCH_SIZE=%q SWEEP_OPD_BC_STEPS=%q SWEEP_OPD_BC_SAVE_STEPS=%q SWEEP_OPD_BC_LOG_INTERVAL_STEPS=%q SWEEP_OPD_FORCE_RETRAIN_TEACHER=%q SWEEP_OPD_TEACHER_LR=%q SWEEP_OPD_SFT_FILTER_FIXED_TASK_IDS=%q SWEEP_OPD_SFT_MATCH_TASK_LANGUAGE=%q SWEEP_OPD_SFT_MATCH_IMAGE_ROTATION=%q SWEEP_OPD_SFT_MATCH_OBS_ACTION_ALIGNMENT=%q SWEEP_OPD_SFT_GRIPPER_FROM_NEG1_0_TO_0_1=%q SWEEP_OPD_NORMALIZE_ADVANTAGES=%q SWEEP_OPD_REWARD_NORMALIZATION=%q SWEEP_OPD_REWARD_TANH_TAU=%q SWEEP_OPD_REWARD_CLIP_C=%q SWEEP_OPD_SUCCESS_GATE_TEACHER_LAMBDA=%q SWEEP_OPD_SUCCESS_GATE_REWARD_THRESHOLD=%q SWEEP_OPD_SUCCESS_GATE_ENV_NORMALIZE_ADVANTAGES=%q SWEEP_OPD_TEACHER_MICRO_BATCH_SIZE=%q SWEEP_OPD_PRECOMPUTE_TEACHER_IN_ROLLOUT=%q SWEEP_OPD_TEACHER_STASH_LOGPROBS_ON_CPU=%q SWEEP_OPD_RL_TEACHER=%q SWEEP_OPD_MODE=%q SWEEP_OPD_TEACHER_HF_REPO=%q SWEEP_OPD_LOSS_TYPE=%q' \
    "$1" "$2" "$3" "${TRAIN_OPD_BC_SAVE_STEPS}" "${TRAIN_OPD_BC_LOG_INTERVAL_STEPS}" "${TRAIN_OPD_FORCE_RETRAIN_TEACHER}" "$4" "$5" "$6" "${TRAIN_OPD_SFT_MATCH_IMAGE_ROTATION}" "$7" "${TRAIN_OPD_SFT_GRIPPER_FROM_NEG1_0_TO_0_1}" "$8" "$9" "${10}" "${11}" "${12}" "${13}" "${14}" "${15}" "${16}" "${17}" "${18}" "${19}" "${20}" "${21}"
}

opd_variant_tag() {
  local norm_adv="$1"
  local reward_norm="$2"
  local loss_type="$3"
  local tag=""

  if [[ "${norm_adv}" == "0" ]]; then
    tag="${tag}_nonorm"
  else
    if [[ -n "${reward_norm}" ]]; then
      local safe_norm="${reward_norm//[^a-zA-Z0-9._-]/_}"
      tag="${tag}_${safe_norm}"
    else
      tag="${tag}_norm_empty"
    fi
  fi

  case "${loss_type}" in
    embodied_opd)
      tag="${tag}_grpo_loss"
      ;;
    embodied_opd_success_gate)
      tag="${tag}_success_gate"
      ;;
    embodied_opd_grpo_plus_success_gate)
      tag="${tag}_grpo_plus_success_gate"
      ;;
    embodied_opd_reinforce)
      ;;
    *)
      local safe_loss="${loss_type//[^a-zA-Z0-9._-]/_}"
      tag="${tag}_${safe_loss}"
      ;;
  esac

  printf '%s' "${tag}"
}

echo "OPD embodied SLURM sweep — RUN_MODE=${RUN_MODE}"
echo "BASE_MODEL=${BASE_MODEL} (1 = force first task in each job to use base SFT checkpoint)"
echo "GRPO_HP_FROM_SWEEP=${GRPO_HP_FROM_SWEEP} (1 = override group_size, num_group_envs, rollout_epoch, global_batch_size)"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "SLURM_LOG_DIR=${SLURM_LOG_DIR}"
echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
echo "OPD_TEACHER_MAPPING_JSON=${OPD_TEACHER_MAPPING_JSON}"
echo "OPD_TEACHER_MAPPING_GROUP=${OPD_TEACHER_MAPPING_GROUP}"
echo "OPD_USE_TEACHER_MAPPING=${OPD_USE_TEACHER_MAPPING}"
echo "OPD_REQUIRE_MAPPED_TEACHER=${OPD_REQUIRE_MAPPED_TEACHER}"
echo "TRAIN_OPD_FORCE_RETRAIN_TEACHER=${TRAIN_OPD_FORCE_RETRAIN_TEACHER}"
echo "TRAIN_OPD_SFT_MATCH_IMAGE_ROTATION=${TRAIN_OPD_SFT_MATCH_IMAGE_ROTATION} (default 1 = rotate SFT images to match rollout preprocessing)"
echo "TRAIN_OPD_SFT_GRIPPER_FROM_NEG1_0_TO_0_1=${TRAIN_OPD_SFT_GRIPPER_FROM_NEG1_0_TO_0_1}"
if [[ -n "${LIBERO_REPO_PATH}" ]]; then
  echo "LIBERO_REPO_PATH (in jobs)=${LIBERO_REPO_PATH}"
else
  echo "LIBERO_REPO_PATH: (unset — jobs use default in run_embodiment.sh: \${REPO_PATH}/LIBERO)"
fi
if [[ "${USE_MINIMAL_SBATCH_RESOURCES}" == "1" ]]; then
  echo "Resources (minimal sbatch mode): TIME=${TIME} — no #SBATCH cpus/mem/gpus; account=${NEW_CLUSTER_ACCOUNT}"
else
  echo "Resources: TIME=${TIME} CPUS_PER_TASK=${CPUS_PER_TASK} MEM_PER_CPU=${MEM_PER_CPU} GPU=${GPU}"
fi
echo "=================================="

job_count=0

if [[ "${RUN_MODE}" == "train" ]]; then
  for TASK in "${TRAIN_TASK_INPUTS[@]}"; do
    TASK_MAPPED_TEACHER_PATH=""
    if [[ "${OPD_USE_TEACHER_MAPPING}" == "1" ]]; then
      TASK_MAPPED_TEACHER_PATH="$(lookup_mapped_teacher_path "${TASK}")"
    fi
    TASK_MAPPED_TEACHER_EX=""
    if [[ -n "${TASK_MAPPED_TEACHER_PATH}" ]]; then
      if [[ -d "${TASK_MAPPED_TEACHER_PATH}" ]]; then
        TASK_MAPPED_TEACHER_EX="SWEEP_OPD_TEACHER_MODEL_PATH=$(printf '%q' "${TASK_MAPPED_TEACHER_PATH}")"
        echo "Task ${TASK}: mapped SFT teacher -> ${TASK_MAPPED_TEACHER_PATH}"
      else
        echo "WARN: Task ${TASK} has mapped teacher path but directory is missing: ${TASK_MAPPED_TEACHER_PATH}"
        if [[ "${OPD_REQUIRE_MAPPED_TEACHER}" == "1" ]]; then
          echo "ERROR: OPD_REQUIRE_MAPPED_TEACHER=1, refusing to submit OPD without this teacher."
          exit 1
        fi
      fi
    else
      echo "Task ${TASK}: no mapped SFT teacher found; using existing default OPD flow."
      if [[ "${OPD_REQUIRE_MAPPED_TEACHER}" == "1" ]]; then
        echo "ERROR: OPD_REQUIRE_MAPPED_TEACHER=1, refusing to submit OPD without a mapped teacher."
        echo "       Check OPD_TEACHER_MAPPING_JSON=${OPD_TEACHER_MAPPING_JSON}"
        echo "       Check OPD_TEACHER_MAPPING_GROUP=${OPD_TEACHER_MAPPING_GROUP} and task=${TASK}"
        exit 1
      fi
    fi
    for CKPT in "${TRAIN_MANUAL_CHECKPOINT[@]}"; do
      for MAX_EP in "${TRAIN_MAX_EPOCHS[@]}"; do
        for CFG in "${TRAIN_CONFIG_NAMES[@]}"; do
          if [[ "${GRPO_HP_FROM_SWEEP}" == "1" ]]; then
            for GS in "${TRAIN_GROUP_SIZES[@]}"; do
              for NGE in "${TRAIN_NUM_GROUP_ENVS[@]}"; do
                for RE in "${TRAIN_ROLLOUT_EPOCHS[@]}"; do
                  G_BATCH=$((GS * NGE * RE * 64))
                  ROLLOUTS_PER_STEP=$((G_BATCH / 64))
                  for OPD_GBS in "${TRAIN_OPD_BC_GLOBAL_BATCH_SIZES[@]}"; do
                    for OPD_MBS in "${TRAIN_OPD_BC_BATCH_SIZES[@]}"; do
                      for OPD_STEPS in "${TRAIN_OPD_BC_STEPS[@]}"; do
                        for OPD_TLR in "${TRAIN_OPD_TEACHER_LRS[@]}"; do
                          for OPD_SFT_FILTER in "${TRAIN_OPD_SFT_FILTER_FIXED_TASK_IDS[@]}"; do
                            for OPD_SFT_LANG in "${TRAIN_OPD_SFT_MATCH_TASK_LANGUAGE[@]}"; do
                              for OPD_SFT_ALIGN in "${TRAIN_OPD_SFT_MATCH_OBS_ACTION_ALIGNMENT[@]}"; do
                                for OPD_NORM_ADV in "${TRAIN_OPD_NORMALIZE_ADVANTAGES[@]}"; do
                                  for OPD_REWARD_NORM in "${TRAIN_OPD_REWARD_NORMALIZATIONS[@]}"; do
                                    for OPD_TMB in "${TRAIN_OPD_TEACHER_MICRO_BATCH_SIZES[@]}"; do
                                      for OPD_LOSS in "${TRAIN_OPD_LOSS_TYPES[@]}"; do
                                        for OPD_SG_LAMBDA in "${TRAIN_OPD_SUCCESS_GATE_TEACHER_LAMBDAS[@]}"; do
                                          for OPD_SG_THRESHOLD in "${TRAIN_OPD_SUCCESS_GATE_REWARD_THRESHOLDS[@]}"; do
                                            for SEED in "${TRAIN_SEEDS[@]}"; do
                            if [[ "${OPD_TEACHER_MAPPING_GROUP}" == "teacher_rl_by_task" ]]; then
                              TEACHER_TAG="rlteacher"
                            else
                              TEACHER_TAG="sftteacher"
                            fi
                            VARIANT_TAG="$(opd_variant_tag "${OPD_NORM_ADV}" "${OPD_REWARD_NORM}" "${OPD_LOSS}")"
                            if [[ "${OPD_LOSS}" == "embodied_opd_success_gate" || "${OPD_LOSS}" == "embodied_opd_grpo_plus_success_gate" ]]; then
                              SAFE_SG_LAMBDA="${OPD_SG_LAMBDA//./p}"
                              SAFE_SG_THRESHOLD="${OPD_SG_THRESHOLD//./p}"
                              VARIANT_TAG="${VARIANT_TAG}_lam${SAFE_SG_LAMBDA}_thr${SAFE_SG_THRESHOLD}"
                            fi
                            JOB_NAME="opd_${TEACHER_TAG}_adv${OPD_NORM_ADV}${VARIANT_TAG}_rps${ROLLOUTS_PER_STEP}_t${TASK}_s${SEED}"
                            JOB_NAME="${JOB_NAME//[^a-zA-Z0-9._-]/_}"
                            if ((${#JOB_NAME} > 40)); then
                              JOB_NAME="${JOB_NAME:0:40}"
                            fi

                            ARGS=(bash examples/crl_experiment/run_embodiment_opd_sequential.sh "${TASK}")
                            [[ -n "${CKPT}" ]] && ARGS+=("${CKPT}") || ARGS+=("")
                            [[ -n "${MAX_EP}" ]] && ARGS+=("${MAX_EP}") || ARGS+=("")
                            ARGS+=("${CFG}" "${SEED}")
                            OPD_EX="$(build_opd_sweep_exports "${OPD_GBS}" "${OPD_MBS}" "${OPD_STEPS}" "${OPD_TLR}" "${OPD_SFT_FILTER}" "${OPD_SFT_LANG}" "${OPD_SFT_ALIGN}" "${OPD_NORM_ADV}" "${OPD_REWARD_NORM}" "${TRAIN_OPD_REWARD_TANH_TAU}" "${TRAIN_OPD_REWARD_CLIP_C}" "${OPD_SG_LAMBDA}" "${OPD_SG_THRESHOLD}" "${TRAIN_OPD_SUCCESS_GATE_ENV_NORMALIZE_ADVANTAGES}" "${OPD_TMB}" "${TRAIN_OPD_PRECOMPUTE_TEACHER_IN_ROLLOUT}" "${TRAIN_OPD_TEACHER_STASH_LOGPROBS_ON_CPU}" "${TRAIN_OPD_RL_TEACHER}" "${TRAIN_OPD_MODE}" "${TRAIN_OPD_TEACHER_HF_REPO}" "${OPD_LOSS}")"
                            SAVE_INTERVAL_OVERRIDE="${SWEEP_SAVE_INTERVAL:-20}"
                            WANDB_PREFIX="opd_${TEACHER_TAG}_adv${OPD_NORM_ADV}${VARIANT_TAG}_rps${ROLLOUTS_PER_STEP}_"
                            WANDB_PREFIX="$(append_wandb_extra_tag "${WANDB_PREFIX}")"
                            CMD="EXPERIMENT_NAME_PREFIX=${WANDB_PREFIX} SKIP_POST_TRAIN_EVAL=1 ${TASK_MAPPED_TEACHER_EX} ${OPD_EX} SWEEP_GROUP_SIZE=${GS} SWEEP_NUM_GROUP_ENVS=${NGE} SWEEP_ROLLOUT_EPOCH=${RE} SWEEP_GLOBAL_BATCH_SIZE=${G_BATCH} SWEEP_SAVE_INTERVAL=${SAVE_INTERVAL_OVERRIDE} $(printf '%q ' "${ARGS[@]}")"
                            echo "Submit OPD train: task=${TASK} seed=${SEED} cfg=${CFG} max_epoch=${MAX_EP:-default} ckpt=${CKPT:-none} group_size=${GS} num_group_envs=${NGE} rollout_epoch=${RE} global_batch_size=${G_BATCH} rollouts_per_step=${ROLLOUTS_PER_STEP} opd_mode=${TRAIN_OPD_MODE} opd_teacher_repo=${TRAIN_OPD_TEACHER_HF_REPO} opd_teacher_model_path=${TASK_MAPPED_TEACHER_PATH:-auto} opd_loss=${OPD_LOSS} opd_norm_adv=${OPD_NORM_ADV} opd_reward_norm=${OPD_REWARD_NORM} opd_reward_tanh_tau=${TRAIN_OPD_REWARD_TANH_TAU} opd_reward_clip_c=${TRAIN_OPD_REWARD_CLIP_C} opd_success_gate_lambda=${OPD_SG_LAMBDA} opd_success_gate_threshold=${OPD_SG_THRESHOLD} opd_success_gate_env_norm=${TRAIN_OPD_SUCCESS_GATE_ENV_NORMALIZE_ADVANTAGES:-default} opd_teacher_micro_batch=${OPD_TMB} opd_precompute_teacher_in_rollout=${TRAIN_OPD_PRECOMPUTE_TEACHER_IN_ROLLOUT} opd_teacher_stash_logprobs_on_cpu=${TRAIN_OPD_TEACHER_STASH_LOGPROBS_ON_CPU} opd_bc_gbs=${OPD_GBS} opd_bc_bs=${OPD_MBS} opd_bc_steps=${OPD_STEPS} opd_teacher_lr=${OPD_TLR} sft_filter=${OPD_SFT_FILTER} sft_lang=${OPD_SFT_LANG} sft_rot=${TRAIN_OPD_SFT_MATCH_IMAGE_ROTATION} sft_align=${OPD_SFT_ALIGN} sft_grip01=${TRAIN_OPD_SFT_GRIPPER_FROM_NEG1_0_TO_0_1}"

                            submit_job "${JOB_NAME}" "${CMD}"
                            job_count=$((job_count + 1))
                                            done
                                          done
                                        done
                                      done
                                done
                                done
                              done
                              done
                            done
                          done
                        done
                      done
                    done
                  done
                done
              done
            done
          else
            for OPD_GBS in "${TRAIN_OPD_BC_GLOBAL_BATCH_SIZES[@]}"; do
              for OPD_MBS in "${TRAIN_OPD_BC_BATCH_SIZES[@]}"; do
                for OPD_STEPS in "${TRAIN_OPD_BC_STEPS[@]}"; do
                  for OPD_TLR in "${TRAIN_OPD_TEACHER_LRS[@]}"; do
                    for OPD_SFT_FILTER in "${TRAIN_OPD_SFT_FILTER_FIXED_TASK_IDS[@]}"; do
                      for OPD_SFT_LANG in "${TRAIN_OPD_SFT_MATCH_TASK_LANGUAGE[@]}"; do
                        for OPD_SFT_ALIGN in "${TRAIN_OPD_SFT_MATCH_OBS_ACTION_ALIGNMENT[@]}"; do
                          for OPD_NORM_ADV in "${TRAIN_OPD_NORMALIZE_ADVANTAGES[@]}"; do
                            for OPD_REWARD_NORM in "${TRAIN_OPD_REWARD_NORMALIZATIONS[@]}"; do
                              for OPD_TMB in "${TRAIN_OPD_TEACHER_MICRO_BATCH_SIZES[@]}"; do
                                for OPD_LOSS in "${TRAIN_OPD_LOSS_TYPES[@]}"; do
                                  for OPD_SG_LAMBDA in "${TRAIN_OPD_SUCCESS_GATE_TEACHER_LAMBDAS[@]}"; do
                                    for OPD_SG_THRESHOLD in "${TRAIN_OPD_SUCCESS_GATE_REWARD_THRESHOLDS[@]}"; do
                                      for SEED in "${TRAIN_SEEDS[@]}"; do
                      if [[ "${OPD_TEACHER_MAPPING_GROUP}" == "teacher_rl_by_task" ]]; then
                        TEACHER_TAG="rlteacher"
                      else
                        TEACHER_TAG="sftteacher"
                      fi
                      VARIANT_TAG="$(opd_variant_tag "${OPD_NORM_ADV}" "${OPD_REWARD_NORM}" "${OPD_LOSS}")"
                      if [[ "${OPD_LOSS}" == "embodied_opd_success_gate" || "${OPD_LOSS}" == "embodied_opd_grpo_plus_success_gate" ]]; then
                        SAFE_SG_LAMBDA="${OPD_SG_LAMBDA//./p}"
                        SAFE_SG_THRESHOLD="${OPD_SG_THRESHOLD//./p}"
                        VARIANT_TAG="${VARIANT_TAG}_lam${SAFE_SG_LAMBDA}_thr${SAFE_SG_THRESHOLD}"
                      fi
                      JOB_NAME="opd_${TEACHER_TAG}_adv${OPD_NORM_ADV}${VARIANT_TAG}_rps${DEFAULT_ROLLOUTS_PER_STEP}_t${TASK}_s${SEED}"
                      JOB_NAME="${JOB_NAME//[^a-zA-Z0-9._-]/_}"
                      if ((${#JOB_NAME} > 40)); then
                        JOB_NAME="${JOB_NAME:0:40}"
                      fi

                      ARGS=(bash examples/crl_experiment/run_embodiment_opd_sequential.sh "${TASK}")
                      [[ -n "${CKPT}" ]] && ARGS+=("${CKPT}") || ARGS+=("")
                      [[ -n "${MAX_EP}" ]] && ARGS+=("${MAX_EP}") || ARGS+=("")
                      ARGS+=("${CFG}" "${SEED}")

                      OPD_EX="$(build_opd_sweep_exports "${OPD_GBS}" "${OPD_MBS}" "${OPD_STEPS}" "${OPD_TLR}" "${OPD_SFT_FILTER}" "${OPD_SFT_LANG}" "${OPD_SFT_ALIGN}" "${OPD_NORM_ADV}" "${OPD_REWARD_NORM}" "${TRAIN_OPD_REWARD_TANH_TAU}" "${TRAIN_OPD_REWARD_CLIP_C}" "${OPD_SG_LAMBDA}" "${OPD_SG_THRESHOLD}" "${TRAIN_OPD_SUCCESS_GATE_ENV_NORMALIZE_ADVANTAGES}" "${OPD_TMB}" "${TRAIN_OPD_PRECOMPUTE_TEACHER_IN_ROLLOUT}" "${TRAIN_OPD_TEACHER_STASH_LOGPROBS_ON_CPU}" "${TRAIN_OPD_RL_TEACHER}" "${TRAIN_OPD_MODE}" "${TRAIN_OPD_TEACHER_HF_REPO}" "${OPD_LOSS}")"
                      SAVE_INTERVAL_OVERRIDE="${MAX_EP}"
                      WANDB_PREFIX="opd_${TEACHER_TAG}_adv${OPD_NORM_ADV}${VARIANT_TAG}_rps${DEFAULT_ROLLOUTS_PER_STEP}_"
                      WANDB_PREFIX="$(append_wandb_extra_tag "${WANDB_PREFIX}")"
                      CMD="EXPERIMENT_NAME_PREFIX=${WANDB_PREFIX} SKIP_POST_TRAIN_EVAL=1 ${TASK_MAPPED_TEACHER_EX} ${OPD_EX} SWEEP_SAVE_INTERVAL=${SAVE_INTERVAL_OVERRIDE} $(printf '%q ' "${ARGS[@]}")"
                      echo "Submit OPD train: task=${TASK} seed=${SEED} cfg=${CFG} max_epoch=${MAX_EP:-default} ckpt=${CKPT:-none} rollouts_per_step=${DEFAULT_ROLLOUTS_PER_STEP} opd_mode=${TRAIN_OPD_MODE} opd_teacher_repo=${TRAIN_OPD_TEACHER_HF_REPO} opd_teacher_model_path=${TASK_MAPPED_TEACHER_PATH:-auto} opd_loss=${OPD_LOSS} opd_norm_adv=${OPD_NORM_ADV} opd_reward_norm=${OPD_REWARD_NORM} opd_reward_tanh_tau=${TRAIN_OPD_REWARD_TANH_TAU} opd_reward_clip_c=${TRAIN_OPD_REWARD_CLIP_C} opd_success_gate_lambda=${OPD_SG_LAMBDA} opd_success_gate_threshold=${OPD_SG_THRESHOLD} opd_success_gate_env_norm=${TRAIN_OPD_SUCCESS_GATE_ENV_NORMALIZE_ADVANTAGES:-default} opd_teacher_micro_batch=${OPD_TMB} opd_precompute_teacher_in_rollout=${TRAIN_OPD_PRECOMPUTE_TEACHER_IN_ROLLOUT} opd_teacher_stash_logprobs_on_cpu=${TRAIN_OPD_TEACHER_STASH_LOGPROBS_ON_CPU} opd_bc_gbs=${OPD_GBS} opd_bc_bs=${OPD_MBS} opd_bc_steps=${OPD_STEPS} opd_teacher_lr=${OPD_TLR} sft_filter=${OPD_SFT_FILTER} sft_lang=${OPD_SFT_LANG} sft_rot=${TRAIN_OPD_SFT_MATCH_IMAGE_ROTATION} sft_align=${OPD_SFT_ALIGN} sft_grip01=${TRAIN_OPD_SFT_GRIPPER_FROM_NEG1_0_TO_0_1}"

                      submit_job "${JOB_NAME}" "${CMD}"
                      job_count=$((job_count + 1))
                                      done
                                    done
                                  done
                                done
                            done
                          done
                          done
                        done
                      done
                    done
                  done
                done
              done
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
