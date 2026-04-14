#! /bin/bash
#
# Setup script for running embodied agent evaluation
#
# Use pipefail + PIPESTATUS so Ctrl+C / SIGQUIT on python is not masked by tee exiting 0.
set -o pipefail
#
# Environment Variables (optional overrides):
#   - LIBERO_REPO_PATH: Path to LIBERO repository (defaults to ${REPO_PATH}/LIBERO)
#   - RLINF_LIBERO_USE_OSMESA: Set to 1 for OSMesa (slow); else EGL + venv.py Ray remap (see run_embodiment.sh).
#   - RLINF_LIBERO_EGL_REMAP: Set to 0 to disable EGL remapping (see run_embodiment.sh).
#   - RLINF_CUDA_LAUNCH_BLOCKING: Set to 1 for synchronous CUDA (debug only; much slower).
#
# Note: REPO_PATH is automatically set to the parent directory of examples/
#       If you need to override it, set it before running this script.
#
# Avoid filling $PWD with HPC core dumps (e.g. core_nid*). Enable with RLINF_ALLOW_CORE_DUMPS=1.
if [ "${RLINF_ALLOW_CORE_DUMPS:-0}" != "1" ]; then
    ulimit -c 0 2>/dev/null || true
fi

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/eval_embodied_agent.py"

if [ "${RLINF_LIBERO_USE_OSMESA:-0}" = "1" ] || [ "${RLINF_LIBERO_USE_OSMESA:-}" = "true" ]; then
    export MUJOCO_GL="osmesa"
    export PYOPENGL_PLATFORM="osmesa"
    unset MUJOCO_EGL_DEVICE_ID
else
    export MUJOCO_GL="egl"
    export PYOPENGL_PLATFORM="egl"
fi

export PYTHONPATH=${REPO_PATH}:$PYTHONPATH
# NOTE: set LIBERO_REPO_PATH to the path of the LIBERO repo
# Defaults to ${REPO_PATH}/LIBERO if not set
export LIBERO_REPO_PATH="${LIBERO_REPO_PATH:-${REPO_PATH}/LIBERO}"
# NOTE: set LIBERO_CONFIG_PATH for libero/libero/__init__.py
export LIBERO_CONFIG_PATH=${LIBERO_REPO_PATH}

export PYTHONPATH=${LIBERO_REPO_PATH}:$PYTHONPATH
if [ "${RLINF_CUDA_LAUNCH_BLOCKING:-0}" = "1" ] || [ "${RLINF_CUDA_LAUNCH_BLOCKING:-}" = "true" ]; then
    export CUDA_LAUNCH_BLOCKING=1
fi
export HYDRA_FULL_ERROR=1
# Line-buffered python stdout/stderr so logs show progress before ray.init returns (tee-safe).
export PYTHONUNBUFFERED=1
# Some deps load TensorFlow for ancillary code; reduce stderr spam (training path often sets this too).
export TF_CPP_MIN_LOG_LEVEL=3

# Match examples/embodiment/run_embodiment.sh so eval sees the same Ray driver environment as training
# (sequential scripts run training then eval; missing these can leave eval stuck inside ray.init()).
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_DISABLE_DASHBOARD=1
export RLINF_RAY_INCLUDE_DASHBOARD=0
# Plasma/raylet sockets: Linux AF_UNIX paths must stay <=107 bytes; long paths under $HOME can break or hang ray.init.
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_${USER}}"
mkdir -p "$RAY_TMPDIR"
export PYTORCH_DISTRIBUTED_BACKEND=nccl
if [ -z "${RLINF_RAY_NUM_CPUS:-}" ] && [ -n "${SLURM_CPUS_PER_TASK:-}" ]; then
    export RLINF_RAY_NUM_CPUS="${SLURM_CPUS_PER_TASK}"
fi
if [ -z "${RLINF_RAY_NUM_GPUS:-}" ] && [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
    export RLINF_RAY_NUM_GPUS="${SLURM_GPUS_ON_NODE}"
fi

# Optional: if ray.init hangs on HPC, set RLINF_RAY_SKIP_RUNTIME_ENV=1, RLINF_RAY_MINIMAL_INIT=1,
# RLINF_RAY_INCLUDE_DASHBOARD=0, or RLINF_RAY_LOCAL_ONLY=1 (see rlinf/utils/embodied_ray_env.py).

if [ -z "$1" ]; then
    CONFIG_NAME="maniskill_ppo_openvlaoft"
    CONFIG_PATH="${EMBODIED_PATH}/config/"
else
    # Check if config name contains a path (subdirectory)
    if [[ "$1" == *"/"* ]]; then
        # Extract directory and config name
        CONFIG_DIR=$(dirname "$1")
        CONFIG_NAME=$(basename "$1")
        CONFIG_PATH="${EMBODIED_PATH}/config/${CONFIG_DIR}/"
        shift
    else
        # No subdirectory, use root config directory
        CONFIG_NAME=$1
        CONFIG_PATH="${EMBODIED_PATH}/config/"
        shift
    fi
fi

# Get remaining arguments (Hydra overrides)
HYDRA_OVERRIDES="$@"

# If we pass actor.model.lora_path=..., store eval logs under:
#   ${REPO_PATH}/logs/evals/<three path components before "checkpoints">/
# Example:
#   .../RLinf/logs/base_to_task0_LoRA_grpo/checkpoints/... -> logs/evals/RLinf/logs/base_to_task0_LoRA_grpo
#
# Otherwise, store under:
#   ${REPO_PATH}/logs/evals/base
#
# If actor.model.lora_scale is provided, append it to the log subdirectory:
#   logs/evals/.../lora_scale_0_5/...
LORA_PATH=""
if [[ " ${HYDRA_OVERRIDES} " =~ actor\.model\.lora_path=([^[:space:]]+) ]]; then
    LORA_PATH="${BASH_REMATCH[1]}"
    # Strip optional single/double quotes around the value
    LORA_PATH="${LORA_PATH%\"}"; LORA_PATH="${LORA_PATH#\"}"
    LORA_PATH="${LORA_PATH%\'}"; LORA_PATH="${LORA_PATH#\'}"
fi

# Extract lora_scale from Hydra overrides
LORA_SCALE=""
if [[ " ${HYDRA_OVERRIDES} " =~ actor\.model\.lora_scale=([^[:space:]]+) ]]; then
    LORA_SCALE="${BASH_REMATCH[1]}"
    # Strip optional single/double quotes around the value
    LORA_SCALE="${LORA_SCALE%\"}"; LORA_SCALE="${LORA_SCALE#\"}"
    LORA_SCALE="${LORA_SCALE%\'}"; LORA_SCALE="${LORA_SCALE#\'}"
fi

# Extract previous_lora_merge_coefficient from Hydra overrides (for multi-LoRA)
PREVIOUS_LORA_COEFF=""
if [[ " ${HYDRA_OVERRIDES} " =~ actor\.model\.previous_lora_merge_coefficient=([^[:space:]]+) ]]; then
    PREVIOUS_LORA_COEFF="${BASH_REMATCH[1]}"
    # Strip optional single/double quotes around the value
    PREVIOUS_LORA_COEFF="${PREVIOUS_LORA_COEFF%\"}"; PREVIOUS_LORA_COEFF="${PREVIOUS_LORA_COEFF#\"}"
    PREVIOUS_LORA_COEFF="${PREVIOUS_LORA_COEFF%\'}"; PREVIOUS_LORA_COEFF="${PREVIOUS_LORA_COEFF#\'}"
fi

# Extract temperature_eval from Hydra overrides
TEMPERATURE_EVAL=""
if [[ " ${HYDRA_OVERRIDES} " =~ algorithm\.sampling_params\.temperature_eval=([^[:space:]]+) ]]; then
    TEMPERATURE_EVAL="${BASH_REMATCH[1]}"
    # Strip optional single/double quotes around the value
    TEMPERATURE_EVAL="${TEMPERATURE_EVAL%\"}"; TEMPERATURE_EVAL="${TEMPERATURE_EVAL#\"}"
    TEMPERATURE_EVAL="${TEMPERATURE_EVAL%\'}"; TEMPERATURE_EVAL="${TEMPERATURE_EVAL#\'}"
fi

# Check if this is multi-LoRA (has lora_paths)
IS_MULTILORA=false
if [[ " ${HYDRA_OVERRIDES} " =~ actor\.model\.lora_paths= ]]; then
    IS_MULTILORA=true
fi

LOG_SUBDIR="base"
if [ -n "${LORA_PATH}" ]; then
    LORA_PATH="${LORA_PATH%/}" # normalize trailing slash
    # Take everything before the first "/checkpoints" occurrence
    BEFORE_CHECKPOINTS="${LORA_PATH%%/checkpoints*}"
    if [ -n "${BEFORE_CHECKPOINTS}" ] && [ "${BEFORE_CHECKPOINTS}" != "${LORA_PATH}" ]; then
        # Use the last 3 path components from BEFORE_CHECKPOINTS (e.g., RLinf/logs/exp_name)
        TMP="${BEFORE_CHECKPOINTS#/}"
        IFS='/' read -r -a PARTS <<< "${TMP}"
        N=${#PARTS[@]}
        if [ "${N}" -ge 3 ]; then
            LOG_SUBDIR="${PARTS[N-3]}/${PARTS[N-2]}/${PARTS[N-1]}"
        elif [ "${N}" -ge 1 ]; then
            LOG_SUBDIR="${PARTS[N-1]}"
        fi
    fi
fi

# Append coefficients to LOG_SUBDIR for multi-LoRA
if [ "$IS_MULTILORA" = true ]; then
    if [ -n "${PREVIOUS_LORA_COEFF}" ]; then
        PREV_COEFF_PATH=$(echo "$PREVIOUS_LORA_COEFF" | tr '.' '_')
        LOG_SUBDIR="${LOG_SUBDIR}_prev_coeff_${PREV_COEFF_PATH}"
    fi
    if [ -n "${LORA_SCALE}" ] && [ "${LORA_SCALE}" != "1.0" ]; then
        LORA_SCALE_PATH=$(echo "$LORA_SCALE" | tr '.' '_')
        LOG_SUBDIR="${LOG_SUBDIR}_curr_scale_${LORA_SCALE_PATH}"
    fi
elif [ -n "${LORA_SCALE}" ]; then
    # Single LoRA: append lora_scale to LOG_SUBDIR if provided
    # Format lora_scale for use in path (replace dot with underscore, e.g., 0.5 -> 0_5)
    LORA_SCALE_PATH=$(echo "$LORA_SCALE" | tr '.' '_')
    LOG_SUBDIR="${LOG_SUBDIR}_lora_scale/lora_scale_${LORA_SCALE_PATH}"
fi

# Append temperature_eval to LOG_SUBDIR if provided
if [ -n "${TEMPERATURE_EVAL}" ]; then
    # Format temperature_eval for use in path (replace dot with underscore, e.g., 2.0 -> 2_0)
    TEMPERATURE_PATH=$(echo "$TEMPERATURE_EVAL" | tr '.' '_')
    LOG_SUBDIR="${LOG_SUBDIR}_temp_${TEMPERATURE_PATH}"
fi

# Extract config tag from CONFIG_NAME
# If config ends with _openvlaoft or _eval, don't set CONFIG_TAG
# Otherwise, extract the part after the last _
CONFIG_TAG=""
if [[ ! "${CONFIG_NAME}" =~ _openvlaoft$ ]] && [[ ! "${CONFIG_NAME}" =~ _eval$ ]]; then
    if [[ "${CONFIG_NAME}" =~ _([^_/]+)$ ]]; then
        CONFIG_TAG="${BASH_REMATCH[1]}"
    fi
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Extract step number from lora_path if available, or use environment variable
STEP_NUMBER=""
if [[ "${LORA_PATH}" =~ global_step_([0-9]+) ]]; then
    STEP_NUMBER="${BASH_REMATCH[1]}"
elif [ -n "${EVAL_STEP_NUMBER}" ]; then
    STEP_NUMBER="${EVAL_STEP_NUMBER}"
fi

# Build log directory path with config tag if present
if [ -n "${CONFIG_TAG}" ]; then
    LOGS_BASE="${REPO_PATH}/logs_${CONFIG_TAG}/evals"
else
    LOGS_BASE="${REPO_PATH}/logs/evals"
fi

# Include step number in log directory name if available
if [ -n "${STEP_NUMBER}" ]; then
    LOG_DIR="${LOGS_BASE}/${LOG_SUBDIR}_step_${STEP_NUMBER}_${TIMESTAMP}"
else
    LOG_DIR="${LOGS_BASE}/${LOG_SUBDIR}_${TIMESTAMP}"
fi

MEGA_LOG_FILE="${LOG_DIR}/eval_embodiment.log"
mkdir -p "${LOG_DIR}"
CMD="python -u ${SRC_FILE} --config-path ${CONFIG_PATH} --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR} ${HYDRA_OVERRIDES}"
echo ${CMD}
${CMD} 2>&1 | tee ${MEGA_LOG_FILE}
# Exit with python's status, not tee's (otherwise SIGQUIT/kill still looks like success).
PYTHON_EXIT=${PIPESTATUS[0]}
exit "${PYTHON_EXIT}"
