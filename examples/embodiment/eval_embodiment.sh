#! /bin/bash
#
# Setup script for running embodied agent evaluation
#
# Environment Variables (optional overrides):
#   - LIBERO_REPO_PATH: Path to LIBERO repository (defaults to ${REPO_PATH}/LIBERO)
#
# Note: REPO_PATH is automatically set to the parent directory of examples/
#       If you need to override it, set it before running this script.

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/eval_embodied_agent.py"

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
# export MUJOCO_GL="osmesa"
# export PYOPENGL_PLATFORM="osmesa"

export PYTHONPATH=${REPO_PATH}:$PYTHONPATH
# NOTE: set LIBERO_REPO_PATH to the path of the LIBERO repo
# Defaults to ${REPO_PATH}/LIBERO if not set
export LIBERO_REPO_PATH="${LIBERO_REPO_PATH:-${REPO_PATH}/LIBERO}"
# NOTE: set LIBERO_CONFIG_PATH for libero/libero/__init__.py
export LIBERO_CONFIG_PATH=${LIBERO_REPO_PATH}

export PYTHONPATH=${LIBERO_REPO_PATH}:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1


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

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${REPO_PATH}/logs/evals/${LOG_SUBDIR}_${TIMESTAMP}"
MEGA_LOG_FILE="${LOG_DIR}/eval_embodiment.log"
mkdir -p "${LOG_DIR}"
CMD="python ${SRC_FILE} --config-path ${CONFIG_PATH} --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR} ${HYDRA_OVERRIDES}"
echo ${CMD}
${CMD} 2>&1 | tee ${MEGA_LOG_FILE}
