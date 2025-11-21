#! /bin/bash
#
# Setup script for running embodied agent training
#
# Environment Variables (optional overrides):
#   - LIBERO_REPO_PATH: Path to LIBERO repository (defaults to ${REPO_PATH}/LIBERO)
#
# Note: REPO_PATH is automatically set to the parent directory of examples/
#       If you need to override it, set it before running this script.

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH
# NOTE: set LIBERO_REPO_PATH to the path of the LIBERO repo
# Defaults to ${REPO_PATH}/LIBERO if not set
export LIBERO_REPO_PATH="${LIBERO_REPO_PATH:-${REPO_PATH}/LIBERO}"
# NOTE: set LIBERO_CONFIG_PATH for libero/libero/__init__.py
export LIBERO_CONFIG_PATH=${LIBERO_REPO_PATH}

export PYTHONPATH=${LIBERO_REPO_PATH}:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1

export RAY_DISABLE_IMPORT_WARNING=1
# export RAY_LOG_TO_STDERR=1
# export RAY_BACKEND_LOG_LEVEL=DEBUG 
export RAY_DISABLE_DASHBOARD=1
export PYTORCH_DISTRIBUTED_BACKEND=nccl
 
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
    else
        # No subdirectory, use root config directory
        CONFIG_NAME=$1
        CONFIG_PATH="${EMBODIED_PATH}/config/"
	shift
    fi
fi

if [ -z "${LOG_DIR}" ]; then
    LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')" #/$(date +'%Y%m%d-%H:%M:%S')"
fi
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
mkdir -p "${LOG_DIR}"
CMD="python ${SRC_FILE} --config-path ${CONFIG_PATH} --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR} $@" 
echo ${CMD} > ${MEGA_LOG_FILE}
${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}
