#! /bin/bash
#
# Setup script for running embodied agent training
#
# Environment Variables (optional overrides):
#   - LIBERO_REPO_PATH: Path to LIBERO repository (defaults to ${REPO_PATH}/LIBERO)
#   - RLINF_LIBERO_USE_OSMESA: Set to 1 for OSMesa (CPU render, slow). Default: EGL; LIBERO env
#     subprocesses remap CUDA_VISIBLE_DEVICES for Ray (see rlinf/envs/libero/venv.py).
#   - RLINF_LIBERO_EGL_REMAP: Set to 0 to disable EGL remapping in env workers if it causes issues.
#   - RLINF_CUDA_LAUNCH_BLOCKING: Set to 1 to enable CUDA_LAUNCH_BLOCKING (debug only; makes rollout
#     / inference far slower — leave unset for normal training).
#
# Note: REPO_PATH is automatically set to the parent directory of examples/
#       If you need to override it, set it before running this script.
#
# Core dumps: on many HPC systems failed workers write huge files like core_nid<PID>_<...>
# in the current directory. Disable unless debugging (RLINF_ALLOW_CORE_DUMPS=1).
if [ "${RLINF_ALLOW_CORE_DUMPS:-0}" != "1" ]; then
    ulimit -c 0 2>/dev/null || true
fi

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"

# MuJoCo offscreen: EGL is faster; on many clusters only EGL device 0 exists while Ray sets
# CUDA_VISIBLE_DEVICES to a single physical GPU index → MuJoCo EGL errors. OSMesa is CPU/software GL.
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
# Do not set CUDA_LAUNCH_BLOCKING by default: it serializes GPU work and tanks VLA rollout speed.
if [ "${RLINF_CUDA_LAUNCH_BLOCKING:-0}" = "1" ] || [ "${RLINF_CUDA_LAUNCH_BLOCKING:-}" = "true" ]; then
    export CUDA_LAUNCH_BLOCKING=1
fi
export HYDRA_FULL_ERROR=1

export RAY_DISABLE_IMPORT_WARNING=1
# export RAY_LOG_TO_STDERR=1
# export RAY_BACKEND_LOG_LEVEL=DEBUG 
export RAY_DISABLE_DASHBOARD=1
# cluster.py reads RLINF_RAY_INCLUDE_DASHBOARD (default on); align with dashboard off above
export RLINF_RAY_INCLUDE_DASHBOARD=0
# Ray plasma/raylet sockets: Linux AF_UNIX paths must stay <=107 bytes. Long paths under $HOME/repo fail ray.init.
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_${USER}}"
mkdir -p "$RAY_TMPDIR"
export PYTORCH_DISTRIBUTED_BACKEND=nccl

# On Slurm shared GPU nodes, Ray otherwise sees all host CPUs (e.g. 192) and spawns a
# massive default worker pool → memory pressure and "Failed to register worker to Raylet".
# Default to the current task's allocation when Slurm env vars are set.
if [ -z "${RLINF_RAY_NUM_CPUS:-}" ] && [ -n "${SLURM_CPUS_PER_TASK:-}" ]; then
    export RLINF_RAY_NUM_CPUS="${SLURM_CPUS_PER_TASK}"
fi
if [ -z "${RLINF_RAY_NUM_GPUS:-}" ] && [ -n "${SLURM_GPUS_ON_NODE:-}" ]; then
    export RLINF_RAY_NUM_GPUS="${SLURM_GPUS_ON_NODE}"
fi

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

# Extract config tag from CONFIG_NAME
# If config ends with _openvlaoft or _eval, don't set CONFIG_TAG
# Otherwise, extract the part after the last _
CONFIG_TAG=""
if [[ ! "${CONFIG_NAME}" =~ _openvlaoft$ ]] && [[ ! "${CONFIG_NAME}" =~ _eval$ ]]; then
    if [[ "${CONFIG_NAME}" =~ _([^_/]+)$ ]]; then
        CONFIG_TAG="${BASH_REMATCH[1]}"
    fi
fi

# Build log directory path with config tag if present
# If LOG_DIR is already set (e.g., by crl_experiment scripts), use it as-is
# Otherwise, create default path with config tag if present
if [ -z "${LOG_DIR}" ]; then
    if [ -n "${CONFIG_TAG}" ]; then
        LOG_DIR="${REPO_PATH}/logs_${CONFIG_TAG}/temp/run_$(date +'%Y%m%d-%H:%M:%S')"
    else
        LOG_DIR="${REPO_PATH}/logs/temp/run_$(date +'%Y%m%d-%H:%M:%S')"
    fi
fi
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
mkdir -p "${LOG_DIR}"
# -u: unbuffered stdout/stderr so [train_embodied_agent] lines keep order vs Ray logs when piped to tee
CMD="python -u ${SRC_FILE} --config-path ${CONFIG_PATH} --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR} $@" 
echo ${CMD} > ${MEGA_LOG_FILE}
# Without pipefail, tee's exit code (0) hides python failures — sequential script would report false success.
set -o pipefail
${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}
