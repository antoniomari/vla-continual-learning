#!/bin/bash
#
# Direct evaluation script for embodied agent with temperature override (without SLURM)
# This script can be run directly on the terminal
#
# Usage: ./examples/mll_cluster/eval_temperature.sh CHECKPOINT_LOCATION TEMPERATURE [STEP_NUMBER]
# Example: ./examples/mll_cluster/eval_temperature.sh logs/naive_LoRA/task_0 2.0
# Example: ./examples/mll_cluster/eval_temperature.sh logs/bcrl_logit/0.3/task_0 1.5 20
#
# Note: CHECKPOINT_LOCATION should be relative to workspace root (e.g., logs/naive_LoRA/task_0)
#       The script will construct the full path: ${WORKSPACE_ROOT}/${CHECKPOINT_LOCATION}/checkpoints/global_step_${STEP_NUMBER}/actor/
#       TEMPERATURE: evaluation temperature to use (default: 2.0)
#       STEP_NUMBER: global step number (default: 10)

CHECKPOINT_LOCATION=$1
TEMPERATURE=$2
STEP_NUMBER=$3

# Default STEP_NUMBER to 10 if not provided
if [ -z "$STEP_NUMBER" ]; then
    STEP_NUMBER=10
fi

# Get workspace root (script directory's parent's parent)
SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
WORKSPACE_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$WORKSPACE_ROOT"

if [ -z "$CHECKPOINT_LOCATION" ] || [ -z "$TEMPERATURE" ]; then
    echo "ERROR: Missing required argument(s)"
    echo "Usage: ./examples/mll_cluster/eval_temperature.sh CHECKPOINT_LOCATION TEMPERATURE [STEP_NUMBER]"
    echo "Example: ./examples/mll_cluster/eval_temperature.sh logs/naive_LoRA/task_0 2.0"
    echo "Example: ./examples/mll_cluster/eval_temperature.sh logs/bcrl_logit/0.3/task_0 1.5 20"
    exit 1
fi

# Validate TEMPERATURE
if ! [[ "$TEMPERATURE" =~ ^[0-9]+\.?[0-9]*$ ]]; then
    echo "ERROR: TEMPERATURE must be a number, got: $TEMPERATURE"
    exit 1
fi

# Validate STEP_NUMBER
if ! [[ "$STEP_NUMBER" =~ ^[0-9]+$ ]]; then
    echo "ERROR: STEP_NUMBER must be a positive integer, got: $STEP_NUMBER"
    exit 1
fi

# Construct full checkpoint path (remove any trailing slashes from CHECKPOINT_LOCATION first)
CHECKPOINT_LOCATION="${CHECKPOINT_LOCATION%/}"
CHECKPOINT_PATH="${WORKSPACE_ROOT}/${CHECKPOINT_LOCATION}/checkpoints/global_step_${STEP_NUMBER}/actor/"

# Verify checkpoint exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT_PATH"
    exit 1
fi

# Print job information
echo "Working Directory: $(pwd)"
echo "Checkpoint Location: $CHECKPOINT_LOCATION"
echo "Full Checkpoint Path: $CHECKPOINT_PATH"
echo "Global Step Number: $STEP_NUMBER"
echo "Evaluation Temperature: $TEMPERATURE"
echo "Start Time: $(date)"
echo ""
echo "Note: Log path will be automatically constructed by eval_embodiment.sh"
echo "      and will include temperature in the directory name."
echo ""

# Run evaluation with temperature override
bash examples/embodiment/eval_embodiment.sh mll_cluster/libero_spatial_grpo_openvlaoft_eval \
    +actor.model.lora_path="${CHECKPOINT_PATH}" \
    algorithm.sampling_params.temperature_eval=${TEMPERATURE}

# Check if evaluation succeeded
if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Evaluation failed"
    echo "Checkpoint path: ${CHECKPOINT_PATH}"
    echo "Temperature: ${TEMPERATURE}"
    exit 1
fi

# Print completion time
echo ""
echo "Evaluation completed successfully!"
echo "End Time: $(date)"
