#!/bin/bash
#
# Direct evaluation script for embodied agent (without SLURM)
# This script can be run directly on the terminal
#
# Usage: ./examples/mll_cluster/eval_embodiment.sh CHECKPOINT_LOCATION [STEP_NUMBER]
# Example: ./examples/mll_cluster/eval_embodiment.sh logs/bcrl_logit/0.3/task_0
# Example: ./examples/mll_cluster/eval_embodiment.sh logs/bcrl_logit/0.3/task_0 20
#
# Note: CHECKPOINT_LOCATION should be relative to workspace root (e.g., logs/bcrl_logit/0.3/task_0)
#       The script will construct the full path: ${WORKSPACE_ROOT}/${CHECKPOINT_LOCATION}/checkpoints/global_step_${STEP_NUMBER}/actor/
#       STEP_NUMBER: global step number (default: 10)

CHECKPOINT_LOCATION=$1
STEP_NUMBER=$2

# Default STEP_NUMBER to 10 if not provided
if [ -z "$STEP_NUMBER" ]; then
    STEP_NUMBER=10
fi

if [ -z "$CHECKPOINT_LOCATION" ]; then
    echo "ERROR: Missing required argument"
    echo "Usage: ./examples/mll_cluster/eval_embodiment.sh CHECKPOINT_LOCATION [STEP_NUMBER]"
    echo "Example: ./examples/mll_cluster/eval_embodiment.sh logs/bcrl_logit/0.3/task_0"
    echo "Example: ./examples/mll_cluster/eval_embodiment.sh logs/bcrl_logit/0.3/task_0 20"
    exit 1
fi

# Validate STEP_NUMBER
if ! [[ "$STEP_NUMBER" =~ ^[0-9]+$ ]]; then
    echo "ERROR: STEP_NUMBER must be a positive integer, got: $STEP_NUMBER"
    exit 1
fi

# Get workspace root (script directory's parent's parent)
SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
WORKSPACE_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$WORKSPACE_ROOT"

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
echo "Start Time: $(date)"
echo ""

# Create a wrapper script that modifies the log path to include step number
# We'll pass an environment variable to the eval script
export EVAL_STEP_NUMBER="${STEP_NUMBER}"

# Run the evaluation
bash examples/embodiment/eval_embodiment.sh mll_cluster/libero_spatial_grpo_openvlaoft_eval +actor.model.lora_path="${CHECKPOINT_PATH}"

EXIT_CODE=$?

# Print completion time
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "Evaluation completed successfully!"
else
    echo "Evaluation failed with exit code $EXIT_CODE"
fi
echo "End Time: $(date)"
echo "Checkpoint evaluated: global_step_${STEP_NUMBER}"

exit $EXIT_CODE