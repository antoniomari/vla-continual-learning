#!/bin/bash
#
# Direct evaluation script for embodied agent (without SLURM)
# This script can be run directly on the terminal
#
# Usage: ./examples/mll_cluster/eval_embodiment.sh CHECKPOINT_LOCATION
# Example: ./examples/mll_cluster/eval_embodiment.sh logs/bcrl_logit/0.3/task_0
#
# Note: CHECKPOINT_LOCATION should be relative to workspace root (e.g., logs/bcrl_logit/0.3/task_0)
#       The script will construct the full path: ${WORKSPACE_ROOT}/${CHECKPOINT_LOCATION}/checkpoints/global_step_10/actor/

CHECKPOINT_LOCATION=$1

if [ -z "$CHECKPOINT_LOCATION" ]; then
    echo "ERROR: Missing required argument"
    echo "Usage: ./examples/mll_cluster/eval_embodiment.sh CHECKPOINT_LOCATION"
    echo "Example: ./examples/mll_cluster/eval_embodiment.sh logs/bcrl_logit/0.3/task_0"
    exit 1
fi

# Get workspace root (script directory's parent's parent)
SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
WORKSPACE_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$WORKSPACE_ROOT"

# Construct full checkpoint path (remove any trailing slashes from CHECKPOINT_LOCATION first)
CHECKPOINT_LOCATION="${CHECKPOINT_LOCATION%/}"
CHECKPOINT_PATH="${WORKSPACE_ROOT}/${CHECKPOINT_LOCATION}/checkpoints/global_step_10/actor/"

# Verify checkpoint exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT_PATH"
    exit 1
fi

# Print job information
echo "Working Directory: $(pwd)"
echo "Checkpoint Location: $CHECKPOINT_LOCATION"
echo "Full Checkpoint Path: $CHECKPOINT_PATH"
echo "Start Time: $(date)"
echo ""

# Run the evaluation
bash examples/embodiment/eval_embodiment.sh mll_cluster/libero_spatial_grpo_openvlaoft_eval +actor.model.lora_path="${CHECKPOINT_PATH}"

# Print completion time
echo ""
echo "End Time: $(date)"
