#!/bin/bash
#
# Direct evaluation script for embodied agent
# This script can be run directly on the terminal
#
# Usage: ./examples/crl_experiment/eval_embodiment.sh CHECKPOINT_LOCATION [STEP_NUMBER] [CONFIG_NAME]
# Example (LoRA): ./examples/crl_experiment/eval_embodiment.sh logs/bcrl_logit/0.3/task_0
# Example (LoRA): ./examples/crl_experiment/eval_embodiment.sh logs/bcrl_logit/0.3/task_0 20
# Example (simple_cnn): ./examples/crl_experiment/eval_embodiment.sh logs/simple_cnn/task_0_seed1234 10 crl_experiment/libero_spatial_grpo_simple_cnn_eval
# Example (cam suite config): ./examples/crl_experiment/eval_embodiment.sh logs/bcrl_logit/0.3/task_0 20 crl_experiment/libero_spatial_grpo_openvlaoft_eval_cam
#
# Note: CHECKPOINT_LOCATION should be relative to workspace root (e.g., logs/bcrl_logit/0.3/task_0)
#       For LoRA: The script will construct: ${WORKSPACE_ROOT}/${CHECKPOINT_LOCATION}/checkpoints/global_step_${STEP_NUMBER}/actor/
#       For simple_cnn: The script will construct: ${WORKSPACE_ROOT}/${CHECKPOINT_LOCATION}/checkpoints/global_step_${STEP_NUMBER}/actor/model.pt
#       STEP_NUMBER: global step folder index (default: 10 if omitted; use final training step, usually max_epochs)
# Optional env: EVAL_HYDRA_OVERRIDES — extra Hydra overrides appended to eval (e.g. seed, task, W&B name, GRPO sweep)

CHECKPOINT_LOCATION=$1
STEP_NUMBER=$2
CONFIG_NAME=$3

# Default STEP_NUMBER to 10 if not provided
if [ -z "$STEP_NUMBER" ]; then
    STEP_NUMBER=10
fi

# Default CONFIG_NAME to original libero_spatial eval config if not provided
if [ -z "$CONFIG_NAME" ]; then
    CONFIG_NAME="crl_experiment/libero_spatial_grpo_openvlaoft_eval_spatial"
fi

if [ -z "$CHECKPOINT_LOCATION" ]; then
    echo "ERROR: Missing required argument"
    echo "Usage: ./examples/crl_experiment/eval_embodiment.sh CHECKPOINT_LOCATION|base [STEP_NUMBER] [CONFIG_NAME]"
    echo "Example (LoRA eval): ./examples/crl_experiment/eval_embodiment.sh logs/bcrl_logit/0.3/task_0 20"
    echo "Example (cam suite config): ./examples/crl_experiment/eval_embodiment.sh logs/bcrl_logit/0.3/task_0 20 crl_experiment/libero_spatial_grpo_openvlaoft_eval_cam"
    echo "Example (base model eval): ./examples/crl_experiment/eval_embodiment.sh base 0 crl_experiment/libero_spatial_grpo_openvlaoft_eval_spatial"
    exit 1
fi

# Validate STEP_NUMBER
if ! [[ "$STEP_NUMBER" =~ ^[0-9]+$ ]]; then
    echo "ERROR: STEP_NUMBER must be a positive integer, got: $STEP_NUMBER"
    exit 1
fi

# Get workspace root (assume we're already in the workspace root)
WORKSPACE_ROOT=$(pwd)

# Base eval mode (no LoRA path)
IS_BASE_EVAL=false
if [ "$CHECKPOINT_LOCATION" = "base" ]; then
    IS_BASE_EVAL=true
fi

CHECKPOINT_PATH=""
IS_SIMPLE_CNN=false
if [[ "$CONFIG_NAME" == *"simple_cnn"* ]]; then
    IS_SIMPLE_CNN=true
    # Set environment variable for CNN models
    export USE_CNN_UTILS=1
fi

if [ "$IS_BASE_EVAL" = false ]; then
    # Construct full checkpoint path (remove any trailing slashes from CHECKPOINT_LOCATION first)
    CHECKPOINT_LOCATION="${CHECKPOINT_LOCATION%/}"
    
    if [ "$IS_SIMPLE_CNN" = true ]; then
        # For simple_cnn, checkpoint is a file (model.pt), not a directory
        CHECKPOINT_PATH="${WORKSPACE_ROOT}/${CHECKPOINT_LOCATION}/checkpoints/global_step_${STEP_NUMBER}/actor/model.pt"
        
        # Verify checkpoint file exists
        if [ ! -f "$CHECKPOINT_PATH" ]; then
            echo "ERROR: Checkpoint file not found at $CHECKPOINT_PATH"
            exit 1
        fi
    else
        # For LoRA models, checkpoint is a directory
        CHECKPOINT_PATH="${WORKSPACE_ROOT}/${CHECKPOINT_LOCATION}/checkpoints/global_step_${STEP_NUMBER}/actor/"
        
        # Verify checkpoint directory exists
        if [ ! -d "$CHECKPOINT_PATH" ]; then
            echo "ERROR: Checkpoint directory not found at $CHECKPOINT_PATH"
            exit 1
        fi
    fi
fi

# Print job information
echo "Working Directory: $(pwd)"
echo "Checkpoint Location: $CHECKPOINT_LOCATION"
if [ "$IS_BASE_EVAL" = true ]; then
    echo "Full Checkpoint Path: (base model; no LoRA)"
else
    echo "Full Checkpoint Path: $CHECKPOINT_PATH"
fi
echo "Global Step Number: $STEP_NUMBER"
echo "Config Name: $CONFIG_NAME"
echo "Start Time: $(date)"
echo ""

# Create a wrapper script that modifies the log path to include step number
# and a short config tag. We'll pass environment variables to the eval script.
export EVAL_STEP_NUMBER="${STEP_NUMBER}"


if [ "$IS_BASE_EVAL" = true ]; then
    bash examples/embodiment/eval_embodiment.sh ${CONFIG_NAME} actor.model.is_lora=False ${EVAL_HYDRA_OVERRIDES:-}
elif [ "$IS_SIMPLE_CNN" = true ]; then
    # For simple_cnn, use rollout.checkpoint_path (points to model.pt file)
    bash examples/embodiment/eval_embodiment.sh ${CONFIG_NAME} rollout.checkpoint_path="${CHECKPOINT_PATH}" ${EVAL_HYDRA_OVERRIDES:-}
else
    # For LoRA models, use actor.model.lora_path (points to directory)
    bash examples/embodiment/eval_embodiment.sh ${CONFIG_NAME} +actor.model.lora_path="${CHECKPOINT_PATH}" ${EVAL_HYDRA_OVERRIDES:-}
fi

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

# Unset environment variable to avoid affecting subsequent runs
if [ "$IS_SIMPLE_CNN" = true ]; then
    unset USE_CNN_UTILS
fi

exit $EXIT_CODE
