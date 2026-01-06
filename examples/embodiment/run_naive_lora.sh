#!/bin/bash
#
# Sequential Task Training Script
# Trains on tasks sequentially, loading checkpoints from previous task
#
# Usage:
#   ./run_lifelong.sh [config_name] [bc_coeff] [num_tasks]

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

# Get script directory
SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
RUN_EMBODIMENT_SCRIPT="${SCRIPT_DIR}/run_embodiment.sh"

# Check if run_embodiment.sh exists
if [ ! -f "$RUN_EMBODIMENT_SCRIPT" ]; then
    echo "ERROR: run_embodiment.sh not found at: $RUN_EMBODIMENT_SCRIPT"
    exit 1
fi

# Default values
DEFAULT_CONFIG="libero_spatial_grpo_openvlaoft"
DEFAULT_BC_COEFF=0.00
DEFAULT_NUM_TASKS=5

# Get arguments
CONFIG_NAME="${1:-$DEFAULT_CONFIG}"
BC_COEFF="${2:-$DEFAULT_BC_COEFF}"
NUM_TASKS="${3:-$DEFAULT_NUM_TASKS}"

# Get REPO_PATH (run_embodiment.sh will set this, but we need it for log dir)
export EMBODIED_PATH="$SCRIPT_DIR"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))

# Format BC coefficient for directory name (e.g., 0.03 -> 03, 0.3 -> 3, 0.005 -> 005)
BC_COEFF_FORMATTED=$(echo "$BC_COEFF" | sed 's/^0\.//' | sed 's/\.//g')

# Determine padding based on value
if (( $(echo "$BC_COEFF < 0.01" | bc -l) )); then
    # For very small values like 0.005, use 3 digits
    BC_COEFF_FORMATTED=$(printf "%03d" "$BC_COEFF_FORMATTED")
elif (( $(echo "$BC_COEFF < 0.1" | bc -l) )); then
    # For values like 0.03, use 2 digits
    BC_COEFF_FORMATTED=$(printf "%02d" "$BC_COEFF_FORMATTED")
else
    # For values like 0.3, use 1 digit
    BC_COEFF_FORMATTED=$(printf "%d" "$BC_COEFF_FORMATTED")
fi

# Base log directory
BASE_LOG_DIR="${REPO_PATH}/logs/naive_LoRA/"
mkdir -p "${BASE_LOG_DIR}"

# ============================================================================
# Print Configuration
# ============================================================================

echo "========================================================================"
echo "Sequential Task Training"
echo "========================================================================"
echo "Config:         $CONFIG_NAME"
echo "BC Coefficient: $BC_COEFF"
echo "Num Tasks:      $NUM_TASKS (0 to $((NUM_TASKS-1)))"
echo "Base Log Dir:   $BASE_LOG_DIR"
echo "========================================================================"
echo ""

# ============================================================================
# Training Loop
# ============================================================================

PREV_CHECKPOINT_PATH=""

for TASK_ID in $(seq 0 $((NUM_TASKS-1))); do
    echo ""
    echo "========================================================================"
    echo "Training on Task ${TASK_ID}"
    echo "========================================================================"
    
    # Create task-specific log directory
    TASK_LOG_DIR="${BASE_LOG_DIR}/task_${TASK_ID}"
    mkdir -p "${TASK_LOG_DIR}"
    
    # Build hydra overrides
    OVERRIDES="env.fixed_task_ids=[${TASK_ID}] algorithm.bc_coeff=${BC_COEFF}"
    
    # For tasks after 0, add the checkpoint path from previous task
    if [ $TASK_ID -gt 0 ]; then
        if [ -z "$PREV_CHECKPOINT_PATH" ]; then
            echo "ERROR: Previous checkpoint path is empty for task ${TASK_ID}"
            exit 1
        fi
        
        if [ ! -d "$PREV_CHECKPOINT_PATH" ]; then
            echo "ERROR: Previous checkpoint does not exist: $PREV_CHECKPOINT_PATH"
            exit 1
        fi
        
        echo "Loading checkpoint from: $PREV_CHECKPOINT_PATH"
        OVERRIDES="${OVERRIDES} +actor.model.lora_path=${PREV_CHECKPOINT_PATH}"
    fi
    
    echo "Task ${TASK_ID} overrides: ${OVERRIDES}"
    echo "Logging to: ${TASK_LOG_DIR}"
    echo ""
    
    # Set LOG_DIR environment variable for run_embodiment.sh
    export LOG_DIR="${TASK_LOG_DIR}"
    
    # Run training via run_embodiment.sh
    bash ${RUN_EMBODIMENT_SCRIPT} ${CONFIG_NAME} ${OVERRIDES}
    
    # Check if training succeeded
    if [ $? -ne 0 ]; then
        echo ""
        echo "ERROR: Training failed for task ${TASK_ID}"
        echo "Check log file: ${TASK_LOG_DIR}/run_embodiment.log"
        exit 1
    fi
    
    # Find the checkpoint directory for this task
    # Assumes checkpoint is saved at: {TASK_LOG_DIR}/checkpoints/global_step_10/actor/
    CHECKPOINT_DIR="${TASK_LOG_DIR}/checkpoints/global_step_10/actor"
    
    if [ ! -d "$CHECKPOINT_DIR" ]; then
        echo ""
        echo "ERROR: Checkpoint not found at expected location: $CHECKPOINT_DIR"
        echo "Training may have completed but checkpoint was not saved correctly"
        exit 1
    fi
    
    echo ""
    echo "Task ${TASK_ID} completed successfully"
    echo "Checkpoint saved at: $CHECKPOINT_DIR"
    
    # Set checkpoint path for next iteration
    PREV_CHECKPOINT_PATH="$CHECKPOINT_DIR"
    
    echo "========================================================================"
done

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "========================================================================"
echo "All tasks completed successfully!"
echo "========================================================================"
echo "Results saved in: $BASE_LOG_DIR"
echo ""
echo "Task directories:"
for TASK_ID in $(seq 0 $((NUM_TASKS-1))); do
    echo "  Task ${TASK_ID}: ${BASE_LOG_DIR}/task_${TASK_ID}"
done
echo "========================================================================"
