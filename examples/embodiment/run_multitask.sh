#!/bin/bash
#
# Multi-Task Training Script
# Trains on all tasks simultaneously, checkpointing every 5 epochs
#
# Usage:
#   ./run_multitask.sh [config_name] [bc_coeff] [num_epochs_per_session]

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
DEFAULT_EPOCHS_PER_SESSION=5

# Get arguments
CONFIG_NAME="${1:-$DEFAULT_CONFIG}"
BC_COEFF="${2:-$DEFAULT_BC_COEFF}"
EPOCHS_PER_SESSION="${3:-$DEFAULT_EPOCHS_PER_SESSION}"

# Get REPO_PATH
export EMBODIED_PATH="$SCRIPT_DIR"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))

# Base log directory
BASE_LOG_DIR="${REPO_PATH}/logs/multitask"
mkdir -p "${BASE_LOG_DIR}"

# Task IDs to train on
TASK_IDS="3,4,5,6,7"
NUM_SESSIONS=5  # Train 5 times (epoch 5, 10, 15, 20, 25)

# ============================================================================
# Print Configuration
# ============================================================================

echo "========================================================================"
echo "Multi-Task Training"
echo "========================================================================"
echo "Config:              $CONFIG_NAME"
echo "BC Coefficient:      $BC_COEFF"
echo "Task IDs:            $TASK_IDS"
echo "Epochs per session:  $EPOCHS_PER_SESSION"
echo "Total sessions:      $NUM_SESSIONS"
echo "Total epochs:        $(($NUM_SESSIONS * $EPOCHS_PER_SESSION))"
echo "Base Log Dir:        $BASE_LOG_DIR"
echo "========================================================================"
echo ""

# ============================================================================
# Training Loop
# ============================================================================

PREV_CHECKPOINT_PATH=""

for SESSION in $(seq 1 $NUM_SESSIONS); do
    CURRENT_EPOCH=$(($SESSION * $EPOCHS_PER_SESSION))
    
    echo ""
    echo "========================================================================"
    echo "Training Session ${SESSION}/${NUM_SESSIONS} - Epoch ${CURRENT_EPOCH}"
    echo "========================================================================"
    
    # Create log directory for this session
    SESSION_LOG_DIR="${BASE_LOG_DIR}/epoch_${CURRENT_EPOCH}"
    mkdir -p "${SESSION_LOG_DIR}"
    
    # Build hydra overrides
    OVERRIDES="env.fixed_task_ids=[${TASK_IDS}] \
               runner.max_epochs=${EPOCHS_PER_SESSION} \
               runner.save_interval=${EPOCHS_PER_SESSION} \
	       rollout.enable_offload=True
	       actor.seed=${BC_COEFF}
               actor.preallocate=0"
    
    # Load checkpoint from previous session (if not first session)
    if [ $SESSION -gt 1 ]; then
        if [ -z "$PREV_CHECKPOINT_PATH" ]; then
            echo "ERROR: Previous checkpoint path is empty for session ${SESSION}"
            exit 1
        fi
        
        if [ ! -d "$PREV_CHECKPOINT_PATH" ]; then
            echo "ERROR: Previous checkpoint does not exist: $PREV_CHECKPOINT_PATH"
            exit 1
        fi
        
        echo "Loading checkpoint from: $PREV_CHECKPOINT_PATH"
        OVERRIDES="${OVERRIDES} +actor.model.lora_path=${PREV_CHECKPOINT_PATH}"
    fi
    
    echo "Session ${SESSION} overrides: ${OVERRIDES}"
    echo "Logging to: ${SESSION_LOG_DIR}"
    echo ""
    
    # Set LOG_DIR environment variable for run_embodiment.sh
    export LOG_DIR="${SESSION_LOG_DIR}"
    
    # Run training via run_embodiment.sh
    bash ${RUN_EMBODIMENT_SCRIPT} ${CONFIG_NAME} ${OVERRIDES}
    
    # Check if training succeeded
    if [ $? -ne 0 ]; then
        echo ""
        echo "ERROR: Training failed for session ${SESSION} (epoch ${CURRENT_EPOCH})"
        echo "Check log file: ${SESSION_LOG_DIR}/run_embodiment.log"
        exit 1
    fi
    
    # Find the checkpoint directory for this session
    # Since we train for EPOCHS_PER_SESSION epochs, checkpoint is at global_step_EPOCHS_PER_SESSION
    CHECKPOINT_DIR="${SESSION_LOG_DIR}/checkpoints/global_step_${EPOCHS_PER_SESSION}/actor"
    
    if [ ! -d "$CHECKPOINT_DIR" ]; then
        echo ""
        echo "ERROR: Checkpoint not found at expected location: $CHECKPOINT_DIR"
        echo "Training may have completed but checkpoint was not saved correctly"
        exit 1
    fi
    
    echo ""
    echo "Session ${SESSION} completed successfully (epoch ${CURRENT_EPOCH})"
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
echo "All training sessions completed successfully!"
echo "========================================================================"
echo "Results saved in: $BASE_LOG_DIR"
echo ""
echo "Session checkpoints:"
for SESSION in $(seq 1 $NUM_SESSIONS); do
    CURRENT_EPOCH=$(($SESSION * $EPOCHS_PER_SESSION))
    echo "  Epoch ${CURRENT_EPOCH}: ${BASE_LOG_DIR}/epoch_${CURRENT_EPOCH}"
done
echo "========================================================================"
