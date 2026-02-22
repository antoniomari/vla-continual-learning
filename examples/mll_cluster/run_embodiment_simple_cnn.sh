#!/bin/bash
#
# Sequential RL training script for simple_cnn policy
# This script trains tasks sequentially, loading checkpoints from previous tasks
#
# Usage: ./examples/mll_cluster/run_embodiment_simple_cnn.sh [TASK_ID_OR_RANGE] [CHECKPOINT_PATH] [MAX_EPOCH] [CONFIG_NAME] [SEED]
# Example (single task): ./examples/mll_cluster/run_embodiment_simple_cnn.sh 0
# Example (task range): ./examples/mll_cluster/run_embodiment_simple_cnn.sh "0,3"
# Example (with max_epoch): ./examples/mll_cluster/run_embodiment_simple_cnn.sh 0 "" 15
# Example (continue from checkpoint): ./examples/mll_cluster/run_embodiment_simple_cnn.sh 0 ./logs/simple_cnn/task_0_seed1234/checkpoints/global_step_10/actor/model.pt 20
# Example (with seed): ./examples/mll_cluster/run_embodiment_simple_cnn.sh 0 "" "" "" 42
# Note: TASK_ID_OR_RANGE can be:
#       - A single task ID (e.g., "0") - trains that task only
#       - A tuple "a,b" where a < b (e.g., "0,3") - trains tasks from a to b sequentially
#       CHECKPOINT_PATH is optional and will be auto-generated from previous task if not provided
#       For simple_cnn, CHECKPOINT_PATH should point to the model.pt file (not a directory)
#       If CHECKPOINT_PATH is provided for a range, it will only be used for the first task
#       MAX_EPOCH is optional and can always be specified to override the default max_epochs
#       SEED is optional and defaults to 1234 if not provided

TASK_INPUT=${1:-0}
MANUAL_CHECKPOINT_PATH=$2
MAX_EPOCH=$3
CONFIG_NAME=${4:-mll_cluster/libero_spatial_grpo_simple_cnn}
SEED=${5:-1234}

# Set environment variable to use CNN utils (no PrismaticProjector import)
export USE_CNN_UTILS=1

# Source common functions
source "examples/mll_cluster/common_functions.sh"

# Extract config tag and derive eval config name
CONFIG_TAG=$(extract_config_tag "$CONFIG_NAME")
EVAL_CONFIG_NAME=$(derive_eval_config_name "$CONFIG_NAME")
# derive_eval_config_name doesn't handle simple_cnn - fix it here
if [[ "$CONFIG_NAME" == *"simple_cnn" ]]; then
    EVAL_CONFIG_NAME="${CONFIG_NAME}_eval"
fi
GLOBAL_STEP=$(get_default_global_step "$CONFIG_NAME")
FIRST_TASK_ID=$(get_first_task_id "$CONFIG_NAME")

# Parse TASK_INPUT to determine if it's a single task or a range
if [[ "$TASK_INPUT" == *,* ]]; then
    # It's a tuple (a,b)
    IFS=',' read -r TASK_START TASK_END <<< "$TASK_INPUT"
    TASK_START=$(echo "$TASK_START" | tr -d '()[] ')
    TASK_END=$(echo "$TASK_END" | tr -d '()[] ')
    
    # Validate that both values are numeric and start < end
    if ! [[ "$TASK_START" =~ ^[0-9]+$ ]] || ! [[ "$TASK_END" =~ ^[0-9]+$ ]]; then
        echo "ERROR: Task range must contain two numeric values: \"a,b\" where a and b are integers"
        echo "       Example: \"0,3\" or \"1,5\""
        exit 1
    fi
    
    if [ "$TASK_START" -ge "$TASK_END" ]; then
        echo "ERROR: First task ID ($TASK_START) must be smaller than second task ID ($TASK_END)"
        echo "       Example: \"0,3\" (trains tasks 0, 1, 2, 3)"
        exit 1
    fi
    
    IS_RANGE=true
    NUM_TASKS=$((TASK_END - TASK_START + 1))
else
    # It's a single task ID
    if ! [[ "$TASK_INPUT" =~ ^[0-9]+$ ]]; then
        echo "ERROR: Task ID must be a numeric value"
        echo "       Example: 0 or \"0,3\" for a range"
        exit 1
    fi
    IS_RANGE=false
    TASK_START=$TASK_INPUT
    TASK_END=$TASK_INPUT
    NUM_TASKS=1
fi

# Validate seed is a number
if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
    echo "ERROR: SEED must be a non-negative integer, got: $SEED"
    exit 1
fi

mkdir -p logs/slurm
mkdir -p logs/simple_cnn

# Print job information
echo "Working Directory: $(pwd)"
echo "Start Time: $(date)"
echo ""

# Main training loop
OVERALL_EXIT_CODE=0

for TASK_ID in $(seq $TASK_START $TASK_END); do
    echo ""
    echo "========================================="
    if [ "$IS_RANGE" = true ]; then
        echo "Sequential Training - Task ${TASK_ID} (${TASK_START} to ${TASK_END})"
    else
        echo "Lifelong Learning - Single Task Training (Simple CNN)"
    fi
    echo "========================================="
    
    # Determine checkpoint path
    if [ "$TASK_ID" -eq "$TASK_START" ] && [ -n "$MANUAL_CHECKPOINT_PATH" ]; then
        # Use manual checkpoint path for first task if provided
        CHECKPOINT_PATH="$MANUAL_CHECKPOINT_PATH"
    elif [ "$TASK_ID" -eq $FIRST_TASK_ID ]; then
        # First task in sequence, use initial checkpoint from config
        # The config should have checkpoint_load_path set to the initial simple_cnn checkpoint
        CHECKPOINT_PATH=""
    else
        # Use checkpoint from previous task
        PREV_TASK_ID=$((TASK_ID - 1))
        PREV_LOG_DIR="./logs/simple_cnn/task_${PREV_TASK_ID}_seed${SEED}"
        # Inject config tag into PREV_LOG_DIR to match where previous task saved checkpoint
        if [ -n "$CONFIG_TAG" ]; then
            # CONFIG_TAG is set, transform the path
            PREV_LOG_DIR_TRANSFORMED=$(inject_config_tag_into_log_path "$PREV_LOG_DIR" "$CONFIG_TAG")
            # Validate transformation result
            if [ -z "$PREV_LOG_DIR_TRANSFORMED" ]; then
                echo "  ERROR: Failed to transform previous task log directory for task $PREV_TASK_ID"
                echo "         Original PREV_LOG_DIR: [$PREV_LOG_DIR]"
                echo "         CONFIG_TAG: [$CONFIG_TAG]"
                OVERALL_EXIT_CODE=1
                break
            fi
        else
            # CONFIG_TAG is empty, use path as-is (no transformation needed)
            PREV_LOG_DIR_TRANSFORMED="$PREV_LOG_DIR"
        fi
        
        # For simple_cnn, checkpoint is saved as model.pt in the actor directory
        CHECKPOINT_PATH="${PREV_LOG_DIR_TRANSFORMED}/checkpoints/global_step_${GLOBAL_STEP}/actor/model.pt"
        
        # Additional validation: ensure CHECKPOINT_PATH is a valid relative or absolute path
        if [[ "$CHECKPOINT_PATH" =~ ^/checkpoints/ ]]; then
            echo "  ERROR: Invalid checkpoint path construction detected"
            echo "         PREV_LOG_DIR was likely empty or malformed"
            echo "         PREV_LOG_DIR: [$PREV_LOG_DIR]"
            echo "         CHECKPOINT_PATH: [$CHECKPOINT_PATH]"
            OVERALL_EXIT_CODE=1
            break
        fi
    fi
    
    # Determine LOG_DIR based on checkpoint path
    if [ "$TASK_ID" -eq "$TASK_START" ] && [ -n "$MANUAL_CHECKPOINT_PATH" ]; then
        # Extract task ID and global_step from checkpoint path
        # e.g., .../task_0/checkpoints/global_step_10/actor/model.pt -> task_0, step_10
        if [[ "$CHECKPOINT_PATH" =~ task_([0-9]+) ]]; then
            SOURCE_TASK="${BASH_REMATCH[1]}"
            if [[ "$CHECKPOINT_PATH" =~ global_step_([0-9]+) ]]; then
                SOURCE_STEP="${BASH_REMATCH[1]}"
                LOG_DIR="./logs/simple_cnn/task_${TASK_ID}_from_task_${SOURCE_TASK}_step_${SOURCE_STEP}_seed${SEED}"
            else
                echo "ERROR: Could not extract global_step from checkpoint path: $CHECKPOINT_PATH"
                echo "       Expected format: .../checkpoints/global_step_<M>/actor/model.pt"
                OVERALL_EXIT_CODE=1
                break
            fi
        else
            echo "ERROR: Could not extract task ID from checkpoint path: $CHECKPOINT_PATH"
            echo "       Expected format: .../task_<N>/checkpoints/global_step_<M>/actor/model.pt"
            OVERALL_EXIT_CODE=1
            break
        fi
    else
        # Standard case: use TASK_ID
        LOG_DIR="./logs/simple_cnn/task_${TASK_ID}_seed${SEED}"
    fi
    
    # Inject config tag into LOG_DIR before exporting
    if [ -n "$CONFIG_TAG" ]; then
        # CONFIG_TAG is set, transform the path
        LOG_DIR_TRANSFORMED=$(inject_config_tag_into_log_path "$LOG_DIR" "$CONFIG_TAG")
        # Validate transformation result
        if [ -z "$LOG_DIR_TRANSFORMED" ]; then
            echo "  ERROR: Failed to transform LOG_DIR with config tag"
            echo "         Original LOG_DIR: [$LOG_DIR]"
            echo "         CONFIG_TAG: [$CONFIG_TAG]"
            OVERALL_EXIT_CODE=1
            break
        fi
        LOG_DIR="$LOG_DIR_TRANSFORMED"
    else
        # CONFIG_TAG is empty, use path as-is (no transformation needed)
        # This is valid - it means no config tag was specified
        :
    fi
    
    # Validate LOG_DIR is not empty
    if [ -z "$LOG_DIR" ]; then
        echo "  ERROR: LOG_DIR is empty after path construction"
        OVERALL_EXIT_CODE=1
        break
    fi
    
    export LOG_DIR
    mkdir -p "${LOG_DIR}"
    
    # Set experiment name based on LOG_DIR (for wandb)
    EXPERIMENT_NAME=$(basename "$LOG_DIR")
    # Append CONFIG_TAG to experiment name if set
    if [ -n "$CONFIG_TAG" ]; then
        EXPERIMENT_NAME="${EXPERIMENT_NAME}_${CONFIG_TAG}"
    fi
    
    echo "Configuration:"
    echo "  Task ID: $TASK_ID"
    if [ "$IS_RANGE" = true ]; then
        echo "  Task Range: ${TASK_START} to ${TASK_END}"
    fi
    echo "  Experiment Name: $EXPERIMENT_NAME"
    echo "  Checkpoint Save Path: $LOG_DIR"
    echo "  Config Name: $CONFIG_NAME"
    echo "  Random Seed: $SEED"
    
    if [ -n "$CHECKPOINT_PATH" ]; then
        if [ ! -f "$CHECKPOINT_PATH" ]; then
            echo "  ERROR: Checkpoint not found at $CHECKPOINT_PATH"
            if [ "$TASK_ID" -gt $FIRST_TASK_ID ]; then
                PREV_TASK_ID=$((TASK_ID - 1))
                echo "         Task $TASK_ID requires checkpoint from task $PREV_TASK_ID"
                echo "         Make sure task $PREV_TASK_ID has been trained first, or provide a manual checkpoint path"
            fi
            OVERALL_EXIT_CODE=1
            break
        fi
        echo "  Loading from checkpoint: $CHECKPOINT_PATH"
    else
        echo "  Training from initial checkpoint (from config) - First task (task $FIRST_TASK_ID)"
    fi
    
    if [ -n "$MAX_EPOCH" ]; then
        if ! [[ "$MAX_EPOCH" =~ ^[0-9]+$ ]] || [ "$MAX_EPOCH" -le 0 ]; then
            echo "  ERROR: MAX_EPOCH must be a positive integer, got: $MAX_EPOCH"
            OVERALL_EXIT_CODE=1
            break
        fi
        echo "  Max epochs: $MAX_EPOCH"
    fi
    
    echo "========================================="
    echo ""
    
    # Build Hydra overrides
    # For simple_cnn, we use checkpoint_load_path (not lora_path) and it should point to the .pt file
    OVERRIDES="env.fixed_task_ids=[${TASK_ID}] \
    	runner.logger.experiment_name=${EXPERIMENT_NAME} \
    	actor.seed=${SEED}"
    
    if [ -n "$CHECKPOINT_PATH" ]; then
        # For simple_cnn, set checkpoint_load_path to the model.pt file
        # Only override if CHECKPOINT_PATH is provided (not empty)
        OVERRIDES="$OVERRIDES actor.checkpoint_load_path=${CHECKPOINT_PATH}"
    fi
    # If CHECKPOINT_PATH is empty, use the value from config (don't override)
    
    if [ -n "$MAX_EPOCH" ]; then
        # Override max_epochs if MAX_EPOCH is provided
        OVERRIDES="$OVERRIDES runner.max_epochs=${MAX_EPOCH}"
    fi
    
    echo "Running with Hydra overrides:"
    echo "$OVERRIDES"
    echo ""
    
    bash examples/embodiment/run_embodiment.sh ${CONFIG_NAME} $OVERRIDES
    
    EXIT_CODE=$?
    echo ""
    echo "========================================="
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Task $TASK_ID completed successfully"
        echo ""
        echo "Checkpoint saved to: ${LOG_DIR}"
        
        # Run evaluation
        # LOG_DIR already has the config tag injected, so use it directly
        CHECKPOINT_LOCATION=$(echo "$LOG_DIR" | sed 's|^\./||')
        echo ""
        echo "Running evaluation for: ${CHECKPOINT_LOCATION}"
        bash examples/mll_cluster/eval_embodiment.sh "${CHECKPOINT_LOCATION}" "" "${EVAL_CONFIG_NAME}"
    else
        echo "✗ Task $TASK_ID failed with exit code $EXIT_CODE"
        OVERALL_EXIT_CODE=$EXIT_CODE
        if [ "$IS_RANGE" = true ]; then
            echo "  Stopping sequential training due to failure"
            break
        fi
    fi
    echo "========================================="
done

echo ""
echo "========================================="
if [ "$IS_RANGE" = true ]; then
    if [ $OVERALL_EXIT_CODE -eq 0 ]; then
        echo "All tasks (${TASK_START} to ${TASK_END}) completed successfully!"
    else
        echo "Sequential training failed. Completed up to task $((TASK_ID - 1))"
    fi
else
    if [ $OVERALL_EXIT_CODE -eq 0 ]; then
        echo "Task $TASK_START completed successfully"
    else
        echo "Task $TASK_START failed"
    fi
fi
echo "Finished at: $(date)"
echo "========================================="

# Unset environment variable to avoid affecting subsequent runs
unset USE_CNN_UTILS

exit $OVERALL_EXIT_CODE
