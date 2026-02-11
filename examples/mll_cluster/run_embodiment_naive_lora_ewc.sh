#!/bin/bash
### Usage: bash examples/mll_cluster/run_embodiment_naive_lora_ewc.sh TASK_ID_OR_RANGE [CHECKPOINT_PATH] [MAX_EPOCH] [CONFIG_NAME] [SEED]
### Example (single task): bash examples/mll_cluster/run_embodiment_naive_lora_ewc.sh 0
### Example (task range): bash examples/mll_cluster/run_embodiment_naive_lora_ewc.sh "0,3"
### Example (with max_epoch): bash examples/mll_cluster/run_embodiment_naive_lora_ewc.sh 0 "" 15
### Example (continue from checkpoint): bash examples/mll_cluster/run_embodiment_naive_lora_ewc.sh 0 ./logs/naive_lora_ewc/task_0_seed1234/checkpoints/global_step_10/actor 20
### Example (with seed): bash examples/mll_cluster/run_embodiment_naive_lora_ewc.sh 0 "" "" "" 42
### Note: TASK_ID_OR_RANGE can be:
###       - A single task ID (e.g., "0") - trains that task only
###       - A tuple "a,b" where a < b (e.g., "0,3") - trains tasks from a to b sequentially
###       CHECKPOINT_PATH is optional and will be auto-generated from previous task if not provided
###       If CHECKPOINT_PATH is provided for a range, it will only be used for the first task
###       MAX_EPOCH is optional and can always be specified to override the default max_epochs
###       SEED is optional and defaults to 1234 if not provided

TASK_INPUT=${1:-0}
MANUAL_CHECKPOINT_PATH=$2
MAX_EPOCH=$3
CONFIG_NAME=${4:-mll_cluster/libero_spatial_grpo_openvlaoft}
SEED=${5:-1234}

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
mkdir -p logs/naive_lora_ewc

# Print job information (only if running under SLURM)
if [ -n "$SLURM_JOB_ID" ]; then
    echo "Job ID: $SLURM_JOB_ID"
    echo "Job Name: $SLURM_JOB_NAME"
    echo "Node: $SLURM_NODELIST"
    echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
    echo "GPUs allocated: $SLURM_GPUS_ON_NODE"
fi
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo ""

# Change to script directory if running under SLURM, otherwise use current directory
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    cd "$SLURM_SUBMIT_DIR"
else
    # Get the directory where this script is located
    SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
    REPO_ROOT=$(dirname $(dirname "$SCRIPT_DIR"))
    cd "$REPO_ROOT"
fi

# Source common functions
source "examples/mll_cluster/common_functions.sh"

# Extract config tag and derive eval config name
CONFIG_TAG=$(extract_config_tag "$CONFIG_NAME")
EVAL_CONFIG_NAME=$(derive_eval_config_name "$CONFIG_NAME")
FIRST_TASK_ID=$(get_first_task_id "$CONFIG_NAME")

# Main training loop
OVERALL_EXIT_CODE=0

for TASK_ID in $(seq $TASK_START $TASK_END); do
    echo ""
    echo "========================================="
    if [ "$IS_RANGE" = true ]; then
        echo "Sequential Training with EWC - Task ${TASK_ID} (${TASK_START} to ${TASK_END})"
    else
        echo "Lifelong Learning with EWC - Single Task Training (LoRA + EWC)"
    fi
    echo "========================================="
    
    # Determine checkpoint path
    USE_EXISTING_FIRST_TASK=false  # Initialize flag
    if [ "$TASK_ID" -eq "$TASK_START" ] && [ -n "$MANUAL_CHECKPOINT_PATH" ]; then
        # Use manual checkpoint path for first task if provided
        CHECKPOINT_PATH="$MANUAL_CHECKPOINT_PATH"
    elif [ "$TASK_ID" -eq $FIRST_TASK_ID ]; then
        # Check if first task weights already exist (from naive_lora training)
        PREV_LOG_DIR="./logs/naive_lora_ewc/task_${FIRST_TASK_ID}_seed${SEED}"
        # Inject config tag into PREV_LOG_DIR to match where previous task saved checkpoint
        if [ -n "$CONFIG_TAG" ]; then
            # CONFIG_TAG is set, transform the path
            PREV_LOG_DIR_TRANSFORMED=$(inject_config_tag_into_log_path "$PREV_LOG_DIR" "$CONFIG_TAG")
            # Validate transformation result
            if [ -z "$PREV_LOG_DIR_TRANSFORMED" ]; then
                echo "  ERROR: Failed to transform previous task log directory for task $FIRST_TASK_ID"
                echo "         Original PREV_LOG_DIR: [$PREV_LOG_DIR]"
                echo "         CONFIG_TAG: [$CONFIG_TAG]"
                OVERALL_EXIT_CODE=1
                break
            fi
        else
            # CONFIG_TAG is empty, use path as-is (no transformation needed)
            PREV_LOG_DIR_TRANSFORMED="$PREV_LOG_DIR"
        fi
        EXISTING_FIRST_TASK_CHECKPOINT="${PREV_LOG_DIR_TRANSFORMED}/checkpoints/global_step_10/actor"
        if [ -d "$EXISTING_FIRST_TASK_CHECKPOINT" ]; then
            # First task weights exist - load them and train for 1 epoch to generate rollouts for Fisher
            CHECKPOINT_PATH="$EXISTING_FIRST_TASK_CHECKPOINT"
            USE_EXISTING_FIRST_TASK=true
        else
            # First task in sequence, no checkpoint - train from scratch
            CHECKPOINT_PATH=""
        fi
    else
        # Use checkpoint from previous task
        PREV_TASK_ID=$((TASK_ID - 1))
        PREV_LOG_DIR="./logs/naive_lora_ewc/task_${PREV_TASK_ID}_seed${SEED}"
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
        
        # Use the transformed path
        CHECKPOINT_PATH="${PREV_LOG_DIR_TRANSFORMED}/checkpoints/global_step_10/actor"
        
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
    
    # Determine EWC path from previous task
    if [ "$TASK_ID" -gt $FIRST_TASK_ID ]; then
        PREV_TASK_ID=$((TASK_ID - 1))
        # Check standard path first
        PREV_LOG_DIR_STANDARD="./logs/naive_lora_ewc/task_${PREV_TASK_ID}_seed${SEED}"
        PREV_LOG_DIR_EXISTING="./logs/naive_lora_ewc/task_${PREV_TASK_ID}_from_existing_seed${SEED}"
        
        # Inject config tag into both paths
        if [ -n "$CONFIG_TAG" ]; then
            PREV_LOG_DIR_STANDARD=$(inject_config_tag_into_log_path "$PREV_LOG_DIR_STANDARD" "$CONFIG_TAG")
            PREV_LOG_DIR_EXISTING=$(inject_config_tag_into_log_path "$PREV_LOG_DIR_EXISTING" "$CONFIG_TAG")
        fi
        
        STANDARD_EWC_PATH="${PREV_LOG_DIR_STANDARD}/ewc_data.pt"
        EXISTING_EWC_PATH="${PREV_LOG_DIR_EXISTING}/ewc_data.pt"
        
        if [ -f "$STANDARD_EWC_PATH" ]; then
            EWC_PATH="$STANDARD_EWC_PATH"
        elif [ -f "$EXISTING_EWC_PATH" ]; then
            EWC_PATH="$EXISTING_EWC_PATH"
        else
            echo "  ERROR: EWC data not found for task $TASK_ID"
            echo "  Checked locations:"
            echo "    - $STANDARD_EWC_PATH"
            echo "    - $EXISTING_EWC_PATH"
            echo "  EWC data is required for tasks after the first task. Cannot continue."
            OVERALL_EXIT_CODE=1
            break
        fi
    else
        EWC_PATH=""
    fi
    
    # Determine LOG_DIR based on checkpoint path
    if [ "$TASK_ID" -eq "$TASK_START" ] && [ -n "$MANUAL_CHECKPOINT_PATH" ]; then
        # Extract task ID and global_step from checkpoint path
        # e.g., .../task_0/checkpoints/global_step_10/actor -> task_0, step_10
        if [[ "$CHECKPOINT_PATH" =~ task_([0-9]+) ]]; then
            SOURCE_TASK="${BASH_REMATCH[1]}"
            if [[ "$CHECKPOINT_PATH" =~ global_step_([0-9]+) ]]; then
                SOURCE_STEP="${BASH_REMATCH[1]}"
                LOG_DIR="./logs/naive_lora_ewc/task_${TASK_ID}_from_task_${SOURCE_TASK}_step_${SOURCE_STEP}_seed${SEED}"
            else
                echo "ERROR: Could not extract global_step from checkpoint path: $CHECKPOINT_PATH"
                echo "       Expected format: .../checkpoints/global_step_<M>/actor"
                OVERALL_EXIT_CODE=1
                break
            fi
        else
            echo "ERROR: Could not extract task ID from checkpoint path: $CHECKPOINT_PATH"
            echo "       Expected format: .../task_<N>/checkpoints/global_step_<M>/actor"
            OVERALL_EXIT_CODE=1
            break
        fi
    elif [ "$USE_EXISTING_FIRST_TASK" = true ]; then
        # Using existing first task weights - create a special log dir to indicate this
        LOG_DIR="./logs/naive_lora_ewc/task_${TASK_ID}_from_existing_seed${SEED}"
    else
        # Standard case: use TASK_ID
        LOG_DIR="./logs/naive_lora_ewc/task_${TASK_ID}_seed${SEED}"
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
    
    # Update SLURM job name to match experiment name (only for first task and if running under SLURM)
    if [ "$TASK_ID" -eq "$TASK_START" ] && [ -n "$SLURM_JOB_ID" ] && command -v scontrol &> /dev/null; then
        if [ "$IS_RANGE" = true ]; then
            if [ -n "$CONFIG_TAG" ]; then
                scontrol update job=$SLURM_JOB_ID name="ewc_lora_tasks_${TASK_START}_to_${TASK_END}_${CONFIG_TAG}" 2>/dev/null || true
            else
                scontrol update job=$SLURM_JOB_ID name="ewc_lora_tasks_${TASK_START}_to_${TASK_END}" 2>/dev/null || true
            fi
        else
            scontrol update job=$SLURM_JOB_ID name="${EXPERIMENT_NAME}" 2>/dev/null || true
        fi
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
    echo "  EWC: Enabled"
    
    if [ -n "$CHECKPOINT_PATH" ]; then
        if [ ! -d "$CHECKPOINT_PATH" ]; then
            echo "  ERROR: Checkpoint not found at $CHECKPOINT_PATH"
            OVERALL_EXIT_CODE=1
            break
        fi
        if [ "$USE_EXISTING_FIRST_TASK" = true ]; then
            echo "  Loading existing first task weights from: $CHECKPOINT_PATH"
            echo "  (Will train for 1 epoch to generate rollouts for Fisher computation)"
        else
            echo "  Loading from checkpoint: $CHECKPOINT_PATH"
        fi
    else
        echo "  Training from base model (SFT checkpoint) - First task (task $FIRST_TASK_ID)"
    fi
    
    if [ -n "$EWC_PATH" ]; then
        echo "  Loading EWC data from: $EWC_PATH"
    else
        echo "  No EWC data (first task or EWC data not found)"
    fi
    
    # Handle max_epochs: if using existing first task, train for 1 epoch only (unless overridden)
    if [ "$USE_EXISTING_FIRST_TASK" = true ] && [ -z "$MAX_EPOCH" ]; then
        # Using existing first task weights - train for 1 epoch to generate rollouts for Fisher
        MAX_EPOCH=1
        echo "  Max epochs: 1 (using existing weights, only need rollouts for Fisher computation)"
    elif [ -n "$MAX_EPOCH" ]; then
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
    OVERRIDES="env.fixed_task_ids=[${TASK_ID}] \
    	runner.logger.experiment_name=${EXPERIMENT_NAME} \
    	actor.seed=${SEED} \
    	+algorithm.use_ewc=True"
    
    if [ -n "$CHECKPOINT_PATH" ]; then
        OVERRIDES="$OVERRIDES +actor.model.lora_path=${CHECKPOINT_PATH}"
    fi
    
    if [ -n "$EWC_PATH" ]; then
        OVERRIDES="$OVERRIDES +algorithm.previous_task_ewc_path=${EWC_PATH}"
    fi
    
    if [ -n "$MAX_EPOCH" ]; then
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
        echo "EWC data (if enabled) saved to: ${LOG_DIR}/ewc_data.pt"
        
        # Run evaluation
        # LOG_DIR already has the config tag injected, so use it directly
        CHECKPOINT_LOCATION=$(echo "$LOG_DIR" | sed 's|^\./||')
        echo ""
        echo "Running evaluation for: ${CHECKPOINT_LOCATION}"
        bash examples/mll_cluster/eval_embodiment.sh "${CHECKPOINT_LOCATION}" "" "${EVAL_CONFIG_NAME}"
    else
        echo "✗ Task $TASK_ID failed with exit code $EXIT_CODE"
        if [ -n "$SLURM_JOB_ID" ]; then
            echo "  Check logs at: logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
        fi
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

exit $OVERALL_EXIT_CODE
