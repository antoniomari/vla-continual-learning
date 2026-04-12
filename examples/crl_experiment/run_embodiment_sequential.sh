#!/bin/bash
### Usage: bash examples/crl_experiment/run_embodiment_sequential.sh TASK_ID_OR_RANGE [CHECKPOINT_PATH] [MAX_EPOCH] [CONFIG_NAME] [SEED]
### Example (single task): bash examples/crl_experiment/run_embodiment_sequential.sh 0
### Example (task range): bash examples/crl_experiment/run_embodiment_sequential.sh "0,3"
### Example (with max_epoch): bash examples/crl_experiment/run_embodiment_sequential.sh 0 "" 15
### Example (continue from checkpoint): bash examples/crl_experiment/run_embodiment_sequential.sh 0 ./logs/sequential/task_0_seed1234/checkpoints/global_step_10/actor 20
### Example (with seed): bash examples/crl_experiment/run_embodiment_sequential.sh 0 "" "" "" 42
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
CONFIG_NAME=${4:-crl_experiment/libero_spatial_grpo_openvlaoft_spatial}
SEED=${5:-1234}

# Log subdirectory for this experiment type
EXPERIMENT_TYPE="sequential"

# Parse TASK_INPUT to determine if it's a single task or a range
if [[ "$TASK_INPUT" == *,* ]]; then
    IFS=',' read -r TASK_START TASK_END <<< "$TASK_INPUT"
    TASK_START=$(echo "$TASK_START" | tr -d '()[] ')
    TASK_END=$(echo "$TASK_END" | tr -d '()[] ')

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

if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
    echo "ERROR: SEED must be a non-negative integer, got: $SEED"
    exit 1
fi

mkdir -p "logs/${EXPERIMENT_TYPE}"

echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo ""

# Change to repo root
SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT=$(dirname $(dirname "$SCRIPT_DIR"))
cd "$REPO_ROOT"

# Source common functions
source "examples/crl_experiment/common_functions.sh"

# Extract config tag and derive eval config name
CONFIG_TAG=$(extract_config_tag "$CONFIG_NAME")
EVAL_CONFIG_NAME=$(derive_eval_config_name "$CONFIG_NAME")
GLOBAL_STEP=$(get_default_global_step "$CONFIG_NAME")
FIRST_TASK_ID=$(get_first_task_id "$CONFIG_NAME")

# Main training loop
OVERALL_EXIT_CODE=0

for TASK_ID in $(seq $TASK_START $TASK_END); do
    echo ""
    echo "========================================="
    if [ "$IS_RANGE" = true ]; then
        echo "Sequential Training - Task ${TASK_ID} (${TASK_START} to ${TASK_END})"
    else
        echo "Sequential Training - Single Task (LoRA)"
    fi
    echo "========================================="

    # Determine checkpoint path
    if [ "$TASK_ID" -eq "$TASK_START" ] && [ -n "$MANUAL_CHECKPOINT_PATH" ]; then
        CHECKPOINT_PATH="$MANUAL_CHECKPOINT_PATH"
    elif [ "$TASK_ID" -eq $FIRST_TASK_ID ]; then
        CHECKPOINT_PATH=""
    else
        PREV_TASK_ID=$((TASK_ID - 1))
        PREV_LOG_DIR="./logs/${EXPERIMENT_TYPE}/task_${PREV_TASK_ID}_seed${SEED}"
        if [ -n "$CONFIG_TAG" ]; then
            PREV_LOG_DIR_TRANSFORMED=$(inject_config_tag_into_log_path "$PREV_LOG_DIR" "$CONFIG_TAG")
            if [ -z "$PREV_LOG_DIR_TRANSFORMED" ]; then
                echo "  ERROR: Failed to transform previous task log directory for task $PREV_TASK_ID"
                echo "         Original PREV_LOG_DIR: [$PREV_LOG_DIR]"
                echo "         CONFIG_TAG: [$CONFIG_TAG]"
                OVERALL_EXIT_CODE=1
                break
            fi
        else
            PREV_LOG_DIR_TRANSFORMED="$PREV_LOG_DIR"
        fi
        CHECKPOINT_PATH="${PREV_LOG_DIR_TRANSFORMED}/checkpoints/global_step_${GLOBAL_STEP}/actor"

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
        if [[ "$CHECKPOINT_PATH" =~ task_([0-9]+) ]]; then
            SOURCE_TASK="${BASH_REMATCH[1]}"
            if [[ "$CHECKPOINT_PATH" =~ global_step_([0-9]+) ]]; then
                SOURCE_STEP="${BASH_REMATCH[1]}"
                LOG_DIR="./logs/${EXPERIMENT_TYPE}/task_${TASK_ID}_from_task_${SOURCE_TASK}_step_${SOURCE_STEP}_seed${SEED}"
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
    else
        LOG_DIR="./logs/${EXPERIMENT_TYPE}/task_${TASK_ID}_seed${SEED}"
    fi

    # Inject config tag into LOG_DIR
    if [ -n "$CONFIG_TAG" ]; then
        LOG_DIR_TRANSFORMED=$(inject_config_tag_into_log_path "$LOG_DIR" "$CONFIG_TAG")
        if [ -z "$LOG_DIR_TRANSFORMED" ]; then
            echo "  ERROR: Failed to transform LOG_DIR with config tag"
            echo "         Original LOG_DIR: [$LOG_DIR]"
            echo "         CONFIG_TAG: [$CONFIG_TAG]"
            OVERALL_EXIT_CODE=1
            break
        fi
        LOG_DIR="$LOG_DIR_TRANSFORMED"
    fi

    if [ -z "$LOG_DIR" ]; then
        echo "  ERROR: LOG_DIR is empty after path construction"
        OVERALL_EXIT_CODE=1
        break
    fi

    export LOG_DIR
    mkdir -p "${LOG_DIR}"

    EXPERIMENT_NAME=$(basename "$LOG_DIR")
    if [ -n "$CONFIG_TAG" ]; then
        EXPERIMENT_NAME="${EXPERIMENT_NAME}_${CONFIG_TAG}"
    fi

    echo "Configuration:"
    echo "  Task ID: $TASK_ID"
    if [ "$IS_RANGE" = true ]; then
        echo "  Task Range: ${TASK_START} to ${TASK_END}"
    fi
    echo "  Experiment Name: $EXPERIMENT_NAME"
    echo "  Experiment Type: $EXPERIMENT_TYPE"
    echo "  Checkpoint Save Path: $LOG_DIR"
    echo "  Config Name: $CONFIG_NAME"
    echo "  Random Seed: $SEED"

    if [ -n "$CHECKPOINT_PATH" ]; then
        if [ ! -d "$CHECKPOINT_PATH" ]; then
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
        echo "  Training from base model (SFT checkpoint) - First task (task $FIRST_TASK_ID)"
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
    OVERRIDES="env.fixed_task_ids=[${TASK_ID}] \
    	runner.logger.experiment_name=${EXPERIMENT_NAME} \
    	actor.seed=${SEED}"

    if [ -n "$CHECKPOINT_PATH" ]; then
        OVERRIDES="$OVERRIDES +actor.model.lora_path=${CHECKPOINT_PATH}"
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

        CHECKPOINT_LOCATION=$(echo "$LOG_DIR" | sed 's|^\./||')
        echo ""
        echo "Running evaluation for: ${CHECKPOINT_LOCATION}"
        bash examples/crl_experiment/eval_embodiment.sh "${CHECKPOINT_LOCATION}" "" "${EVAL_CONFIG_NAME}"
    else
        echo "Task $TASK_ID failed with exit code $EXIT_CODE"
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
