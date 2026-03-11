#!/bin/bash
### Usage: bash examples/crl_experiment/run_embodiment_multitask.sh TASK_IDS [CHECKPOINT_PATH] [MAX_EPOCH] [CONFIG_NAME] [SEED]
### Example: bash examples/crl_experiment/run_embodiment_multitask.sh "0,2,4"
### Example: bash examples/crl_experiment/run_embodiment_multitask.sh "0 2 4"
### Example (with max_epoch): bash examples/crl_experiment/run_embodiment_multitask.sh "0,2,4" "" 15
### Example (continue from checkpoint): bash examples/crl_experiment/run_embodiment_multitask.sh "0,2,4" ./logs/multitask/tasks_0_2_4_seed1234/checkpoints/global_step_10/actor 20
### Example (with seed): bash examples/crl_experiment/run_embodiment_multitask.sh "0,2,4" "" "" "" 42
### Note: TASK_IDS can be comma-separated (e.g., "0,2,4") or space-separated (e.g., "0 2 4")
###       CHECKPOINT_PATH is optional and will load model weights (LoRA adapter)
###       MAX_EPOCH is optional and can override the default max_epochs
###       SEED is optional and defaults to 1234 if not provided

TASK_IDS_STR=$1
CHECKPOINT_PATH=$2
MAX_EPOCH=$3
CONFIG_NAME=${4:-crl_experiment/libero_spatial_grpo_openvlaoft_spatial}
SEED=${5:-1234}

# Log subdirectory for this experiment type
EXPERIMENT_TYPE="multitask"

if [ -z "$TASK_IDS_STR" ]; then
    echo "ERROR: Missing required argument"
    echo "Usage: bash examples/crl_experiment/run_embodiment_multitask.sh TASK_IDS [CHECKPOINT_PATH] [MAX_EPOCH] [CONFIG_NAME] [SEED]"
    echo "Example: bash examples/crl_experiment/run_embodiment_multitask.sh \"0,2,4\""
    exit 1
fi

if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
    echo "ERROR: SEED must be a non-negative integer, got: $SEED"
    exit 1
fi

# Parse TASK_IDS: convert comma-separated to space-separated, then to array
TASK_IDS_STR=$(echo "$TASK_IDS_STR" | tr ',' ' ')
read -ra TASK_IDS_ARRAY <<< "$TASK_IDS_STR"

for task_id in "${TASK_IDS_ARRAY[@]}"; do
    if ! [[ "$task_id" =~ ^[0-9]+$ ]]; then
        echo "ERROR: Invalid task ID: $task_id (must be a non-negative integer)"
        exit 1
    fi
done

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

# Sort task IDs and create formatted strings
IFS=$'\n' sorted_task_ids=($(sort -n <<<"${TASK_IDS_ARRAY[*]}"))
unset IFS

TASK_LIST_STR=$(IFS=','; echo "${sorted_task_ids[*]}")
TASK_DIR_STR=$(IFS='_'; echo "${sorted_task_ids[*]}")

# Determine LOG_DIR
if [ -z "$LOG_DIR" ]; then
    if [ -n "$CHECKPOINT_PATH" ]; then
        if [[ "$CHECKPOINT_PATH" =~ global_step_([0-9]+) ]]; then
            SOURCE_STEP="${BASH_REMATCH[1]}"
            LOG_DIR="./logs/${EXPERIMENT_TYPE}/tasks_${TASK_DIR_STR}_from_checkpoint_step_${SOURCE_STEP}_seed${SEED}"
        else
            LOG_DIR="./logs/${EXPERIMENT_TYPE}/tasks_${TASK_DIR_STR}_from_checkpoint_seed${SEED}"
        fi
    else
        LOG_DIR="./logs/${EXPERIMENT_TYPE}/tasks_${TASK_DIR_STR}_seed${SEED}"
    fi
    if [ -n "$CONFIG_TAG" ]; then
        LOG_DIR_TRANSFORMED=$(inject_config_tag_into_log_path "$LOG_DIR" "$CONFIG_TAG")
        if [ -n "$LOG_DIR_TRANSFORMED" ]; then
            LOG_DIR="$LOG_DIR_TRANSFORMED"
        fi
    fi
fi

export LOG_DIR

EXPERIMENT_NAME=$(basename "$LOG_DIR")
if [ -n "$CONFIG_TAG" ]; then
    EXPERIMENT_NAME="${EXPERIMENT_NAME}_${CONFIG_TAG}"
fi

echo "========================================="
echo "Multi-Task Training"
echo "========================================="
echo "Configuration:"
echo "  Task IDs: ${sorted_task_ids[*]}"
echo "  Tasks (formatted): [${TASK_LIST_STR}]"
echo "  Experiment Name: $EXPERIMENT_NAME"
echo "  Experiment Type: $EXPERIMENT_TYPE"
echo "  Checkpoint Save Path: $LOG_DIR"
echo "  Config Name: $CONFIG_NAME"
echo "  Random Seed: $SEED"

if [ -n "$CHECKPOINT_PATH" ]; then
    if [ ! -d "$CHECKPOINT_PATH" ]; then
        echo "  ERROR: Checkpoint not found at $CHECKPOINT_PATH"
        exit 1
    fi
    echo "  Loading from checkpoint: $CHECKPOINT_PATH"
else
    echo "  Training from base model - Multi-task setup (SFT checkpoint)"
fi

if [ -n "$MAX_EPOCH" ]; then
    if ! [[ "$MAX_EPOCH" =~ ^[0-9]+$ ]] || [ "$MAX_EPOCH" -le 0 ]; then
        echo "  ERROR: MAX_EPOCH must be a positive integer, got: $MAX_EPOCH"
        exit 1
    fi
    echo "  Max epochs: $MAX_EPOCH"
fi

echo "========================================="
echo ""

# Build Hydra overrides
OVERRIDES="env.fixed_task_ids=[${TASK_LIST_STR}] runner.logger.experiment_name=${EXPERIMENT_NAME} actor.seed=${SEED}"

if [ -n "$CHECKPOINT_PATH" ]; then
    OVERRIDES="$OVERRIDES +actor.model.lora_path=${CHECKPOINT_PATH}"
fi

if [ -n "$MAX_EPOCH" ]; then
    OVERRIDES="$OVERRIDES runner.max_epochs=${MAX_EPOCH}"
fi

echo "Running with Hydra overrides:"
echo "$OVERRIDES"
echo ""

bash examples/embodiment/run_embodiment.sh "${CONFIG_NAME}" ${OVERRIDES}

EXIT_CODE=$?
echo ""
echo "========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Multi-task training completed successfully"
    echo ""
    echo "Tasks trained: ${sorted_task_ids[*]}"
    echo "Checkpoint saved to: ${LOG_DIR}"
else
    echo "Multi-task training failed with exit code $EXIT_CODE"
    echo "  Tasks: ${sorted_task_ids[*]}"
fi
echo "Finished at: $(date)"
echo "========================================="

exit $EXIT_CODE
