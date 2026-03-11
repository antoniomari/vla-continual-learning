#!/bin/bash
### NOTE: this script is meant for sequential launching of tasks in hand-crafted order.
###       The first task will by default load from base model. If INIT_CHECKPOINT is provided, it will load from the checkpoint.
### Usage: bash examples/crl_experiment/run_embodiment_sequential_reorder.sh TASK_IDS [RUN_ID] [MAX_EPOCH] [CONFIG_NAME] [SEED] [INIT_CHECKPOINT]
### Example: bash examples/crl_experiment/run_embodiment_sequential_reorder.sh "0,1,2"
### Example: bash examples/crl_experiment/run_embodiment_sequential_reorder.sh "0 1 2"
### Example (with run_id): bash examples/crl_experiment/run_embodiment_sequential_reorder.sh "0,1,2" "run1"
### Example (with max_epoch): bash examples/crl_experiment/run_embodiment_sequential_reorder.sh "0,1,2" "" 15
### Example (with seed): bash examples/crl_experiment/run_embodiment_sequential_reorder.sh "0,1,2" "" "" "" 42
### Example (with initial checkpoint): bash examples/crl_experiment/run_embodiment_sequential_reorder.sh "0,1,2" "" "" "" 42 logs/sequential_reorder/seq_0_1_2_seed42/task_0/checkpoints/global_step_10/actor
### Note: TASK_IDS can be comma-separated (e.g., "0,1,2") or space-separated (e.g., "0 1 2")
###       IMPORTANT: Order matters! Tasks are trained in the exact order provided
###       RUN_ID is optional - if not provided, uses task sequence (e.g., "0_1_2") to identify the sequence run
###       INIT_CHECKPOINT is optional; if provided, the first task will load from this checkpoint instead of the base model

TASK_IDS_STR=$1
RUN_ID=$2
MAX_EPOCH=$3
CONFIG_NAME=${4:-crl_experiment/libero_spatial_grpo_openvlaoft_spatial}
SEED=${5:-1234}
INIT_CHECKPOINT=$6

# Log subdirectory for this experiment type
EXPERIMENT_TYPE="sequential_reorder"

if [ -z "$TASK_IDS_STR" ]; then
    echo "ERROR: Missing required argument"
    echo "Usage: bash examples/crl_experiment/run_embodiment_sequential_reorder.sh TASK_IDS [RUN_ID] [MAX_EPOCH] [CONFIG_NAME] [SEED] [INIT_CHECKPOINT]"
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

TASK_SEQ_STR=$(IFS='_'; echo "${TASK_IDS_ARRAY[*]}")

if [ -n "$INIT_CHECKPOINT" ] && [ ! -d "$INIT_CHECKPOINT" ]; then
    echo "  ERROR: INIT_CHECKPOINT directory not found at: $INIT_CHECKPOINT"
    exit 1
fi

if [ -z "$RUN_ID" ]; then
    RUN_ID="seq_${TASK_SEQ_STR}"
fi

if [ -n "$INIT_CHECKPOINT" ]; then
    if [[ "$INIT_CHECKPOINT" =~ task_([0-9]+) ]]; then
        SOURCE_TASK="${BASH_REMATCH[1]}"
        if [[ "$INIT_CHECKPOINT" =~ global_step_([0-9]+) ]]; then
            SOURCE_STEP="${BASH_REMATCH[1]}"
            RUN_ID="${RUN_ID}_from_task_${SOURCE_TASK}_step_${SOURCE_STEP}"
        fi
    fi
fi

mkdir -p "logs/${EXPERIMENT_TYPE}"

echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo ""

# Change to repo root
SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT=$(dirname $(dirname "$SCRIPT_DIR"))
cd "$REPO_ROOT"

source "examples/crl_experiment/common_functions.sh"

CONFIG_TAG=$(extract_config_tag "$CONFIG_NAME")
EVAL_CONFIG_NAME=$(derive_eval_config_name "$CONFIG_NAME")
GLOBAL_STEP=$(get_default_global_step "$CONFIG_NAME")

echo "========================================="
echo "Sequential Task Training (Custom Order)"
echo "========================================="
echo "Configuration:"
echo "  Task Sequence (order preserved): ${TASK_IDS_ARRAY[*]}"
echo "  Run ID: $RUN_ID"
echo "  Experiment Type: $EXPERIMENT_TYPE"
echo "  Config Name: $CONFIG_NAME"
echo "  Random Seed: $SEED"

if [ -n "$MAX_EPOCH" ]; then
    if ! [[ "$MAX_EPOCH" =~ ^[0-9]+$ ]] || [ "$MAX_EPOCH" -le 0 ]; then
        echo "  ERROR: MAX_EPOCH must be a positive integer, got: $MAX_EPOCH"
        exit 1
    fi
    echo "  Max epochs per task: $MAX_EPOCH"
fi

echo "========================================="
echo ""

OVERALL_EXIT_CODE=0
PREV_CHECKPOINT_PATH=""
NUM_TASKS=${#TASK_IDS_ARRAY[@]}

for i in "${!TASK_IDS_ARRAY[@]}"; do
    TASK_ID=${TASK_IDS_ARRAY[$i]}
    TASK_NUM=$((i + 1))

    echo ""
    echo "========================================="
    echo "Training Task ${TASK_ID} (${TASK_NUM}/${NUM_TASKS} in sequence)"
    echo "Run ID: $RUN_ID"
    echo "========================================="

    TASK_LOG_DIR="./logs/${EXPERIMENT_TYPE}/${RUN_ID}_seed${SEED}/task_${TASK_ID}"
    if [ -n "$CONFIG_TAG" ]; then
        TASK_LOG_DIR=$(inject_config_tag_into_log_path "$TASK_LOG_DIR" "$CONFIG_TAG")
    fi
    mkdir -p "${TASK_LOG_DIR}"
    export LOG_DIR="${TASK_LOG_DIR}"

    EXPERIMENT_NAME=$(basename "$TASK_LOG_DIR")
    if [ -n "$CONFIG_TAG" ]; then
        EXPERIMENT_NAME="${EXPERIMENT_NAME}_${CONFIG_TAG}"
    fi

    echo "Configuration:"
    echo "  Task ID: $TASK_ID"
    echo "  Task Position: ${TASK_NUM}/${NUM_TASKS}"
    echo "  Experiment Name: $EXPERIMENT_NAME"
    echo "  Experiment Type: $EXPERIMENT_TYPE"
    echo "  Checkpoint Save Path: $TASK_LOG_DIR"
    echo "  Config Name: $CONFIG_NAME"
    echo "  Random Seed: $SEED"

    if [ $TASK_NUM -eq 1 ]; then
        if [ -n "$INIT_CHECKPOINT" ]; then
            CHECKPOINT_PATH="$INIT_CHECKPOINT"
            echo "  Loading from: Initial checkpoint (provided via argument)"
        else
            CHECKPOINT_PATH=""
            echo "  Loading from: Base model (SFT checkpoint) - First task in sequence"
        fi
    else
        CHECKPOINT_PATH="$PREV_CHECKPOINT_PATH"
        if [ ! -d "$CHECKPOINT_PATH" ]; then
            echo "  ERROR: Previous checkpoint not found at $CHECKPOINT_PATH"
            OVERALL_EXIT_CODE=1
            break
        fi
        echo "  Loading from: Previous task checkpoint"
    fi
    echo "  Checkpoint path: $CHECKPOINT_PATH"

    if [ -n "$MAX_EPOCH" ]; then
        echo "  Max epochs: $MAX_EPOCH"
    fi

    echo "========================================="
    echo ""

    OVERRIDES="env.fixed_task_ids=[${TASK_ID}] \
    	runner.logger.experiment_name=${EXPERIMENT_NAME} \
    	actor.seed=${SEED}"
    [ -n "$CHECKPOINT_PATH" ] && OVERRIDES="$OVERRIDES +actor.model.lora_path=${CHECKPOINT_PATH}"
    [ -n "$MAX_EPOCH" ] && OVERRIDES="$OVERRIDES runner.max_epochs=${MAX_EPOCH}"

    echo "Running with Hydra overrides:"
    echo "$OVERRIDES"
    echo ""

    bash examples/embodiment/run_embodiment.sh ${CONFIG_NAME} $OVERRIDES

    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo ""
        echo "========================================="
        echo "Task $TASK_ID failed with exit code $EXIT_CODE"
        OVERALL_EXIT_CODE=$EXIT_CODE
        break
    fi

    PREV_CHECKPOINT_PATH="${TASK_LOG_DIR}/checkpoints/global_step_${GLOBAL_STEP}/actor"

    echo ""
    echo "========================================="
    echo "Task $TASK_ID completed successfully"
    echo "  Checkpoint saved to: ${TASK_LOG_DIR}"
    echo "========================================="

    CHECKPOINT_LOCATION=$(echo "$TASK_LOG_DIR" | sed 's|^\./||')
    echo ""
    echo "Running evaluation for task $TASK_ID..."
    bash examples/crl_experiment/eval_embodiment.sh "${CHECKPOINT_LOCATION}" "" "${EVAL_CONFIG_NAME}"

    EVAL_EXIT_CODE=$?
    if [ $EVAL_EXIT_CODE -ne 0 ]; then
        echo "  ERROR: Evaluation for task $TASK_ID failed, but continuing sequence..."
        OVERALL_EXIT_CODE=$EVAL_EXIT_CODE
        break
    else
        echo "  Evaluation for task $TASK_ID completed"
    fi
    echo ""
done

echo ""
echo "========================================="
if [ $OVERALL_EXIT_CODE -eq 0 ]; then
    echo "Sequential training completed successfully"
    echo ""
    echo "Tasks trained (in order): ${TASK_IDS_ARRAY[*]}"
    echo "Run ID: $RUN_ID"
    echo "Results directory: ./logs/${EXPERIMENT_TYPE}/${RUN_ID}_seed${SEED}/"
else
    echo "Sequential training failed"
fi
echo "Finished at: $(date)"
echo "========================================="

exit $OVERALL_EXIT_CODE
