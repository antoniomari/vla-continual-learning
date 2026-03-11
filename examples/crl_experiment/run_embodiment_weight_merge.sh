#!/bin/bash
### Usage: bash examples/crl_experiment/run_embodiment_weight_merge.sh TASK_ID_OR_RANGE [MERGE_COEFFICIENT] [CONFIG_NAME] [SEED]
### Example (single task): bash examples/crl_experiment/run_embodiment_weight_merge.sh 0
### Example (single task): bash examples/crl_experiment/run_embodiment_weight_merge.sh 1 0.8
### Example (with seed):   bash examples/crl_experiment/run_embodiment_weight_merge.sh 0 "" "" 42
### Example (range):       bash examples/crl_experiment/run_embodiment_weight_merge.sh "0,3" 0.8
###
### Behavior:
###   - First task ID: Train first LoRA adapter on top of base model (same as sequential).
###   - Subsequent tasks: Load and merge ALL previous task adapters, then create a NEW LoRA adapter.
### MERGE_COEFFICIENT (optional, default=0.5): Coefficient applied when merging LoRA adapters (0.0 to 1.0)
### SEED (optional, default=1234): Random seed for reproducibility

TASK_INPUT=${1:-}
MERGE_COEFFICIENT=$2
CONFIG_NAME=${3:-crl_experiment/libero_spatial_grpo_openvlaoft_spatial}
SEED=${4:-1234}

# Log subdirectory for this experiment type
EXPERIMENT_TYPE="weight_merge"

# Change to repo root
SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT=$(dirname $(dirname "$SCRIPT_DIR"))
cd "$REPO_ROOT"

source "examples/crl_experiment/common_functions.sh"

CONFIG_TAG=$(extract_config_tag "$CONFIG_NAME")
EVAL_CONFIG_NAME=$(derive_eval_config_name "$CONFIG_NAME")
GLOBAL_STEP=$(get_default_global_step "$CONFIG_NAME")
FIRST_TASK_ID=$(get_first_task_id "$CONFIG_NAME")

TASK_INPUT=${TASK_INPUT:-$FIRST_TASK_ID}

# Parse TASK_ID_OR_RANGE
if [[ "$TASK_INPUT" == *,* ]]; then
    IFS=',' read -r TASK_START TASK_END <<< "$TASK_INPUT"
    TASK_START=$(echo "$TASK_START" | tr -d '()[] ')
    TASK_END=$(echo "$TASK_END" | tr -d '()[] ')
    if ! [[ "$TASK_START" =~ ^[0-9]+$ ]] || ! [[ "$TASK_END" =~ ^[0-9]+$ ]]; then
        echo "ERROR: Task range must contain two numeric values: \"a,b\" where a and b are integers"
        exit 1
    fi
    if [ "$TASK_START" -ge "$TASK_END" ]; then
        echo "ERROR: First task ID ($TASK_START) must be smaller than second task ID ($TASK_END)"
        exit 1
    fi
    IS_RANGE=true
else
    if ! [[ "$TASK_INPUT" =~ ^[0-9]+$ ]]; then
        echo "ERROR: Task ID must be a numeric value"
        exit 1
    fi
    IS_RANGE=false
    TASK_START=$TASK_INPUT
    TASK_END=$TASK_INPUT
fi

if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
    echo "ERROR: SEED must be a non-negative integer, got: $SEED"
    exit 1
fi

# Validate MERGE_COEFFICIENT if provided
if [ -n "$MERGE_COEFFICIENT" ]; then
    if ! [[ "$MERGE_COEFFICIENT" =~ ^[0-9]+\.?[0-9]*$ ]]; then
        echo "ERROR: MERGE_COEFFICIENT must be a number, got: $MERGE_COEFFICIENT"
        exit 1
    fi
    if ! awk "BEGIN {exit !($MERGE_COEFFICIENT >= 0 && $MERGE_COEFFICIENT <= 1)}"; then
        echo "ERROR: MERGE_COEFFICIENT must be between 0.0 and 1.0, got: $MERGE_COEFFICIENT"
        exit 1
    fi
fi

if [ -z "$MERGE_COEFFICIENT" ]; then
    MERGE_COEFFICIENT="0.5"
fi

COEFFICIENT_PATH=$(echo "$MERGE_COEFFICIENT" | tr '.' '_')

mkdir -p "logs/${EXPERIMENT_TYPE}"

echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo ""

OVERALL_EXIT_CODE=0

for TASK_ID in $(seq $TASK_START $TASK_END); do
    if [ "$TASK_ID" -eq $FIRST_TASK_ID ]; then
        LOG_DIR="./logs/${EXPERIMENT_TYPE}/task_${TASK_ID}_seed${SEED}"
        EXPERIMENT_NAME="${EXPERIMENT_TYPE}_task_${TASK_ID}_seed${SEED}"
    else
        LOG_DIR="./logs/${EXPERIMENT_TYPE}/task_${TASK_ID}_coeff_${COEFFICIENT_PATH}_seed${SEED}"
        EXPERIMENT_NAME="${EXPERIMENT_TYPE}_task_${TASK_ID}_coeff_${COEFFICIENT_PATH}_seed${SEED}"
    fi

    if [ -n "$CONFIG_TAG" ]; then
        EXPERIMENT_NAME="${EXPERIMENT_NAME}_${CONFIG_TAG}"
        LOG_DIR_TRANSFORMED=$(inject_config_tag_into_log_path "$LOG_DIR" "$CONFIG_TAG")
        if [ -z "$LOG_DIR_TRANSFORMED" ]; then
            echo "  ERROR: Failed to transform LOG_DIR with config tag"
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
    mkdir -p "$LOG_DIR"

    echo "========================================="
    echo "Weight Merge Training"
    echo "========================================="
    echo "Configuration:"
    echo "  Task ID: $TASK_ID"
    if [ "$IS_RANGE" = true ]; then
        echo "  Task Range: ${TASK_START} to ${TASK_END}"
    fi
    echo "  Experiment Name: $EXPERIMENT_NAME"
    echo "  Experiment Type: $EXPERIMENT_TYPE"
    echo "  Checkpoint Save Path: $LOG_DIR"
    echo "  Config Name: $CONFIG_NAME"
    echo "  Merge Coefficient: ${MERGE_COEFFICIENT}"
    echo "  Random Seed: $SEED"

    LORA_PATHS_ARG=""
    MERGE_COEFFICIENT_ARG="+actor.model.previous_lora_merge_coefficient=${MERGE_COEFFICIENT}"

    if [ "$TASK_ID" -gt $FIRST_TASK_ID ]; then
        echo "  Mode: WEIGHT MERGE - Loading and merging all previous task adapters ($FIRST_TASK_ID..$((TASK_ID - 1)))"
        LORA_PATHS=()
        for prev_task in $(seq $FIRST_TASK_ID $((TASK_ID - 1))); do
            if [ "$prev_task" -eq $FIRST_TASK_ID ]; then
                if [ -z "$CONFIG_TAG" ]; then
                    prev_log_dir="./logs/sequential/task_${prev_task}_seed${SEED}"
                else
                    prev_log_dir="./logs/${EXPERIMENT_TYPE}/task_${prev_task}_seed${SEED}"
                fi
            else
                prev_log_dir="./logs/${EXPERIMENT_TYPE}/task_${prev_task}_coeff_${COEFFICIENT_PATH}_seed${SEED}"
            fi
            if [ -n "$CONFIG_TAG" ]; then
                prev_log_dir_transformed=$(inject_config_tag_into_log_path "$prev_log_dir" "$CONFIG_TAG")
            else
                prev_log_dir_transformed="$prev_log_dir"
            fi
            if [ -z "$prev_log_dir_transformed" ]; then
                echo "    ERROR: Failed to construct log directory for task $prev_task"
                continue
            fi
            prev_log_dir="$prev_log_dir_transformed"
            prev_adapter_path="${prev_log_dir}/checkpoints/global_step_${GLOBAL_STEP}/actor"
            if [[ "$prev_adapter_path" =~ ^/checkpoints/ ]]; then
                echo "    ERROR: Invalid adapter path construction detected for task $prev_task"
                continue
            fi
            if [ -d "$prev_adapter_path" ]; then
                LORA_PATHS+=("$prev_adapter_path")
                echo "    Found adapter for task $prev_task: $prev_adapter_path"
            else
                echo "    ERROR: Adapter not found for task $prev_task: $prev_adapter_path"
                OVERALL_EXIT_CODE=1
                break 2
            fi
        done

        if [ ${#LORA_PATHS[@]} -gt 0 ]; then
            LORA_PATHS_STR=$(IFS=','; echo "${LORA_PATHS[*]}")
            LORA_PATHS_ARG="+actor.model.lora_paths=[${LORA_PATHS_STR}]"
        else
            echo "  ERROR: No previous adapters found; cannot proceed."
            OVERALL_EXIT_CODE=1
            break
        fi
    else
        echo "  Mode: FIRST TASK - Training first LoRA adapter on base model."
    fi

    echo "========================================="
    echo ""

    OVERRIDES="env.fixed_task_ids=[${TASK_ID}] \
    	runner.logger.experiment_name=${EXPERIMENT_NAME} \
    	actor.seed=${SEED}"
    [ -n "$LORA_PATHS_ARG" ] && OVERRIDES="$OVERRIDES ${LORA_PATHS_ARG}"
    [ -n "$MERGE_COEFFICIENT_ARG" ] && OVERRIDES="$OVERRIDES ${MERGE_COEFFICIENT_ARG}"

    echo "Running with Hydra overrides:"
    echo "$OVERRIDES"
    echo ""

    bash examples/embodiment/run_embodiment.sh ${CONFIG_NAME} $OVERRIDES

    EXIT_CODE=$?
    echo ""
    echo "========================================="
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Task $TASK_ID completed successfully (Weight Merge)"
        echo ""
        echo "Checkpoint saved to: ${LOG_DIR}"

        EVAL_LORA_PATH="${LOG_DIR}/checkpoints/global_step_${GLOBAL_STEP}/actor"
        echo ""
        echo "Running evaluation for Weight Merge checkpoint:"
        EVAL_OVERRIDES="+actor.model.lora_path=${EVAL_LORA_PATH} +actor.model.lora_scale=${MERGE_COEFFICIENT}"
        [ -n "$LORA_PATHS_ARG" ] && EVAL_OVERRIDES="${EVAL_OVERRIDES} ${LORA_PATHS_ARG} ${MERGE_COEFFICIENT_ARG}"

        bash examples/embodiment/eval_embodiment.sh ${EVAL_CONFIG_NAME} ${EVAL_OVERRIDES}
    else
        echo "Task $TASK_ID (Weight Merge) failed with exit code $EXIT_CODE"
        OVERALL_EXIT_CODE=$EXIT_CODE
        [ "$IS_RANGE" = true ] && break
    fi
done

echo ""
echo "========================================="
if [ "$IS_RANGE" = true ]; then
    if [ $OVERALL_EXIT_CODE -eq 0 ]; then
        echo "All Weight Merge tasks (${TASK_START} to ${TASK_END}) completed successfully!"
    else
        echo "Weight Merge training failed."
    fi
else
    if [ $OVERALL_EXIT_CODE -eq 0 ]; then
        echo "Task $TASK_START completed successfully (Weight Merge)"
    else
        echo "Task $TASK_START failed (Weight Merge)"
    fi
fi
echo "Finished at: $(date)"
echo "========================================="

exit $OVERALL_EXIT_CODE
