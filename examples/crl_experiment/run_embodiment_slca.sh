#!/bin/bash
### Usage: bash examples/crl_experiment/run_embodiment_slca.sh TASK_ID_OR_RANGE [LR_STRING] [CONFIG_NAME] [SEED]
### Example (single task): bash examples/crl_experiment/run_embodiment_slca.sh 0
### Example (task range): bash examples/crl_experiment/run_embodiment_slca.sh "0,3"
### Example (with LR): bash examples/crl_experiment/run_embodiment_slca.sh "1,4" "2e-6,2e-6,1e-5"
### Example (with config): bash examples/crl_experiment/run_embodiment_slca.sh "0,2" "2e-6,2e-6,1e-5" crl_experiment/libero_spatial_grpo_openvlaoft_spatial
### Example (with seed): bash examples/crl_experiment/run_embodiment_slca.sh "0,2" "" "" 42
### Note: TASK_ID_OR_RANGE can be:
###       - A single task ID (e.g., "0") - trains that task only
###       - A tuple "a,b" where a < b (e.g., "0,3") - trains tasks from a to b sequentially
###       LR_STRING is comma-separated: "vision_lora_lr,llm_lora_lr,llm_head_lora_lr"
###       If not provided, uses default values: 4.0e-6,4.0e-6,4.0e-5
###       SEED is optional and defaults to 1234 if not provided

TASK_INPUT=${1:-0}
LR_STRING=$2
CONFIG_NAME=${3:-crl_experiment/libero_spatial_grpo_openvlaoft_spatial}
SEED=${4:-1234}

# Log subdirectory for this experiment type
EXPERIMENT_TYPE="slca"

# Default learning rates if not provided
if [ -z "$LR_STRING" ]; then
    LR_STRING="4.0e-6,4.0e-6,4.0e-5"
fi

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

# Parse learning rates
IFS=',' read -r VISION_LR LLM_LR HEAD_LR <<< "$LR_STRING"
if [ -z "$VISION_LR" ] || [ -z "$LLM_LR" ] || [ -z "$HEAD_LR" ]; then
    echo "ERROR: LR_STRING must contain exactly 3 comma-separated values: vision_lora_lr,llm_lora_lr,llm_head_lora_lr"
    echo "       Example: \"2e-6,2e-6,1e-5\""
    exit 1
fi

# Create short strings for directory naming
V_LR_STR=$(echo "$VISION_LR" | sed 's/\.0*e/e/g' | sed 's/\.//g' | sed 's/e-/e/g')
L_LR_STR=$(echo "$LLM_LR" | sed 's/\.0*e/e/g' | sed 's/\.//g' | sed 's/e-/e/g')
H_LR_STR=$(echo "$HEAD_LR" | sed 's/\.0*e/e/g' | sed 's/\.//g' | sed 's/e-/e/g')

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
FIRST_TASK_ID=$(get_first_task_id "$CONFIG_NAME")

# Main training loop
OVERALL_EXIT_CODE=0

for TASK_ID in $(seq $TASK_START $TASK_END); do
    echo ""
    echo "========================================="
    if [ "$IS_RANGE" = true ]; then
        echo "SLCA Training - Task ${TASK_ID} (${TASK_START} to ${TASK_END})"
    else
        echo "SLCA Training - Single Task (Learning Rate Experiment)"
    fi
    echo "========================================="

    TASK_LOG_DIR="./logs/${EXPERIMENT_TYPE}/task_${TASK_ID}_lr_v${V_LR_STR}_l${L_LR_STR}_h${H_LR_STR}_seed${SEED}"

    if [ -n "$CONFIG_TAG" ]; then
        TASK_LOG_DIR_TRANSFORMED=$(inject_config_tag_into_log_path "$TASK_LOG_DIR" "$CONFIG_TAG")
        if [ -z "$TASK_LOG_DIR_TRANSFORMED" ]; then
            echo "  ERROR: Failed to transform TASK_LOG_DIR with config tag"
            OVERALL_EXIT_CODE=1
            break
        fi
        TASK_LOG_DIR="$TASK_LOG_DIR_TRANSFORMED"
    fi

    if [ -z "$TASK_LOG_DIR" ]; then
        echo "  ERROR: TASK_LOG_DIR is empty after path construction"
        OVERALL_EXIT_CODE=1
        break
    fi

    export LOG_DIR="${TASK_LOG_DIR}"
    mkdir -p "${TASK_LOG_DIR}"

    EXPERIMENT_NAME=$(basename "$TASK_LOG_DIR")
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
    echo "  Checkpoint Save Path: $TASK_LOG_DIR"
    echo "  Config Name: $CONFIG_NAME"
    echo "  Random Seed: $SEED"
    echo "  Vision LoRA LR: $VISION_LR"
    echo "  LLM LoRA LR: $LLM_LR"
    echo "  LLM Head LoRA LR: $HEAD_LR"

    if [ "$TASK_ID" -eq "$FIRST_TASK_ID" ]; then
        CHECKPOINT_PATH=""
        echo "  Training from base model (SFT checkpoint) - no LoRA path"
    else
        PREV_TASK_ID=$((TASK_ID - 1))
        PREV_TASK_LOG_DIR="./logs/${EXPERIMENT_TYPE}/task_${PREV_TASK_ID}_lr_v${V_LR_STR}_l${L_LR_STR}_h${H_LR_STR}_seed${SEED}"
        if [ -n "$CONFIG_TAG" ]; then
            PREV_TASK_LOG_DIR_TRANSFORMED=$(inject_config_tag_into_log_path "$PREV_TASK_LOG_DIR" "$CONFIG_TAG")
        else
            PREV_TASK_LOG_DIR_TRANSFORMED="$PREV_TASK_LOG_DIR"
        fi
        if [ -z "$PREV_TASK_LOG_DIR_TRANSFORMED" ]; then
            echo "  ERROR: Failed to construct previous task log directory for task $PREV_TASK_ID"
            OVERALL_EXIT_CODE=1
            break
        fi
        PREV_TASK_LOG_DIR="$PREV_TASK_LOG_DIR_TRANSFORMED"
        CHECKPOINT_PATH="${PREV_TASK_LOG_DIR}/checkpoints/global_step_${GLOBAL_STEP}/actor"
        if [[ "$CHECKPOINT_PATH" =~ ^/checkpoints/ ]]; then
            echo "  ERROR: Invalid checkpoint path construction detected"
            OVERALL_EXIT_CODE=1
            break
        fi
        if [ ! -d "$CHECKPOINT_PATH" ]; then
            echo "ERROR: Previous checkpoint does not exist: $CHECKPOINT_PATH"
            OVERALL_EXIT_CODE=1
            break
        fi
        echo "  Loading checkpoint from previous task: $CHECKPOINT_PATH"
    fi
    echo "========================================="
    echo ""

    # Build Hydra overrides
    # Learning rates are set via hydra overrides (not a separate config)
    OVERRIDES="env.fixed_task_ids=[${TASK_ID}] \
    	runner.logger.experiment_name=${EXPERIMENT_NAME} \
    	actor.seed=${SEED} \
    	actor.optim.vision_lora_lr=${VISION_LR} \
    	actor.optim.llm_lora_lr=${LLM_LR} \
    	actor.optim.llm_head_lora_lr=${HEAD_LR}"

    [ -n "$CHECKPOINT_PATH" ] && OVERRIDES="${OVERRIDES} +actor.model.lora_path=${CHECKPOINT_PATH}"

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
        echo "Checkpoint saved to: ${TASK_LOG_DIR}"
        CHECKPOINT_LOCATION=$(echo "$TASK_LOG_DIR" | sed 's|^\./||')
        echo ""
        echo "Running evaluation for: ${CHECKPOINT_LOCATION}"
        bash examples/crl_experiment/eval_embodiment.sh "${CHECKPOINT_LOCATION}" "${GLOBAL_STEP}" "${EVAL_CONFIG_NAME}"
    else
        echo "Task $TASK_ID failed with exit code $EXIT_CODE"
        OVERALL_EXIT_CODE=$EXIT_CODE
        if [ "$IS_RANGE" = true ]; then
            echo "  Stopping training due to failure"
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
        echo "SLCA training failed. Completed up to task $((TASK_ID - 1))"
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
