#!/bin/bash
### Usage: bash examples/crl_experiment/eval_embodiment_lora_scale.sh CHECKPOINT_LOCATION [CURRENT_LORA_SCALE] [PREVIOUS_LORA_COEFF] [STEP_NUMBER] [CONFIG_NAME]
### Example (single LoRA): bash examples/crl_experiment/eval_embodiment_lora_scale.sh logs/sequential/task_0_seed1234 0.5
### Example (weight merge): bash examples/crl_experiment/eval_embodiment_lora_scale.sh logs/weight_merge/task_2_coeff_0_9
### Example (weight merge with current scale): bash examples/crl_experiment/eval_embodiment_lora_scale.sh logs/weight_merge/task_2_coeff_0_9 0.7
### Example (weight merge with both overrides): bash examples/crl_experiment/eval_embodiment_lora_scale.sh logs/weight_merge/task_2_coeff_0_9 0.7 0.8
### Example (with custom step): bash examples/crl_experiment/eval_embodiment_lora_scale.sh logs/sequential/task_0_seed1234 0.5 "" 20
###
### Note: CHECKPOINT_LOCATION should be relative to workspace root
###       For weight merge checkpoints, PREVIOUS_LORA_COEFF is auto-extracted from path
###       CURRENT_LORA_SCALE: required for single LoRA, defaults to 1.0 for weight merge

CHECKPOINT_LOCATION=$1
CURRENT_LORA_SCALE=$2
PREVIOUS_LORA_COEFF=$3
STEP_NUMBER=${4:-10}
CONFIG_NAME=${5:-crl_experiment/libero_spatial_grpo_openvlaoft_eval_spatial}

if [ -z "$CHECKPOINT_LOCATION" ]; then
    echo "ERROR: Missing required argument"
    echo "Usage: bash examples/crl_experiment/eval_embodiment_lora_scale.sh CHECKPOINT_LOCATION [CURRENT_LORA_SCALE] [PREVIOUS_LORA_COEFF] [STEP_NUMBER] [CONFIG_NAME]"
    exit 1
fi

if ! [[ "$STEP_NUMBER" =~ ^[0-9]+$ ]]; then
    echo "ERROR: STEP_NUMBER must be a positive integer, got: $STEP_NUMBER"
    exit 1
fi

# Change to repo root
SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT=$(dirname $(dirname "$SCRIPT_DIR"))
cd "$REPO_ROOT"

source "examples/crl_experiment/common_functions.sh"
FIRST_TASK_ID=$(get_first_task_id "$CONFIG_NAME")

# Check if this is a weight merge checkpoint
IS_MULTILORA=false
EXTRACTED_COEFF=""
if [[ "$CHECKPOINT_LOCATION" == *"weight_merge"* ]] && [[ "$CHECKPOINT_LOCATION" == *"coeff_"* ]]; then
    IS_MULTILORA=true
    if [[ "$CHECKPOINT_LOCATION" =~ coeff_([0-9]+_[0-9]+) ]]; then
        COEFF_STR="${BASH_REMATCH[1]}"
        EXTRACTED_COEFF=$(echo "$COEFF_STR" | tr '_' '.')
        echo "Detected weight merge checkpoint with coefficient: $EXTRACTED_COEFF"
    fi
fi

# Set defaults based on checkpoint type
if [ "$IS_MULTILORA" = true ]; then
    if [ -z "$PREVIOUS_LORA_COEFF" ]; then
        if [ -n "$EXTRACTED_COEFF" ]; then
            PREVIOUS_LORA_COEFF="$EXTRACTED_COEFF"
            echo "Using PREVIOUS_LORA_COEFF extracted from weight merge path: $PREVIOUS_LORA_COEFF"
        else
            echo "ERROR: Could not extract coefficient from weight merge checkpoint path: $CHECKPOINT_LOCATION"
            exit 1
        fi
    fi
    [ -z "$CURRENT_LORA_SCALE" ] && CURRENT_LORA_SCALE="1.0"
else
    if [ -z "$CURRENT_LORA_SCALE" ]; then
        echo "ERROR: For single LoRA checkpoints, CURRENT_LORA_SCALE (2nd argument) is required"
        exit 1
    fi
    PREVIOUS_LORA_COEFF=""
fi

# Validate PREVIOUS_LORA_COEFF if provided
if [ -n "$PREVIOUS_LORA_COEFF" ]; then
    if ! [[ "$PREVIOUS_LORA_COEFF" =~ ^[0-9]+\.?[0-9]*$ ]]; then
        echo "ERROR: PREVIOUS_LORA_COEFF must be a number, got: $PREVIOUS_LORA_COEFF"
        exit 1
    fi
    if ! awk "BEGIN {exit !($PREVIOUS_LORA_COEFF >= 0 && $PREVIOUS_LORA_COEFF <= 1)}"; then
        echo "ERROR: PREVIOUS_LORA_COEFF must be between 0.0 and 1.0, got: $PREVIOUS_LORA_COEFF"
        exit 1
    fi
fi

# Validate CURRENT_LORA_SCALE
if ! [[ "$CURRENT_LORA_SCALE" =~ ^[0-9]+\.?[0-9]*$ ]]; then
    echo "ERROR: CURRENT_LORA_SCALE must be a number, got: $CURRENT_LORA_SCALE"
    exit 1
fi
if ! awk "BEGIN {exit !($CURRENT_LORA_SCALE >= 0 && $CURRENT_LORA_SCALE <= 1)}"; then
    echo "ERROR: CURRENT_LORA_SCALE must be between 0.0 and 1.0, got: $CURRENT_LORA_SCALE"
    exit 1
fi

WORKSPACE_ROOT=$(pwd)
CHECKPOINT_LOCATION="${CHECKPOINT_LOCATION%/}"
CHECKPOINT_PATH="${WORKSPACE_ROOT}/${CHECKPOINT_LOCATION}/checkpoints/global_step_${STEP_NUMBER}/actor/"

if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT_PATH"
    exit 1
fi

echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo ""
echo "Checkpoint Location: $CHECKPOINT_LOCATION"
echo "Full Checkpoint Path: $CHECKPOINT_PATH"
echo "Global Step Number: $STEP_NUMBER"
if [ "$IS_MULTILORA" = true ]; then
    echo "Weight Merge Mode: Enabled"
    echo "Previous LoRA Coefficient: $PREVIOUS_LORA_COEFF"
    echo "Current LoRA Scale: $CURRENT_LORA_SCALE"
else
    echo "Single LoRA Mode"
    echo "LoRA Scale: $CURRENT_LORA_SCALE"
fi
echo ""

HYDRA_OVERRIDES="+actor.model.lora_path=${CHECKPOINT_PATH} +actor.model.lora_scale=${CURRENT_LORA_SCALE}"

if [ "$IS_MULTILORA" = true ]; then
    TASK_ID=""
    if [[ "$CHECKPOINT_LOCATION" =~ task_([0-9]+) ]]; then
        TASK_ID="${BASH_REMATCH[1]}"
    else
        echo "ERROR: Could not extract task ID from checkpoint path: $CHECKPOINT_LOCATION"
        exit 1
    fi

    if [ "$TASK_ID" -gt $FIRST_TASK_ID ]; then
        COEFFICIENT_PATH=$(echo "$PREVIOUS_LORA_COEFF" | tr '.' '_')
        LORA_PATHS=()

        for prev_task in $(seq $FIRST_TASK_ID $((TASK_ID - 1))); do
            if [ "$prev_task" -eq $FIRST_TASK_ID ]; then
                prev_adapter_path="${WORKSPACE_ROOT}/logs/sequential/task_${prev_task}/checkpoints/global_step_${STEP_NUMBER}/actor"
            else
                prev_adapter_path="${WORKSPACE_ROOT}/logs/weight_merge/task_${prev_task}_coeff_${COEFFICIENT_PATH}/checkpoints/global_step_${STEP_NUMBER}/actor"
            fi

            if [ ! -d "$prev_adapter_path" ]; then
                echo "ERROR: Previous adapter not found for task $prev_task: $prev_adapter_path"
                exit 1
            fi

            LORA_PATHS+=("$prev_adapter_path")
            echo "  Found previous adapter for task $prev_task: $prev_adapter_path"
        done

        if [ ${#LORA_PATHS[@]} -eq 0 ]; then
            echo "ERROR: No previous adapters found for task $TASK_ID"
            exit 1
        fi

        LORA_PATHS_STR=$(IFS=','; echo "${LORA_PATHS[*]}")
        HYDRA_OVERRIDES="${HYDRA_OVERRIDES} +actor.model.lora_paths=[${LORA_PATHS_STR}] +actor.model.previous_lora_merge_coefficient=${PREVIOUS_LORA_COEFF}"
    elif [ "$TASK_ID" -eq $FIRST_TASK_ID ]; then
        echo "  Task $FIRST_TASK_ID: No previous adapters to load (this is the first task)"
    else
        echo "ERROR: Invalid task ID extracted: $TASK_ID"
        exit 1
    fi
fi

bash examples/embodiment/eval_embodiment.sh ${CONFIG_NAME} ${HYDRA_OVERRIDES}

echo "End Time: $(date)"
