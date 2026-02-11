#!/bin/bash
#
# Launch multiple evaluation jobs with different lora_scale coefficients
#
# Usage: ./examples/mll_cluster/launch_lora_scale_evals.sh CHECKPOINT_LOCATION LORA_SCALE_1 [LORA_SCALE_2 ...] [--step STEP_NUMBER] [--local] [--config CONFIG_NAME]
# Example: ./examples/mll_cluster/launch_lora_scale_evals.sh logs/bcrl_logit/0.3/task_0 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
# Example (with step): ./examples/mll_cluster/launch_lora_scale_evals.sh logs/bcrl_logit/0.3/task_0 0.0 0.1 0.2 --step 20
# Example (local): ./examples/mll_cluster/launch_lora_scale_evals.sh logs/bcrl_logit/0.3/task_0 0.0 0.1 0.2 --local
# Example (local with step): ./examples/mll_cluster/launch_lora_scale_evals.sh logs/bcrl_logit/0.3/task_0 0.0 0.1 0.2 --local --step 20
# Example (libero_10): ./examples/mll_cluster/launch_lora_scale_evals.sh logs/bcrl_logit/0.3/task_0 0.0 0.1 0.2 --config mll_cluster/libero_10_grpo_openvlaoft_eval_long
#
# This script will submit one SLURM job (or run locally if --local is specified) for each lora_scale coefficient provided

if [ $# -lt 2 ]; then
    echo "ERROR: Missing required arguments"
    echo "Usage: ./examples/mll_cluster/launch_lora_scale_evals.sh CHECKPOINT_LOCATION LORA_SCALE_1 [LORA_SCALE_2 ...] [--step STEP_NUMBER] [--local]"
    echo "Example: ./examples/mll_cluster/launch_lora_scale_evals.sh logs/bcrl_logit/0.3/task_0 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"
    echo "Example (with step): ./examples/mll_cluster/launch_lora_scale_evals.sh logs/bcrl_logit/0.3/task_0 0.0 0.1 0.2 --step 20"
    echo "Example (local): ./examples/mll_cluster/launch_lora_scale_evals.sh logs/bcrl_logit/0.3/task_0 0.0 0.1 0.2 --local"
    echo "Example (local with step): ./examples/mll_cluster/launch_lora_scale_evals.sh logs/bcrl_logit/0.3/task_0 0.0 0.1 0.2 --local --step 20"
    exit 1
fi

CHECKPOINT_LOCATION=$1
shift

# Source common functions
source "examples/mll_cluster/common_functions.sh"

# Parse optional arguments
STEP_NUMBER=""
LOCAL_MODE=false
CONFIG_NAME="mll_cluster/libero_spatial_grpo_openvlaoft_eval"
LORA_SCALES=()
while [ $# -gt 0 ]; do
    case "$1" in
        --step)
            shift
            if [ $# -eq 0 ]; then
                echo "ERROR: --step requires a step number"
                exit 1
            fi
            STEP_NUMBER="$1"
            shift
            ;;
        --local)
            LOCAL_MODE=true
            shift
            ;;
        --config)
            shift
            if [ $# -eq 0 ]; then
                echo "ERROR: --config requires a config name"
                exit 1
            fi
            CONFIG_NAME="$1"
            shift
            ;;
        *)
            LORA_SCALES+=("$1")
            shift
            ;;
    esac
done

# Assume we're already in the workspace root
WORKSPACE_ROOT=$(pwd)
SLURM_SCRIPT="examples/mll_cluster/eval_embodiment_lora_scale.slurm"
EVAL_SCRIPT="${WORKSPACE_ROOT}/examples/embodiment/eval_embodiment.sh"

# Verify required scripts exist
if [ "$LOCAL_MODE" = false ] && [ ! -f "$SLURM_SCRIPT" ]; then
    echo "ERROR: SLURM script not found at $SLURM_SCRIPT"
    exit 1
fi
if [ "$LOCAL_MODE" = true ] && [ ! -f "$EVAL_SCRIPT" ]; then
    echo "ERROR: Evaluation script not found at $EVAL_SCRIPT"
    exit 1
fi

# Validate step number if provided
if [ -n "$STEP_NUMBER" ]; then
    if ! [[ "$STEP_NUMBER" =~ ^[0-9]+$ ]]; then
        echo "ERROR: STEP_NUMBER must be a positive integer, got: $STEP_NUMBER"
        exit 1
    fi
fi

# Validate all lora_scale values
for scale in "${LORA_SCALES[@]}"; do
    if ! [[ "$scale" =~ ^[0-9]+\.?[0-9]*$ ]]; then
        echo "ERROR: Invalid lora_scale value: $scale (must be a number)"
        exit 1
    fi
    # Use awk for floating point comparison (more portable than bc)
    if ! awk "BEGIN {exit !($scale >= 0 && $scale <= 1)}"; then
        echo "ERROR: Invalid lora_scale value: $scale (must be between 0.0 and 1.0)"
        exit 1
    fi
done

echo "=========================================="
if [ "$LOCAL_MODE" = true ]; then
    echo "Running LoRA Scale Evaluations Locally"
else
    echo "Launching LoRA Scale Evaluation Jobs (SLURM)"
fi
echo "=========================================="
echo "Checkpoint Location: $CHECKPOINT_LOCATION"
echo "LoRA Scales: ${LORA_SCALES[*]}"
echo "Number of evaluations: ${#LORA_SCALES[@]}"
echo "Config Name: $CONFIG_NAME"
if [ -n "$STEP_NUMBER" ]; then
    echo "Global Step Number: $STEP_NUMBER"
fi
echo "=========================================="
echo ""

# Function to build Hydra overrides (extracted from SLURM script logic)
build_hydra_overrides() {
    local checkpoint_loc="$1"
    local current_scale="$2"
    local prev_coeff="$3"
    local step_num="${4:-10}"
    
    # Normalize checkpoint location
    checkpoint_loc="${checkpoint_loc%/}"
    local checkpoint_path="${WORKSPACE_ROOT}/${checkpoint_loc}/checkpoints/global_step_${step_num}/actor/"
    
    # Check if multi-LoRA
    local is_multilora=false
    local extracted_coeff=""
    if [[ "$checkpoint_loc" == *"multilora"* ]] && [[ "$checkpoint_loc" == *"coeff_"* ]]; then
        is_multilora=true
        if [[ "$checkpoint_loc" =~ coeff_([0-9]+_[0-9]+) ]]; then
            local coeff_str="${BASH_REMATCH[1]}"
            extracted_coeff=$(echo "$coeff_str" | tr '_' '.')
        fi
    fi
    
    # Set defaults for multi-LoRA
    if [ "$is_multilora" = true ]; then
        if [ -z "$prev_coeff" ] && [ -n "$extracted_coeff" ]; then
            prev_coeff="$extracted_coeff"
        fi
        if [ -z "$current_scale" ]; then
            current_scale="1.0"
        fi
    fi
    
    # Build base overrides
    local overrides="+actor.model.lora_path=${checkpoint_path} actor.model.lora_scale=${current_scale}"
    
    # Add multi-LoRA parameters if needed
    if [ "$is_multilora" = true ] && [ -n "$prev_coeff" ]; then
        # Extract task ID
        local task_id=""
        if [[ "$checkpoint_loc" =~ task_([0-9]+) ]]; then
            task_id="${BASH_REMATCH[1]}"
        fi
        
        local first_task_id=$(get_first_task_id "$CONFIG_NAME")
        if [ -n "$task_id" ] && [ "$task_id" -gt $first_task_id ]; then
            local coeff_path=$(echo "$prev_coeff" | tr '.' '_')
            local lora_paths=()
            
            # Build previous adapter paths
            for prev_task in $(seq $first_task_id $((task_id - 1))); do
                if [ "$prev_task" -eq $first_task_id ]; then
                    local prev_path="${WORKSPACE_ROOT}/logs/naive_lora/task_${prev_task}/checkpoints/global_step_${step_num}/actor"
                else
                    local prev_path="${WORKSPACE_ROOT}/logs/naive_lora_multilora/task_${prev_task}_coeff_${coeff_path}/checkpoints/global_step_${step_num}/actor"
                fi
                lora_paths+=("$prev_path")
            done
            
            if [ ${#lora_paths[@]} -gt 0 ]; then
                local paths_str=$(IFS=','; echo "${lora_paths[*]}")
                overrides="${overrides} +actor.model.lora_paths=[${paths_str}] +actor.model.previous_lora_merge_coefficient=${prev_coeff}"
            fi
        fi
    fi
    
    # Return both overrides and checkpoint_path (separated by |)
    echo "${overrides}|${checkpoint_path}"
}

# Run evaluations
JOB_IDS=()
SUCCESS_COUNT=0
FAILED_COUNT=0

for scale in "${LORA_SCALES[@]}"; do
    if [ "$LOCAL_MODE" = true ]; then
        echo "Running evaluation for lora_scale=$scale..."
        # Build Hydra overrides
        RESULT=$(build_hydra_overrides "$CHECKPOINT_LOCATION" "$scale" "" "${STEP_NUMBER:-10}")
        HYDRA_OVERRIDES="${RESULT%|*}"
        CHECKPOINT_PATH="${RESULT#*|}"
        
        # Validate checkpoint exists
        if [ ! -d "$CHECKPOINT_PATH" ]; then
            echo "  ✗ ERROR: Checkpoint not found at $CHECKPOINT_PATH"
            FAILED_COUNT=$((FAILED_COUNT + 1))
            echo ""
            continue
        fi
        
        # Run evaluation locally
        if bash "$EVAL_SCRIPT" ${CONFIG_NAME} ${HYDRA_OVERRIDES}; then
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            echo "  ✓ Evaluation completed successfully for lora_scale=$scale"
        else
            FAILED_COUNT=$((FAILED_COUNT + 1))
            echo "  ✗ Evaluation failed for lora_scale=$scale"
        fi
    else
        echo "Submitting job for lora_scale=$scale..."
        if [ -n "$STEP_NUMBER" ]; then
            JOB_OUTPUT=$(sbatch "$SLURM_SCRIPT" "$CHECKPOINT_LOCATION" "$scale" "" "$STEP_NUMBER" "$CONFIG_NAME" 2>&1)
        else
            JOB_OUTPUT=$(sbatch "$SLURM_SCRIPT" "$CHECKPOINT_LOCATION" "$scale" "" "" "$CONFIG_NAME" 2>&1)
        fi
        
        if [ $? -eq 0 ]; then
            # Extract job ID from output (format: "Submitted batch job 12345")
            JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP '\d+$')
            JOB_IDS+=("$JOB_ID")
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            echo "  ✓ Job submitted successfully (Job ID: $JOB_ID)"
        else
            FAILED_COUNT=$((FAILED_COUNT + 1))
            echo "  ✗ Failed to submit job for lora_scale=$scale"
            echo "  Error: $JOB_OUTPUT"
        fi
    fi
    echo ""
done

echo "=========================================="
if [ "$LOCAL_MODE" = true ]; then
    echo "Local Execution Summary"
    echo "=========================================="
    echo "Total evaluations: ${#LORA_SCALES[@]}"
    echo "Successful: $SUCCESS_COUNT"
    echo "Failed: $FAILED_COUNT"
else
    echo "Job Submission Summary"
    echo "=========================================="
    echo "Total jobs submitted: $SUCCESS_COUNT"
    echo "Failed submissions: $FAILED_COUNT"
    if [ ${#JOB_IDS[@]} -gt 0 ]; then
        echo "Job IDs: ${JOB_IDS[*]}"
        echo ""
        echo "Monitor jobs with: squeue -u \$USER"
        echo "Cancel all jobs with: scancel ${JOB_IDS[*]}"
    fi
fi
echo "=========================================="
