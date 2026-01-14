#!/bin/bash
### Usage: bash examples/mll_cluster/run_embodiment_naive_lora_multitask.sh TASK_IDS [MAX_EPOCH] [CONFIG_NAME]
### Example: bash examples/mll_cluster/run_embodiment_naive_lora_multitask.sh "0,2,4"
### Example: bash examples/mll_cluster/run_embodiment_naive_lora_multitask.sh "0 2 4"
### Example (with max_epoch): bash examples/mll_cluster/run_embodiment_naive_lora_multitask.sh "0,2,4" 15
### Note: TASK_IDS can be comma-separated (e.g., "0,2,4") or space-separated (e.g., "0 2 4")
###       Evaluation runs automatically every 10 global steps during training
###       MAX_EPOCH is optional and can override the default max_epochs

TASK_IDS_STR=$1
MAX_EPOCH=$2
CONFIG_NAME=${3:-mll_cluster/libero_spatial_grpo_openvlaoft}

if [ -z "$TASK_IDS_STR" ]; then
    echo "ERROR: Missing required argument"
    echo "Usage: bash examples/mll_cluster/run_embodiment_naive_lora_multitask.sh TASK_IDS [MAX_EPOCH] [CONFIG_NAME]"
    echo "Example: bash examples/mll_cluster/run_embodiment_naive_lora_multitask.sh \"0,2,4\""
    exit 1
fi

# Parse TASK_IDS: convert comma-separated to space-separated, then to array
TASK_IDS_STR=$(echo "$TASK_IDS_STR" | tr ',' ' ')
read -ra TASK_IDS_ARRAY <<< "$TASK_IDS_STR"

# Validate task IDs
for task_id in "${TASK_IDS_ARRAY[@]}"; do
    if ! [[ "$task_id" =~ ^[0-9]+$ ]]; then
        echo "ERROR: Invalid task ID: $task_id (must be a non-negative integer)"
        exit 1
    fi
done

# Sort task IDs and create formatted string for logging
IFS=$'\n' sorted_task_ids=($(sort -n <<<"${TASK_IDS_ARRAY[*]}"))
unset IFS

# Create task list string for Hydra (comma-separated, no spaces)
TASK_LIST_STR=$(IFS=','; echo "${sorted_task_ids[*]}")

# Create task list string for directory name (underscore-separated)
TASK_DIR_STR=$(IFS='_'; echo "${sorted_task_ids[*]}")

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

mkdir -p logs/slurm
mkdir -p logs/naive_lora_multitask
export LOG_DIR="./logs/naive_lora_multitask/tasks_${TASK_DIR_STR}"

# Set experiment name based on LOG_DIR (for wandb)
EXPERIMENT_NAME=$(basename "$LOG_DIR")

# Update SLURM job name to match experiment name (only if running under SLURM)
if [ -n "$SLURM_JOB_ID" ] && command -v scontrol &> /dev/null; then
    scontrol update job=$SLURM_JOB_ID name="${EXPERIMENT_NAME}" 2>/dev/null || true
fi

echo "========================================="
echo "Multi-Task Training (Naive LoRA)"
echo "========================================="
echo "Configuration:"
echo "  Task IDs: ${sorted_task_ids[*]}"
echo "  Tasks (formatted): [${TASK_LIST_STR}]"
echo "  Experiment Name: $EXPERIMENT_NAME"
echo "  Checkpoint Save Path: $LOG_DIR"
echo "  Config Name: $CONFIG_NAME"
echo "  Evaluation: After training completes (all checkpoints)"

if [ -n "$MAX_EPOCH" ]; then
    if ! [[ "$MAX_EPOCH" =~ ^[0-9]+$ ]] || [ "$MAX_EPOCH" -le 0 ]; then
        echo "  ERROR: MAX_EPOCH must be a positive integer, got: $MAX_EPOCH"
        exit 1
    fi
    echo "  Max epochs: $MAX_EPOCH"
fi

echo "  Training from base model (SFT checkpoint) - Multi-task setup"
echo "========================================="
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

# Build Hydra overrides
# Note: fixed_task_ids expects a list format like [0,2,4]
OVERRIDES="env.fixed_task_ids=[${TASK_LIST_STR}] \
	runner.logger.experiment_name=${EXPERIMENT_NAME} \
	actor.checkpoint_save_path=${LOG_DIR}"

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
    echo "Multi-task training completed successfully"
    echo ""
    echo "Tasks trained: ${sorted_task_ids[*]}"
    echo "Checkpoint saved to: ${LOG_DIR}"
    echo ""
    
    # Discover and evaluate all checkpoints
    CHECKPOINTS_DIR="${LOG_DIR}/checkpoints"
    if [ ! -d "$CHECKPOINTS_DIR" ]; then
        echo "Warning: Checkpoints directory not found at $CHECKPOINTS_DIR"
    else
        echo "Discovering checkpoints for evaluation..."
        # Find all global_step directories and extract step numbers
        # Use find to get directories, extract step numbers, sort numerically
        CHECKPOINT_STEPS=($(find "$CHECKPOINTS_DIR" -maxdepth 1 -type d -name "global_step_*" | sed 's|.*/global_step_||' | sort -n))
        
        if [ ${#CHECKPOINT_STEPS[@]} -eq 0 ]; then
            echo "Warning: No checkpoints found in $CHECKPOINTS_DIR"
        else
            echo "Found ${#CHECKPOINT_STEPS[@]} checkpoint(s) at steps: ${CHECKPOINT_STEPS[*]}"
            echo ""
            echo "Running evaluation for each checkpoint..."
            echo ""
            
            CHECKPOINT_LOCATION=$(echo "$LOG_DIR" | sed 's|^\./||')
            EVAL_EXIT_CODE=0
            
            for STEP in "${CHECKPOINT_STEPS[@]}"; do
                echo "========================================="
                echo "Evaluating checkpoint at global_step_${STEP}"
                echo "========================================="
                bash examples/mll_cluster/eval_embodiment.sh "${CHECKPOINT_LOCATION}" "${STEP}"
                
                STEP_EXIT_CODE=$?
                if [ $STEP_EXIT_CODE -ne 0 ]; then
                    echo "  Warning: Evaluation for step ${STEP} failed (exit code: $STEP_EXIT_CODE)"
                    EVAL_EXIT_CODE=$STEP_EXIT_CODE
                else
                    echo "  ✓ Evaluation for step ${STEP} completed successfully"
                fi
                echo ""
            done
            
            if [ $EVAL_EXIT_CODE -eq 0 ]; then
                echo "All checkpoint evaluations completed successfully"
            else
                echo "Some checkpoint evaluations failed (last exit code: $EVAL_EXIT_CODE)"
            fi
        fi
    fi
else
    echo "✗ Multi-task training failed with exit code $EXIT_CODE"
    echo "  Tasks: ${sorted_task_ids[*]}"
    if [ -n "$SLURM_JOB_ID" ]; then
        echo "  Check logs at: logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
    fi
fi
echo "Finished at: $(date)"
echo "========================================="

exit $EXIT_CODE
