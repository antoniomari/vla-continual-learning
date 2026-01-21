#!/bin/bash
### Usage: bash examples/mll_cluster/run_embodiment_naive_lora_multitask.sh TASK_IDS [CHECKPOINT_PATH] [MAX_EPOCH] [CONFIG_NAME] [SEED]
### Example: bash examples/mll_cluster/run_embodiment_naive_lora_multitask.sh "0,2,4"
### Example: bash examples/mll_cluster/run_embodiment_naive_lora_multitask.sh "0 2 4"
### Example (with max_epoch): bash examples/mll_cluster/run_embodiment_naive_lora_multitask.sh "0,2,4" "" 15
### Example (continue from checkpoint): bash examples/mll_cluster/run_embodiment_naive_lora_multitask.sh "0,2,4" ./logs/naive_lora_multitask/tasks_0_2_4_seed1234/checkpoints/global_step_10/actor 20
### Example (with seed): bash examples/mll_cluster/run_embodiment_naive_lora_multitask.sh "0,2,4" "" "" "" 42
### Note: TASK_IDS can be comma-separated (e.g., "0,2,4") or space-separated (e.g., "0 2 4")
###       CHECKPOINT_PATH is optional and will load model weights (LoRA adapter)
###       Evaluation runs automatically every 10 global steps during training
###       MAX_EPOCH is optional and can override the default max_epochs
###       SEED is optional and defaults to 1234 if not provided

TASK_IDS_STR=$1
CHECKPOINT_PATH=$2
MAX_EPOCH=$3
CONFIG_NAME=${4:-mll_cluster/libero_spatial_grpo_openvlaoft}
SEED=${5:-1234}

if [ -z "$TASK_IDS_STR" ]; then
    echo "ERROR: Missing required argument"
    echo "Usage: bash examples/mll_cluster/run_embodiment_naive_lora_multitask.sh TASK_IDS [CHECKPOINT_PATH] [MAX_EPOCH] [CONFIG_NAME] [SEED]"
    echo "Example: bash examples/mll_cluster/run_embodiment_naive_lora_multitask.sh \"0,2,4\""
    echo "Example (with checkpoint): bash examples/mll_cluster/run_embodiment_naive_lora_multitask.sh \"0,2,4\" ./logs/naive_lora_multitask/tasks_0_2_4_seed1234/checkpoints/global_step_10/actor"
    echo "Example (with seed): bash examples/mll_cluster/run_embodiment_naive_lora_multitask.sh \"0,2,4\" \"\" \"\" \"\" 42"
    exit 1
fi

# Validate seed is a number
if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
    echo "ERROR: SEED must be a non-negative integer, got: $SEED"
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

# Determine LOG_DIR based on checkpoint path
if [ -n "$CHECKPOINT_PATH" ]; then
    # Extract global_step from checkpoint path
    # e.g., .../checkpoints/global_step_10/actor -> step_10
    if [[ "$CHECKPOINT_PATH" =~ global_step_([0-9]+) ]]; then
        SOURCE_STEP="${BASH_REMATCH[1]}"
        LOG_DIR="./logs/naive_lora_multitask/tasks_${TASK_DIR_STR}_from_checkpoint_step_${SOURCE_STEP}_seed${SEED}"
    else
        # Fallback: use a generic name
        LOG_DIR="./logs/naive_lora_multitask/tasks_${TASK_DIR_STR}_from_checkpoint_seed${SEED}"
    fi
else
    # Standard case: use task IDs
    LOG_DIR="./logs/naive_lora_multitask/tasks_${TASK_DIR_STR}_seed${SEED}"
fi

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
export LOG_DIR

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
echo "  Random Seed: $SEED"
echo "  Evaluation: After training completes (all checkpoints)"

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
# Build as space-separated string (not multi-line) to avoid argument splitting issues
OVERRIDES="env.fixed_task_ids=[${TASK_LIST_STR}] runner.logger.experiment_name=${EXPERIMENT_NAME} actor.checkpoint_save_path=${LOG_DIR} actor.seed=${SEED}"

if [ -n "$CHECKPOINT_PATH" ]; then
    # Set lora_path to load LoRA adapter weights (like single-task version)
    OVERRIDES="$OVERRIDES +actor.model.lora_path=${CHECKPOINT_PATH}"
fi

if [ -n "$MAX_EPOCH" ]; then
    OVERRIDES="$OVERRIDES runner.max_epochs=${MAX_EPOCH}"
fi

echo "Running with Hydra overrides:"
echo "$OVERRIDES"
echo ""

# Properly quote arguments to prevent word splitting
bash examples/embodiment/run_embodiment.sh "${CONFIG_NAME}" ${OVERRIDES}

EXIT_CODE=$?
echo ""
echo "========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Multi-task training completed successfully"
    echo ""
    echo "Tasks trained: ${sorted_task_ids[*]}"
    echo "Checkpoint saved to: ${LOG_DIR}"
    echo ""
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
