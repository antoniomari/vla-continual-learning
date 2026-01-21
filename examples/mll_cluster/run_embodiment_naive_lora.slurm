#!/bin/bash
#SBATCH --job-name=train_embodiment      # Job name
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks-per-node=1             # Number of tasks per node
#SBATCH --cpus-per-task=90              # Number of CPUs per task
#SBATCH --gres=gpu:8                    # Number of GPUs (RECOMMENDED: 4+ GPUs)
#SBATCH --time=70:00:00                  # Maximum execution time (HH:MM:SS)
#SBATCH --mem=900G                      # Memory per node (adjust as needed)
#SBATCH --partition=mll                 # Partition name (adjust to your cluster's partition)
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=jiahengh@utexas.edu
#SBATCH --exclude=slurm-node-008,slurm-node-011
#SBATCH --output=logs/slurm/%x-%j.out              # Output file name (%x = job name, %j = job ID)
#SBATCH --error=logs/slurm/%x-%j.err               # Error file name


### Usage: sbatch examples/mll_cluster/run_embodiment_naive_lora.slurm [TASK_ID_OR_RANGE] [CHECKPOINT_PATH] [MAX_EPOCH] [CONFIG_NAME] [SEED]
### Example (single task): sbatch examples/mll_cluster/run_embodiment_naive_lora.slurm 0
### Example (task range): sbatch examples/mll_cluster/run_embodiment_naive_lora.slurm "0,3"
### Example (with max_epoch): sbatch examples/mll_cluster/run_embodiment_naive_lora.slurm 0 "" 15
### Example (continue from checkpoint): sbatch examples/mll_cluster/run_embodiment_naive_lora.slurm 0 ./logs/naive_lora/task_0_seed1234/checkpoints/global_step_10/actor 20
### Example (with seed): sbatch examples/mll_cluster/run_embodiment_naive_lora.slurm 0 "" "" "" 42
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
mkdir -p logs/naive_lora

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "GPUs allocated: $SLURM_GPUS_ON_NODE"
echo ""

cd $SLURM_SUBMIT_DIR

# Main training loop
OVERALL_EXIT_CODE=0

for TASK_ID in $(seq $TASK_START $TASK_END); do
    echo ""
    echo "========================================="
    if [ "$IS_RANGE" = true ]; then
        echo "Sequential Training - Task ${TASK_ID} (${TASK_START} to ${TASK_END})"
    else
        echo "Lifelong Learning - Single Task Training (Naive LoRA)"
    fi
    echo "========================================="
    
    # Determine checkpoint path
    if [ "$TASK_ID" -eq "$TASK_START" ] && [ -n "$MANUAL_CHECKPOINT_PATH" ]; then
        # Use manual checkpoint path for first task if provided
        CHECKPOINT_PATH="$MANUAL_CHECKPOINT_PATH"
    elif [ "$TASK_ID" -eq 0 ]; then
        # First task in sequence, no checkpoint
        CHECKPOINT_PATH=""
    else
        # Use checkpoint from previous task
        PREV_TASK_ID=$((TASK_ID - 1))
        CHECKPOINT_PATH="./logs/naive_lora/task_${PREV_TASK_ID}_seed${SEED}/checkpoints/global_step_10/actor"
    fi
    
    # Determine LOG_DIR based on checkpoint path
    if [ "$TASK_ID" -eq "$TASK_START" ] && [ -n "$MANUAL_CHECKPOINT_PATH" ]; then
        # Extract task ID and global_step from checkpoint path
        # e.g., .../task_0/checkpoints/global_step_10/actor -> task_0, step_10
        if [[ "$CHECKPOINT_PATH" =~ task_([0-9]+) ]]; then
            SOURCE_TASK="${BASH_REMATCH[1]}"
            if [[ "$CHECKPOINT_PATH" =~ global_step_([0-9]+) ]]; then
                SOURCE_STEP="${BASH_REMATCH[1]}"
                LOG_DIR="./logs/naive_lora/task_${TASK_ID}_from_task_${SOURCE_TASK}_step_${SOURCE_STEP}_seed${SEED}"
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
        # Standard case: use TASK_ID
        LOG_DIR="./logs/naive_lora/task_${TASK_ID}_seed${SEED}"
    fi
    
    export LOG_DIR
    mkdir -p "${LOG_DIR}"
    
    # Set experiment name based on LOG_DIR (for wandb)
    EXPERIMENT_NAME=$(basename "$LOG_DIR")
    
    # Update SLURM job name to match experiment name (only for first task)
    if [ "$TASK_ID" -eq "$TASK_START" ] && command -v scontrol &> /dev/null && [ -n "$SLURM_JOB_ID" ]; then
        if [ "$IS_RANGE" = true ]; then
            scontrol update job=$SLURM_JOB_ID name="naive_lora_tasks_${TASK_START}_to_${TASK_END}" 2>/dev/null || true
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
    
    if [ -n "$CHECKPOINT_PATH" ]; then
        if [ ! -d "$CHECKPOINT_PATH" ]; then
            echo "  ERROR: Checkpoint not found at $CHECKPOINT_PATH"
            OVERALL_EXIT_CODE=1
            break
        fi
        echo "  Loading from checkpoint: $CHECKPOINT_PATH"
    else
        echo "  Training from base model (SFT checkpoint) - Task 0"
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
    	actor.checkpoint_save_path=${LOG_DIR} \
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
        
        # Run evaluation
        CHECKPOINT_LOCATION=$(echo "$LOG_DIR" | sed 's|^\./||')
        echo ""
        echo "Running evaluation for: ${CHECKPOINT_LOCATION}"
        bash examples/mll_cluster/eval_embodiment.sh "${CHECKPOINT_LOCATION}"
    else
        echo "✗ Task $TASK_ID failed with exit code $EXIT_CODE"
        echo "  Check logs at: logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
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
