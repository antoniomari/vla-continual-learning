#!/bin/bash
#SBATCH --job-name=seq_lora      # Job name
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

### NOTE: this script is meant for sequential launching of tasks in hand-crafted order. 
###       The first task will always load from base model.
### Usage: sbatch examples/mll_cluster/run_embodiment_naive_lora_sequential.slurm TASK_IDS [RUN_ID] [MAX_EPOCH] [CONFIG_NAME] [SEED]
### Example: sbatch examples/mll_cluster/run_embodiment_naive_lora_sequential.slurm "0,1,2"
### Example: sbatch examples/mll_cluster/run_embodiment_naive_lora_sequential.slurm "0 1 2"
### Example (with run_id): sbatch examples/mll_cluster/run_embodiment_naive_lora_sequential.slurm "0,1,2" "run1"
### Example (with max_epoch): sbatch examples/mll_cluster/run_embodiment_naive_lora_sequential.slurm "0,1,2" "" 15
### Example (with seed): sbatch examples/mll_cluster/run_embodiment_naive_lora_sequential.slurm "0,1,2" "" "" "" 42
### Note: TASK_IDS can be comma-separated (e.g., "0,1,2") or space-separated (e.g., "0 1 2")
###       IMPORTANT: Order matters! Tasks are trained in the exact order provided
###       Tasks are trained sequentially, with each task loading from the previous one
###       Evaluation runs after each task completes training
###       RUN_ID is optional - if not provided, uses task sequence (e.g., "0_1_2") to identify the sequence run
###       MAX_EPOCH is optional and can override the default max_epochs for all tasks
###       SEED is optional and defaults to 1234 if not provided

TASK_IDS_STR=$1
RUN_ID=$2
MAX_EPOCH=$3
CONFIG_NAME=${4:-mll_cluster/libero_spatial_grpo_openvlaoft}
SEED=${5:-1234}

if [ -z "$TASK_IDS_STR" ]; then
    echo "ERROR: Missing required argument"
    echo "Usage: sbatch examples/mll_cluster/run_embodiment_naive_lora_sequential.slurm TASK_IDS [RUN_ID] [MAX_EPOCH] [CONFIG_NAME] [SEED]"
    echo "Example: sbatch examples/mll_cluster/run_embodiment_naive_lora_sequential.slurm \"0,1,2\""
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

# Preserve input order (order matters!)
# Create task sequence string for directory names
TASK_SEQ_STR=$(IFS='_'; echo "${TASK_IDS_ARRAY[*]}")

# Generate RUN_ID if not provided (use task sequence)
if [ -z "$RUN_ID" ]; then
    RUN_ID="seq_${TASK_SEQ_STR}"
fi

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "GPUs allocated: $SLURM_GPUS_ON_NODE"
echo ""

mkdir -p logs/slurm
mkdir -p logs/naive_lora_sequential

echo "========================================="
echo "Sequential Task Training (Naive LoRA)"
echo "========================================="
echo "Configuration:"
echo "  Task Sequence (order preserved): ${TASK_IDS_ARRAY[*]}"
echo "  Run ID: $RUN_ID"
echo "  Config Name: $CONFIG_NAME"

if [ -n "$MAX_EPOCH" ]; then
    if ! [[ "$MAX_EPOCH" =~ ^[0-9]+$ ]] || [ "$MAX_EPOCH" -le 0 ]; then
        echo "  ERROR: MAX_EPOCH must be a positive integer, got: $MAX_EPOCH"
        exit 1
    fi
    echo "  Max epochs per task: $MAX_EPOCH"
fi

echo "  Training mode: Sequential (each task loads from previous)"
echo "  Evaluation: After each task completes training"
echo "========================================="
echo ""

cd $SLURM_SUBMIT_DIR

# Main sequential training loop
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
    
    # Create task-specific log directory with run_id and seed to reflect non-ordered runs
    TASK_LOG_DIR="./logs/naive_lora_sequential/${RUN_ID}_seed${SEED}/task_${TASK_ID}"
    mkdir -p "${TASK_LOG_DIR}"
    export LOG_DIR="${TASK_LOG_DIR}"
    
    # Set experiment name based on LOG_DIR (for wandb)
    EXPERIMENT_NAME=$(basename "$TASK_LOG_DIR")
    
    # Update SLURM job name (only for first task to avoid spam)
    if [ $TASK_NUM -eq 1 ] && command -v scontrol &> /dev/null && [ -n "$SLURM_JOB_ID" ]; then
        scontrol update job=$SLURM_JOB_ID name="seq_${RUN_ID}_tasks_${TASK_SEQ_STR}" 2>/dev/null || true
    fi
    
    echo "Configuration:"
    echo "  Task ID: $TASK_ID"
    echo "  Task Position: ${TASK_NUM}/${NUM_TASKS}"
    echo "  Experiment Name: $EXPERIMENT_NAME"
    echo "  Checkpoint Save Path: $TASK_LOG_DIR"
    echo "  Config Name: $CONFIG_NAME"
    echo "  Random Seed: $SEED"
    
    # Determine checkpoint path
    if [ $TASK_NUM -eq 1 ]; then
        CHECKPOINT_PATH=""
        echo "  Loading from: Base model (SFT checkpoint) - First task in sequence"
    else
        CHECKPOINT_PATH="$PREV_CHECKPOINT_PATH"
        if [ ! -d "$CHECKPOINT_PATH" ]; then
            echo "  ERROR: Previous checkpoint not found at $CHECKPOINT_PATH"
            OVERALL_EXIT_CODE=1
            break
        fi
        echo "  Loading from: Previous task checkpoint"
        echo "  Checkpoint path: $CHECKPOINT_PATH"
    fi
    
    if [ -n "$MAX_EPOCH" ]; then
        echo "  Max epochs: $MAX_EPOCH"
    fi
    
    echo "========================================="
    echo ""
    
    # Build Hydra overrides
    OVERRIDES="env.fixed_task_ids=[${TASK_ID}] \
    	runner.logger.experiment_name=${EXPERIMENT_NAME} \
    	actor.checkpoint_save_path=${TASK_LOG_DIR} \
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
    
    # Run training
    bash examples/embodiment/run_embodiment.sh ${CONFIG_NAME} $OVERRIDES
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -ne 0 ]; then
        echo ""
        echo "========================================="
        echo "✗ Task $TASK_ID failed with exit code $EXIT_CODE"
        echo "  Stopping sequence at task ${TASK_NUM}/${NUM_TASKS}"
        OVERALL_EXIT_CODE=$EXIT_CODE
        break
    fi
    
    # Update previous checkpoint path for next iteration
    PREV_CHECKPOINT_PATH="${TASK_LOG_DIR}/checkpoints/global_step_10/actor"
    
    echo ""
    echo "========================================="
    echo "✓ Task $TASK_ID completed successfully"
    echo "  Checkpoint saved to: ${TASK_LOG_DIR}"
    echo "========================================="
    
    # Run evaluation after each task
    CHECKPOINT_LOCATION=$(echo "$TASK_LOG_DIR" | sed 's|^\./||')
    echo ""
    echo "Running evaluation for task $TASK_ID..."
    echo "Checkpoint location: ${CHECKPOINT_LOCATION}"
    bash examples/mll_cluster/eval_embodiment.sh "${CHECKPOINT_LOCATION}"
    
    EVAL_EXIT_CODE=$?
    if [ $EVAL_EXIT_CODE -ne 0 ]; then
        echo "  Warning: Evaluation for task $TASK_ID failed, but continuing sequence..."
    else
        echo "  ✓ Evaluation for task $TASK_ID completed"
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
    echo "Results directory: ./logs/naive_lora_sequential/${RUN_ID}_seed${SEED}/"
    echo ""
    echo "All tasks completed and evaluated"
else
    echo "✗ Sequential training failed"
    echo "  Tasks trained: ${TASK_IDS_ARRAY[*]}"
    echo "  Run ID: $RUN_ID"
    echo "  Check logs at: logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
fi
echo "Finished at: $(date)"
echo "========================================="

exit $OVERALL_EXIT_CODE
