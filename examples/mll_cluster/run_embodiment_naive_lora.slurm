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


### Usage: sbatch examples/mll_cluster/run_embodiment_naive_lora.slurm TASK_ID [CHECKPOINT_PATH] [MAX_EPOCH] [CONFIG_NAME]
### Example: sbatch examples/mll_cluster/run_embodiment_naive_lora.slurm 0
### Example: sbatch examples/mll_cluster/run_embodiment_naive_lora.slurm 1
### Example (with max_epoch): sbatch examples/mll_cluster/run_embodiment_naive_lora.slurm 0 "" 15
### Example (continue from checkpoint): sbatch examples/mll_cluster/run_embodiment_naive_lora.slurm 0 ./logs/naive_lora/task_0/checkpoints/global_step_10/actor 20
### Note: CHECKPOINT_PATH is optional and will be auto-generated from previous task if not provided
###       MAX_EPOCH is optional and can always be specified to override the default max_epochs

TASK_ID=$1
MANUAL_CHECKPOINT_PATH=$2
MAX_EPOCH=$3
CONFIG_NAME=${4:-mll_cluster/libero_spatial_grpo_openvlaoft}

if [ -z "$TASK_ID" ]; then
    echo "ERROR: Missing required argument"
    echo "Usage: sbatch examples/mll_cluster/run_embodiment_naive_lora.slurm TASK_ID [CHECKPOINT_PATH] [MAX_EPOCH] [CONFIG_NAME]"
    exit 1
fi

# Automatically determine checkpoint path if not manually provided
if [ -z "$MANUAL_CHECKPOINT_PATH" ]; then
    if [ "$TASK_ID" -eq 0 ]; then
        CHECKPOINT_PATH=""
    else
        PREV_TASK_ID=$((TASK_ID - 1))
        CHECKPOINT_PATH="./logs/naive_lora/task_${PREV_TASK_ID}/checkpoints/global_step_10/actor"
    fi
else
    CHECKPOINT_PATH="$MANUAL_CHECKPOINT_PATH"
fi

# Determine LOG_DIR based on checkpoint path
if [ -n "$MANUAL_CHECKPOINT_PATH" ]; then
    # Extract task ID and global_step from checkpoint path
    # e.g., .../task_0/checkpoints/global_step_10/actor -> task_0, step_10
    if [[ "$CHECKPOINT_PATH" =~ task_([0-9]+) ]]; then
        SOURCE_TASK="${BASH_REMATCH[1]}"
        if [[ "$CHECKPOINT_PATH" =~ global_step_([0-9]+) ]]; then
            SOURCE_STEP="${BASH_REMATCH[1]}"
            LOG_DIR="./logs/naive_lora/task_${TASK_ID}_from_task_${SOURCE_TASK}_step_${SOURCE_STEP}"
        else
            echo "ERROR: Could not extract global_step from checkpoint path: $CHECKPOINT_PATH"
            echo "       Expected format: .../checkpoints/global_step_<M>/actor"
            exit 1
        fi
    else
        echo "ERROR: Could not extract task ID from checkpoint path: $CHECKPOINT_PATH"
        echo "       Expected format: .../task_<N>/checkpoints/global_step_<M>/actor"
        exit 1
    fi
else
    # Standard case: use TASK_ID
    LOG_DIR="./logs/naive_lora/task_${TASK_ID}"
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
mkdir -p logs/naive_lora
export LOG_DIR

# Set experiment name based on LOG_DIR (for wandb)
EXPERIMENT_NAME=$(basename "$LOG_DIR")

# Update SLURM job name to match experiment name
if command -v scontrol &> /dev/null && [ -n "$SLURM_JOB_ID" ]; then
    scontrol update job=$SLURM_JOB_ID name="${EXPERIMENT_NAME}" 2>/dev/null || true
fi

echo "========================================="
echo "Lifelong Learning - Single Task Training (Naive LoRA)"
echo "========================================="
echo "Configuration:"
echo "  Task ID: $TASK_ID"
echo "  Experiment Name: $EXPERIMENT_NAME"
echo "  Checkpoint Save Path: $LOG_DIR"
echo "  Config Name: $CONFIG_NAME"

if [ -n "$CHECKPOINT_PATH" ]; then
    if [ ! -d "$CHECKPOINT_PATH" ]; then
        echo "  ERROR: Checkpoint not found at $CHECKPOINT_PATH"
        exit 1
    fi
    echo "  Loading from checkpoint: $CHECKPOINT_PATH"
else
    echo "  Training from base model (SFT checkpoint) - Task 0"
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

cd $SLURM_SUBMIT_DIR

# Build Hydra overrides
OVERRIDES="env.fixed_task_ids=[${TASK_ID}] \
	runner.logger.experiment_name=${EXPERIMENT_NAME} \
	actor.checkpoint_save_path=${LOG_DIR}"

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
fi
echo "Finished at: $(date)"
echo "========================================="

exit $EXIT_CODE
