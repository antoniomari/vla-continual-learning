#!/bin/bash
#SBATCH --job-name=train_embodiment      # Job name
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks-per-node=1             # Number of tasks per node
#SBATCH --cpus-per-task=90              # Number of CPUs per task
#SBATCH --gres=gpu:8                    # Number of GPUs (RECOMMENDED: 4+ GPUs)
#SBATCH --time=35:00:00                  # Maximum execution time (HH:MM:SS)
#SBATCH --mem=900G                      # Memory per node (adjust as needed)
#SBATCH --partition=mll                 # Partition name (adjust to your cluster's partition)
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=jiahengh@utexas.edu
#SBATCH --exclude=slurm-node-004,slurm-node-011


### Usage: sbatch run_embodiment_naive_lora.slurm TASK_ID [CHECKPOINT_PATH] [CONFIG_NAME]
### Example: sbatch run_embodiment_naive_lora.slurm 0
### Example: sbatch run_embodiment_naive_lora.slurm 1
### Note: CHECKPOINT_PATH is now optional and will be auto-generated from previous task if not provided

TASK_ID=$1
MANUAL_CHECKPOINT_PATH=$2
CONFIG_NAME=${3:-mll_cluster/libero_spatial_grpo_openvlaoft}  # Default config if not provided

if [ -z "$TASK_ID" ]; then
    echo "ERROR: Missing required argument"
    echo "Usage: sbatch run_embodiment_naive_lora.slurm TASK_ID [CHECKPOINT_PATH] [CONFIG_NAME]"
    exit 1
fi

# Automatically determine checkpoint path if not manually provided
if [ -z "$MANUAL_CHECKPOINT_PATH" ]; then
    if [ "$TASK_ID" -eq 0 ]; then
        # Task 0 uses base model, no checkpoint
        CHECKPOINT_PATH=""
    else
        # Use checkpoint from previous task
        PREV_TASK_ID=$((TASK_ID - 1))
        CHECKPOINT_PATH="./logs/naive_lora/task_${PREV_TASK_ID}/checkpoints/global_step_10/actor"
    fi
else
    # Use manually provided checkpoint path
    CHECKPOINT_PATH="$MANUAL_CHECKPOINT_PATH"
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

mkdir -p logs/naive_lora
LOG_DIR="./logs/naive_lora/task_${TASK_ID}"
export LOG_DIR

EXPERIMENT_NAME="naive_lora_task_${TASK_ID}"

# Update SLURM job name to be more descriptive (based on LOG_DIR)
# Extract the last component of LOG_DIR (e.g., "task_0")
LOG_DIR_BASENAME=$(basename "$LOG_DIR")
if command -v scontrol &> /dev/null && [ -n "$SLURM_JOB_ID" ]; then
    # Use the directory name from LOG_DIR (e.g., "task_0")
    scontrol update job=$SLURM_JOB_ID name="${LOG_DIR_BASENAME}" 2>/dev/null || true
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
    if [ -n "$MANUAL_CHECKPOINT_PATH" ]; then
        echo "  Loading from checkpoint (manually specified): $CHECKPOINT_PATH"
    else
        echo "  Loading from checkpoint (auto-generated from task $((TASK_ID - 1))): $CHECKPOINT_PATH"
    fi

    # Verify checkpoint exists
    if [ ! -d "$CHECKPOINT_PATH" ]; then
        echo "  ERROR: Checkpoint not found at $CHECKPOINT_PATH"
        exit 1
    fi
    CHECKPOINT_ARG="+actor.model.lora_path=${CHECKPOINT_PATH}"
else
    echo "  Training from base model (SFT checkpoint) - Task 0"
fi

echo "========================================="
echo ""

cd $SLURM_SUBMIT_DIR

# hydra overrides
OVERRIDES="env.fixed_task_ids=[${TASK_ID}] \
	runner.logger.experiment_name=${EXPERIMENT_NAME} \
	actor.checkpoint_save_path=${LOG_DIR}"

if [ -n "$CHECKPOINT_PATH" ]; then
    OVERRIDES="$OVERRIDES ${CHECKPOINT_ARG}"
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
else
    echo "✗ Task $TASK_ID failed with exit code $EXIT_CODE"
    echo "  Check logs at: ${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
fi
echo "Finished at: $(date)"
echo "========================================="

exit $EXIT_CODE

