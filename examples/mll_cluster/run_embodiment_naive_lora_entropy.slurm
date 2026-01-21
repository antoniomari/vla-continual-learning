#!/bin/bash
#SBATCH --job-name=train_embodiment_entropy      # Job name
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


### Usage: sbatch examples/mll_cluster/run_embodiment_naive_lora_entropy.slurm TASK_ID ENTROPY_BONUS [CHECKPOINT_PATH] [MAX_EPOCH] [SEED]
### Example: sbatch examples/mll_cluster/run_embodiment_naive_lora_entropy.slurm 0 0.01
### Example: sbatch examples/mll_cluster/run_embodiment_naive_lora_entropy.slurm 1 0.01
### Example (with max_epoch): sbatch examples/mll_cluster/run_embodiment_naive_lora_entropy.slurm 0 0.01 "" 15
### Example (continue from checkpoint): sbatch examples/mll_cluster/run_embodiment_naive_lora_entropy.slurm 0 0.01 ./logs/naive_lora_entropy/task_0_ent0.01_seed1234/checkpoints/global_step_10/actor 20
### Example (with seed): sbatch examples/mll_cluster/run_embodiment_naive_lora_entropy.slurm 0 0.01 "" "" 42
### Note: ENTROPY_BONUS is required (e.g., 0.01, 0.001, 0.0)
###       CHECKPOINT_PATH is optional and will be auto-generated from previous task if not provided
###       MAX_EPOCH is optional and can always be specified to override the default max_epochs
###       SEED is optional and defaults to 1234 if not provided

TASK_ID=$1
ENTROPY_BONUS=$2
MANUAL_CHECKPOINT_PATH=$3
MAX_EPOCH=$4
SEED=${5:-1234}
CONFIG_NAME="mll_cluster/libero_spatial_grpo_openvlaoft"

if [ -z "$TASK_ID" ] || [ -z "$ENTROPY_BONUS" ]; then
    echo "ERROR: Missing required arguments"
    echo "Usage: sbatch examples/mll_cluster/run_embodiment_naive_lora_entropy.slurm TASK_ID ENTROPY_BONUS [CHECKPOINT_PATH] [MAX_EPOCH] [SEED]"
    exit 1
fi

# Validate entropy_bonus is a valid number
if ! [[ "$ENTROPY_BONUS" =~ ^[0-9]+\.?[0-9]*$ ]]; then
    echo "ERROR: ENTROPY_BONUS must be a valid number, got: $ENTROPY_BONUS"
    exit 1
fi

# Validate seed is a number
if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
    echo "ERROR: SEED must be a non-negative integer, got: $SEED"
    exit 1
fi

# Automatically determine checkpoint path if not manually provided
if [ -z "$MANUAL_CHECKPOINT_PATH" ]; then
    if [ "$TASK_ID" -eq 0 ]; then
        CHECKPOINT_PATH=""
    else
        PREV_TASK_ID=$((TASK_ID - 1))
        CHECKPOINT_PATH="./logs/naive_lora_entropy/task_${PREV_TASK_ID}_ent${ENTROPY_BONUS}_seed${SEED}/checkpoints/global_step_10/actor"
    fi
else
    CHECKPOINT_PATH="$MANUAL_CHECKPOINT_PATH"
fi

# Determine LOG_DIR - always include entropy bonus and seed in the name
LOG_DIR="./logs/naive_lora_entropy/task_${TASK_ID}_ent${ENTROPY_BONUS}_seed${SEED}"

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
mkdir -p logs/naive_lora_entropy
export LOG_DIR

# Set experiment name based on LOG_DIR (for wandb)
EXPERIMENT_NAME=$(basename "$LOG_DIR")

# Update SLURM job name to match experiment name
if command -v scontrol &> /dev/null && [ -n "$SLURM_JOB_ID" ]; then
    scontrol update job=$SLURM_JOB_ID name="${EXPERIMENT_NAME}" 2>/dev/null || true
fi

echo "========================================="
echo "Lifelong Learning - Single Task Training (Naive LoRA with Entropy Bonus)"
echo "========================================="
echo "Configuration:"
echo "  Task ID: $TASK_ID"
echo "  Entropy Bonus: $ENTROPY_BONUS"
echo "  Experiment Name: $EXPERIMENT_NAME"
echo "  Checkpoint Save Path: $LOG_DIR"
echo "  Config Name: $CONFIG_NAME"
echo "  Random Seed: $SEED"

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
	actor.checkpoint_save_path=${LOG_DIR} \
	algorithm.entropy_bonus=${ENTROPY_BONUS} \
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
fi
echo "Finished at: $(date)"
echo "========================================="

exit $EXIT_CODE
