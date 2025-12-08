#!/bin/bash
#SBATCH --job-name=train_embodiment      # Job name
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks-per-node=1             # Number of tasks per node
#SBATCH --cpus-per-task=90              # Number of CPUs per task
#SBATCH --gres=gpu:8                    # Number of GPUs (RECOMMENDED: 4+ GPUs)
#SBATCH --time=35:00:00                  # Maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out              # Output file name (%x = job name, %j = job ID)
#SBATCH --error=%x-%j.err               # Error file name
#SBATCH --mem=900G                      # Memory per node (adjust as needed)
#SBATCH --partition=mll                 # Partition name (adjust to your cluster's partition)
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=jiahengh@utexas.edu
#SBATCH --exclude=slurm-node-004


### Usage: sbatch run_embodiment.slurm TASK_ID BC_COEFF [CHECKPOINT_PATH] [CONFIG_NAME]
### Example: sbatch run_embodiment.slurm 0 0.1 ./logs/... libero_spatial_grpo_openvlaoft_bcrl_logit

TASK_ID=$1
BC_COEFF=$2
CHECKPOINT_PATH=$3
CONFIG_NAME=${4:-libero_spatial_grpo_openvlaoft_bcrl_logit}  # Default config if not provided

if [ -z "$TASK_ID" ] || [ -z "$BC_COEFF" ]; then
    echo "ERROR: Missing required arguments"
    echo "Usage: sbatch run_embodiment.slurm TASK_ID BC_COEFF [CHECKPOINT_PATH] [CONFIG_NAME]"
    exit 1
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

mkdir -p logs/bcrl_logit
mkdir -p logs/bcrl_logit/${BC_COEFF}
LOG_DIR="./logs/bcrl_logit/${BC_COEFF}/task_${TASK_ID}"
export LOG_DIR

EXPERIMENT_NAME="bcrl_task_${TASK_ID}_${BC_COEFF}"

echo "========================================="
echo "Lifelong Learning - Single Task Training"
echo "========================================="
echo "Configuration:"
echo "  Task ID: $TASK_ID"
echo "  BC Coefficient: $BC_COEFF"
echo "  Experiment Name: $EXPERIMENT_NAME"
echo "  Checkpoint Save Path: $LOG_DIR"
echo "  Config Name: $CONFIG_NAME"

if [ -n "$CHECKPOINT_PATH" ]; then
    echo "  Loading from checkpoint: $CHECKPOINT_PATH"

    # Verify checkpoint exists
    if [ ! -d "$CHECKPOINT_PATH" ]; then
        echo "  ERROR: Checkpoint not found at $CHECKPOINT_PATH"
        exit 1
    fi
    CHECKPOINT_ARG="actor.lora_path=${CHECKPOINT_PATH}"
else
    echo "  Training from base model (SFT checkpoint)"
fi

echo "========================================="
echo ""

cd $SLURM_SUBMIT_DIR

# hydra overrides
OVERRIDES="algorithm.bc_coeff=${BC_COEFF} \
	env.fixed_task_ids=[${TASK_ID}] \
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
    echo "Task $TASK_ID with BC_COEFF=$BC_COEFF completed successfully"
    echo ""
    echo "Checkpoint saved to: ${LOG_DIR}"
else
    echo "✗ Task $TASK_ID failed with exit code $EXIT_CODE"
    echo "  Check logs at: ${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
fi
echo "Finished at: $(date)"
echo "========================================="

exit $EXIT_CODE

