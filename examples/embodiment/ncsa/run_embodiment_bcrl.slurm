#!/bin/bash
#SBATCH --job-name=ll_t%t_bc%b
#SBATCH --output=logs/task_%t_bc_%b_%j.out
#SBATCH --error=logs/task_%t_bc_%b_%j.err
#SBATCH --partition=gpuA100x4
#SBATCH --mem=240G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest
#SBATCH --account=bfuf-delta-gpu
#SBATCH --constraint="scratch"
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --time=24:00:00


### Usage: sbatch run_embodiment_bcrl.sh TASK_ID BC_COEFF [CHECKPOINT_PATH]
### Example: sbatch run_embodiment_bcrl.sh 0 0.1 ./logs/...

TASK_ID=$1
BC_COEFF=$2
CHECKPOINT_PATH=$3

if [ -z "$TASK_ID" ] || [ -z "$BC_COEFF" ]; then
    echo "ERROR: Missing required arguments"
    echo "Usage: sbatch run_embodiment_bcrl.sh TASK_ID BC_COEFF [CHECKPOINT_PATH]"
    exit 1
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rlinf

mkdir -p logs/slurm
mkdir -p logs/bcrl
mkdir -p logs/bcrl/${BC_COEFF}
LOG_DIR="./logs/bcrl/${BC_COEFF}/task_${TASK_ID}"
export LOG_DIR

EXPERIMENT_NAME="bcrl_task_${TASK_ID}_${BC_COEFF}"

echo "========================================="
echo "Lifelong Learning - Single Task Training"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo ""
echo "Configuration:"
echo "  Task ID: $TASK_ID"
echo "  BC Coefficient: $BC_COEFF"
echo "  Experiment Name: $EXPERIMENT_NAME"
echo "  Checkpoint Save Path: $LOG_DIR"

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
	runner.max_epochs=10 \
	runner.save_interval=10 \
	runner.logger.experiment_name=${EXPERIMENT_NAME} \
	actor.checkpoint_save_path=${LOG_DIR}"

if [ -n "$CHECKPOINT_PATH" ]; then
    OVERRIDES="$OVERRIDES ${CHECKPOINT_ARG}"
fi

echo "Running with Hydra overrides:"
echo "$OVERRIDES"
echo ""

bash examples/embodiment/run_embodiment.sh libero_spatial_grpo_openvlaoft_bcrl $OVERRIDES

EXIT_CODE=$?
echo ""
echo "========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Task $TASK_ID with BC_COEFF=$BC_COEFF completed successfully"
    echo ""
    echo "Checkpoint saved to: ${LOG_DIR}/global_step_10"
else
    echo "✗ Task $TASK_ID failed with exit code $EXIT_CODE"
    echo "  Check logs at: logs/slurm/task_${TASK_ID}_bc_${BC_COEFF}_${SLURM_JOB_ID}.out"
fi
echo "Finished at: $(date)"
echo "========================================="

exit $EXIT_CODE
