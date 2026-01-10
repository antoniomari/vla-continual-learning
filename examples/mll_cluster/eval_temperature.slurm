#!/bin/bash
#SBATCH --job-name=eval_temperature      # Job name
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks-per-node=1             # Number of tasks per node
#SBATCH --cpus-per-task=80              # Number of CPUs per task
#SBATCH --gres=gpu:4                    # Number of GPUs (RECOMMENDED: 4+ GPUs)
#SBATCH --time=24:00:00                  # Maximum execution time (HH:MM:SS)
#SBATCH --mem=490G                       # Memory per node (adjust as needed)
# #SBATCH --partition=mll            # Partition name (adjust to your cluster's partition)
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=jiahengh@utexas.edu
#SBATCH --exclude=slurm-node-008,slurm-node-011
#SBATCH --output=logs/slurm/%x-%j.out              # Output file name (%x = job name, %j = job ID)
#SBATCH --error=logs/slurm/%x-%j.err               # Error file name

# =============================================================================
# MODULE LOADING
# =============================================================================
# Common modules to load (uncomment and adjust for your cluster):
# module purge                          # Clear existing modules
# module load cuda/11.8                 # CUDA toolkit (adjust version)
# module load python/3.11               # Python (if not using conda/venv)
# module load gcc/9.3.0                 # Compiler (if needed)
# module list                           # Verify loaded modules

### Usage: sbatch examples/mll_cluster/eval_temperature.slurm TASK_ID1 [TASK_ID2 ...]
### Example: sbatch examples/mll_cluster/eval_temperature.slurm 0 1 2 3 4
### Example: sbatch examples/mll_cluster/eval_temperature.slurm 2 3
### Note: Task IDs should correspond to directories in logs/naive_LoRA/task_{TASK_ID}/
###       The script will automatically find checkpoints at:
###       logs/naive_LoRA/task_{TASK_ID}/checkpoints/global_step_10/actor/
###       Temperature is set to 2.0 to match training temperature_train from libero_spatial_grpo_openvlaoft.yaml

# Get workspace root (SLURM_SUBMIT_DIR is the directory where sbatch was run from)
cd $SLURM_SUBMIT_DIR
WORKSPACE_ROOT=$(pwd)
mkdir -p logs/slurm

# Get task IDs from command line arguments
TASK_IDS=("$@")

if [ ${#TASK_IDS[@]} -eq 0 ]; then
    echo "ERROR: Missing required argument(s)"
    echo "Usage: sbatch examples/mll_cluster/eval_temperature.slurm TASK_ID1 [TASK_ID2 ...]"
    echo "Example: sbatch examples/mll_cluster/eval_temperature.slurm 0 1 2 3 4"
    echo "Example: sbatch examples/mll_cluster/eval_temperature.slurm 2 3"
    exit 1
fi

# Base log directory for naive LoRA checkpoints
BASE_LOG_DIR="${WORKSPACE_ROOT}/logs/naive_LoRA"

# Temperature to use for evaluation (matches temperature_train from training config)
EVAL_TEMPERATURE=2.0

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "GPUs allocated: $SLURM_GPUS_ON_NODE"
echo ""
echo "Task IDs to evaluate: ${TASK_IDS[@]}"
echo "Base Log Directory: ${BASE_LOG_DIR}"
echo "Evaluation Temperature: ${EVAL_TEMPERATURE} (matching training temperature_train)"
echo ""

# Loop through each task ID and evaluate
for TASK_ID in "${TASK_IDS[@]}"; do
    echo ""
    echo "========================================================================"
    echo "Evaluating Task ${TASK_ID}"
    echo "========================================================================"
    
    # Construct checkpoint path
    TASK_LOG_DIR="${BASE_LOG_DIR}/task_${TASK_ID}"
    CHECKPOINT_PATH="${TASK_LOG_DIR}/checkpoints/global_step_10/actor/"
    
    # Verify checkpoint exists
    if [ ! -d "$CHECKPOINT_PATH" ]; then
        echo "ERROR: Checkpoint not found at $CHECKPOINT_PATH"
        echo "Skipping task ${TASK_ID}..."
        continue
    fi
    
    echo "Task ID: ${TASK_ID}"
    echo "Checkpoint Path: ${CHECKPOINT_PATH}"
    echo ""
    
    # Run evaluation with temperature override to match training temperature
    # Using libero_spatial_grpo_openvlaoft_eval config with temperature_eval override
    bash examples/embodiment/eval_embodiment.sh mll_cluster/libero_spatial_grpo_openvlaoft_eval \
        +actor.model.lora_path="${CHECKPOINT_PATH}" \
        algorithm.sampling_params.temperature_eval=${EVAL_TEMPERATURE}
    
    # Check if evaluation succeeded
    if [ $? -ne 0 ]; then
        echo ""
        echo "ERROR: Evaluation failed for task ${TASK_ID}"
        echo "Checkpoint path: ${CHECKPOINT_PATH}"
        # Continue with next task instead of exiting
        continue
    fi
    
    echo ""
    echo "Task ${TASK_ID} evaluation completed successfully"
    echo "========================================================================"
done

# Print completion time
echo ""
echo "All evaluations completed!"
echo "End Time: $(date)"
