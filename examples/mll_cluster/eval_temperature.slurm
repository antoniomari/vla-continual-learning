#!/bin/bash
#SBATCH --job-name=eval_temperature      # Job name
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks-per-node=1             # Number of tasks per node
#SBATCH --cpus-per-task=96              # Number of CPUs per task
#SBATCH --gres=gpu:8                    # Number of GPUs (RECOMMENDED: 8 GPUs for evaluation)
#SBATCH --time=24:00:00                  # Maximum execution time (HH:MM:SS)
#SBATCH --mem=980G                       # Memory per node (adjusted as needed)
# #SBATCH --partition=mll            # Partition name (adjust to your cluster's partition)
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=jiahengh@utexas.edu
# #SBATCH --exclude=slurm-node-008,slurm-node-011
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

### Usage: sbatch examples/mll_cluster/eval_temperature.slurm CHECKPOINT_LOCATION TEMPERATURE [STEP_NUMBER]
### Example: sbatch examples/mll_cluster/eval_temperature.slurm logs/naive_LoRA/task_0 2.0
### Example: sbatch examples/mll_cluster/eval_temperature.slurm logs/bcrl_logit/0.3/task_0 1.5 20
### Note: CHECKPOINT_LOCATION should be relative to workspace root (e.g., logs/naive_LoRA/task_0)
###       The script will construct the full path: ${WORKSPACE_ROOT}/${CHECKPOINT_LOCATION}/checkpoints/global_step_${STEP_NUMBER}/actor/
###       TEMPERATURE: evaluation temperature to use (default: 2.0)
###       STEP_NUMBER: global step number (default: 10)

CHECKPOINT_LOCATION=$1
TEMPERATURE=$2
STEP_NUMBER=$3

# Default STEP_NUMBER to 10 if not provided
if [ -z "$STEP_NUMBER" ]; then
    STEP_NUMBER=10
fi

# Get workspace root (SLURM_SUBMIT_DIR is the directory where sbatch was run from)
cd $SLURM_SUBMIT_DIR
WORKSPACE_ROOT=$(pwd)
mkdir -p logs/slurm

if [ -z "$CHECKPOINT_LOCATION" ] || [ -z "$TEMPERATURE" ]; then
    echo "ERROR: Missing required argument(s)"
    echo "Usage: sbatch examples/mll_cluster/eval_temperature.slurm CHECKPOINT_LOCATION TEMPERATURE [STEP_NUMBER]"
    echo "Example: sbatch examples/mll_cluster/eval_temperature.slurm logs/naive_LoRA/task_0 2.0"
    echo "Example: sbatch examples/mll_cluster/eval_temperature.slurm logs/bcrl_logit/0.3/task_0 1.5 20"
    exit 1
fi

# Validate TEMPERATURE
if ! [[ "$TEMPERATURE" =~ ^[0-9]+\.?[0-9]*$ ]]; then
    echo "ERROR: TEMPERATURE must be a number, got: $TEMPERATURE"
    exit 1
fi

# Validate STEP_NUMBER
if ! [[ "$STEP_NUMBER" =~ ^[0-9]+$ ]]; then
    echo "ERROR: STEP_NUMBER must be a positive integer, got: $STEP_NUMBER"
    exit 1
fi

# Construct full checkpoint path (remove any trailing slashes from CHECKPOINT_LOCATION first)
CHECKPOINT_LOCATION="${CHECKPOINT_LOCATION%/}"
CHECKPOINT_PATH="${WORKSPACE_ROOT}/${CHECKPOINT_LOCATION}/checkpoints/global_step_${STEP_NUMBER}/actor/"

# Verify checkpoint exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT_PATH"
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
echo "Checkpoint Location: $CHECKPOINT_LOCATION"
echo "Full Checkpoint Path: $CHECKPOINT_PATH"
echo "Global Step Number: $STEP_NUMBER"
echo "Evaluation Temperature: $TEMPERATURE"
echo ""
echo "Note: Log path will be automatically constructed by eval_embodiment.sh"
echo "      and will include temperature in the directory name."
echo ""

# Update SLURM job name to be more descriptive (based on CHECKPOINT_LOCATION and temperature)
# Convert CHECKPOINT_LOCATION path to job name, then add temperature info
# e.g., "logs/naive_LoRA/task_0" -> "naive_LoRA_task_0_temp_2_0"
if command -v scontrol &> /dev/null && [ -n "$SLURM_JOB_ID" ]; then
    # Convert path to job name by replacing slashes with underscores and removing "logs/" prefix if present
    JOB_NAME_BASE=$(echo "$CHECKPOINT_LOCATION" | sed 's|^logs/||' | tr '/' '_')
    # Add temperature info
    TEMP_STR=$(echo "$TEMPERATURE" | tr '.' '_')
    JOB_NAME="${JOB_NAME_BASE}_temp_${TEMP_STR}"
    scontrol update job=$SLURM_JOB_ID name="${JOB_NAME}" 2>/dev/null || true
fi

# Run evaluation with temperature override
bash examples/embodiment/eval_embodiment.sh mll_cluster/libero_spatial_grpo_openvlaoft_eval \
    +actor.model.lora_path="${CHECKPOINT_PATH}" \
    algorithm.sampling_params.temperature_eval=${TEMPERATURE}

# Check if evaluation succeeded
if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Evaluation failed"
    echo "Checkpoint path: ${CHECKPOINT_PATH}"
    echo "Temperature: ${TEMPERATURE}"
    exit 1
fi

# Print completion time
echo ""
echo "Evaluation completed successfully!"
echo "End Time: $(date)"
