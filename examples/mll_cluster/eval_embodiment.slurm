#!/bin/bash
#SBATCH --job-name=eval_embodiment      # Job name
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

### Usage: sbatch examples/mll_cluster/eval_embodiment.slurm CHECKPOINT_LOCATION
### Example: sbatch examples/mll_cluster/eval_embodiment.slurm logs/bcrl_logit/0.3/task_0
### Note: CHECKPOINT_LOCATION should be relative to workspace root (e.g., logs/bcrl_logit/0.3/task_0)
###       The script will construct the full path: ${WORKSPACE_ROOT}/${CHECKPOINT_LOCATION}/checkpoints/global_step_10/actor/

CHECKPOINT_LOCATION=$1

if [ -z "$CHECKPOINT_LOCATION" ]; then
    echo "ERROR: Missing required argument"
    echo "Usage: sbatch examples/mll_cluster/eval_embodiment.slurm CHECKPOINT_LOCATION"
    echo "Example: sbatch examples/mll_cluster/eval_embodiment.slurm logs/bcrl_logit/0.3/task_0"
    exit 1
fi

# Get workspace root (SLURM_SUBMIT_DIR is the directory where sbatch was run from)
cd $SLURM_SUBMIT_DIR
WORKSPACE_ROOT=$(pwd)
mkdir -p logs/slurm

# Update SLURM job name to be more descriptive (based on CHECKPOINT_LOCATION)
# Extract meaningful path components from CHECKPOINT_LOCATION
# e.g., "logs/bcrl_logit/0.3/task_0" -> "bcrl_logit_0.3_task_0"
if command -v scontrol &> /dev/null && [ -n "$SLURM_JOB_ID" ]; then
    # Convert path to job name by replacing slashes with underscores and removing "logs/" prefix
    JOB_NAME=$(echo "$CHECKPOINT_LOCATION" | sed 's|^logs/||' | tr '/' '_')
    scontrol update job=$SLURM_JOB_ID name="${JOB_NAME}" 2>/dev/null || true
fi

# Print SLURM job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "GPUs allocated: $SLURM_GPUS_ON_NODE"
echo ""

# Call the bash script that contains the core evaluation logic
bash examples/mll_cluster/eval_embodiment.sh "$CHECKPOINT_LOCATION"

# Print completion time
echo ""
echo "SLURM Job End Time: $(date)"

