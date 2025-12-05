#!/bin/bash
#SBATCH --job-name=eval_embodiment      # Job name
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks-per-node=1             # Number of tasks per node
#SBATCH --cpus-per-task=80              # Number of CPUs per task
#SBATCH --gres=gpu:4                    # Number of GPUs (RECOMMENDED: 4+ GPUs)
#SBATCH --time=4:00:00                  # Maximum execution time (HH:MM:SS)
#SBATCH --error=logs/%x-%j.err               # Error file name
#SBATCH --mem=510G                       # Memory per node (adjust as needed)
#SBATCH --partition=mll            # Partition name (adjust to your cluster's partition)
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=jiahengh@utexas.edu
#SBATCH --exclude=slurm-node-004

# =============================================================================
# MODULE LOADING
# =============================================================================
# #SBATCH --output=%x-%j.out              # Output file name (%x = job name, %j = job ID)
# Common modules to load (uncomment and adjust for your cluster):
# module purge                          # Clear existing modules
# module load cuda/11.8                 # CUDA toolkit (adjust version)
# module load python/3.11               # Python (if not using conda/venv)
# module load gcc/9.3.0                 # Compiler (if needed)
# module list                           # Verify loaded modules

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "GPUs allocated: $SLURM_GPUS_ON_NODE"

LORA_PATH=${1} # "run_1.0"
rlinf  # Starts the conda environment
bash examples/embodiment/eval_embodiment.sh mll_cluster/libero_spatial_grpo_openvlaoft_eval actor.model.lora_path="logs/${LORA_PATH}/checkpoints/global_step_10/actor/"
 # For low memory setup

# Print completion time
echo "End Time: $(date)"

