#!/bin/bash
#SBATCH --job-name=multitask_lora      # Job name
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks-per-node=1             # Number of tasks per node
#SBATCH --cpus-per-task=90              # Number of CPUs per task
#SBATCH --gres=gpu:8                    # Number of GPUs (RECOMMENDED: 4+ GPUs)
#SBATCH --time=70:00:00                  # Maximum execution time (HH:MM:SS)
#SBATCH --mem=900G                      # Memory per node (adjust as needed)
#SBATCH --partition=mll                 # Partition name (adjust to your cluster's partition)
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=jiahengh@utexas.edu
#SBATCH --exclude=slurm-node-007
#SBATCH --output=logs/slurm/%x-%j.out              # Output file name (%x = job name, %j = job ID)
#SBATCH --error=logs/slurm/%x-%j.err               # Error file name


### Usage: sbatch examples/mll_cluster/run_embodiment_naive_lora_multitask.slurm TASK_IDS [MAX_EPOCH] [CONFIG_NAME]
### Example: sbatch examples/mll_cluster/run_embodiment_naive_lora_multitask.slurm "0,2,4"
### Example: sbatch examples/mll_cluster/run_embodiment_naive_lora_multitask.slurm "0 2 4"
### Example (with max_epoch): sbatch examples/mll_cluster/run_embodiment_naive_lora_multitask.slurm "0,2,4" 15
### Note: TASK_IDS can be comma-separated (e.g., "0,2,4") or space-separated (e.g., "0 2 4")
###       Evaluation runs automatically every 10 global steps during training
###       MAX_EPOCH is optional and can override the default max_epochs

# Call the shared script with all arguments (path relative to workspace root)
bash examples/mll_cluster/run_embodiment_naive_lora_multitask.sh "$@"
