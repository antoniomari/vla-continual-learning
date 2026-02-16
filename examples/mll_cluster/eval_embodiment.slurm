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

### Usage: sbatch examples/mll_cluster/eval_embodiment.slurm CHECKPOINT_LOCATION [STEP_NUMBER ...] [CONFIG_NAME]
### Example: sbatch examples/mll_cluster/eval_embodiment.slurm logs/bcrl_logit/0.3/task_0
### Example: sbatch examples/mll_cluster/eval_embodiment.slurm logs/bcrl_logit/0.3/task_0 20
### Example: sbatch examples/mll_cluster/eval_embodiment.slurm logs/bcrl_logit/0.3/task_0 10 20 30
### Example (with config): sbatch examples/mll_cluster/eval_embodiment.slurm logs/bcrl_logit/0.3/task_0 20 mll_cluster/libero_spatial_grpo_openvlaoft_eval_cam
### Note: CHECKPOINT_LOCATION should be relative to workspace root (e.g., logs/bcrl_logit/0.3/task_0)
###       The script will construct the full path: ${WORKSPACE_ROOT}/${CHECKPOINT_LOCATION}/checkpoints/global_step_${STEP_NUMBER}/actor/
###       STEP_NUMBER: global step number(s) - if multiple provided, all will be evaluated sequentially (default: 10)
###       CONFIG_NAME: optional config name (default: mll_cluster/libero_spatial_grpo_openvlaoft_eval)

CHECKPOINT_LOCATION=$1
shift  # Remove CHECKPOINT_LOCATION from $@, leaving remaining arguments

if [ -z "$CHECKPOINT_LOCATION" ]; then
    echo "ERROR: Missing required argument"
    echo "Usage: sbatch examples/mll_cluster/eval_embodiment.slurm CHECKPOINT_LOCATION [STEP_NUMBER ...] [CONFIG_NAME]"
    echo "Example: sbatch examples/mll_cluster/eval_embodiment.slurm logs/bcrl_logit/0.3/task_0"
    echo "Example: sbatch examples/mll_cluster/eval_embodiment.slurm logs/bcrl_logit/0.3/task_0 20"
    echo "Example: sbatch examples/mll_cluster/eval_embodiment.slurm logs/bcrl_logit/0.3/task_0 10 20 30"
    echo "Example (with config): sbatch examples/mll_cluster/eval_embodiment.slurm logs/bcrl_logit/0.3/task_0 20 mll_cluster/libero_spatial_grpo_openvlaoft_eval_cam"
    exit 1
fi

# Parse remaining arguments: STEP_NUMBERS and optional CONFIG_NAME
# CONFIG_NAME should be the last non-numeric argument (if any)
STEP_NUMBERS=()
CONFIG_NAME=""
NON_NUMERIC_ARGS=()

for arg in "$@"; do
    if [[ "$arg" =~ ^[0-9]+$ ]]; then
        # It's a numeric step number
        STEP_NUMBERS+=("$arg")
    else
        # It's a non-numeric argument (could be CONFIG_NAME)
        NON_NUMERIC_ARGS+=("$arg")
    fi
done

# Validate: should have at most one non-numeric argument (the CONFIG_NAME)
if [ ${#NON_NUMERIC_ARGS[@]} -gt 1 ]; then
    echo "ERROR: Multiple non-numeric arguments found: ${NON_NUMERIC_ARGS[*]}"
    echo "       Only one CONFIG_NAME argument is allowed (should be the last argument)"
    exit 1
elif [ ${#NON_NUMERIC_ARGS[@]} -eq 1 ]; then
    CONFIG_NAME="${NON_NUMERIC_ARGS[0]}"
fi

# If no STEP_NUMBER arguments provided, default to 10
if [ ${#STEP_NUMBERS[@]} -eq 0 ]; then
    STEP_NUMBERS=(10)
fi

# Validate all STEP_NUMBER arguments (should already be validated above, but double-check)
for STEP_NUM in "${STEP_NUMBERS[@]}"; do
    if ! [[ "$STEP_NUM" =~ ^[0-9]+$ ]]; then
        echo "ERROR: All STEP_NUMBER arguments must be positive integers, got: $STEP_NUM"
        exit 1
    fi
done

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
echo "Checkpoint Location: $CHECKPOINT_LOCATION"
if [ -n "$CONFIG_NAME" ]; then
    echo "Config Name: $CONFIG_NAME"
else
    echo "Config Name: (default: mll_cluster/libero_spatial_grpo_openvlaoft_eval)"
fi
echo "Step Numbers: ${STEP_NUMBERS[*]}"
echo ""

# Call the bash script that contains the core evaluation logic for each STEP_NUMBER
TOTAL_STEPS=${#STEP_NUMBERS[@]}
CURRENT_STEP=0

for STEP_NUMBER in "${STEP_NUMBERS[@]}"; do
    CURRENT_STEP=$((CURRENT_STEP + 1))
    echo "================================================================================"
    echo "Evaluating step $STEP_NUMBER ($CURRENT_STEP of $TOTAL_STEPS)"
    echo "================================================================================"
    echo ""
    
    # Call the bash script with the current STEP_NUMBER and CONFIG_NAME (if provided)
    if [ -n "$CONFIG_NAME" ]; then
        bash examples/mll_cluster/eval_embodiment.sh "$CHECKPOINT_LOCATION" "$STEP_NUMBER" "$CONFIG_NAME"
    else
        bash examples/mll_cluster/eval_embodiment.sh "$CHECKPOINT_LOCATION" "$STEP_NUMBER"
    fi
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -ne 0 ]; then
        echo ""
        echo "ERROR: Evaluation failed for step $STEP_NUMBER"
        echo "Aborting remaining evaluations."
        exit $EXIT_CODE
    fi
    
    echo ""
    echo "Successfully completed evaluation for step $STEP_NUMBER"
    echo ""
done

echo "================================================================================"
echo "All evaluations completed successfully!"
echo "Evaluated steps: ${STEP_NUMBERS[*]}"
echo "================================================================================"

# Print completion time
echo ""
echo "SLURM Job End Time: $(date)"

