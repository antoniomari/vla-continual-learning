#!/bin/bash
#SBATCH --job-name=train_embodiment_multilora   # Job name
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --ntasks-per-node=1                     # Number of tasks per node
#SBATCH --cpus-per-task=90                      # Number of CPUs per task
#SBATCH --gres=gpu:8                            # Number of GPUs
#SBATCH --time=70:00:00                         # Maximum execution time (HH:MM:SS)
#SBATCH --mem=900G                              # Memory per node (adjust as needed)
#SBATCH --partition=mll                         # Partition name
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=jiahengh@utexas.edu
#SBATCH --exclude=slurm-node-004,slurm-node-011
#SBATCH --output=logs/slurm/%x-%j.out              # Output file name (%x = job name, %j = job ID)
#SBATCH --error=logs/slurm/%x-%j.err               # Error file name

### Usage: sbatch examples/mll_cluster/run_embodiment_multilora.slurm TASK_ID [MERGE_COEFFICIENT] [CONFIG_NAME]
### Example: sbatch examples/mll_cluster/run_embodiment_multilora.slurm 0
### Example: sbatch examples/mll_cluster/run_embodiment_multilora.slurm 1 0.8
### Example: sbatch examples/mll_cluster/run_embodiment_multilora.slurm 2 0.8
###
### Behavior:
###   - TASK_ID=0: Train first LoRA adapter on top of base model (same as naive LoRA)
###   - TASK_ID>0: Load and merge ALL previous task adapters (0..TASK_ID-1) into base,
###                then create a NEW LoRA adapter for this task (only this one is trainable).
###                Note: Task 0 adapter is loaded from naive_lora folder (reuses existing training),
###                      while tasks 1+ are loaded from naive_lora_multilora folder.
###
### MERGE_COEFFICIENT (optional, default=1.0):
###   - Coefficient applied when merging previous LoRA adapters (0.0 to 1.0)
###   - 1.0 = full weight of previous adapters, 0.0 = ignore previous adapters
###   - The coefficient is included in the checkpoint path (e.g., task_1_coeff_0_8)
###   - When loading previous adapters, automatically uses adapters trained with the same coefficient
###   - If not specified, defaults to 1.0

TASK_ID=$1
MERGE_COEFFICIENT=$2
CONFIG_NAME=${3:-mll_cluster/libero_spatial_grpo_openvlaoft}  # Default config if not provided

if [ -z "$TASK_ID" ]; then
    echo "ERROR: Missing required argument"
    echo "Usage: sbatch examples/mll_cluster/run_embodiment_multilora.slurm TASK_ID [MERGE_COEFFICIENT] [CONFIG_NAME]"
    exit 1
fi

# Validate MERGE_COEFFICIENT if provided
if [ -n "$MERGE_COEFFICIENT" ]; then
    if ! [[ "$MERGE_COEFFICIENT" =~ ^[0-9]+\.?[0-9]*$ ]]; then
        echo "ERROR: MERGE_COEFFICIENT must be a number, got: $MERGE_COEFFICIENT"
        exit 1
    fi
    if ! awk "BEGIN {exit !($MERGE_COEFFICIENT >= 0 && $MERGE_COEFFICIENT <= 1)}"; then
        echo "ERROR: MERGE_COEFFICIENT must be between 0.0 and 1.0, got: $MERGE_COEFFICIENT"
        exit 1
    fi
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

# Determine merge coefficient (default to 1.0 if not specified)
if [ -z "$MERGE_COEFFICIENT" ]; then
    MERGE_COEFFICIENT="1.0"
fi

# Format coefficient for use in path (replace dot with underscore, e.g., 0.8 -> 0_8)
COEFFICIENT_PATH=$(echo "$MERGE_COEFFICIENT" | tr '.' '_')

# Include coefficient in the log directory path
mkdir -p logs/slurm
mkdir -p logs/naive_lora_multilora
if [ "$TASK_ID" -eq 0 ]; then
    # Task 0 doesn't use coefficient in path (no previous adapters to merge)
    LOG_DIR="./logs/naive_lora_multilora/task_${TASK_ID}"
    EXPERIMENT_NAME="multilora_task_${TASK_ID}"
else
    # Tasks 1+ include coefficient in path and experiment name
    LOG_DIR="./logs/naive_lora_multilora/task_${TASK_ID}_coeff_${COEFFICIENT_PATH}"
    EXPERIMENT_NAME="multilora_task_${TASK_ID}_coeff_${COEFFICIENT_PATH}"
fi
export LOG_DIR

# Update SLURM job name to be more descriptive (based on LOG_DIR)
# Extract the last component of LOG_DIR (e.g., "task_0" or "task_1_coeff_0_8")
# This makes the job name match the log directory structure exactly
LOG_DIR_BASENAME=$(basename "$LOG_DIR")
if command -v scontrol &> /dev/null && [ -n "$SLURM_JOB_ID" ]; then
    # Use the full directory name from LOG_DIR as the job name
    # This will show as "task_0" or "task_1_coeff_0_8" in SLURM, matching LOG_DIR exactly
    scontrol update job=$SLURM_JOB_ID name="${LOG_DIR_BASENAME}" 2>/dev/null || true
fi

echo "========================================="
echo "Lifelong Learning - Multi-LoRA Training"
echo "========================================="
echo "Configuration:"
echo "  Task ID: $TASK_ID"
echo "  Experiment Name: $EXPERIMENT_NAME"
echo "  Checkpoint Save Path: $LOG_DIR"
echo "  Config Name: $CONFIG_NAME"
echo "  Merge Coefficient: ${MERGE_COEFFICIENT}"

LORA_PATHS_ARG=""
MERGE_COEFFICIENT_ARG="+actor.model.previous_lora_merge_coefficient=${MERGE_COEFFICIENT}"

if [ "$TASK_ID" -gt 0 ]; then
    echo "  Mode: MULTI-LORA - Loading and merging all previous task adapters (0..$((TASK_ID - 1)))"
    echo "  Looking for previous adapters with coefficient ${MERGE_COEFFICIENT}..."
    LORA_PATHS=()
    for prev_task in $(seq 0 $((TASK_ID - 1))); do
        # Task 0 adapter comes from naive_lora folder (no coefficient in path)
        if [ "$prev_task" -eq 0 ]; then
            prev_adapter_path="./logs/naive_lora/task_${prev_task}/checkpoints/global_step_10/actor"
        else
            # Tasks 1+ use coefficient in path to find matching adapters
            prev_adapter_path="./logs/naive_lora_multilora/task_${prev_task}_coeff_${COEFFICIENT_PATH}/checkpoints/global_step_10/actor"
        fi
        
        if [ -d "$prev_adapter_path" ]; then
            LORA_PATHS+=("$prev_adapter_path")
            echo "    Found adapter for task $prev_task: $prev_adapter_path"
        else
            echo "    Warning: Adapter not found for task $prev_task: $prev_adapter_path"
        fi
    done

    if [ ${#LORA_PATHS[@]} -gt 0 ]; then
        # Convert array to comma-separated string for Hydra
        LORA_PATHS_STR=$(IFS=','; echo "${LORA_PATHS[*]}")
        LORA_PATHS_ARG="+actor.model.lora_paths=[${LORA_PATHS_STR}]"
        echo "  Will merge ${#LORA_PATHS[@]} previous adapter(s) into base model, then train new adapter."
    else
        echo "  Warning: No previous adapters found; training from base model."
    fi
else
    echo "  Mode: FIRST TASK - Training first LoRA adapter on base model."
fi

echo "========================================="
echo ""

cd $SLURM_SUBMIT_DIR

# Hydra overrides
OVERRIDES="env.fixed_task_ids=[${TASK_ID}] \
	runner.logger.experiment_name=${EXPERIMENT_NAME} \
	actor.checkpoint_save_path=${LOG_DIR}"

if [ -n "$LORA_PATHS_ARG" ]; then
    OVERRIDES="$OVERRIDES ${LORA_PATHS_ARG}"
fi

if [ -n "$MERGE_COEFFICIENT_ARG" ]; then
    OVERRIDES="$OVERRIDES ${MERGE_COEFFICIENT_ARG}"
fi

echo "Running with Hydra overrides:"
echo "$OVERRIDES"
echo ""

bash examples/embodiment/run_embodiment.sh ${CONFIG_NAME} $OVERRIDES

EXIT_CODE=$?
echo ""
echo "========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Task $TASK_ID completed successfully (Multi-LoRA)"
    echo ""
    echo "Checkpoint saved to: ${LOG_DIR}"
    echo "  (Merge coefficient is automatically saved with checkpoint)"
else
    echo "✗ Task $TASK_ID (Multi-LoRA) failed with exit code $EXIT_CODE"
    echo "  Check logs at: logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
fi
echo "Finished at: $(date)"
echo "========================================="

exit $EXIT_CODE

