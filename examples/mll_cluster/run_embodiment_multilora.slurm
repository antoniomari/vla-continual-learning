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
#SBATCH --exclude=slurm-node-008,slurm-node-011
#SBATCH --output=logs/slurm/%x-%j.out              # Output file name (%x = job name, %j = job ID)
#SBATCH --error=logs/slurm/%x-%j.err               # Error file name

### Usage: sbatch examples/mll_cluster/run_embodiment_multilora.slurm TASK_ID [MERGE_COEFFICIENT] [CONFIG_NAME] [SEED]
### Example: sbatch examples/mll_cluster/run_embodiment_multilora.slurm 0
### Example: sbatch examples/mll_cluster/run_embodiment_multilora.slurm 1 0.8
### Example: sbatch examples/mll_cluster/run_embodiment_multilora.slurm 2 0.8
### Example (with seed): sbatch examples/mll_cluster/run_embodiment_multilora.slurm 0 "" "" 42
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
### SEED (optional, default=1234):
###   - Random seed for reproducibility
###   - Included in the checkpoint path to prevent overwriting runs with different seeds

TASK_ID=$1
MERGE_COEFFICIENT=$2
CONFIG_NAME=${3:-mll_cluster/libero_spatial_grpo_openvlaoft}  # Default config if not provided
SEED=${4:-1234}

# Source common functions
source "examples/mll_cluster/common_functions.sh"

# Extract config tag and derive eval config name
CONFIG_TAG=$(extract_config_tag "$CONFIG_NAME")
EVAL_CONFIG_NAME=$(derive_eval_config_name "$CONFIG_NAME")
FIRST_TASK_ID=$(get_first_task_id "$CONFIG_NAME")

if [ -z "$TASK_ID" ]; then
    echo "ERROR: Missing required argument"
    echo "Usage: sbatch examples/mll_cluster/run_embodiment_multilora.slurm TASK_ID [MERGE_COEFFICIENT] [CONFIG_NAME] [SEED]"
    exit 1
fi

# Validate seed is a number
if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
    echo "ERROR: SEED must be a non-negative integer, got: $SEED"
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

# Include coefficient and seed in the log directory path
mkdir -p logs/slurm
mkdir -p logs/naive_lora_multilora
if [ "$TASK_ID" -eq $FIRST_TASK_ID ]; then
    # First task doesn't use coefficient in path (no previous adapters to merge)
    LOG_DIR="./logs/naive_lora_multilora/task_${TASK_ID}_seed${SEED}"
    EXPERIMENT_NAME="multilora_task_${TASK_ID}_seed${SEED}"
else
    # Tasks 1+ include coefficient and seed in path and experiment name
    LOG_DIR="./logs/naive_lora_multilora/task_${TASK_ID}_coeff_${COEFFICIENT_PATH}_seed${SEED}"
    EXPERIMENT_NAME="multilora_task_${TASK_ID}_coeff_${COEFFICIENT_PATH}_seed${SEED}"
fi
# Append CONFIG_TAG to experiment name if set
if [ -n "$CONFIG_TAG" ]; then
    EXPERIMENT_NAME="${EXPERIMENT_NAME}_${CONFIG_TAG}"
fi
# Inject config tag into LOG_DIR before exporting
if [ -n "$CONFIG_TAG" ]; then
    LOG_DIR_TRANSFORMED=$(inject_config_tag_into_log_path "$LOG_DIR" "$CONFIG_TAG")
    if [ -z "$LOG_DIR_TRANSFORMED" ]; then
        echo "  ERROR: Failed to transform LOG_DIR with config tag"
        echo "         Original LOG_DIR: [$LOG_DIR]"
        echo "         CONFIG_TAG: [$CONFIG_TAG]"
        exit 1
    fi
    LOG_DIR="$LOG_DIR_TRANSFORMED"
fi

# Validate LOG_DIR is not empty
if [ -z "$LOG_DIR" ]; then
    echo "  ERROR: LOG_DIR is empty after path construction"
    exit 1
fi

export LOG_DIR

# Update SLURM job name to be more descriptive (based on LOG_DIR)
# Extract the last component of LOG_DIR (e.g., "task_0" or "task_1_coeff_0_8")
# This makes the job name match the log directory structure exactly
LOG_DIR_BASENAME=$(basename "$LOG_DIR")
if command -v scontrol &> /dev/null && [ -n "$SLURM_JOB_ID" ]; then
    # Use the full directory name from LOG_DIR as the job name
    # This will show as "task_0" or "task_1_coeff_0_8" in SLURM, matching LOG_DIR exactly
    JOB_NAME="${LOG_DIR_BASENAME}"
    if [ -n "$CONFIG_TAG" ]; then
        JOB_NAME="${JOB_NAME}_${CONFIG_TAG}"
    fi
    scontrol update job=$SLURM_JOB_ID name="${JOB_NAME}" 2>/dev/null || true
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
echo "  Random Seed: $SEED"

LORA_PATHS_ARG=""
MERGE_COEFFICIENT_ARG="+actor.model.previous_lora_merge_coefficient=${MERGE_COEFFICIENT}"

if [ "$TASK_ID" -gt $FIRST_TASK_ID ]; then
    echo "  Mode: MULTI-LORA - Loading and merging all previous task adapters ($FIRST_TASK_ID..$((TASK_ID - 1)))"
    echo "  Looking for previous adapters with coefficient ${MERGE_COEFFICIENT}..."
    LORA_PATHS=()
    for prev_task in $(seq $FIRST_TASK_ID $((TASK_ID - 1))); do
        # First task adapter comes from naive_lora folder (no coefficient in path, but includes seed)
        if [ "$prev_task" -eq $FIRST_TASK_ID ]; then
            prev_log_dir="./logs/naive_lora/task_${prev_task}_seed${SEED}"
        else
            # Tasks 1+ use coefficient and seed in path to find matching adapters
            prev_log_dir="./logs/naive_lora_multilora/task_${prev_task}_coeff_${COEFFICIENT_PATH}_seed${SEED}"
        fi
        # Inject config tag into prev_log_dir to match where previous task saved checkpoint
        if [ -z "$CONFIG_TAG" ]; then
            # No config tag, use path as-is
            prev_log_dir_transformed="$prev_log_dir"
        else
            prev_log_dir_transformed=$(inject_config_tag_into_log_path "$prev_log_dir" "$CONFIG_TAG")
        fi
        
        # Validate that prev_log_dir_transformed is not empty
        if [ -z "$prev_log_dir_transformed" ]; then
            echo "    ERROR: Failed to construct log directory for task $prev_task"
            echo "           Original prev_log_dir: [$prev_log_dir]"
            echo "           CONFIG_TAG: [$CONFIG_TAG]"
            continue
        fi
        
        prev_log_dir="$prev_log_dir_transformed"
        prev_adapter_path="${prev_log_dir}/checkpoints/global_step_10/actor"
        
        # Additional validation: ensure prev_adapter_path is a valid relative or absolute path
        if [[ "$prev_adapter_path" =~ ^/checkpoints/ ]]; then
            echo "    ERROR: Invalid adapter path construction detected for task $prev_task"
            echo "           prev_log_dir was likely empty or malformed"
            echo "           prev_log_dir: [$prev_log_dir]"
            echo "           prev_adapter_path: [$prev_adapter_path]"
            continue
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

# Hydra overrides
OVERRIDES="env.fixed_task_ids=[${TASK_ID}] \
	runner.logger.experiment_name=${EXPERIMENT_NAME} \
	actor.seed=${SEED}"

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
    
    # Run evaluation
    # LOG_DIR already has the config tag injected, so use it directly
    CHECKPOINT_LOCATION=$(echo "$LOG_DIR" | sed 's|^\./||')
    echo ""
    echo "Running evaluation for: ${CHECKPOINT_LOCATION}"
    bash examples/mll_cluster/eval_embodiment.sh "${CHECKPOINT_LOCATION}" "" "${EVAL_CONFIG_NAME}"
else
    echo "✗ Task $TASK_ID (Multi-LoRA) failed with exit code $EXIT_CODE"
    echo "  Check logs at: logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
fi
echo "Finished at: $(date)"
echo "========================================="

exit $EXIT_CODE

