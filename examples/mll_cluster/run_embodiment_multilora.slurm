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

### Usage: sbatch examples/mll_cluster/run_embodiment_multilora.slurm TASK_ID_OR_RANGE [MERGE_COEFFICIENT] [CONFIG_NAME] [SEED]
### Example (single task): sbatch examples/mll_cluster/run_embodiment_multilora.slurm 0
### Example (single task): sbatch examples/mll_cluster/run_embodiment_multilora.slurm 1 0.8
### Example (single task): sbatch examples/mll_cluster/run_embodiment_multilora.slurm 2 0.8
### Example (with seed):   sbatch examples/mll_cluster/run_embodiment_multilora.slurm 0 "" "" 42
### Example (range):       sbatch examples/mll_cluster/run_embodiment_multilora.slurm "0,3" 0.8
###
### Behavior:
###   - First task ID (e.g., 0 for spatial, 3 for libero_10 long):
###       Train first LoRA adapter on top of base model (same as naive LoRA).
###   - Subsequent tasks: Load and merge ALL previous task adapters (FIRST_TASK_ID..TASK_ID-1)
###       into base, then create a NEW LoRA adapter for this task (only this one is trainable).
###       Note: The first task adapter is loaded from naive_lora folder only when CONFIG_TAG is empty
###             (libero spatial case). Otherwise, it's loaded from naive_lora_multilora folder.
###             Later tasks are always loaded from naive_lora_multilora folder.
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

MERGE_COEFFICIENT=$2
CONFIG_NAME=${3:-mll_cluster/libero_spatial_grpo_openvlaoft}  # Default config if not provided
SEED=${4:-1234}

# Source common functions
source "examples/mll_cluster/common_functions.sh"

# Extract config tag and derive eval config name
CONFIG_TAG=$(extract_config_tag "$CONFIG_NAME")
EVAL_CONFIG_NAME=$(derive_eval_config_name "$CONFIG_NAME")
GLOBAL_STEP=$(get_default_global_step "$CONFIG_NAME")
FIRST_TASK_ID=$(get_first_task_id "$CONFIG_NAME")

# Parse TASK_ID_OR_RANGE to determine if it's a single task or a range
TASK_INPUT=${1:-$FIRST_TASK_ID}
if [[ "$TASK_INPUT" == *,* ]]; then
    # It's a tuple (a,b)
    IFS=',' read -r TASK_START TASK_END <<< "$TASK_INPUT"
    TASK_START=$(echo "$TASK_START" | tr -d '()[] ')
    TASK_END=$(echo "$TASK_END" | tr -d '()[] ')
    
    # Validate that both values are numeric and start < end
    if ! [[ "$TASK_START" =~ ^[0-9]+$ ]] || ! [[ "$TASK_END" =~ ^[0-9]+$ ]]; then
        echo "ERROR: Task range must contain two numeric values: \"a,b\" where a and b are integers"
        echo "       Example: \"0,3\" or \"1,5\""
        exit 1
    fi
    
    if [ "$TASK_START" -ge "$TASK_END" ]; then
        echo "ERROR: First task ID ($TASK_START) must be smaller than second task ID ($TASK_END)"
        echo "       Example: \"0,3\" (trains tasks 0, 1, 2, 3)"
        exit 1
    fi
    
    IS_RANGE=true
    NUM_TASKS=$((TASK_END - TASK_START + 1))
else
    # It's a single task ID
    if ! [[ "$TASK_INPUT" =~ ^[0-9]+$ ]]; then
        echo "ERROR: Task ID must be a numeric value"
        echo "       Example: 0 or \"0,3\" for a range"
        exit 1
    fi
    IS_RANGE=false
    TASK_START=$TASK_INPUT
    TASK_END=$TASK_INPUT
    NUM_TASKS=1
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

# Create base log directories
mkdir -p logs/slurm
mkdir -p logs/naive_lora_multilora

OVERALL_EXIT_CODE=0

for TASK_ID in $(seq $TASK_START $TASK_END); do
    # Include coefficient and seed in the log directory path
    if [ "$TASK_ID" -eq $FIRST_TASK_ID ]; then
        # First task doesn't use coefficient in path (no previous adapters to merge)
        LOG_DIR="./logs/naive_lora_multilora/task_${TASK_ID}_seed${SEED}"
        EXPERIMENT_NAME="multilora_task_${TASK_ID}_seed${SEED}"
    else
        # Tasks after FIRST_TASK_ID include coefficient and seed in path and experiment name
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
            OVERALL_EXIT_CODE=1
            break
        fi
        LOG_DIR="$LOG_DIR_TRANSFORMED"
    fi

    # Validate LOG_DIR is not empty
    if [ -z "$LOG_DIR" ]; then
        echo "  ERROR: LOG_DIR is empty after path construction"
        OVERALL_EXIT_CODE=1
        break
    fi

    export LOG_DIR
    mkdir -p "$LOG_DIR"

    # Update SLURM job name to be more descriptive (based on LOG_DIR) for the first task only
    LOG_DIR_BASENAME=$(basename "$LOG_DIR")
    if [ "$TASK_ID" -eq "$TASK_START" ] && command -v scontrol &> /dev/null && [ -n "$SLURM_JOB_ID" ]; then
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
    if [ "$IS_RANGE" = true ]; then
        echo "  Task Range: ${TASK_START} to ${TASK_END}"
    fi
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
            # First task adapter: only look in naive_lora folder when CONFIG_TAG is empty (libero spatial case)
            # Otherwise, look in naive_lora_multilora folder (with config_tag injected)
            if [ "$prev_task" -eq $FIRST_TASK_ID ]; then
                if [ -z "$CONFIG_TAG" ]; then
                    # No config tag (libero spatial): look in naive_lora folder
                    prev_log_dir="./logs/naive_lora/task_${prev_task}_seed${SEED}"
                else
                    # Has config tag: look in naive_lora_multilora folder (will inject config_tag below)
                    prev_log_dir="./logs/naive_lora_multilora/task_${prev_task}_seed${SEED}"
                fi
            else
                # Tasks after FIRST_TASK_ID use coefficient and seed in path to find matching adapters
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
            prev_adapter_path="${prev_log_dir}/checkpoints/global_step_${GLOBAL_STEP}/actor"
            
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
                echo "    ERROR: Adapter not found for task $prev_task: $prev_adapter_path"
                OVERALL_EXIT_CODE=1
                break
            fi
        done

        if [ ${#LORA_PATHS[@]} -gt 0 ]; then
            # Convert array to comma-separated string for Hydra
            LORA_PATHS_STR=$(IFS=','; echo "${LORA_PATHS[*]}")
            LORA_PATHS_ARG="+actor.model.lora_paths=[${LORA_PATHS_STR}]"
            echo "  Will merge ${#LORA_PATHS[@]} previous adapter(s) into base model, then train new adapter."
        else
            echo "  ERROR: No previous adapters found; training from base model."
            OVERALL_EXIT_CODE=1
            break
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
        
        #
        # Run evaluation for the newly trained checkpoint.
        # Reuse the same previous-adapter paths (LORA_PATHS_ARG) that were
        # already collected for training, and apply MERGE_COEFFICIENT to
        # all adapters (previous + latest) via:
        #   - actor.model.lora_scale (latest adapter)
        #   - actor.model.previous_lora_merge_coefficient (previous adapters)
        #
        EVAL_LORA_PATH="${LOG_DIR}/checkpoints/global_step_${GLOBAL_STEP}/actor"
        echo ""
        echo "Running evaluation for Multi-LoRA checkpoint:"
        echo "  Current adapter path: ${EVAL_LORA_PATH}"
        echo "  Merge coefficient applied to ALL adapters (including latest): ${MERGE_COEFFICIENT}"

        EVAL_OVERRIDES="+actor.model.lora_path=${EVAL_LORA_PATH} actor.model.lora_scale=${MERGE_COEFFICIENT}"
        if [ -n "$LORA_PATHS_ARG" ]; then
            # Reuse LORA_PATHS_ARG (previous adapters) and MERGE_COEFFICIENT_ARG
            EVAL_OVERRIDES="${EVAL_OVERRIDES} ${LORA_PATHS_ARG} ${MERGE_COEFFICIENT_ARG}"
        fi

        echo ""
        echo "Launching eval_embodiment with overrides:"
        echo "  Config: ${EVAL_CONFIG_NAME}"
        echo "  Overrides: ${EVAL_OVERRIDES}"

        bash examples/embodiment/eval_embodiment.sh ${EVAL_CONFIG_NAME} ${EVAL_OVERRIDES}
    else
        echo "✗ Task $TASK_ID (Multi-LoRA) failed with exit code $EXIT_CODE"
        echo "  Check logs at: logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
        OVERALL_EXIT_CODE=$EXIT_CODE
        # For ranges, stop on first failure
        if [ "$IS_RANGE" = true ]; then
            echo "  Stopping sequential Multi-LoRA training due to failure"
            break
        fi
    fi
done

echo "Finished at: $(date)"
echo "========================================="
if [ "$IS_RANGE" = true ]; then
    if [ $OVERALL_EXIT_CODE -eq 0 ]; then
        echo "All Multi-LoRA tasks (${TASK_START} to ${TASK_END}) completed successfully!"
    else
        echo "Multi-LoRA range training failed. Completed up to task $TASK_ID"
    fi
else
    if [ $OVERALL_EXIT_CODE -eq 0 ]; then
        echo "Task $TASK_START completed successfully (Multi-LoRA)"
    else
        echo "Task $TASK_START failed (Multi-LoRA)"
    fi
fi

exit $OVERALL_EXIT_CODE

