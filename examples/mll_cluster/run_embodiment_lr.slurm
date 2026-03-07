#!/bin/bash
#SBATCH --job-name=train_embodiment_lr      # Job name
#SBATCH --nodes=1                           # Number of nodes
#SBATCH --ntasks-per-node=1                 # Number of tasks per node
#SBATCH --cpus-per-task=90                  # Number of CPUs per task
#SBATCH --gres=gpu:8                        # Number of GPUs (RECOMMENDED: 4+ GPUs)
#SBATCH --time=70:00:00                     # Maximum execution time (HH:MM:SS)
#SBATCH --mem=900G                          # Memory per node (adjust as needed)
#SBATCH --partition=mll                     # Partition name (adjust to your cluster's partition)
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=jiahengh@utexas.edu
#SBATCH --exclude=slurm-node-008,slurm-node-011
#SBATCH --output=logs/slurm/%x-%j.out              # Output file name (%x = job name, %j = job ID)
#SBATCH --error=logs/slurm/%x-%j.err               # Error file name

### Usage: sbatch examples/mll_cluster/run_embodiment_lr.slurm [TASK_ID_OR_RANGE] [LR_STRING] [CONFIG_NAME] [SEED]
### Example (single task): sbatch examples/mll_cluster/run_embodiment_lr.slurm 0
### Example (task range): sbatch examples/mll_cluster/run_embodiment_lr.slurm "0,3" 
### Example (with LR): sbatch examples/mll_cluster/run_embodiment_lr.slurm "1,4" "2e-6,2e-6,1e-5"
### Example (with config): sbatch examples/mll_cluster/run_embodiment_lr.slurm "0,2" "2e-6,2e-6,1e-5" mll_cluster/libero_spatial_grpo_openvlaoft_lr
### Example (with seed): sbatch examples/mll_cluster/run_embodiment_lr.slurm "0,2" "" "" 42
### Note: TASK_ID_OR_RANGE can be:
###       - A single task ID (e.g., "0") - trains that task only
###       - A tuple "a,b" where a < b (e.g., "0,3") - trains tasks from a to b sequentially
###       LR_STRING is comma-separated: "vision_lora_lr,llm_lora_lr,llm_head_lora_lr"
###       If not provided, uses default values from config
###       SEED is optional and defaults to 1234 if not provided

TASK_INPUT=${1:-0}
LR_STRING=$2
CONFIG_NAME=${3:-mll_cluster/libero_spatial_grpo_openvlaoft_lr}
SEED=${4:-1234}

# Source common functions
source "examples/mll_cluster/common_functions.sh"

# Extract config tag and derive eval config name
CONFIG_TAG=$(extract_config_tag "$CONFIG_NAME")
EVAL_CONFIG_NAME=$(derive_eval_config_name "$CONFIG_NAME")
GLOBAL_STEP=$(get_default_global_step "$CONFIG_NAME")
FIRST_TASK_ID=$(get_first_task_id "$CONFIG_NAME")

# For _lr configs, use the existing eval config (without _lr)
# e.g., libero_spatial_grpo_openvlaoft_lr -> libero_spatial_grpo_openvlaoft_eval
# e.g., libero_10_grpo_openvlaoft_lr_long -> libero_10_grpo_openvlaoft_eval_long
# e.g., libero_object_grpo_openvlaoft_lr_object -> libero_object_grpo_openvlaoft_eval_object
if [[ "$CONFIG_NAME" =~ _lr ]]; then
    # Check if config ends with _lr_{tag} (e.g., _lr_long, _lr_object)
    if [[ "$CONFIG_NAME" =~ _lr_([^/]+)$ ]]; then
        # Extract the tag part (e.g., "long" or "object" from "_lr_long" or "_lr_object")
        LR_TAG="${BASH_REMATCH[1]}"
        # Replace _lr_{tag} with _eval_{tag}
        EVAL_CONFIG_NAME=$(echo "$CONFIG_NAME" | sed "s|_lr_${LR_TAG}$|_eval_${LR_TAG}|")
    # Check if config ends with just _lr (no tag)
    elif [[ "$CONFIG_NAME" =~ _lr$ ]]; then
        EVAL_CONFIG_NAME=$(echo "$CONFIG_NAME" | sed 's|_lr$|_eval|')
    fi
fi

# Parse TASK_INPUT to determine if it's a single task or a range
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

mkdir -p logs/slurm
mkdir -p logs/lr_experiment

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "GPUs allocated: $SLURM_GPUS_ON_NODE"
echo ""

# Parse learning rates if provided (parse once, use for all tasks in range)
if [ -n "$LR_STRING" ]; then
    # Split comma-separated string: "2e-6,2e-6,1e-5" -> vision_lora_lr,llm_lora_lr,llm_head_lora_lr
    IFS=',' read -r VISION_LR LLM_LR HEAD_LR <<< "$LR_STRING"
    
    # Validate that we got exactly 3 values
    if [ -z "$VISION_LR" ] || [ -z "$LLM_LR" ] || [ -z "$HEAD_LR" ]; then
        echo "ERROR: LR_STRING must contain exactly 3 comma-separated values: vision_lora_lr,llm_lora_lr,llm_head_lora_lr"
        echo "       Example: \"2e-6,2e-6,1e-5\""
        exit 1
    fi
    
    # Create concise folder name: convert to compact format
    # e.g., 2.0e-6 -> 2e6, 2e-6 -> 2e6, 1e-5 -> 1e5
    V_LR_STR=$(echo "$VISION_LR" | sed 's/\.0*e/e/g' | sed 's/\.//g' | sed 's/e-/e/g')
    L_LR_STR=$(echo "$LLM_LR" | sed 's/\.0*e/e/g' | sed 's/\.//g' | sed 's/e-/e/g')
    H_LR_STR=$(echo "$HEAD_LR" | sed 's/\.0*e/e/g' | sed 's/\.//g' | sed 's/e-/e/g')
fi

# Main training loop
OVERALL_EXIT_CODE=0

for TASK_ID in $(seq $TASK_START $TASK_END); do
    echo ""
    echo "========================================="
    if [ "$IS_RANGE" = true ]; then
        echo "Sequential Training - Task ${TASK_ID} (${TASK_START} to ${TASK_END})"
    else
        echo "Lifelong Learning - Learning Rate Experiment"
    fi
    echo "========================================="
    
    # Determine task log directory
    if [ -n "$LR_STRING" ]; then
        TASK_LOG_DIR="./logs/lr_experiment/task_${TASK_ID}_lr_v${V_LR_STR}_l${L_LR_STR}_h${H_LR_STR}_seed${SEED}"
    else
        TASK_LOG_DIR="./logs/lr_experiment/task_${TASK_ID}_lr_default_seed${SEED}"
    fi
    
    # Inject config tag into TASK_LOG_DIR before exporting
    if [ -n "$CONFIG_TAG" ]; then
        TASK_LOG_DIR_TRANSFORMED=$(inject_config_tag_into_log_path "$TASK_LOG_DIR" "$CONFIG_TAG")
        if [ -z "$TASK_LOG_DIR_TRANSFORMED" ]; then
            echo "  ERROR: Failed to transform TASK_LOG_DIR with config tag"
            echo "         Original TASK_LOG_DIR: [$TASK_LOG_DIR]"
            echo "         CONFIG_TAG: [$CONFIG_TAG]"
            OVERALL_EXIT_CODE=1
            break
        fi
        TASK_LOG_DIR="$TASK_LOG_DIR_TRANSFORMED"
    fi
    
    # Validate TASK_LOG_DIR is not empty
    if [ -z "$TASK_LOG_DIR" ]; then
        echo "  ERROR: TASK_LOG_DIR is empty after path construction"
        OVERALL_EXIT_CODE=1
        break
    fi
    
    export LOG_DIR="${TASK_LOG_DIR}"
    mkdir -p "${TASK_LOG_DIR}"
    
    # Set experiment name based on LOG_DIR (for wandb)
    EXPERIMENT_NAME=$(basename "$TASK_LOG_DIR")
    # Append CONFIG_TAG to experiment name if set
    if [ -n "$CONFIG_TAG" ]; then
        EXPERIMENT_NAME="${EXPERIMENT_NAME}_${CONFIG_TAG}"
    fi
    
    # Update SLURM job name to match experiment name (only for first task)
    if [ "$TASK_ID" -eq "$TASK_START" ] && command -v scontrol &> /dev/null && [ -n "$SLURM_JOB_ID" ]; then
        if [ "$IS_RANGE" = true ]; then
            if [ -n "$CONFIG_TAG" ]; then
                scontrol update job=$SLURM_JOB_ID name="lr_tasks_${TASK_START}_to_${TASK_END}_${CONFIG_TAG}" 2>/dev/null || true
            else
                scontrol update job=$SLURM_JOB_ID name="lr_tasks_${TASK_START}_to_${TASK_END}" 2>/dev/null || true
            fi
        else
            scontrol update job=$SLURM_JOB_ID name="${EXPERIMENT_NAME}" 2>/dev/null || true
        fi
    fi
    
    echo "Configuration:"
    echo "  Task ID: $TASK_ID"
    if [ "$IS_RANGE" = true ]; then
        echo "  Task Range: ${TASK_START} to ${TASK_END}"
    fi
    echo "  Experiment Name: $EXPERIMENT_NAME"
    echo "  Checkpoint Save Path: $TASK_LOG_DIR"
    echo "  Config Name: $CONFIG_NAME"
    echo "  Random Seed: $SEED"
    
    if [ -n "$LR_STRING" ]; then
        echo "  Vision LoRA LR: $VISION_LR"
        echo "  LLM LoRA LR: $LLM_LR"
        echo "  LLM Head LoRA LR: $HEAD_LR"
    else
        echo "  Using default learning rates from config"
    fi
    
    # Determine checkpoint path from previous task (similar to run_embodiment_naive_lora.slurm)
    # Only load from base model if this is the first task in the benchmark (FIRST_TASK_ID)
    # For all other tasks, automatically load from previous task's checkpoint
    if [ "$TASK_ID" -eq "$FIRST_TASK_ID" ]; then
        # First task in benchmark, load from base model (SFT checkpoint)
        CHECKPOINT_PATH=""
        echo "  Training from base model (SFT checkpoint) - no LoRA path"
    else
        # Use checkpoint from previous task
        PREV_TASK_ID=$((TASK_ID - 1))
        if [ -n "$LR_STRING" ]; then
            PREV_TASK_LOG_DIR="./logs/lr_experiment/task_${PREV_TASK_ID}_lr_v${V_LR_STR}_l${L_LR_STR}_h${H_LR_STR}_seed${SEED}"
        else
            PREV_TASK_LOG_DIR="./logs/lr_experiment/task_${PREV_TASK_ID}_lr_default_seed${SEED}"
        fi
        # Inject config tag into PREV_TASK_LOG_DIR to match where previous task saved checkpoint
        if [ -z "$CONFIG_TAG" ]; then
            # No config tag, use path as-is
            PREV_TASK_LOG_DIR_TRANSFORMED="$PREV_TASK_LOG_DIR"
        else
            PREV_TASK_LOG_DIR_TRANSFORMED=$(inject_config_tag_into_log_path "$PREV_TASK_LOG_DIR" "$CONFIG_TAG")
        fi
        
        # Validate that PREV_TASK_LOG_DIR_TRANSFORMED is not empty
        if [ -z "$PREV_TASK_LOG_DIR_TRANSFORMED" ]; then
            echo "  ERROR: Failed to construct previous task log directory for task $PREV_TASK_ID"
            echo "         Original PREV_TASK_LOG_DIR: [$PREV_TASK_LOG_DIR]"
            echo "         CONFIG_TAG: [$CONFIG_TAG]"
            OVERALL_EXIT_CODE=1
            break
        fi
        
        PREV_TASK_LOG_DIR="$PREV_TASK_LOG_DIR_TRANSFORMED"
        CHECKPOINT_PATH="${PREV_TASK_LOG_DIR}/checkpoints/global_step_${GLOBAL_STEP}/actor"
        
        # Additional validation: ensure CHECKPOINT_PATH is a valid relative or absolute path
        if [[ "$CHECKPOINT_PATH" =~ ^/checkpoints/ ]]; then
            echo "  ERROR: Invalid checkpoint path construction detected"
            echo "         PREV_TASK_LOG_DIR was likely empty or malformed"
            echo "         PREV_TASK_LOG_DIR: [$PREV_TASK_LOG_DIR]"
            echo "         CHECKPOINT_PATH: [$CHECKPOINT_PATH]"
            OVERALL_EXIT_CODE=1
            break
        fi
        
        if [ ! -d "$CHECKPOINT_PATH" ]; then
            echo "ERROR: Previous checkpoint does not exist: $CHECKPOINT_PATH"
            echo "       Task $TASK_ID requires checkpoint from task $PREV_TASK_ID"
            echo "       Make sure task $PREV_TASK_ID has been trained first"
            OVERALL_EXIT_CODE=1
            break
        fi
        echo "  Loading checkpoint from previous task: $CHECKPOINT_PATH"
    fi
    echo "========================================="
    echo ""
    
    # Build Hydra overrides
    OVERRIDES="env.fixed_task_ids=[${TASK_ID}] \
    	runner.logger.experiment_name=${EXPERIMENT_NAME} \
    	actor.seed=${SEED}"
    
    if [ -n "$LR_STRING" ]; then
        OVERRIDES="$OVERRIDES actor.optim.vision_lora_lr=${VISION_LR} \
    	actor.optim.llm_lora_lr=${LLM_LR} \
    	actor.optim.llm_head_lora_lr=${HEAD_LR}"
    fi
    
    # Add checkpoint path from previous task if available
    if [ -n "$CHECKPOINT_PATH" ]; then
        OVERRIDES="${OVERRIDES} +actor.model.lora_path=${CHECKPOINT_PATH}"
    fi
    
    echo "Running with Hydra overrides:"
    echo "$OVERRIDES"
    echo ""
    
    bash examples/embodiment/run_embodiment.sh ${CONFIG_NAME} $OVERRIDES
    
    EXIT_CODE=$?
    echo ""
    echo "========================================="
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Task $TASK_ID completed successfully"
        echo ""
        echo "Checkpoint saved to: ${TASK_LOG_DIR}"
        
        # Run evaluation
        # TASK_LOG_DIR already has the config tag injected, so use it directly
        CHECKPOINT_LOCATION=$(echo "$TASK_LOG_DIR" | sed 's|^\./||')
        echo ""
        echo "Running evaluation for: ${CHECKPOINT_LOCATION}"
        echo "Using global_step: ${GLOBAL_STEP}"
        bash examples/mll_cluster/eval_embodiment.sh "${CHECKPOINT_LOCATION}" "${GLOBAL_STEP}" "${EVAL_CONFIG_NAME}"
    else
        echo "✗ Task $TASK_ID failed with exit code $EXIT_CODE"
        echo "  Check logs at: logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
        OVERALL_EXIT_CODE=$EXIT_CODE
        if [ "$IS_RANGE" = true ]; then
            echo "  Stopping sequential training due to failure"
            break
        fi
    fi
    echo "========================================="
done

echo ""
echo "========================================="
if [ "$IS_RANGE" = true ]; then
    if [ $OVERALL_EXIT_CODE -eq 0 ]; then
        echo "All tasks (${TASK_START} to ${TASK_END}) completed successfully!"
    else
        echo "Sequential training failed. Completed up to task $((TASK_ID - 1))"
    fi
else
    if [ $OVERALL_EXIT_CODE -eq 0 ]; then
        echo "Task $TASK_START completed successfully"
    else
        echo "Task $TASK_START failed"
    fi
fi
echo "Finished at: $(date)"
echo "========================================="

exit $OVERALL_EXIT_CODE

