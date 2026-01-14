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
#SBATCH --exclude=slurm-node-007
#SBATCH --output=logs/slurm/%x-%j.out              # Output file name (%x = job name, %j = job ID)
#SBATCH --error=logs/slurm/%x-%j.err               # Error file name

### Usage: sbatch examples/mll_cluster/run_embodiment_lr.slurm [TASK_ID_OR_RANGE] [LR_STRING] [CONFIG_NAME]
### Example (single task): sbatch examples/mll_cluster/run_embodiment_lr.slurm 0
### Example (task range): sbatch examples/mll_cluster/run_embodiment_lr.slurm "0,3" 
### Example (with LR): sbatch examples/mll_cluster/run_embodiment_lr.slurm "1,4" "2e-6,2e-6,1e-5"
### Example (with config): sbatch examples/mll_cluster/run_embodiment_lr.slurm "0,2" "2e-6,2e-6,1e-5" mll_cluster/libero_spatial_grpo_openvlaoft_lr
### Note: TASK_ID_OR_RANGE can be:
###       - A single task ID (e.g., "0") - trains that task only
###       - A tuple "a,b" where a < b (e.g., "0,3") - trains tasks from a to b sequentially
###       LR_STRING is comma-separated: "vision_lora_lr,llm_lora_lr,llm_head_lora_lr"
###       If not provided, uses default values from config

TASK_INPUT=${1:-0}
LR_STRING=$2
CONFIG_NAME=${3:-mll_cluster/libero_spatial_grpo_openvlaoft_lr}

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

cd $SLURM_SUBMIT_DIR

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
        TASK_LOG_DIR="./logs/lr_experiment/task_${TASK_ID}_lr_v${V_LR_STR}_l${L_LR_STR}_h${H_LR_STR}"
    else
        TASK_LOG_DIR="./logs/lr_experiment/task_${TASK_ID}_lr_default"
    fi
    
    export LOG_DIR="${TASK_LOG_DIR}"
    mkdir -p "${TASK_LOG_DIR}"
    
    # Set experiment name based on LOG_DIR (for wandb)
    EXPERIMENT_NAME=$(basename "$TASK_LOG_DIR")
    
    # Update SLURM job name to match experiment name (only for first task)
    if [ "$TASK_ID" -eq "$TASK_START" ] && command -v scontrol &> /dev/null && [ -n "$SLURM_JOB_ID" ]; then
        if [ "$IS_RANGE" = true ]; then
            scontrol update job=$SLURM_JOB_ID name="lr_tasks_${TASK_START}_to_${TASK_END}" 2>/dev/null || true
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
    
    if [ -n "$LR_STRING" ]; then
        echo "  Vision LoRA LR: $VISION_LR"
        echo "  LLM LoRA LR: $LLM_LR"
        echo "  LLM Head LoRA LR: $HEAD_LR"
    else
        echo "  Using default learning rates from config"
    fi
    
    # Determine checkpoint path from previous task (similar to run_embodiment_naive_lora.slurm)
    if [ $TASK_ID -gt $TASK_START ]; then
        PREV_TASK_ID=$((TASK_ID - 1))
        if [ -n "$LR_STRING" ]; then
            PREV_TASK_LOG_DIR="./logs/lr_experiment/task_${PREV_TASK_ID}_lr_v${V_LR_STR}_l${L_LR_STR}_h${H_LR_STR}"
        else
            PREV_TASK_LOG_DIR="./logs/lr_experiment/task_${PREV_TASK_ID}_lr_default"
        fi
        CHECKPOINT_PATH="${PREV_TASK_LOG_DIR}/checkpoints/global_step_10/actor"
        
        if [ ! -d "$CHECKPOINT_PATH" ]; then
            echo "ERROR: Previous checkpoint does not exist: $CHECKPOINT_PATH"
            OVERALL_EXIT_CODE=1
            break
        fi
        echo "  Loading checkpoint from previous task: $CHECKPOINT_PATH"
    else
        CHECKPOINT_PATH=""
        echo "  Training from base model (SFT checkpoint) - no LoRA path"
    fi
    echo "========================================="
    echo ""
    
    # Build Hydra overrides
    OVERRIDES="env.fixed_task_ids=[${TASK_ID}] \
    	runner.logger.experiment_name=${EXPERIMENT_NAME} \
    	actor.checkpoint_save_path=${TASK_LOG_DIR}"
    
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
        CHECKPOINT_LOCATION=$(echo "$TASK_LOG_DIR" | sed 's|^\./||')
        echo ""
        echo "Running evaluation for: ${CHECKPOINT_LOCATION}"
        bash examples/mll_cluster/eval_embodiment.sh "${CHECKPOINT_LOCATION}"
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

