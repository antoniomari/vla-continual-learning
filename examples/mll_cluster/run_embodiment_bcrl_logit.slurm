#!/bin/bash
#SBATCH --job-name=train_embodiment      # Job name
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks-per-node=1             # Number of tasks per node
#SBATCH --cpus-per-task=90              # Number of CPUs per task
#SBATCH --gres=gpu:8                    # Number of GPUs (RECOMMENDED: 4+ GPUs)
#SBATCH --time=70:00:00                  # Maximum execution time (HH:MM:SS)
#SBATCH --mem=900G                      # Memory per node (adjust as needed)
#SBATCH --partition=mll                 # Partition name (adjust to your cluster's partition)
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=jiahengh@utexas.edu
#SBATCH --exclude=slurm-node-008,slurm-node-011
#SBATCH --output=logs/slurm/%x-%j.out              # Output file name (%x = job name, %j = job ID)
#SBATCH --error=logs/slurm/%x-%j.err               # Error file name


### Usage: sbatch examples/mll_cluster/run_embodiment_bcrl_logit.slurm TASK_ID BC_COEFF [CHECKPOINT_PATH] [CONFIG_NAME] [SEED]
### Example: sbatch examples/mll_cluster/run_embodiment_bcrl_logit.slurm 0 0.1 libero_spatial_grpo_openvlaoft_bcrl_logit
### Example (with seed): sbatch examples/mll_cluster/run_embodiment_bcrl_logit.slurm 0 0.1 "" "" 42
### Note: CHECKPOINT_PATH is now optional and will be auto-generated from previous task if not provided
###       SEED is optional and defaults to 1234 if not provided

TASK_ID=$1
BC_COEFF=$2
MANUAL_CHECKPOINT_PATH=$3
CONFIG_NAME=${4:-libero_spatial_grpo_openvlaoft_bcrl_logit}  # Default config if not provided
SEED=${5:-1234}

# Source common functions
source "examples/mll_cluster/common_functions.sh"

# Extract config tag and derive eval config name
CONFIG_TAG=$(extract_config_tag "$CONFIG_NAME")
EVAL_CONFIG_NAME=$(derive_eval_config_name "$CONFIG_NAME")
FIRST_TASK_ID=$(get_first_task_id "$CONFIG_NAME")

if [ -z "$TASK_ID" ] || [ -z "$BC_COEFF" ]; then
    echo "ERROR: Missing required arguments"
    echo "Usage: sbatch examples/mll_cluster/run_embodiment_bcrl_logit.slurm TASK_ID BC_COEFF [CHECKPOINT_PATH] [CONFIG_NAME] [SEED]"
    exit 1
fi

# Validate seed is a number
if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
    echo "ERROR: SEED must be a non-negative integer, got: $SEED"
    exit 1
fi

# Automatically determine checkpoint path if not manually provided
if [ -z "$MANUAL_CHECKPOINT_PATH" ]; then
    if [ "$TASK_ID" -eq $FIRST_TASK_ID ]; then
        # First task uses base model, no checkpoint
        CHECKPOINT_PATH=""
    else
        # Use checkpoint from previous task
        PREV_TASK_ID=$((TASK_ID - 1))
        PREV_LOG_DIR="./logs/bcrl_logit/${BC_COEFF}/task_${PREV_TASK_ID}_seed${SEED}"
        # Inject config tag into PREV_LOG_DIR to match where previous task saved checkpoint
        if [ -z "$CONFIG_TAG" ]; then
            # No config tag, use path as-is
            PREV_LOG_DIR_TRANSFORMED="$PREV_LOG_DIR"
        else
            PREV_LOG_DIR_TRANSFORMED=$(inject_config_tag_into_log_path "$PREV_LOG_DIR" "$CONFIG_TAG")
        fi
        
        # Validate that PREV_LOG_DIR_TRANSFORMED is not empty
        if [ -z "$PREV_LOG_DIR_TRANSFORMED" ]; then
            echo "  ERROR: Failed to construct previous task log directory for task $PREV_TASK_ID"
            echo "         Original PREV_LOG_DIR: [$PREV_LOG_DIR]"
            echo "         CONFIG_TAG: [$CONFIG_TAG]"
            exit 1
        fi
        
        PREV_LOG_DIR="$PREV_LOG_DIR_TRANSFORMED"
        CHECKPOINT_PATH="${PREV_LOG_DIR}/checkpoints/global_step_10/actor"
        
        # Additional validation: ensure CHECKPOINT_PATH is a valid relative or absolute path
        if [[ "$CHECKPOINT_PATH" =~ ^/checkpoints/ ]]; then
            echo "  ERROR: Invalid checkpoint path construction detected"
            echo "         PREV_LOG_DIR was likely empty or malformed"
            echo "         PREV_LOG_DIR: [$PREV_LOG_DIR]"
            echo "         CHECKPOINT_PATH: [$CHECKPOINT_PATH]"
            exit 1
        fi
    fi
else
    # Use manually provided checkpoint path
    CHECKPOINT_PATH="$MANUAL_CHECKPOINT_PATH"
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

mkdir -p logs/slurm
mkdir -p logs/bcrl_logit
mkdir -p logs/bcrl_logit/${BC_COEFF}
LOG_DIR="./logs/bcrl_logit/${BC_COEFF}/task_${TASK_ID}_seed${SEED}"
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

EXPERIMENT_NAME="bcrl_logit_task_${TASK_ID}_${BC_COEFF}"
# Append CONFIG_TAG to experiment name if set
if [ -n "$CONFIG_TAG" ]; then
    EXPERIMENT_NAME="${EXPERIMENT_NAME}_${CONFIG_TAG}"
fi

# Update SLURM job name to be more descriptive (based on LOG_DIR)
# Extract meaningful path components: bcrl_logit/${BC_COEFF}/task_${TASK_ID}
# Use parent directory and basename for a descriptive name
LOG_DIR_PARENT=$(basename "$(dirname "$LOG_DIR")")  # e.g., "0.3"
LOG_DIR_BASENAME=$(basename "$LOG_DIR")  # e.g., "task_0"
if command -v scontrol &> /dev/null && [ -n "$SLURM_JOB_ID" ]; then
    # Combine to show both BC_COEFF and task: "bcrl_logit_0.3_task_0"
    JOB_NAME="bcrl_logit_${LOG_DIR_PARENT}_${LOG_DIR_BASENAME}"
    if [ -n "$CONFIG_TAG" ]; then
        JOB_NAME="${JOB_NAME}_${CONFIG_TAG}"
    fi
    scontrol update job=$SLURM_JOB_ID name="${JOB_NAME}" 2>/dev/null || true
fi

echo "========================================="
echo "Lifelong Learning - Single Task Training"
echo "========================================="
echo "Configuration:"
echo "  Task ID: $TASK_ID"
echo "  BC Coefficient: $BC_COEFF"
echo "  Experiment Name: $EXPERIMENT_NAME"
echo "  Checkpoint Save Path: $LOG_DIR"
echo "  Config Name: $CONFIG_NAME"
echo "  Random Seed: $SEED"

if [ -n "$CHECKPOINT_PATH" ]; then
    if [ -n "$MANUAL_CHECKPOINT_PATH" ]; then
        echo "  Loading from checkpoint (manually specified): $CHECKPOINT_PATH"
    else
        echo "  Loading from checkpoint (auto-generated from task $((TASK_ID - 1))): $CHECKPOINT_PATH"
    fi

    # Verify checkpoint exists
    if [ ! -d "$CHECKPOINT_PATH" ]; then
        echo "  ERROR: Checkpoint not found at $CHECKPOINT_PATH"
        if [ "$TASK_ID" -gt $FIRST_TASK_ID ]; then
            PREV_TASK_ID=$((TASK_ID - 1))
            echo "         Task $TASK_ID requires checkpoint from task $PREV_TASK_ID"
            echo "         Make sure task $PREV_TASK_ID has been trained first, or provide a manual checkpoint path"
        fi
        exit 1
    fi
    CHECKPOINT_ARG="+actor.model.lora_path=${CHECKPOINT_PATH}"
else
    echo "  Training from base model (SFT checkpoint) - First task (task $FIRST_TASK_ID)"
fi

echo "========================================="
echo ""

# hydra overrides
OVERRIDES="algorithm.bc_coeff=${BC_COEFF} \
	env.fixed_task_ids=[${TASK_ID}] \
	runner.logger.experiment_name=${EXPERIMENT_NAME} \
	actor.seed=${SEED}"

if [ -n "$CHECKPOINT_PATH" ]; then
    OVERRIDES="$OVERRIDES ${CHECKPOINT_ARG}"
fi

echo "Running with Hydra overrides:"
echo "$OVERRIDES"
echo ""

bash examples/embodiment/run_embodiment.sh ${CONFIG_NAME} $OVERRIDES

EXIT_CODE=$?
echo ""
echo "========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Task $TASK_ID with BC_COEFF=$BC_COEFF completed successfully"
    echo ""
    echo "Checkpoint saved to: ${LOG_DIR}"
    
    # Run evaluation
    # LOG_DIR already has the config tag injected, so use it directly
    CHECKPOINT_LOCATION=$(echo "$LOG_DIR" | sed 's|^\./||')
    echo ""
    echo "Running evaluation for: ${CHECKPOINT_LOCATION}"
    bash examples/mll_cluster/eval_embodiment.sh "${CHECKPOINT_LOCATION}" "" "${EVAL_CONFIG_NAME}"
else
    echo "✗ Task $TASK_ID failed with exit code $EXIT_CODE"
    echo "  Check logs at: logs/slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
fi
echo "Finished at: $(date)"
echo "========================================="

exit $EXIT_CODE

