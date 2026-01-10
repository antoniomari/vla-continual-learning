#!/bin/bash
#SBATCH --job-name=eval_lora_scale      # Job name
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks-per-node=1             # Number of tasks per node
#SBATCH --cpus-per-task=80              # Number of CPUs per task
#SBATCH --gres=gpu:4                    # Number of GPUs (RECOMMENDED: 4+ GPUs)
#SBATCH --time=24:00:00                  # Maximum execution time (HH:MM:SS)
#SBATCH --mem=490G                       # Memory per node (adjust as needed)
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

### Usage: sbatch examples/mll_cluster/eval_embodiment_lora_scale.slurm CHECKPOINT_LOCATION [CURRENT_LORA_SCALE] [PREVIOUS_LORA_COEFF]
### Example (single LoRA): sbatch examples/mll_cluster/eval_embodiment_lora_scale.slurm logs/bcrl_logit/0.3/task_0 0.5
### Example (multi-LoRA): sbatch examples/mll_cluster/eval_embodiment_lora_scale.slurm logs/naive_lora_multilora/task_2_coeff_0_9
### Example (multi-LoRA with current scale): sbatch examples/mll_cluster/eval_embodiment_lora_scale.slurm logs/naive_lora_multilora/task_2_coeff_0_9 0.7
### Example (multi-LoRA with both overrides): sbatch examples/mll_cluster/eval_embodiment_lora_scale.slurm logs/naive_lora_multilora/task_2_coeff_0_9 0.7 0.8
### Note: CHECKPOINT_LOCATION should be relative to workspace root (e.g., logs/bcrl_logit/0.3/task_0)
###       The script will construct the full path: ${WORKSPACE_ROOT}/${CHECKPOINT_LOCATION}/checkpoints/global_step_10/actor/
###       For multi-LoRA checkpoints (path contains "multilora" and "coeff_"), the script will:
###         - Extract PREVIOUS_LORA_COEFF from the path (e.g., "coeff_0_9" -> 0.9) automatically
###         - Default CURRENT_LORA_SCALE to 1.0 if not specified
###       CURRENT_LORA_SCALE: scale for the current adapter (0.0-1.0, default 1.0 for multi-LoRA, required for single LoRA)
###       PREVIOUS_LORA_COEFF: coefficient for merging previous adapters (0.0-1.0, auto-extracted from path for multi-LoRA)

CHECKPOINT_LOCATION=$1
CURRENT_LORA_SCALE=$2
PREVIOUS_LORA_COEFF=$3

if [ -z "$CHECKPOINT_LOCATION" ]; then
    echo "ERROR: Missing required argument"
    echo "Usage: sbatch examples/mll_cluster/eval_embodiment_lora_scale.slurm CHECKPOINT_LOCATION [CURRENT_LORA_SCALE] [PREVIOUS_LORA_COEFF]"
    echo "Example (single LoRA): sbatch examples/mll_cluster/eval_embodiment_lora_scale.slurm logs/bcrl_logit/0.3/task_0 0.5"
    echo "Example (multi-LoRA): sbatch examples/mll_cluster/eval_embodiment_lora_scale.slurm logs/naive_lora_multilora/task_2_coeff_0_9"
    echo "Example (multi-LoRA with scale): sbatch examples/mll_cluster/eval_embodiment_lora_scale.slurm logs/naive_lora_multilora/task_2_coeff_0_9 0.7"
    exit 1
fi

# Check if this is a multi-LoRA checkpoint (contains "multilora" and "coeff_")
IS_MULTILORA=false
EXTRACTED_COEFF=""
if [[ "$CHECKPOINT_LOCATION" == *"multilora"* ]] && [[ "$CHECKPOINT_LOCATION" == *"coeff_"* ]]; then
    IS_MULTILORA=true
    # Extract coefficient from path (e.g., "task_2_coeff_0_9" -> "0_9" -> "0.9")
    if [[ "$CHECKPOINT_LOCATION" =~ coeff_([0-9]+_[0-9]+) ]]; then
        COEFF_STR="${BASH_REMATCH[1]}"
        EXTRACTED_COEFF=$(echo "$COEFF_STR" | tr '_' '.')
        echo "Detected multi-LoRA checkpoint with coefficient: $EXTRACTED_COEFF"
    fi
fi

# Set defaults based on checkpoint type
if [ "$IS_MULTILORA" = true ]; then
    # For multi-LoRA: PREVIOUS_LORA_COEFF defaults to extracted coefficient
    if [ -z "$PREVIOUS_LORA_COEFF" ]; then
        if [ -n "$EXTRACTED_COEFF" ]; then
            PREVIOUS_LORA_COEFF="$EXTRACTED_COEFF"
            echo "Using PREVIOUS_LORA_COEFF extracted from path: $PREVIOUS_LORA_COEFF"
        else
            echo "ERROR: Could not extract coefficient from multi-LoRA checkpoint path: $CHECKPOINT_LOCATION"
            echo "       Expected path format: .../task_<N>_coeff_<X>_<Y>/..."
            echo "       Please either fix the path or manually specify PREVIOUS_LORA_COEFF as 3rd argument"
            exit 1
        fi
    else
        echo "Using manually specified PREVIOUS_LORA_COEFF: $PREVIOUS_LORA_COEFF"
    fi
    # CURRENT_LORA_SCALE defaults to 1.0 for multi-LoRA
    if [ -z "$CURRENT_LORA_SCALE" ]; then
        CURRENT_LORA_SCALE="1.0"
    fi
else
    # For single LoRA: CURRENT_LORA_SCALE is required
    if [ -z "$CURRENT_LORA_SCALE" ]; then
        echo "ERROR: For single LoRA checkpoints, CURRENT_LORA_SCALE (2nd argument) is required"
        echo "Usage: sbatch examples/mll_cluster/eval_embodiment_lora_scale.slurm CHECKPOINT_LOCATION CURRENT_LORA_SCALE"
        exit 1
    fi
    # PREVIOUS_LORA_COEFF is not used for single LoRA (silently ignore if provided)
    PREVIOUS_LORA_COEFF=""
fi

# Validate PREVIOUS_LORA_COEFF if provided
if [ -n "$PREVIOUS_LORA_COEFF" ]; then
    if ! [[ "$PREVIOUS_LORA_COEFF" =~ ^[0-9]+\.?[0-9]*$ ]]; then
        echo "ERROR: PREVIOUS_LORA_COEFF must be a number, got: $PREVIOUS_LORA_COEFF"
        exit 1
    fi
    if ! awk "BEGIN {exit !($PREVIOUS_LORA_COEFF >= 0 && $PREVIOUS_LORA_COEFF <= 1)}"; then
        echo "ERROR: PREVIOUS_LORA_COEFF must be between 0.0 and 1.0, got: $PREVIOUS_LORA_COEFF"
        exit 1
    fi
fi

# Validate CURRENT_LORA_SCALE
if ! [[ "$CURRENT_LORA_SCALE" =~ ^[0-9]+\.?[0-9]*$ ]]; then
    echo "ERROR: CURRENT_LORA_SCALE must be a number, got: $CURRENT_LORA_SCALE"
    exit 1
fi
if ! awk "BEGIN {exit !($CURRENT_LORA_SCALE >= 0 && $CURRENT_LORA_SCALE <= 1)}"; then
    echo "ERROR: CURRENT_LORA_SCALE must be between 0.0 and 1.0, got: $CURRENT_LORA_SCALE"
    exit 1
fi

# Get workspace root (SLURM_SUBMIT_DIR is the directory where sbatch was run from)
cd $SLURM_SUBMIT_DIR
WORKSPACE_ROOT=$(pwd)
mkdir -p logs/slurm

# Construct full checkpoint path (remove any trailing slashes from CHECKPOINT_LOCATION first)
CHECKPOINT_LOCATION="${CHECKPOINT_LOCATION%/}"
CHECKPOINT_PATH="${WORKSPACE_ROOT}/${CHECKPOINT_LOCATION}/checkpoints/global_step_10/actor/"

# Verify checkpoint exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT_PATH"
    exit 1
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
echo "Checkpoint Location: $CHECKPOINT_LOCATION"
echo "Full Checkpoint Path: $CHECKPOINT_PATH"
if [ "$IS_MULTILORA" = true ]; then
    echo "Multi-LoRA Mode: Enabled"
    echo "Previous LoRA Coefficient: $PREVIOUS_LORA_COEFF"
    echo "Current LoRA Scale: $CURRENT_LORA_SCALE"
else
    echo "Single LoRA Mode"
    echo "LoRA Scale: $CURRENT_LORA_SCALE"
fi
echo ""
echo "Note: Log path will be automatically constructed by eval_embodiment.sh"
echo "      and will include coefficients in the directory name."
echo ""

# Update SLURM job name to be more descriptive (based on CHECKPOINT_LOCATION and parameters)
# Convert CHECKPOINT_LOCATION path to job name, then add scale info
# e.g., "logs/naive_lora_multilora/task_2_coeff_0_9" -> "naive_lora_multilora_task_2_coeff_0_9_scale_0_7"
if command -v scontrol &> /dev/null && [ -n "$SLURM_JOB_ID" ]; then
    # Convert path to job name by replacing slashes with underscores and removing "logs/" prefix if present
    JOB_NAME_BASE=$(echo "$CHECKPOINT_LOCATION" | sed 's|^logs/||' | tr '/' '_')
    # Add scale info
    SCALE_STR=$(echo "$CURRENT_LORA_SCALE" | tr '.' '_')
    JOB_NAME="${JOB_NAME_BASE}_scale_${SCALE_STR}"
    scontrol update job=$SLURM_JOB_ID name="${JOB_NAME}" 2>/dev/null || true
fi

# Build Hydra overrides
HYDRA_OVERRIDES="+actor.model.lora_path=${CHECKPOINT_PATH} actor.model.lora_scale=${CURRENT_LORA_SCALE}"

# For multi-LoRA, add previous_lora_merge_coefficient and lora_paths
if [ "$IS_MULTILORA" = true ]; then
    # Extract task ID from checkpoint path (e.g., "task_2_coeff_0_9" -> "2")
    TASK_ID=""
    if [[ "$CHECKPOINT_LOCATION" =~ task_([0-9]+) ]]; then
        TASK_ID="${BASH_REMATCH[1]}"
    else
        echo "ERROR: Could not extract task ID from checkpoint path: $CHECKPOINT_LOCATION"
        echo "       Expected path format: .../task_<N>_coeff_<X>_<Y>/..."
        exit 1
    fi
    
    if [ "$TASK_ID" -gt 0 ]; then
        # Need to load previous adapters
        COEFFICIENT_PATH=$(echo "$PREVIOUS_LORA_COEFF" | tr '.' '_')
        LORA_PATHS=()
        
        # Load previous adapters (0 to TASK_ID-1)
        for prev_task in $(seq 0 $((TASK_ID - 1))); do
            if [ "$prev_task" -eq 0 ]; then
                # Task 0 comes from naive_lora folder (no coefficient in path)
                prev_adapter_path="${WORKSPACE_ROOT}/logs/naive_lora/task_${prev_task}/checkpoints/global_step_10/actor"
            else
                # Tasks 1+ use coefficient in path
                prev_adapter_path="${WORKSPACE_ROOT}/logs/naive_lora_multilora/task_${prev_task}_coeff_${COEFFICIENT_PATH}/checkpoints/global_step_10/actor"
            fi
            
            if [ ! -d "$prev_adapter_path" ]; then
                echo "ERROR: Previous adapter not found for task $prev_task: $prev_adapter_path"
                echo "       This is required for multi-LoRA evaluation. Please ensure all previous task adapters exist."
                exit 1
            fi
            
            LORA_PATHS+=("$prev_adapter_path")
            echo "  Found previous adapter for task $prev_task: $prev_adapter_path"
        done
        
        if [ ${#LORA_PATHS[@]} -eq 0 ]; then
            echo "ERROR: No previous adapters found for task $TASK_ID"
            echo "       Expected to find adapters for tasks 0 to $((TASK_ID - 1))"
            exit 1
        fi
        
        # Convert array to comma-separated string for Hydra
        LORA_PATHS_STR=$(IFS=','; echo "${LORA_PATHS[*]}")
        HYDRA_OVERRIDES="${HYDRA_OVERRIDES} +actor.model.lora_paths=[${LORA_PATHS_STR}] +actor.model.previous_lora_merge_coefficient=${PREVIOUS_LORA_COEFF}"
        echo "  Will merge ${#LORA_PATHS[@]} previous adapter(s) with coefficient ${PREVIOUS_LORA_COEFF}"
    elif [ "$TASK_ID" -eq 0 ]; then
        echo "  Task 0: No previous adapters to load (this is the first task)"
    else
        echo "ERROR: Invalid task ID extracted: $TASK_ID"
        exit 1
    fi
fi

# Pass overrides to the eval script
bash examples/embodiment/eval_embodiment.sh mll_cluster/libero_spatial_grpo_openvlaoft_eval ${HYDRA_OVERRIDES}

# Print completion time
echo "End Time: $(date)"
