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
#SBATCH --exclude=slurm-node-008,slurm-node-011
#SBATCH --output=logs/slurm/%x-%j.out              # Output file name (%x = job name, %j = job ID)
#SBATCH --error=logs/slurm/%x-%j.err               # Error file name


### Usage: sbatch examples/mll_cluster/run_embodiment_naive_lora_multitask.slurm TASK_IDS [CHECKPOINT_PATH] [MAX_EPOCH] [CONFIG_NAME] [SEED]
### Example: sbatch examples/mll_cluster/run_embodiment_naive_lora_multitask.slurm "0,2,4"
### Example: sbatch examples/mll_cluster/run_embodiment_naive_lora_multitask.slurm "0 2 4"
### Example (with max_epoch): sbatch examples/mll_cluster/run_embodiment_naive_lora_multitask.slurm "0,2,4" "" 15
### Example (continue from checkpoint): sbatch examples/mll_cluster/run_embodiment_naive_lora_multitask.slurm "0,2,4" ./logs/naive_lora_multitask/tasks_0_2_4_seed1234/checkpoints/global_step_10/actor 20
### Example (with seed): sbatch examples/mll_cluster/run_embodiment_naive_lora_multitask.slurm "0,2,4" "" "" "" 42
### Note: TASK_IDS can be comma-separated (e.g., "0,2,4") or space-separated (e.g., "0 2 4")
###       CHECKPOINT_PATH is optional and will load model weights (LoRA adapter)
###       Evaluation runs automatically every 10 global steps during training
###       MAX_EPOCH is optional and can override the default max_epochs
###       SEED is optional and defaults to 1234 if not provided

# Call the shared script with all arguments (path relative to workspace root)
bash examples/mll_cluster/run_embodiment_naive_lora_multitask.sh "$@"

TRAINING_EXIT_CODE=$?

# If training succeeded, run evaluation in this script (not in the .sh file)
# This avoids nested buffering issues when eval is called from within the .sh subprocess
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "Running evaluation for all checkpoints..."
    echo "========================================="
    
    # Reconstruct LOG_DIR from arguments (same logic as in .sh file)
    TASK_IDS_STR=$1
    CHECKPOINT_PATH=$2
    SEED=${5:-1234}
    TASK_IDS_STR=$(echo "$TASK_IDS_STR" | tr ',' ' ')
    read -ra TASK_IDS_ARRAY <<< "$TASK_IDS_STR"
    IFS=$'\n' sorted_task_ids=($(sort -n <<<"${TASK_IDS_ARRAY[*]}"))
    unset IFS
    TASK_DIR_STR=$(IFS='_'; echo "${sorted_task_ids[*]}")
    
    if [ -n "$CHECKPOINT_PATH" ] && [[ "$CHECKPOINT_PATH" =~ global_step_([0-9]+) ]]; then
        SOURCE_STEP="${BASH_REMATCH[1]}"
        LOG_DIR="./logs/naive_lora_multitask/tasks_${TASK_DIR_STR}_from_checkpoint_step_${SOURCE_STEP}_seed${SEED}"
    elif [ -n "$CHECKPOINT_PATH" ]; then
        LOG_DIR="./logs/naive_lora_multitask/tasks_${TASK_DIR_STR}_from_checkpoint_seed${SEED}"
    else
        LOG_DIR="./logs/naive_lora_multitask/tasks_${TASK_DIR_STR}_seed${SEED}"
    fi
    
    CONFIG_NAME=${4:-mll_cluster/libero_spatial_grpo_openvlaoft}
    
    # Skip evaluation if using test config
    if [[ "$CONFIG_NAME" == *"test"* ]] || [[ "$CONFIG_NAME" == *"libero_spatial_grpo_openvlaoft_test"* ]]; then
        echo "Skipping evaluation (test config detected: $CONFIG_NAME)"
    else
        # Discover and evaluate all checkpoints
        CHECKPOINTS_DIR="${LOG_DIR}/checkpoints"
        if [ ! -d "$CHECKPOINTS_DIR" ]; then
            echo "Warning: Checkpoints directory not found at $CHECKPOINTS_DIR"
        else
            echo "Discovering checkpoints for evaluation..."
            CHECKPOINT_STEPS=($(find "$CHECKPOINTS_DIR" -maxdepth 1 -type d -name "global_step_*" | sed 's|.*/global_step_||' | sort -n))
            
            if [ ${#CHECKPOINT_STEPS[@]} -eq 0 ]; then
                echo "Warning: No checkpoints found in $CHECKPOINTS_DIR"
            else
                echo "Found ${#CHECKPOINT_STEPS[@]} checkpoint(s) at steps: ${CHECKPOINT_STEPS[*]}"
                echo ""
                echo "Running evaluation for each checkpoint..."
                echo ""
                
                CHECKPOINT_LOCATION=$(echo "$LOG_DIR" | sed 's|^\./||')
                EVAL_EXIT_CODE=0
                
                for STEP in "${CHECKPOINT_STEPS[@]}"; do
                    echo "========================================="
                    echo "Evaluating checkpoint at global_step_${STEP}"
                    echo "========================================="
                    bash examples/mll_cluster/eval_embodiment.sh "${CHECKPOINT_LOCATION}" "${STEP}"
                    
                    STEP_EXIT_CODE=$?
                    if [ $STEP_EXIT_CODE -ne 0 ]; then
                        echo "  Warning: Evaluation for step ${STEP} failed (exit code: $STEP_EXIT_CODE)"
                        EVAL_EXIT_CODE=$STEP_EXIT_CODE
                    else
                        echo "  ✓ Evaluation for step ${STEP} completed successfully"
                    fi
                    echo ""
                done
                
                if [ $EVAL_EXIT_CODE -eq 0 ]; then
                    echo "All checkpoint evaluations completed successfully"
                else
                    echo "Some checkpoint evaluations failed (last exit code: $EVAL_EXIT_CODE)"
                fi
            fi
        fi
    fi
fi

exit $TRAINING_EXIT_CODE
