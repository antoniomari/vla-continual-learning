#!/bin/bash
### Usage: bash examples/crl_experiment/run_embodiment_sequential.sh TASK_ID_OR_RANGE [CHECKPOINT_PATH] [MAX_EPOCH] [CONFIG_NAME] [SEED]
### OPD default config: bash examples/crl_experiment/run_embodiment_opd_sequential.sh TASK_ID_OR_RANGE [...same optional args...]
### Example (single task): bash examples/crl_experiment/run_embodiment_sequential.sh 0
### Example (task range): bash examples/crl_experiment/run_embodiment_sequential.sh "0,3"
### Example (with max_epoch): bash examples/crl_experiment/run_embodiment_sequential.sh 0 "" 15
### Example (continue from checkpoint): bash examples/crl_experiment/run_embodiment_sequential.sh 0 ./logs/sequential/task_0_seed1234/checkpoints/global_step_50/actor 20
### Example (force base model): bash examples/crl_experiment/run_embodiment_sequential.sh 4 base 50
### Example (with seed): bash examples/crl_experiment/run_embodiment_sequential.sh 0 "" "" "" 42
### Note: TASK_ID_OR_RANGE can be:
###       - A single task ID (e.g., "0") - trains that task only
###       - A tuple "a,b" where a < b (e.g., "0,3") - trains tasks from a to b sequentially
###       CHECKPOINT_PATH is optional and will be auto-generated from previous task if not provided
###       If CHECKPOINT_PATH is provided for a range, it will only be used for the first task
###       MAX_EPOCH is optional and can always be specified to override the default max_epochs
###       SEED is optional and defaults to 1234 if not provided
### Optional: EXPERIMENT_NAME_PREFIX=opd_ prepended to runner.logger.experiment_name (WandB run name);
###           run_embodiment_opd_sequential.sh sets this by default.
### Optional (Slurm sweep): SWEEP_GROUP_SIZE, SWEEP_NUM_GROUP_ENVS, SWEEP_ROLLOUT_EPOCH,
###           SWEEP_GLOBAL_BATCH_SIZE — appended as Hydra overrides for GRPO / actor batch sizing.
### Optional (OPD Slurm sweep): SWEEP_OPD_BC_GLOBAL_BATCH_SIZE, SWEEP_OPD_BC_BATCH_SIZE,
###           SWEEP_OPD_BC_STEPS, SWEEP_OPD_TEACHER_LR — BC warmup overrides (see jobs/embodiment_slurm_opd_sweep.sh).
###           SWEEP_OPD_NORMALIZE_ADVANTAGES — override algorithm.normalize_advantages.
###           SWEEP_OPD_RL_TEACHER — override algorithm.rl_teacher (0/1).
### Eval after each task: passes global_step (= MAX_EPOCH or get_default_global_step), seed,
###           env.fixed_task_ids=null (all suite tasks, e.g. 10 for LIBERO spatial), same SWEEP_* as training,
###           and runner.logger.experiment_name=eval_<train_name>_step_<N>. Training still uses one task per stage.
### Optional: EVAL_HYDRA_OVERRIDES is consumed by eval_embodiment.sh (do not set manually unless extending).

TASK_INPUT=${1:-0}
MANUAL_CHECKPOINT_PATH=$2
MAX_EPOCH=$3
CONFIG_NAME=${4:-crl_experiment/libero_spatial_grpo_openvlaoft_spatial}
SEED=${5:-1234}

# Special token: if CHECKPOINT_PATH is literal "base", force first task to start
# from base model (no previous-task checkpoint), even when TASK_ID > FIRST_TASK_ID.
USE_BASE_MODEL_START=0
if [[ "${MANUAL_CHECKPOINT_PATH,,}" == "base" ]]; then
    USE_BASE_MODEL_START=1
    MANUAL_CHECKPOINT_PATH=""
fi

# Log subdirectory for this experiment type
EXPERIMENT_TYPE="sequential"

# Parse TASK_INPUT to determine if it's a single task or a range
if [[ "$TASK_INPUT" == *,* ]]; then
    IFS=',' read -r TASK_START TASK_END <<< "$TASK_INPUT"
    TASK_START=$(echo "$TASK_START" | tr -d '()[] ')
    TASK_END=$(echo "$TASK_END" | tr -d '()[] ')

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

if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
    echo "ERROR: SEED must be a non-negative integer, got: $SEED"
    exit 1
fi

mkdir -p "logs/${EXPERIMENT_TYPE}"

echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo ""

# Change to repo root
SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT=$(dirname $(dirname "$SCRIPT_DIR"))
cd "$REPO_ROOT"

# Source common functions
source "examples/crl_experiment/common_functions.sh"

# Extract config tag and derive eval config name
CONFIG_TAG=$(extract_config_tag "$CONFIG_NAME")
EVAL_CONFIG_NAME=$(derive_eval_config_name "$CONFIG_NAME")
# Final checkpoint folder is global_step_<N> with N == max_epochs for embodied training.
if [ -n "$MAX_EPOCH" ]; then
    GLOBAL_STEP="$MAX_EPOCH"
else
    GLOBAL_STEP=$(get_default_global_step "$CONFIG_NAME")
fi
FIRST_TASK_ID=$(get_first_task_id "$CONFIG_NAME")

# Main training loop
OVERALL_EXIT_CODE=0

for TASK_ID in $(seq $TASK_START $TASK_END); do
    echo ""
    echo "========================================="
    if [ "$IS_RANGE" = true ]; then
        echo "Sequential Training - Task ${TASK_ID} (${TASK_START} to ${TASK_END})"
    else
        echo "Sequential Training - Single Task (LoRA)"
    fi
    echo "========================================="

    # Determine checkpoint path
    if [ "$TASK_ID" -eq "$TASK_START" ] && [ "$USE_BASE_MODEL_START" -eq 1 ]; then
        CHECKPOINT_PATH=""
    elif [ "$TASK_ID" -eq "$TASK_START" ] && [ -n "$MANUAL_CHECKPOINT_PATH" ]; then
        CHECKPOINT_PATH="$MANUAL_CHECKPOINT_PATH"
    elif [ "$TASK_ID" -eq $FIRST_TASK_ID ]; then
        CHECKPOINT_PATH=""
    else
        PREV_TASK_ID=$((TASK_ID - 1))
        PREV_LOG_DIR="./logs/${EXPERIMENT_TYPE}/task_${PREV_TASK_ID}_seed${SEED}"
        if [ -n "$CONFIG_TAG" ]; then
            PREV_LOG_DIR_TRANSFORMED=$(inject_config_tag_into_log_path "$PREV_LOG_DIR" "$CONFIG_TAG")
            if [ -z "$PREV_LOG_DIR_TRANSFORMED" ]; then
                echo "  ERROR: Failed to transform previous task log directory for task $PREV_TASK_ID"
                echo "         Original PREV_LOG_DIR: [$PREV_LOG_DIR]"
                echo "         CONFIG_TAG: [$CONFIG_TAG]"
                OVERALL_EXIT_CODE=1
                break
            fi
        else
            PREV_LOG_DIR_TRANSFORMED="$PREV_LOG_DIR"
        fi
        CHECKPOINT_PATH="${PREV_LOG_DIR_TRANSFORMED}/checkpoints/global_step_${GLOBAL_STEP}/actor"

        if [[ "$CHECKPOINT_PATH" =~ ^/checkpoints/ ]]; then
            echo "  ERROR: Invalid checkpoint path construction detected"
            echo "         PREV_LOG_DIR was likely empty or malformed"
            echo "         PREV_LOG_DIR: [$PREV_LOG_DIR]"
            echo "         CHECKPOINT_PATH: [$CHECKPOINT_PATH]"
            OVERALL_EXIT_CODE=1
            break
        fi
    fi

    # Determine LOG_DIR based on checkpoint path
    if [ "$TASK_ID" -eq "$TASK_START" ] && [ "$USE_BASE_MODEL_START" -eq 1 ]; then
        LOG_DIR="./logs/${EXPERIMENT_TYPE}/task_${TASK_ID}_seed${SEED}"
    elif [ "$TASK_ID" -eq "$TASK_START" ] && [ -n "$MANUAL_CHECKPOINT_PATH" ]; then
        if [[ "$CHECKPOINT_PATH" =~ task_([0-9]+) ]]; then
            SOURCE_TASK="${BASH_REMATCH[1]}"
            if [[ "$CHECKPOINT_PATH" =~ global_step_([0-9]+) ]]; then
                SOURCE_STEP="${BASH_REMATCH[1]}"
                LOG_DIR="./logs/${EXPERIMENT_TYPE}/task_${TASK_ID}_from_task_${SOURCE_TASK}_step_${SOURCE_STEP}_seed${SEED}"
            else
                echo "ERROR: Could not extract global_step from checkpoint path: $CHECKPOINT_PATH"
                echo "       Expected format: .../checkpoints/global_step_<M>/actor"
                OVERALL_EXIT_CODE=1
                break
            fi
        else
            echo "ERROR: Could not extract task ID from checkpoint path: $CHECKPOINT_PATH"
            echo "       Expected format: .../task_<N>/checkpoints/global_step_<M>/actor"
            OVERALL_EXIT_CODE=1
            break
        fi
    else
        LOG_DIR="./logs/${EXPERIMENT_TYPE}/task_${TASK_ID}_seed${SEED}"
    fi

    # Inject config tag into LOG_DIR
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

    if [ -z "$LOG_DIR" ]; then
        echo "  ERROR: LOG_DIR is empty after path construction"
        OVERALL_EXIT_CODE=1
        break
    fi

    export LOG_DIR
    mkdir -p "${LOG_DIR}"

    EXPERIMENT_NAME=$(basename "$LOG_DIR")
    if [ -n "$CONFIG_TAG" ]; then
        EXPERIMENT_NAME="${EXPERIMENT_NAME}_${CONFIG_TAG}"
    fi
    if [ -n "${EXPERIMENT_NAME_PREFIX:-}" ]; then
        EXPERIMENT_NAME="${EXPERIMENT_NAME_PREFIX}${EXPERIMENT_NAME}"
    fi

    echo "Configuration:"
    echo "  Task ID: $TASK_ID"
    if [ "$IS_RANGE" = true ]; then
        echo "  Task Range: ${TASK_START} to ${TASK_END}"
    fi
    echo "  Experiment Name: $EXPERIMENT_NAME"
    echo "  Experiment Type: $EXPERIMENT_TYPE"
    echo "  Checkpoint Save Path: $LOG_DIR"
    echo "  Config Name: $CONFIG_NAME"
    echo "  Random Seed: $SEED"

    if [ -n "$CHECKPOINT_PATH" ]; then
        if [ ! -d "$CHECKPOINT_PATH" ]; then
            echo "  ERROR: Checkpoint not found at $CHECKPOINT_PATH"
            if [ "$TASK_ID" -gt $FIRST_TASK_ID ]; then
                PREV_TASK_ID=$((TASK_ID - 1))
                echo "         Task $TASK_ID requires checkpoint from task $PREV_TASK_ID"
                echo "         Make sure task $PREV_TASK_ID has been trained first, or provide a manual checkpoint path"
            fi
            OVERALL_EXIT_CODE=1
            break
        fi
        echo "  Loading from checkpoint: $CHECKPOINT_PATH"
    else
        if [ "$TASK_ID" -eq "$TASK_START" ] && [ "$USE_BASE_MODEL_START" -eq 1 ]; then
            echo "  Training from base model (SFT checkpoint) - BASE override for task $TASK_ID"
        else
            echo "  Training from base model (SFT checkpoint) - First task (task $FIRST_TASK_ID)"
        fi
    fi

    if [ -n "$MAX_EPOCH" ]; then
        if ! [[ "$MAX_EPOCH" =~ ^[0-9]+$ ]] || [ "$MAX_EPOCH" -le 0 ]; then
            echo "  ERROR: MAX_EPOCH must be a positive integer, got: $MAX_EPOCH"
            OVERALL_EXIT_CODE=1
            break
        fi
        echo "  Max epochs: $MAX_EPOCH"
    fi

    echo "========================================="
    echo ""

    # Build Hydra overrides
    OVERRIDES="env.fixed_task_ids=[${TASK_ID}] \
    	runner.logger.experiment_name=${EXPERIMENT_NAME} \
    	actor.seed=${SEED}"

    if [ -n "$CHECKPOINT_PATH" ]; then
        OVERRIDES="$OVERRIDES +actor.model.lora_path=${CHECKPOINT_PATH}"
    fi

    if [ -n "$MAX_EPOCH" ]; then
        OVERRIDES="$OVERRIDES runner.max_epochs=${MAX_EPOCH}"
    fi

    # Optional: Slurm sweep (examples/crl_experiment/jobs/embodiment_slurm_sweep.sh) exports these
    # to override GRPO rollout geometry and actor.global_batch_size without editing the yaml.
    if [ -n "${SWEEP_GROUP_SIZE:-}" ]; then
        OVERRIDES="$OVERRIDES algorithm.group_size=${SWEEP_GROUP_SIZE}"
    fi
    if [ -n "${SWEEP_NUM_GROUP_ENVS:-}" ]; then
        OVERRIDES="$OVERRIDES algorithm.num_group_envs=${SWEEP_NUM_GROUP_ENVS}"
    fi
    if [ -n "${SWEEP_ROLLOUT_EPOCH:-}" ]; then
        OVERRIDES="$OVERRIDES algorithm.rollout_epoch=${SWEEP_ROLLOUT_EPOCH}"
    fi
    if [ -n "${SWEEP_GLOBAL_BATCH_SIZE:-}" ]; then
        OVERRIDES="$OVERRIDES actor.global_batch_size=${SWEEP_GLOBAL_BATCH_SIZE}"
    fi
    if [ -n "${SWEEP_OPD_BC_GLOBAL_BATCH_SIZE:-}" ]; then
        OVERRIDES="$OVERRIDES algorithm.opd_bc_global_batch_size=${SWEEP_OPD_BC_GLOBAL_BATCH_SIZE}"
    fi
    if [ -n "${SWEEP_OPD_BC_BATCH_SIZE:-}" ]; then
        OVERRIDES="$OVERRIDES algorithm.opd_bc_batch_size=${SWEEP_OPD_BC_BATCH_SIZE}"
    fi
    if [ -n "${SWEEP_OPD_BC_STEPS:-}" ]; then
        OVERRIDES="$OVERRIDES algorithm.opd_bc_steps=${SWEEP_OPD_BC_STEPS}"
    fi
    if [ -n "${SWEEP_OPD_TEACHER_LR:-}" ]; then
        OVERRIDES="$OVERRIDES actor.optim.opd_teacher_lr=${SWEEP_OPD_TEACHER_LR}"
    fi
    if [ -n "${SWEEP_OPD_SFT_FILTER_FIXED_TASK_IDS:-}" ]; then
        OVERRIDES="$OVERRIDES +algorithm.sft_filter_fixed_task_ids_for_opd=${SWEEP_OPD_SFT_FILTER_FIXED_TASK_IDS}"
    fi
    if [ -n "${SWEEP_OPD_SFT_MATCH_TASK_LANGUAGE:-}" ]; then
        OVERRIDES="$OVERRIDES +algorithm.sft_match_rollout_task_language=${SWEEP_OPD_SFT_MATCH_TASK_LANGUAGE}"
    fi
    if [ -n "${SWEEP_OPD_SFT_MATCH_IMAGE_ROTATION:-}" ]; then
        OVERRIDES="$OVERRIDES +algorithm.sft_match_rollout_image_rotation=${SWEEP_OPD_SFT_MATCH_IMAGE_ROTATION}"
    fi
    if [ -n "${SWEEP_OPD_SFT_MATCH_OBS_ACTION_ALIGNMENT:-}" ]; then
        OVERRIDES="$OVERRIDES +algorithm.sft_match_rollout_obs_action_alignment=${SWEEP_OPD_SFT_MATCH_OBS_ACTION_ALIGNMENT}"
    fi
    if [ -n "${SWEEP_OPD_NORMALIZE_ADVANTAGES:-}" ]; then
        OVERRIDES="$OVERRIDES algorithm.normalize_advantages=${SWEEP_OPD_NORMALIZE_ADVANTAGES}"
    fi
    if [ -n "${SWEEP_OPD_RL_TEACHER:-}" ]; then
        OVERRIDES="$OVERRIDES algorithm.rl_teacher=${SWEEP_OPD_RL_TEACHER}"
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
        echo "Checkpoint saved to: ${LOG_DIR}"

        CHECKPOINT_LOCATION=$(echo "$LOG_DIR" | sed 's|^\./||')
        EVAL_GLOBAL_STEP="${GLOBAL_STEP}"
        EVAL_EXPERIMENT_NAME="eval_${EXPERIMENT_NAME}_step_${EVAL_GLOBAL_STEP}"
        EVAL_HYDRA_OVERRIDES="runner.logger.experiment_name=${EVAL_EXPERIMENT_NAME} actor.seed=${SEED} env.fixed_task_ids=null"
        if [ -n "${SWEEP_GROUP_SIZE:-}" ]; then
            EVAL_HYDRA_OVERRIDES="${EVAL_HYDRA_OVERRIDES} algorithm.group_size=${SWEEP_GROUP_SIZE}"
        fi
        if [ -n "${SWEEP_NUM_GROUP_ENVS:-}" ]; then
            EVAL_HYDRA_OVERRIDES="${EVAL_HYDRA_OVERRIDES} algorithm.num_group_envs=${SWEEP_NUM_GROUP_ENVS}"
        fi
        if [ -n "${SWEEP_ROLLOUT_EPOCH:-}" ]; then
            EVAL_HYDRA_OVERRIDES="${EVAL_HYDRA_OVERRIDES} algorithm.rollout_epoch=${SWEEP_ROLLOUT_EPOCH}"
        fi
        if [ -n "${SWEEP_GLOBAL_BATCH_SIZE:-}" ]; then
            EVAL_HYDRA_OVERRIDES="${EVAL_HYDRA_OVERRIDES} actor.global_batch_size=${SWEEP_GLOBAL_BATCH_SIZE}"
        fi
        export EVAL_HYDRA_OVERRIDES
        echo ""
        echo "Running evaluation (all suite tasks; env.fixed_task_ids=null) for: ${CHECKPOINT_LOCATION} global_step=${EVAL_GLOBAL_STEP}"
        echo "  W&B eval run name (experiment_name): ${EVAL_EXPERIMENT_NAME}"
        bash examples/crl_experiment/eval_embodiment.sh "${CHECKPOINT_LOCATION}" "${EVAL_GLOBAL_STEP}" "${EVAL_CONFIG_NAME}"
        unset EVAL_HYDRA_OVERRIDES
    else
        echo "Task $TASK_ID failed with exit code $EXIT_CODE"
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
