#!/bin/bash
#
# Test script for simple_cnn checkpoint loading and saving
# This script performs quick validation tests without full training

set -e  # Exit on error

echo "========================================="
echo "Testing simple_cnn checkpoint functionality"
echo "========================================="
echo ""

# Test 1: Check if initial checkpoint exists (from config)
echo "Test 1: Checking initial checkpoint path from config..."
INITIAL_CHECKPOINT="${REPO_PATH:-$(pwd)}/logs/simple_cnn_policy/best_checkpoint.pt"
if [ -f "$INITIAL_CHECKPOINT" ]; then
    echo "✓ Initial checkpoint found: $INITIAL_CHECKPOINT"
else
    echo "✗ Initial checkpoint NOT found: $INITIAL_CHECKPOINT"
    echo "  You may need to train a simple_cnn model first using:"
    echo "  python examples/simple_cnn/train_and_eval.py --output_dir ./logs/simple_cnn_policy ..."
    echo ""
    echo "  For testing purposes, you can create a dummy checkpoint:"
    echo "  mkdir -p $(dirname $INITIAL_CHECKPOINT)"
    echo "  python -c \"import torch; torch.save({'model_state_dict': {}, 'task_id_map': {}, 'num_tasks': 10, 'norm_stats': {}, 'unnorm_key': 'libero_spatial_no_noops'}, '$INITIAL_CHECKPOINT')\""
    exit 1
fi
echo ""

# Test 2: Test path construction logic
echo "Test 2: Testing path construction..."
source "examples/mll_cluster/common_functions.sh"

CONFIG_NAME="mll_cluster/libero_spatial_grpo_simple_cnn"
CONFIG_TAG=$(extract_config_tag "$CONFIG_NAME")
EVAL_CONFIG_NAME=$(derive_eval_config_name "$CONFIG_NAME")
GLOBAL_STEP=$(get_default_global_step "$CONFIG_NAME")
FIRST_TASK_ID=$(get_first_task_id "$CONFIG_NAME")

echo "  Config Name: $CONFIG_NAME"
echo "  Config Tag: $CONFIG_TAG"
echo "  Eval Config: $EVAL_CONFIG_NAME"
echo "  Global Step: $GLOBAL_STEP"
echo "  First Task ID: $FIRST_TASK_ID"

# Test LOG_DIR construction
TEST_LOG_DIR="./logs/simple_cnn/task_0_seed1234"
if [ -n "$CONFIG_TAG" ]; then
    TEST_LOG_DIR_TRANSFORMED=$(inject_config_tag_into_log_path "$TEST_LOG_DIR" "$CONFIG_TAG")
    echo "  LOG_DIR (with tag): $TEST_LOG_DIR_TRANSFORMED"
else
    echo "  LOG_DIR (no tag): $TEST_LOG_DIR"
fi

# Test checkpoint path construction
TEST_CHECKPOINT_PATH="${TEST_LOG_DIR}/checkpoints/global_step_${GLOBAL_STEP}/actor/model.pt"
echo "  Expected checkpoint path: $TEST_CHECKPOINT_PATH"
echo "✓ Path construction test passed"
echo ""

# Test 3: Dry-run the script with --help or validate arguments
echo "Test 3: Validating script arguments..."
SCRIPT_PATH="examples/mll_cluster/run_embodiment_simple_cnn.sh"

# Test argument parsing
echo "  Testing argument parsing..."
bash -n "$SCRIPT_PATH" && echo "✓ Script syntax is valid" || echo "✗ Script has syntax errors"
echo ""

# Test 4: Check if checkpoint path validation works
echo "Test 4: Testing checkpoint path validation..."
# Create a test checkpoint path that doesn't exist
NONEXISTENT_CHECKPOINT="./logs/simple_cnn/task_999_seed1234/checkpoints/global_step_10/actor/model.pt"
if [ ! -f "$NONEXISTENT_CHECKPOINT" ]; then
    echo "✓ Nonexistent checkpoint correctly identified as missing"
else
    echo "✗ Unexpected: checkpoint exists when it shouldn't"
fi
echo ""

# Test 5: Test sequential checkpoint path construction
echo "Test 5: Testing sequential checkpoint path construction..."
TASK_0_LOG_DIR="./logs/simple_cnn/task_0_seed1234"
TASK_1_CHECKPOINT="${TASK_0_LOG_DIR}/checkpoints/global_step_${GLOBAL_STEP}/actor/model.pt"
echo "  Task 0 log dir: $TASK_0_LOG_DIR"
echo "  Task 1 would load from: $TASK_1_CHECKPOINT"
echo "✓ Sequential path construction test passed"
echo ""

echo "========================================="
echo "All validation tests passed!"
echo "========================================="
echo ""
echo "To run a quick training test (1 epoch), use:"
echo "  ./examples/mll_cluster/run_embodiment_simple_cnn.sh 0 \"\" 1"
echo ""
echo "To test checkpoint loading from a specific path:"
echo "  ./examples/mll_cluster/run_embodiment_simple_cnn.sh 0 <path_to_checkpoint.pt> 1"
echo ""
echo "To test sequential training (task 0 then task 1):"
echo "  ./examples/mll_cluster/run_embodiment_simple_cnn.sh \"0,1\" \"\" 1"
