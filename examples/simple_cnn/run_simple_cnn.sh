#!/bin/bash
# Simple script to train and evaluate CNN policy on LIBERO

# Set paths
DATA_DIR="${LIBERO_REPO_PATH:-../LIBERO}/libero/datasets_with_logits/libero_spatial_simplevla_trajall"
OUTPUT_DIR="./logs/simple_cnn_libero_spatial"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training and evaluation
python examples/simple_cnn/train_and_eval.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --task_suite libero_spatial \
    --num_epochs 10 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --eval_num_trials 10

echo "Training and evaluation complete!"
echo "Checkpoint saved to: $OUTPUT_DIR"
