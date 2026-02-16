#!/usr/bin/env python3
# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Main script to train and evaluate simple CNN policy on LIBERO.

This script:
1. Trains the CNN policy on LIBERO demonstrations
2. Evaluates the trained policy on LIBERO tasks
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate simple CNN policy on LIBERO"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(
            os.environ.get("LIBERO_REPO_PATH", "../LIBERO"),
            "libero/datasets_with_logits/libero_spatial_simplevla_trajall"
        ),
        help="Directory containing LIBERO HDF5 files (default: LIBERO_REPO_PATH/libero/datasets_with_logits/libero_spatial_simplevla_trajall)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--task_suite",
        type=str,
        default="libero_spatial",
        help="LIBERO task suite name for evaluation (default: libero_spatial)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--eval_num_trials",
        type=int,
        default=None,
        help="Number of evaluation trials per task (optional override, uses config default if not set)",
    )
    parser.add_argument(
        "--eval_num_parallel_envs",
        type=int,
        default=None,
        help="Number of parallel environments for evaluation (optional override, uses config default if not set)",
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip training and only run evaluation",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path for evaluation (if skipping training)",
    )
    
    args = parser.parse_args()
    
    # Get script directory
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent
    
    # Add repo to Python path
    sys.path.insert(0, str(repo_root))
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training
    checkpoint_path = None
    if not args.skip_training:
        print("=" * 80)
        print("Starting Training")
        print("=" * 80)
        
        train_script = repo_root / "rlinf" / "training" / "simple_cnn_train.py"
        
        train_cmd = [
            sys.executable,
            str(train_script),
            "--data_dir", args.data_dir,
            "--output_dir", args.output_dir,
            "--num_epochs", str(args.num_epochs),
            "--batch_size", str(args.batch_size),
            "--learning_rate", str(args.learning_rate),
        ]
        
        print(f"Running: {' '.join(train_cmd)}")
        result = subprocess.run(train_cmd, check=True)
        
        # Find best checkpoint
        best_checkpoint = os.path.join(args.output_dir, "best_checkpoint.pt")
        if os.path.exists(best_checkpoint):
            checkpoint_path = best_checkpoint
        else:
            # Find latest checkpoint
            checkpoints = [
                f for f in os.listdir(args.output_dir)
                if f.startswith("checkpoint_epoch_") and f.endswith(".pt")
            ]
            if checkpoints:
                # Sort by epoch number
                checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
                checkpoint_path = os.path.join(args.output_dir, checkpoints[-1])
        
        if checkpoint_path is None:
            raise RuntimeError("No checkpoint found after training!")
        
        print(f"Training complete. Using checkpoint: {checkpoint_path}")
    else:
        if args.checkpoint is None:
            raise ValueError("--checkpoint required when --skip_training is set")
        checkpoint_path = args.checkpoint
    
    # Evaluation using eval_embodiment.sh
    print("=" * 80)
    print("Starting Evaluation (using eval_embodiment.sh)")
    print("=" * 80)
    
    eval_script = repo_root / "examples" / "embodiment" / "eval_embodiment.sh"
    
    # Convert checkpoint path to absolute if relative
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.abspath(checkpoint_path)
    
    # Build eval command
    eval_cmd = [
        "bash",
        str(eval_script),
        "mll_cluster/libero_spatial_cnn_eval",
        f"rollout.checkpoint_path={checkpoint_path}",
    ]
    
    # Add optional overrides if provided
    if args.eval_num_trials is not None:
        eval_cmd.append(f"env.eval.eval_per_task={args.eval_num_trials}")
    
    if args.eval_num_parallel_envs is not None:
        eval_cmd.append(f"env.eval.num_envs={args.eval_num_parallel_envs}")
    
    print(f"Running: {' '.join(eval_cmd)}")
    result = subprocess.run(eval_cmd, check=True, cwd=str(repo_root))
    
    print("=" * 80)
    print("Training and Evaluation Complete!")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Evaluation results: Check logs/evals/ directory for results")


if __name__ == "__main__":
    main()
