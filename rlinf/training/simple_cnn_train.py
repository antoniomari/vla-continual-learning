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
Training script for simple CNN policy.

This script performs supervised finetuning on LIBERO demonstrations.
"""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from omegaconf import DictConfig, OmegaConf

from rlinf.custom.simple_cnn_utils import compute_action_statistics
from rlinf.models.simple_cnn_policy import SimpleCNNPolicy


class SimpleCNNDataset(Dataset):
    """
    Simple CNN dataset that matches CNNRolloutWorker preprocessing.
    
    Key features:
    1. Uses same image preprocessing as CNNRolloutWorker (ImageNet normalization)
    2. Uses task IDs (not one-hot, matching model's task_embedding)
    3. Uses exact same action processing as OpenVLA (normalization + tokenization)
    4. No dependency on OpenVLA processor
    """
    
    def __init__(
        self,
        root_dir: str,
        image_size: int = 224,
        num_action_chunks: int = 1,
        demos_per_task: int = 1,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.root_dir = root_dir
        self.image_size = image_size
        self.num_action_chunks = num_action_chunks
        
        # SAME image transforms as CNNRolloutWorker (ImageNet normalization)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Load all trajectories
        self.task_files = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.endswith(".hdf5")
        ]
        
        self.trajectories = []
        for path in self.task_files:
            filename = os.path.basename(path)
            task_desc = self._extract_task_description(filename)
            with h5py.File(path, "r") as f:
                # Use all demos if demos_per_task <= 0, otherwise limit to demos_per_task
                all_demo_names = sorted(list(f["data"].keys()))
                if demos_per_task > 0:
                    demo_names = all_demo_names[:demos_per_task]
                else:
                    demo_names = all_demo_names  # Use all demos
                for name in demo_names:
                    traj_len = len(f["data"][name]["actions"])
                    self.trajectories.append((path, name, traj_len, task_desc))
        
        # Create sample indices
        self.sample_indices = []
        for path, demo_name, traj_len, task_desc in self.trajectories:
            valid_len = traj_len - self.num_action_chunks + 1
            if valid_len > 0:
                for t in range(valid_len):
                    self.sample_indices.append((path, demo_name, t, task_desc))
        
        self.sample_indices = self.sample_indices[rank::world_size]
        self.file_handles = {}
        
        # Build task ID map (sorted for consistency)
        task_descriptions = sorted(set([desc for _, _, _, desc in self.sample_indices]))
        self.task_id_map = {task: idx for idx, task in enumerate(task_descriptions)}
    
    def _extract_task_description(self, filename):
        """Extract task description from filename."""
        name = filename.replace(".hdf5", "")
        parts = name.split("_")
        if parts[-1] == "demo":
            parts = parts[:-1]
        return " ".join(parts)
    
    def __len__(self):
        return len(self.sample_indices)
    
    def __getitem__(self, idx):
        path, demo_name, timestep, task_desc = self.sample_indices[idx]
        
        if path not in self.file_handles:
            self.file_handles[path] = h5py.File(path, "r")
        f = self.file_handles[path]
        
        demo = f["data"][demo_name]
        
        # Load raw image [H, W, 3] uint8 (0-255)
        obs = np.array(demo["obs"]["agentview_rgb"][timestep])
        
        # TODO: Verify if rotation is needed - use inspect_dataset.py to check
        # Evaluation pipeline applies: img = img[::-1, ::-1] in get_libero_image()
        # We need to verify if dataset images are already rotated or need rotation
        
        # Apply SAME transform as CNNRolloutWorker
        pixel_values = self.transform(obs)  # [3, H, W] float32, ImageNet normalized
        
        # Load actions [num_action_chunks, action_dim]
        actions = np.array(
            demo["actions"][timestep : timestep + self.num_action_chunks]
        )
        actions = torch.from_numpy(actions).float()  # [num_action_chunks, action_dim]
        
        # Get task ID
        task_id = torch.tensor(self.task_id_map[task_desc], dtype=torch.long)
        
        return {
            "pixel_values": pixel_values,
            "actions": actions,  # Raw continuous actions (will be tokenized in training loop)
            "task_id": task_id,
            "task_description": task_desc,
        }
    
    def __del__(self):
        for fh in self.file_handles.values():
            fh.close()


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    task_id_map: dict,
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device)  # [B, C, H, W]
        actions = batch["actions"].to(device)  # [B, num_action_chunks, action_dim]
        
        # Extract task IDs from batch (provided by SimpleCNNDataset)
        task_ids = batch["task_id"].to(device) if "task_id" in batch else None
        
        # Compute bin indices directly from actions
        with torch.no_grad():
            target_bin_indices = model.compute_bin_indices_from_actions(actions)  # [B, num_action_chunks, action_dim]
        
        # Forward pass
        output = model(
            pixel_values=pixel_values,
            task_ids=task_ids,
            return_logprobs=False,
            return_values=False,
        )
        action_logits = output["action_logits"]  # [B, num_action_chunks, action_dim, n_action_bins]
        
        # Target bin indices are already in [0, n_action_bins-1]
        target_indices = target_bin_indices  # [B, num_action_chunks, action_dim]
        
        # Reshape for cross-entropy: [B, num_action_chunks, action_dim, n_action_bins] -> [B*num_action_chunks*action_dim, n_action_bins]
        # Target: [B, num_action_chunks, action_dim] -> [B*num_action_chunks*action_dim]
        logits_flat = action_logits.view(-1, action_logits.shape[-1])  # [B*num_action_chunks*action_dim, n_action_bins]
        targets_flat = target_indices.view(-1)  # [B*num_action_chunks*action_dim]
        
        # Compute loss (cross-entropy on action tokens)
        loss = criterion(logits_flat, targets_flat)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({"loss": loss.item()})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return {"train_loss": avg_loss}


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    task_id_map: dict,
) -> dict:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            pixel_values = batch["pixel_values"].to(device)
            actions = batch["actions"].to(device)  # [B, num_action_chunks, action_dim]
            
            # Get task IDs directly from batch (dataset provides task_id)
            task_ids = batch["task_id"].to(device) if "task_id" in batch else None
            
            # Compute bin indices directly from actions
            with torch.no_grad():
                target_bin_indices = model.compute_bin_indices_from_actions(actions)  # [B, num_action_chunks, action_dim]
            
            # Forward pass
            output = model(
                pixel_values=pixel_values,
                task_ids=task_ids,
                return_logprobs=False,
                return_values=False,
            )
            action_logits = output["action_logits"]  # [B, num_action_chunks, action_dim, n_action_bins]
            
            # Target bin indices are already in [0, n_action_bins-1]
            target_indices = target_bin_indices  # [B, num_action_chunks, action_dim]
            
            # Reshape for cross-entropy
            logits_flat = action_logits.view(-1, action_logits.shape[-1])
            targets_flat = target_indices.view(-1)
            
            # Compute loss
            loss = criterion(logits_flat, targets_flat)
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return {"val_loss": avg_loss}


def main():
    parser = argparse.ArgumentParser(description="Train simple CNN policy")
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
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--action_dim",
        type=int,
        default=7,
        help="Action dimension (default: 7 for LIBERO)",
    )
    parser.add_argument(
        "--num_action_chunks",
        type=int,
        default=8,
        help="Number of action chunks (default: 8)",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Image size (default: 224)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay (default: 1e-5)",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=5,
        help="Save checkpoint every N epochs (default: 5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loader workers (default: 4)",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1)",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=32000,
        help="Vocabulary size for action tokens (default: 32000)",
    )
    parser.add_argument(
        "--n_action_bins",
        type=int,
        default=256,
        help="Number of action bins (default: 256)",
    )
    parser.add_argument(
        "--unnorm_key",
        type=str,
        default="libero_spatial_no_noops",
        help="Key for action normalization stats (default: libero_spatial_no_noops)",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Use args.data_dir directly if it exists, otherwise try to construct path
    if os.path.exists(args.data_dir):
        data_dir_for_stats = args.data_dir
        print(f"Using provided data directory: {data_dir_for_stats}")
    else:
        # Fallback: try to construct path (for backward compatibility)
        if "datasets" in args.data_dir or "datasets_with_logits" in args.data_dir:
            # args.data_dir is something like LIBERO/libero/datasets/... or LIBERO/libero/datasets_with_logits/...
            # Go up to LIBERO/libero, then up to LIBERO
            libero_root = os.path.dirname(os.path.dirname(os.path.dirname(args.data_dir)))
        else:
            # args.data_dir might already be LIBERO or LIBERO/libero
            libero_root = os.path.dirname(args.data_dir) if os.path.basename(args.data_dir) == "libero" else args.data_dir
        
        # Compute action statistics (needed for action normalization, matching OpenVLA)
        # Use the same path structure as datasets_with_logits/libero_spatial_simplevla_trajall
        data_dir_for_stats = os.path.join(libero_root, "libero", "datasets_with_logits", "libero_spatial_simplevla_trajall")
        if not os.path.exists(data_dir_for_stats):
            data_dir_for_stats = args.data_dir
        print(f"Using constructed data directory: {data_dir_for_stats}")
    
    print(f"Computing action statistics from {data_dir_for_stats}...")
    norm_stats = compute_action_statistics(data_dir_for_stats, unnorm_key=args.unnorm_key)
    
    # Load dataset using SimpleCNNDataset (no OpenVLA dependency)
    print(f"Loading dataset from {data_dir_for_stats}...")
    base_dataset = SimpleCNNDataset(
        root_dir=data_dir_for_stats,
        image_size=args.image_size,
        num_action_chunks=args.num_action_chunks,
        demos_per_task=0,  # 0 means use all available demos
        rank=0,
        world_size=1,
    )
    
    task_id_map = base_dataset.task_id_map
    num_tasks = len(task_id_map)
    num_trajectories = len(base_dataset.trajectories)
    print(f"Found {num_trajectories} trajectories")
    print(f"Found {len(base_dataset)} samples across {num_tasks} tasks")
    print(f"Task ID map: {task_id_map}")
    
    full_dataset = base_dataset
    
    # Create model
    print("Creating model...")
    model = SimpleCNNPolicy(
        action_dim=args.action_dim,
        num_action_chunks=args.num_action_chunks,
        image_size=args.image_size,
        num_tasks=num_tasks,
        use_task_embedding=True,
        vocab_size=args.vocab_size,
        n_action_bins=args.n_action_bins,
        norm_stats=norm_stats,
        unnorm_key=args.unnorm_key,
    ).to(device)
    
    # Split into train/val
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,} total, {trainable_params:,} trainable")
    
    # Loss and optimizer (cross-entropy for action tokens)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    # Training loop
    print("Starting training...")
    best_val_loss = float("inf")
    
    for epoch in range(1, args.num_epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, task_id_map
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, task_id_map)
        
        # Print metrics
        print(
            f"Epoch {epoch}/{args.num_epochs}: "
            f"Train Loss: {train_metrics['train_loss']:.6f}, "
            f"Val Loss: {val_metrics['val_loss']:.6f}"
        )
        
        # Save checkpoint
        if epoch % args.save_interval == 0 or val_metrics["val_loss"] < best_val_loss:
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                checkpoint_path = os.path.join(args.output_dir, "best_checkpoint.pt")
            else:
                checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_metrics["train_loss"],
                    "val_loss": val_metrics["val_loss"],
                    "task_id_map": task_id_map,
                    "num_tasks": num_tasks,
                    "num_action_chunks": args.num_action_chunks,
                    "action_dim": args.action_dim,
                    "norm_stats": norm_stats,
                    "unnorm_key": args.unnorm_key,
                    "vocab_size": args.vocab_size,
                    "n_action_bins": args.n_action_bins,
                },
                checkpoint_path,
            )
            print(f"Saved checkpoint to {checkpoint_path}")
    
    print("Training complete!")


if __name__ == "__main__":
    main()
