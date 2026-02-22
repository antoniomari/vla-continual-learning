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
Simple CNN Policy for robotic manipulation tasks.

This is a lightweight policy that:
- Takes RGB images and task IDs/embeddings as input
- Outputs discrete action tokens (compatible with OpenVLA format)
- Designed for supervised learning and later RL (GRPO) finetuning
- No model sharding (single GPU or data parallel only)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any
from prismatic.vla.constants import (
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    NormalizationType,
)


class SimpleCNNPolicy(nn.Module):
    """
    Simple CNN-based policy for robotic manipulation.
    
    Architecture:
    - ResNet-like CNN backbone for image encoding
    - Task embedding layer (for task IDs or descriptions)
    - MLP head to predict actions
    - Outputs actions and logprobs (for GRPO compatibility)
    """
    
    def __init__(
        self,
        action_dim: int = 7,
        num_action_chunks: int = 1,
        image_size: int = 224,
        num_tasks: Optional[int] = None,
        task_embed_dim: int = 128,
        hidden_dim: int = 512,
        use_task_embedding: bool = True,
        vocab_size: int = 32000,
        n_action_bins: int = 256,
        norm_stats: Optional[Dict[str, Any]] = None,
        unnorm_key: Optional[str] = None,
    ):
        """
        Args:
            action_dim: Dimension of action space (default 7 for LIBERO)
            num_action_chunks: Number of action chunks to predict (default 1)
            image_size: Input image size (assumed square)
            num_tasks: Number of tasks (for task ID embedding). If None, uses task descriptions.
            task_embed_dim: Dimension of task embedding
            hidden_dim: Hidden dimension for MLP
            use_task_embedding: Whether to use task embeddings
            vocab_size: Vocabulary size (kept for backward compatibility, not used - we use bin indices 0-255 directly)
            n_action_bins: Number of action bins for discretization (default 256)
            norm_stats: Action normalization statistics (dict with keys like "libero_spatial_no_noops")
            unnorm_key: Key to select normalization stats (e.g., "libero_spatial_no_noops")
        """
        super().__init__()
        
        self.action_dim = action_dim
        self.num_action_chunks = num_action_chunks
        self.image_size = image_size
        self.use_task_embedding = use_task_embedding
        self.vocab_size = vocab_size  # Kept for backward compatibility, not used
        self.n_action_bins = n_action_bins
        
        # Action discretization setup - we use bin indices 0-255 directly (no vocab_size mapping)
        self.bins = np.linspace(-1, 1, n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0  # [n_action_bins - 1]
        
        # Action normalization stats
        self.norm_stats = norm_stats or {}
        self.unnorm_key = unnorm_key
        
        # Dummy config for compatibility
        self.config = type('Config', (object,), {'n_action_bins': n_action_bins})()
        
        # Image encoder (simple ResNet-like CNN)
        self.image_encoder = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # ResNet-like blocks
            self._make_layer(64, 128, num_blocks=2, stride=1),
            self._make_layer(128, 256, num_blocks=2, stride=2),
            self._make_layer(256, 512, num_blocks=2, stride=2),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        
        # Compute image feature dimension
        # After conv layers: image_size / 8 (due to 3 stride-2 operations)
        # With adaptive pooling, we get 512 features
        image_feat_dim = 512
        
        # Task embedding
        if use_task_embedding:
            if num_tasks is not None:
                # Task ID embedding
                self.task_embedding = nn.Embedding(num_tasks, task_embed_dim)
            else:
                # Task description embedding (simple MLP)
                # We'll use a simple hash-based approach or learnable projection
                self.task_embedding = nn.Linear(256, task_embed_dim)  # Assume 256-dim text features
            task_input_dim = task_embed_dim
        else:
            task_input_dim = 0
        
        # Action prediction head - outputs logits over n_action_bins (0-255)
        # Shape: [B, num_action_chunks * action_dim, n_action_bins]
        # We use bin indices directly, no vocab_size mapping needed
        total_feat_dim = image_feat_dim + task_input_dim
        self.action_head = nn.Sequential(
            nn.Linear(total_feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim * num_action_chunks * n_action_bins),
        )
        
        # Value head (for GRPO compatibility, optional)
        self.value_head = nn.Sequential(
            nn.Linear(total_feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
    
    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int):
        """Create a ResNet-like block."""
        layers = []
        layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        )
        
        for _ in range(1, num_blocks):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )
        
        return nn.Sequential(*layers)
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        task_ids: Optional[torch.Tensor] = None,
        task_embeddings: Optional[torch.Tensor] = None,
        action_tokens: Optional[torch.Tensor] = None,
        return_logprobs: bool = True,
        return_values: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.  Handles both inference and training.
        
        When action_tokens is None  → **inference** mode:
            returns actions, bin_indices, action_logits, (logprobs), (values)
        When action_tokens is given  → **training** mode:
            returns logprobs and entropy for the given action_tokens,
            shaped for actor_loss compatibility.
        
        IMPORTANT: This is the single entry point that FSDP's forward() calls,
        which ensures backward hooks for gradient synchronization are set up.
        
        Args:
            pixel_values: Image tensor [B, C, H, W] or [B, num_images, C, H, W]
            task_ids: Task IDs [B] (if using task ID embedding)
            task_embeddings: Task embeddings [B, task_embed_dim]
            action_tokens: [B, n_chunks * action_dim] bin indices for training
            return_logprobs: Whether to return logprobs (inference mode)
            return_values: Whether to return values (inference mode)
        """
        batch_size = pixel_values.shape[0]
        
        # ── Shared feature extraction ─────────────────────────────────
        # Handle multi-image input (flatten if needed)
        if pixel_values.dim() == 5:
            pixel_values = pixel_values.view(-1, *pixel_values.shape[2:])
            image_features = self.image_encoder(pixel_values)
            num_images = pixel_values.shape[0] // batch_size
            image_features = image_features.view(batch_size, num_images, -1).mean(dim=1)
        else:
            image_features = self.image_encoder(pixel_values)  # [B, 512]
        
        # Task embedding
        if self.use_task_embedding:
            if task_ids is not None:
                task_emb = self.task_embedding(task_ids)
            elif task_embeddings is not None:
                if isinstance(self.task_embedding, nn.Linear):
                    task_emb = self.task_embedding(task_embeddings)
                else:
                    task_emb = task_embeddings
            else:
                device = image_features.device
                task_emb = torch.zeros(batch_size, 128, device=device)
            features = torch.cat([image_features, task_emb], dim=1)
        else:
            features = image_features
        
        # ── Training mode (action_tokens provided) ────────────────────
        if action_tokens is not None:
            action_token_len = self.num_action_chunks * self.action_dim
            action_logits_flat = self.action_head(features)
            action_logits = action_logits_flat.view(
                batch_size, action_token_len, self.n_action_bins
            )
            log_probs = F.log_softmax(action_logits, dim=-1)
            action_tokens_clamped = action_tokens.long().clamp(0, self.n_action_bins - 1)
            logprobs = log_probs.gather(
                2, action_tokens_clamped.unsqueeze(-1)
            ).squeeze(-1)   # [B, action_token_len]
            probs = F.softmax(action_logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1)  # [B, action_token_len]
            return {"logprobs": logprobs, "entropy": entropy}
        
        # ── Inference mode (no action_tokens) ─────────────────────────
        action_logits_flat = self.action_head(features)
        action_logits = action_logits_flat.view(
            batch_size, self.num_action_chunks, self.action_dim, self.n_action_bins
        )
        bin_indices = action_logits.argmax(dim=-1)
        actions = self._decode_bin_indices(bin_indices)
        
        logprobs = None
        if return_logprobs:
            logits_flat = action_logits.view(-1, self.n_action_bins)
            log_probs_flat = F.log_softmax(logits_flat, dim=-1)
            indices_flat = bin_indices.view(-1)
            logprobs_flat = log_probs_flat.gather(1, indices_flat.unsqueeze(1)).squeeze(1)
            logprobs = logprobs_flat.view(batch_size, self.num_action_chunks, self.action_dim)
        
        values = None
        if return_values:
            values = self.value_head(features)
        
        output = {
            "actions": actions,
            "bin_indices": bin_indices,
            "action_logits": action_logits,
        }
        if logprobs is not None:
            output["logprobs"] = logprobs
        if values is not None:
            output["values"] = values
        return output
    
    def compute_bin_indices_from_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Compute bin indices directly from continuous actions.
        
        Args:
            actions: [B, num_action_chunks, action_dim] continuous actions
        
        Returns:
            bin_indices: [B, num_action_chunks, action_dim] bin indices in [0, n_action_bins-1]
        """
        B, T, D = actions.shape
        assert D == self.action_dim
        
        # Normalize actions to [-1, 1]
        normalized_actions = self._normalize_actions_for_binning(actions)
        normalized_actions = normalized_actions.reshape(-1, D)
        bin_centers = self.bin_centers
        
        # Find nearest bin for each action dimension
        discretized_actions = []
        for dim in range(D):
            vals = normalized_actions[:, dim][:, None]  # (B*T, 1)
            dists = np.abs(vals - bin_centers[None, :])  # (B*T, n_bins)
            nearest_bins = np.argmin(dists, axis=1)  # (B*T,)
            discretized_actions.append(nearest_bins)
        
        bin_indices = np.stack(discretized_actions, axis=1)  # (B*T, D)
        bin_indices = bin_indices.reshape(B, T, D)
        
        return torch.from_numpy(bin_indices).long().to(actions.device)
    
    def _normalize_actions_for_binning(self, actions: torch.Tensor) -> np.ndarray:
        """Normalize actions to [-1, 1] for binning (same as _normalize_actions but returns numpy)."""
        from rlinf.models.embodiment.model_utils import _normalize_actions
        return _normalize_actions(self, actions, norm_key=self.unnorm_key)
    
    def _decode_bin_indices(self, bin_indices: torch.Tensor) -> torch.Tensor:
        """
        Decode bin indices to continuous actions.
        
        Args:
            bin_indices: [B, num_action_chunks, action_dim] bin indices in [0, n_action_bins-1]
        
        Returns:
            actions: [B, num_action_chunks, action_dim] continuous actions
        """
        # Convert to numpy for processing
        bin_indices_np = bin_indices.cpu().numpy()
        batch_size, num_chunks, action_dim = bin_indices_np.shape
        
        # Reshape to [B*num_chunks*action_dim]
        bin_indices_flat = bin_indices_np.reshape(-1)
        
        # Clip to valid range
        bin_indices_flat = np.clip(
            bin_indices_flat, a_min=0, a_max=self.bin_centers.shape[0] - 1
        )
        
        # Map bin indices to normalized actions (bin centers)
        normalized_actions = np.array([
            self.bin_centers[da] for da in bin_indices_flat
        ])  # [B*num_chunks*action_dim]
        
        # Reshape back to [B, num_chunks, action_dim]
        normalized_actions = normalized_actions.reshape(batch_size, num_chunks, action_dim)
        
        # Unnormalize to original action space
        actions = self._unnormalize_actions(normalized_actions, self.unnorm_key)
        
        return torch.from_numpy(actions).to(bin_indices.device).float()
    
    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict[str, Any]:
        """Get action normalization statistics."""
        if unnorm_key is None:
            unnorm_key = self.unnorm_key
        
        if unnorm_key is None:
            if len(self.norm_stats) == 0:
                raise ValueError("No normalization statistics available!")
            unnorm_key = next(iter(self.norm_stats.keys()))
        
        if unnorm_key not in self.norm_stats:
            # Try with _no_noops suffix
            alt_key = f"{unnorm_key}_no_noops"
            if alt_key in self.norm_stats:
                return self.norm_stats[alt_key]["action"]
            raise ValueError(f"Normalization key '{unnorm_key}' not found in norm_stats!")
        
        return self.norm_stats[unnorm_key]["action"]
    
    def _unnormalize_actions(self, normalized_actions: np.ndarray, unnorm_key: Optional[str] = None) -> np.ndarray:
        """
        Unnormalize actions from [-1, 1] to original action space.
        
        Args:
            normalized_actions: [B, num_chunks, action_dim] normalized actions in [-1, 1]
            unnorm_key: Key to select normalization stats
        
        Returns:
            actions: [B, num_chunks, action_dim] unnormalized actions
        """
        if len(self.norm_stats) == 0:
            raise ValueError(
                "norm_stats is empty! Cannot unnormalize actions. "
                "Make sure norm_stats are loaded when creating the model. "
                f"Current unnorm_key: {unnorm_key}, self.unnorm_key: {self.unnorm_key}"
            )
        
        action_norm_stats = self.get_action_stats(unnorm_key)
        
        if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
            mask = action_norm_stats.get(
                "mask", np.ones_like(action_norm_stats["min"], dtype=bool)
            )
            action_high, action_low = (
                np.array(action_norm_stats["max"]),
                np.array(action_norm_stats["min"]),
            )
        elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
            mask = action_norm_stats.get(
                "mask", np.ones_like(action_norm_stats["q01"], dtype=bool)
            )
            action_high, action_low = (
                np.array(action_norm_stats["q99"]),
                np.array(action_norm_stats["q01"]),
            )
        else:
            raise ValueError("Unsupported action/proprio normalization type detected!")
        
        action_dim = normalized_actions.shape[-1]
        repeat_factor = action_dim // action_high.shape[0]
        action_high = action_high.repeat(repeat_factor)
        action_low = action_low.repeat(repeat_factor)
        mask = np.tile(mask, repeat_factor)
        
        # Reshape for broadcasting
        normalized_flat = normalized_actions.reshape(-1, action_dim)
        
        actions_flat = np.where(
            mask,
            0.5 * (normalized_flat + 1) * (action_high - action_low + 1e-8) + action_low,
            normalized_flat,
        )
        
        return actions_flat.reshape(normalized_actions.shape)
    
    def preprocess_for_train(self, data):
        """Preprocess rollout batch data for training.
        
        Reshapes action_tokens from [B, n_chunks, action_dim] to [B, n_chunks * action_dim]
        to match the VLA convention used by actor_loss.
        """
        for key in ["action_tokens"]:
            value = data[key]
            data[key] = value.reshape(
                value.shape[0],
                self.action_dim * self.num_action_chunks,
                *value.shape[3:],
            )
        return data

    def train_forward(
        self,
        pixel_values: torch.Tensor,
        task_ids: torch.Tensor,
        action_tokens: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Backward-compatible thin wrapper.  Prefer calling forward() directly
        (via self.model(...)) so FSDP hooks are triggered correctly."""
        return self.forward(
            pixel_values=pixel_values,
            task_ids=task_ids,
            action_tokens=action_tokens,
        )

    def predict_action_batch(
        self,
        pixel_values: torch.Tensor,
        task_ids: Optional[torch.Tensor] = None,
        task_embeddings: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict actions in batch (compatible with rollout worker interface).
        
        Returns:
            actions: [B, num_action_chunks, action_dim] continuous actions
            action_tokens: [B, num_action_chunks, action_dim] bin indices (0-255)
            action_logits: [B, num_action_chunks, action_dim, n_action_bins] logits
            last_hidden_state: None
        """
        with torch.no_grad():
            output = self.forward(
                pixel_values=pixel_values,
                task_ids=task_ids,
                task_embeddings=task_embeddings,
                return_logprobs=True,
                return_values=False,
            )
        
        actions = output["actions"]
        action_tokens = output["bin_indices"]  # bin indices [0, n_action_bins-1]
        action_logits = output["action_logits"]
        
        return actions, action_tokens, action_logits, None
