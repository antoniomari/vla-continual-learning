"""
Gradient-based diagnostics for continual learning in RLinf.

This module provides small, self-contained utilities to study:

1. Gradient alignment between tasks
   - Compute average gradients for different task datasets
   - Measure cosine similarity between those gradient vectors
   - This tells you how much *conflict* there is between tasks in parameter space.

2. Fisher-style parameter importance (especially for LoRA)
   - Wraps existing EWC utilities to estimate Fisher information for LoRA parameters
   - Helps you see which LoRA weights are most important for a given task.

The code is written to be:
 - Model-agnostic where possible (works for both SimpleCNN and LoRA models)
 - Easy to adapt: you can plug in your own dataloaders and loss functions
 - Interpretable: each function has a short explanation of what its output means

This file does NOT assume any particular training loop. Instead, you:
 - Load a model from a checkpoint path
 - Build one DataLoader per task (or a saved rollout batch for LoRA)
 - Call the utilities below to compute/visualize diagnostics.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rlinf.algorithms.ewc import (
    get_lora_parameters,
    compute_fisher_information_from_rollout,
)
from rlinf.models.simple_cnn_policy import SimpleCNNPolicy


TensorDict = Dict[str, torch.Tensor]


# ---------------------------------------------------------------------------
# Generic helpers for working with parameter dictionaries
# ---------------------------------------------------------------------------

def _flatten_param_dict(params: TensorDict) -> torch.Tensor:
    """
    Flatten a dict of parameter tensors into a single 1D vector.

    This is used both for:
      - Gradient vectors (per-task average gradient)
      - Fisher vectors (per-parameter importance)

    Args:
        params: Mapping from parameter name to tensor.

    Returns:
        1D tensor containing all parameters concatenated.
    """
    if not params:
        raise ValueError("Received empty parameter dictionary to flatten.")

    flat_tensors: List[torch.Tensor] = []
    for name, t in params.items():
        if t is None:
            continue
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"Expected Tensor for param '{name}', got {type(t)}")
        flat_tensors.append(t.reshape(-1))

    return torch.cat(flat_tensors, dim=0)


def _cosine_similarity(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    """
    Compute cosine similarity between two 1D tensors on CPU.

    Interpretation:
      - +1.0  → gradients fully aligned (tasks want very similar updates)
      - 0.0   → orthogonal (no systematic agreement/conflict)
      - -1.0  → perfectly opposed (strong gradient conflict)
    """
    if vec_a.numel() != vec_b.numel():
        raise ValueError(
            f"Vector size mismatch: {vec_a.numel()} vs {vec_b.numel()}. "
            "Make sure you use the same parameter set for both tasks."
        )
    a = vec_a.float().cpu()
    b = vec_b.float().cpu()
    a_norm = a.norm(p=2)
    b_norm = b.norm(p=2)
    if a_norm.item() == 0.0 or b_norm.item() == 0.0:
        return 0.0
    return float(torch.dot(a, b) / (a_norm * b_norm))


def _select_parameters(
    model: nn.Module,
    mode: str = "all",
    name_filter: Optional[Iterable[str]] = None,
) -> TensorDict:
    """
    Select a subset of model parameters to analyze.

    Args:
        model: PyTorch module (may be wrapped in DDP/FSDP).
        mode:
            - "all":     all trainable parameters
            - "lora":    only parameters whose names contain 'lora' (for LoRA)
            - "custom":  user-provided name filter (list of substrings)
        name_filter:
            When mode == "custom", keep parameters whose name contains ANY
            of the substrings in name_filter.

    Returns:
        Dict mapping parameter name → tensor (live tensors, not cloned).

    Notes:
        For LoRA models, you typically want mode="lora" to focus on the
        adapter space rather than the full base model.
    """
    if hasattr(model, "module"):
        model_to_check = model.module
    else:
        model_to_check = model

    params: TensorDict = {}
    for name, p in model_to_check.named_parameters():
        if not p.requires_grad:
            continue

        if mode == "all":
            params[name] = p
        elif mode == "lora":
            if "lora" in name.lower():
                params[name] = p
        elif mode == "custom":
            if name_filter is None:
                raise ValueError("name_filter must be provided when mode='custom'")
            if any(sub in name for sub in name_filter):
                params[name] = p
        else:
            raise ValueError(f"Unknown parameter selection mode: {mode}")

    if not params:
        raise ValueError(
            f"No parameters selected with mode={mode}. "
            "Check your model and selection criteria."
        )
    return params


# ---------------------------------------------------------------------------
# 1. Gradient alignment diagnostics
# ---------------------------------------------------------------------------

def compute_grpo_gradient_for_task(
    model: nn.Module,
    rollout_data: List[Dict],
    param_mode: str = "all",
    device: str = "cuda",
    group_size: int = 10,
    clip_ratio_low: float = 0.2,
    clip_ratio_high: float = 0.2,
    entropy_bonus: float = 0.0,
) -> TensorDict:
    """
    Compute average gradient using GRPO loss based on success/failure.
    
    This function:
      1. Groups episodes by task
      2. Computes episode-level rewards (success = 1.0, failure = 0.0)
      3. Computes GRPO advantages (group-relative normalization)
      4. Computes policy gradient loss: -log_prob * advantage (with PPO clipping)
      5. Returns average gradients
    
    Args:
        model: PyTorch model.
        rollout_data: List of rollout samples, each containing:
            - pixel_values: [C, H, W]
            - task_id: int
            - actions: [num_chunks, action_dim]
            - logprobs: [num_chunks, action_dim] or [action_dim] (old logprobs from rollout)
            - reward: float
            - done: bool
            - success: bool
        param_mode: Parameter selection mode ("all", "lora", "custom").
        device: Device to compute on.
        group_size: Number of episodes per group for GRPO advantage normalization.
        clip_ratio_low: Lower clipping ratio for PPO.
        clip_ratio_high: Upper clipping ratio for PPO.
        entropy_bonus: Entropy bonus coefficient.
    
    Returns:
        Dict mapping parameter name → *average gradient tensor* (on CPU).
    """
    model.to(device)
    model.eval()  # Start in eval mode, will switch to train() when needed for gradient computation
    
    # Select parameters
    param_tensors = _select_parameters(model, mode=param_mode)
    grad_sums: TensorDict = {
        name: torch.zeros_like(p, device="cpu") for name, p in param_tensors.items()
    }
    
    # Group samples by episode using episode_idx (more reliable than done flags)
    # This ensures we get exactly num_episodes episodes, even if some timed out
    episodes_dict = {}
    for sample in rollout_data:
        episode_idx = sample.get("episode_idx", None)
        if episode_idx is None:
            # Fallback: use done flags if episode_idx not available (for backward compatibility)
            # This shouldn't happen with new data, but handle it gracefully
            if not hasattr(compute_grpo_gradient_for_task, '_warned_no_episode_idx'):
                import warnings
                warnings.warn("Rollout data missing episode_idx, falling back to done flags. "
                            "This may cause incorrect episode grouping.")
                compute_grpo_gradient_for_task._warned_no_episode_idx = True
            # Use done flags as fallback
            if not hasattr(compute_grpo_gradient_for_task, '_current_episode'):
                compute_grpo_gradient_for_task._current_episode = []
            compute_grpo_gradient_for_task._current_episode.append(sample)
            if sample.get("done", False):
                episodes_dict[len(episodes_dict)] = compute_grpo_gradient_for_task._current_episode
                compute_grpo_gradient_for_task._current_episode = []
        else:
            # Use episode_idx to group samples
            if episode_idx not in episodes_dict:
                episodes_dict[episode_idx] = []
            episodes_dict[episode_idx].append(sample)
    
    # Convert to list sorted by episode index
    episodes = [episodes_dict[i] for i in sorted(episodes_dict.keys())]
    
    # Handle any remaining samples (shouldn't happen with episode_idx, but handle fallback case)
    if hasattr(compute_grpo_gradient_for_task, '_current_episode'):
        if compute_grpo_gradient_for_task._current_episode:
            episodes.append(compute_grpo_gradient_for_task._current_episode)
        delattr(compute_grpo_gradient_for_task, '_current_episode')
    
    if len(episodes) == 0:
        raise RuntimeError("No complete episodes found in rollout data.")
    
    # Compute episode rewards (success = 1.0, failure = 0.0)
    episode_rewards = []
    for episode in episodes:
        # Check if any step had success=True
        success = any(s.get("success", False) for s in episode)
        episode_reward = 1.0 if success else 0.0
        episode_rewards.append(episode_reward)
    
    episode_rewards = torch.tensor(episode_rewards, device=device)  # [num_episodes]
    
    # Compute GRPO advantages (group-relative normalization)
    num_episodes = len(episodes)
    num_groups = (num_episodes + group_size - 1) // group_size
    
    # Flatten all samples from all episodes and assign advantages
    all_samples_flat = []
    sample_advantages = []
    sample_old_logprobs = []
    
    for group_idx in range(num_groups):
        start_idx = group_idx * group_size
        end_idx = min(start_idx + group_size, num_episodes)
        group_episodes = episodes[start_idx:end_idx]
        group_rewards = episode_rewards[start_idx:end_idx]
        
        # Normalize advantages within group
        if len(group_rewards) > 1:
            mean_reward = group_rewards.mean()
            std_reward = group_rewards.std()
            if std_reward > 1e-6:
                group_advantages = (group_rewards - mean_reward) / (std_reward + 1e-6)
            else:
                group_advantages = group_rewards - mean_reward
        else:
            group_advantages = group_rewards - group_rewards.mean()
        
        # Assign advantages to all steps in each episode
        for ep_idx, episode in enumerate(group_episodes):
            advantage = group_advantages[ep_idx].item()
            for sample in episode:
                all_samples_flat.append(sample)
                sample_advantages.append(advantage)
                
                # Get old logprob (from rollout)
                old_logprob = sample.get("logprobs", None)
                if old_logprob is not None:
                    if isinstance(old_logprob, torch.Tensor):
                        old_logprob_val = old_logprob
                    else:
                        old_logprob_val = torch.tensor(old_logprob, dtype=torch.float32)
                    # Flatten if needed - average over chunks/dims
                    if old_logprob_val.dim() > 0:
                        old_logprob_val = old_logprob_val.mean() if old_logprob_val.numel() > 0 else torch.tensor(0.0, dtype=torch.float32)
                else:
                    old_logprob_val = torch.tensor(0.0, dtype=torch.float32)
                sample_old_logprobs.append(old_logprob_val)
    
    if len(all_samples_flat) == 0:
        raise RuntimeError("No valid samples for gradient computation.")
    
    # Batch process samples for efficiency
    batch_size = 32
    num_batches = 0
    
    for batch_start in range(0, len(all_samples_flat), batch_size):
        batch_end = min(batch_start + batch_size, len(all_samples_flat))
        batch_samples = all_samples_flat[batch_start:batch_end]
        
        # Collect batch data
        batch_pixel_values = []
        batch_task_ids = []
        batch_actions = []
        batch_advantages = []
        batch_old_logprobs = []
        
        for idx, sample in enumerate(batch_samples):
            global_idx = batch_start + idx
            batch_pixel_values.append(sample["pixel_values"])
            batch_task_ids.append(sample["task_id"])
            batch_actions.append(sample["actions"])
            batch_advantages.append(sample_advantages[global_idx])
            batch_old_logprobs.append(sample_old_logprobs[global_idx])
        
        # Stack into batches
        batch_pixel_values = torch.stack(batch_pixel_values).to(device)  # [B, C, H, W]
        batch_task_ids = torch.stack(batch_task_ids).to(device)  # [B]
        batch_advantages = torch.tensor(batch_advantages, device=device)  # [B]
        batch_old_logprobs = torch.stack(batch_old_logprobs).to(device)  # [B]
        
        # Forward pass to get current logprobs (need train mode for gradients)
        model.train()
        output = model(
            pixel_values=batch_pixel_values,
            task_ids=batch_task_ids,
            return_logprobs=True,
            return_values=False,
        )
        
        # Get logprobs for the actions taken
        action_logits = output["action_logits"]  # [B, num_chunks, action_dim, n_bins]
        log_probs = F.log_softmax(action_logits, dim=-1)  # [B, num_chunks, action_dim, n_bins]
        
        # Get bin indices from actions
        batch_actions_tensor = torch.stack(batch_actions).to(device)  # [B, num_chunks, action_dim]
        with torch.no_grad():
            target_bin_indices = model.compute_bin_indices_from_actions(batch_actions_tensor)
        
        # Gather logprobs for the taken actions
        target_indices = target_bin_indices.unsqueeze(-1)  # [B, num_chunks, action_dim, 1]
        current_logprobs = log_probs.gather(-1, target_indices).squeeze(-1)  # [B, num_chunks, action_dim]
        current_logprobs = current_logprobs.mean(dim=(1, 2))  # [B] - average over chunks and action dims
        
        # Compute GRPO loss
        logratio = current_logprobs - batch_old_logprobs
        ratio = torch.exp(logratio)
        
        # Clipped policy loss
        policy_loss = -batch_advantages * ratio
        policy_loss2 = -batch_advantages * torch.clamp(
            ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high
        )
        policy_loss = torch.max(policy_loss, policy_loss2).mean()
        
        # Add entropy bonus (only compute if needed)
        if entropy_bonus > 0:
            probs = F.softmax(action_logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1)  # [B, num_chunks, action_dim] after sum over n_bins
            entropy = entropy.mean(dim=(1, 2))  # [B] - average over chunks and action dims
            entropy_loss = entropy.mean()
            total_loss = policy_loss - entropy_bonus * entropy_loss
        else:
            total_loss = policy_loss
        
        # Backward
        model.zero_grad(set_to_none=True)
        total_loss.backward()
        
        # Accumulate gradients
        for name, p in param_tensors.items():
            if p.grad is None:
                continue
            grad_sums[name] += p.grad.detach().cpu()
        
        num_batches += 1
    
    if num_batches == 0:
        raise RuntimeError("No batches processed for gradient computation.")
    
    # Average gradients
    for name in grad_sums:
        grad_sums[name] /= float(num_batches)
    
    return grad_sums


def compute_task_gradient_matrix_grpo(
    model: nn.Module,
    rollout_data: List[Dict],
    param_mode: str = "all",
    device: str = "cuda",
    batch_size: int = 32,
    max_batches: Optional[int] = None,
) -> torch.Tensor:
    """
    Collect per-batch GRPO-style gradients for a single task and return them as a matrix.

    Each row of the returned matrix is a flattened gradient vector for one mini-batch.
    This is used to estimate a low-dimensional gradient subspace for the task.
    """
    model.to(device)
    model.eval()

    # Select parameters to analyze
    param_tensors = _select_parameters(model, mode=param_mode)

    if len(rollout_data) == 0:
        raise RuntimeError("No rollout samples provided for gradient subspace computation.")

    grad_rows: List[torch.Tensor] = []
    num_batches = 0

    for batch_start in range(0, len(rollout_data), batch_size):
        batch_end = min(batch_start + batch_size, len(rollout_data))
        batch_samples = rollout_data[batch_start:batch_end]

        batch_pixel_values = []
        batch_task_ids = []
        batch_actions = []
        batch_advantages = []
        batch_old_logprobs = []

        for s in batch_samples:
            batch_pixel_values.append(s["pixel_values"])
            batch_task_ids.append(s["task_id"])
            batch_actions.append(s["actions"])
            # Simple choice: uniform positive advantage for subspace estimation
            batch_advantages.append(1.0)

            old_logprob = s.get("logprobs", None)
            if old_logprob is not None:
                if isinstance(old_logprob, torch.Tensor):
                    old_logprob_val = old_logprob
                else:
                    old_logprob_val = torch.tensor(old_logprob, dtype=torch.float32)
                if old_logprob_val.dim() > 0 and old_logprob_val.numel() > 0:
                    old_logprob_val = old_logprob_val.mean()
            else:
                old_logprob_val = torch.tensor(0.0, dtype=torch.float32)
            batch_old_logprobs.append(old_logprob_val)

        batch_pixel_values = torch.stack(batch_pixel_values).to(device)  # [B, C, H, W]
        batch_task_ids = torch.stack(batch_task_ids).to(device)          # [B]
        batch_advantages = torch.tensor(batch_advantages, device=device) # [B]
        batch_old_logprobs = torch.stack(batch_old_logprobs).to(device)  # [B]
        batch_actions_tensor = torch.stack(batch_actions).to(device)     # [B, num_chunks, action_dim]

        # Forward pass (train mode for gradients)
        model.train()
        out = model(
            pixel_values=batch_pixel_values,
            task_ids=batch_task_ids,
            return_logprobs=True,
            return_values=False,
        )
        action_logits = out["action_logits"]                         # [B, num_chunks, action_dim, n_bins]
        log_probs_full = F.log_softmax(action_logits, dim=-1)        # same shape

        with torch.no_grad():
            target_bin_indices = model.compute_bin_indices_from_actions(batch_actions_tensor)

        target_indices = target_bin_indices.unsqueeze(-1)            # [B, num_chunks, action_dim, 1]
        taken_logprobs = log_probs_full.gather(-1, target_indices).squeeze(-1)  # [B, num_chunks, action_dim]
        taken_logprobs = taken_logprobs.mean(dim=(1, 2))             # [B]

        # Simple GRPO-like loss without clipping for subspace estimation
        logratio = taken_logprobs - batch_old_logprobs
        ratio = torch.exp(logratio)
        policy_loss = -batch_advantages * ratio
        loss = policy_loss.mean()

        model.zero_grad(set_to_none=True)
        loss.backward()

        # Flatten gradient
        grad_dict: TensorDict = {}
        for name, p in param_tensors.items():
            if p.grad is None:
                continue
            grad_dict[name] = p.grad.detach().cpu()
        flat_grad = _flatten_param_dict(grad_dict)  # [D]
        grad_rows.append(flat_grad)
        num_batches += 1

        if max_batches is not None and num_batches >= max_batches:
            break

    if len(grad_rows) == 0:
        raise RuntimeError("No gradient batches collected for subspace computation.")

    return torch.stack(grad_rows, dim=0)  # [num_batches, D]


def compute_subspace_basis(
    grad_matrix: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """
    Compute top-k principal components of grad_matrix.

    Args:
        grad_matrix: [num_batches, D]
        k: number of principal directions.

    Returns:
        U: [D, k_eff] orthonormal basis (columns are principal directions).
    """
    # Center across batches
    X = grad_matrix - grad_matrix.mean(dim=0, keepdim=True)
    X = X.float()

    # SVD: X = U S Vh, PCs are rows of Vh
    U_svd, S_svd, Vh = torch.linalg.svd(X, full_matrices=False)
    V = Vh.transpose(0, 1)  # [D, r]

    k_eff = min(k, V.shape[1])
    return V[:, :k_eff]


def subspace_overlap(
    U_i: torch.Tensor,
    U_j: torch.Tensor,
    normalize: bool = True,
) -> float:
    """
    Compute subspace overlap between two task bases U_i, U_j.

    Overlap(i,j) = || U_i^T U_j ||_F^2
    Optionally normalized by min(k_i, k_j) to be in [0, 1].
    """
    Ui = U_i.float()
    Uj = U_j.float()

    M = Ui.t() @ Uj  # [k_i, k_j]
    overlap = (M * M).sum().item()

    if normalize:
        k_min = min(Ui.shape[1], Uj.shape[1])
        if k_min > 0:
            overlap /= k_min
    return overlap


def compute_cnn_fisher_diagonal_for_task_from_rollouts(
    model: nn.Module,
    rollout_data: List[Dict],
    param_mode: str = "all",
    device: str = "cuda",
    batch_size: int = 32,
    max_batches: Optional[int] = None,
) -> TensorDict:
    """
    Approximate diagonal Fisher information for selected parameters using saved rollouts.

    We use the common diagonal Fisher approximation for policies:

        F_d ≈ E[(∂/∂θ_d log πθ(a|s))^2]

    Operationally, for each minibatch, we:
      - recompute log πθ(a|s) for the action actually taken in the rollout,
      - backprop the negative log-likelihood (-log πθ(a|s)),
      - accumulate squared gradients per-parameter.

    Notes:
      - This does NOT require a Hessian.
      - This is a *data-dependent* importance estimate: it is only “importance on the
        rollout distribution”.

    Returns:
        Dict mapping parameter name -> Fisher diagonal tensor (CPU).
    """
    model.to(device)
    model.eval()

    param_tensors = _select_parameters(model, mode=param_mode)
    fisher_sums: TensorDict = {
        name: torch.zeros_like(p, device="cpu") for name, p in param_tensors.items()
    }

    if len(rollout_data) == 0:
        raise RuntimeError("No rollout samples provided for Fisher computation.")

    num_batches = 0
    num_examples = 0

    for batch_start in range(0, len(rollout_data), batch_size):
        batch_end = min(batch_start + batch_size, len(rollout_data))
        batch_samples = rollout_data[batch_start:batch_end]

        batch_pixel_values = []
        batch_task_ids = []
        batch_actions = []

        for s in batch_samples:
            batch_pixel_values.append(s["pixel_values"])
            batch_task_ids.append(s["task_id"])
            batch_actions.append(s["actions"])

        batch_pixel_values = torch.stack(batch_pixel_values).to(device)  # [B, C, H, W]
        batch_task_ids = torch.stack(batch_task_ids).to(device)          # [B]
        batch_actions_tensor = torch.stack(batch_actions).to(device)     # [B, num_chunks, action_dim]

        model.train()
        out = model(
            pixel_values=batch_pixel_values,
            task_ids=batch_task_ids,
            return_logprobs=True,
            return_values=False,
        )
        action_logits = out["action_logits"]  # [B, num_chunks, action_dim, n_bins]
        log_probs_full = F.log_softmax(action_logits, dim=-1)

        with torch.no_grad():
            target_bin_indices = model.compute_bin_indices_from_actions(batch_actions_tensor)

        target_indices = target_bin_indices.unsqueeze(-1)  # [B, num_chunks, action_dim, 1]
        taken_logprobs = log_probs_full.gather(-1, target_indices).squeeze(-1)  # [B, num_chunks, action_dim]
        taken_logprobs = taken_logprobs.mean(dim=(1, 2))  # [B]

        nll = -taken_logprobs.mean()

        model.zero_grad(set_to_none=True)
        nll.backward()

        for name, p in param_tensors.items():
            if p.grad is None:
                continue
            g = p.grad.detach().cpu()
            fisher_sums[name] += g * g

        num_batches += 1
        num_examples += batch_pixel_values.shape[0]

        if max_batches is not None and num_batches >= max_batches:
            break

    # Average across minibatches (per-example scaling cancels in cosine-like metrics;
    # we still normalize by #batches for stability).
    for name in fisher_sums:
        fisher_sums[name] /= float(max(1, num_batches))

    return fisher_sums


def fisher_weighted_gradient_interference(
    grad_task: TensorDict,
    fisher_diag: TensorDict,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """
    Compute how much a gradient falls on Fisher-important directions.

    Given:
      - gradient g (dict of tensors)
      - diagonal fisher F (dict of tensors, same keys/shapes)

    We compute:
      raw = Σ_d F_d * g_d^2
      grad_energy = Σ_d g_d^2
      fisher_mass = Σ_d F_d
      frac_of_grad_energy = raw / (grad_energy + eps)
      avg_on_fisher = raw / (fisher_mass + eps)

    Returns a dict of scalar metrics.
    """
    if set(grad_task.keys()) != set(fisher_diag.keys()):
        raise ValueError("grad_task and fisher_diag must have identical parameter keys.")

    raw = 0.0
    grad_energy = 0.0
    fisher_mass = 0.0
    fisher_max = 0.0
    for k in grad_task.keys():
        g = grad_task[k].float()
        f = fisher_diag[k].float()
        raw += float((f * (g * g)).sum().item())
        grad_energy += float((g * g).sum().item())
        fisher_mass += float(f.sum().item())
        # Track maximum Fisher entry across all parameters (λ_max for diagonal F)
        if f.numel() > 0:
            local_max = float(f.max().item())
            if local_max > fisher_max:
                fisher_max = local_max

    # Rayleigh quotient g^T F g / g^T g lies in [λ_min, λ_max] for PSD F.
    # Normalizing by λ_max(F) makes this lie in [0, 1] and removes arbitrary scaling of F.
    frac_on_fisher = raw / (grad_energy + eps) if grad_energy > 0.0 else 0.0
    frac_on_fisher_normalized = (
        frac_on_fisher / (fisher_max + eps) if fisher_max > 0.0 else 0.0
    )

    return {
        "raw_fisher_weighted_grad_sq": raw,
        "grad_sq_sum": grad_energy,
        "fisher_sum": fisher_mass,
        "lambda_max_fisher": fisher_max,
        # Rayleigh quotient g^T F g / g^T g (scale depends on Fisher magnitude)
        "frac_of_grad_energy_on_fisher": frac_on_fisher,
        # Scale-invariant version in [0, 1], comparable across models:
        # (g^T F g / g^T g) / λ_max(F)
        "frac_of_grad_energy_on_fisher_normalized": frac_on_fisher_normalized,
        "avg_grad_sq_weighted_by_fisher": raw / (fisher_mass + eps),
    }
def compute_average_gradient_for_task(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    param_mode: str = "all",
    device: str = "cuda",
    max_batches: int = 1,
) -> TensorDict:
    """
    Compute the average gradient over (up to) `max_batches` from a dataloader.

    This is the core primitive for gradient alignment:
      - Run a few mini-batches from a specific *task dataset*
      - Compute gradients w.r.t. a chosen set of parameters
      - Average them to get a single gradient vector per task.

    Args:
        model: PyTorch model.
        dataloader: Yields batches for a single task.
        loss_fn: Loss function mapping (logits, targets) -> scalar loss.
                 For SimpleCNN, you can reuse nn.CrossEntropyLoss.
        param_mode: Parameter selection mode ("all", "lora", "custom").
                    See `_select_parameters` for details.
        device: Device to compute on.
        max_batches: Limit on number of batches to average over.
                     Even 1–5 batches is often enough for a useful signal.

    Returns:
        Dict mapping parameter name → *average gradient tensor* (on CPU).

    Interpretation:
        - Magnitude encodes how strongly this task pulls on each parameter.
        - Direction across tasks is what we use to measure alignment/conflict.
    """
    model.to(device)
    model.train()  # We want normal gradient flow

    # Select which parameters we care about
    param_tensors = _select_parameters(model, mode=param_mode)
    grad_sums: TensorDict = {
        name: torch.zeros_like(p, device="cpu") for name, p in param_tensors.items()
    }

    num_batches = 0
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break

        # You must adapt this part to your batch structure.
        # Here we handle SimpleCNN-style batches with:
        #   batch["pixel_values"], batch["actions"], batch["task_id"]
        if "pixel_values" in batch and "actions" in batch:
            pixel_values = batch["pixel_values"].to(device)
            actions = batch["actions"].to(device)
            task_ids = batch.get("task_id", None)
            if task_ids is not None:
                task_ids = task_ids.to(device)

            # For SimpleCNN, we treat action discretization as the target
            with torch.no_grad():
                target_bin_indices = model.compute_bin_indices_from_actions(actions)

            out = model(
                pixel_values=pixel_values,
                task_ids=task_ids,
                return_logprobs=False,
                return_values=False,
            )
            logits = out["action_logits"]  # [B, num_chunks, action_dim, n_bins]

            logits_flat = logits.view(-1, logits.shape[-1])
            targets_flat = target_bin_indices.view(-1)
            loss = loss_fn(logits_flat, targets_flat)
        else:
            raise ValueError(
                "compute_average_gradient_for_task currently expects "
                "SimpleCNN-style batches with 'pixel_values' and 'actions'. "
                "Adapt this section for other model types."
            )

        model.zero_grad(set_to_none=True)
        loss.backward()

        # Accumulate gradients
        for name, p in param_tensors.items():
            if p.grad is None:
                continue
            grad_sums[name] += p.grad.detach().cpu()

        num_batches += 1

    if num_batches == 0:
        raise RuntimeError("Dataloader yielded zero batches while computing gradients.")

    # Average gradients over batches
    for name in grad_sums:
        grad_sums[name] /= float(num_batches)

    return grad_sums


def gradient_cosine_similarity_between_tasks(
    grad_task_a: TensorDict,
    grad_task_b: TensorDict,
) -> float:
    """
    Compute cosine similarity between average gradients of two tasks.

    Args:
        grad_task_a: Dict name → gradient tensor for task A.
        grad_task_b: Dict name → gradient tensor for task B.

    Returns:
        Scalar cosine similarity in [-1, 1].

    Interpretation:
        - Values near +1 suggest the two tasks *help each other* (aligned gradients)
        - Values near  0 suggest *independent* updates
        - Values near -1 suggest strong *interference* (one task undoes the other)
    """
    # Ensure the same parameter set and ordering for both tasks
    if set(grad_task_a.keys()) != set(grad_task_b.keys()):
        raise ValueError(
            "Gradient dictionaries have different parameter keys. "
            "Make sure they were computed with the same model and parameter subset."
        )

    # Stack in consistent order
    ordered_names = sorted(grad_task_a.keys())
    vec_a = _flatten_param_dict({n: grad_task_a[n] for n in ordered_names})
    vec_b = _flatten_param_dict({n: grad_task_b[n] for n in ordered_names})
    return _cosine_similarity(vec_a, vec_b)


# ---------------------------------------------------------------------------
# 2. Fisher-style importance diagnostics for LoRA
# ---------------------------------------------------------------------------

def compute_lora_fisher_from_rollout(
    model: nn.Module,
    rollout_batch: TensorDict,
    num_samples: int = 128,
    device: str = "cuda",
) -> TensorDict:
    """
    Compute Fisher information for LoRA parameters using an existing rollout batch.

    This is a light wrapper around `rlinf.algorithms.ewc.compute_fisher_information_from_rollout`
    with clearer documentation for analysis use.

    Args:
        model: LoRA model (may be FSDP-wrapped).
        rollout_batch: Dict with keys used during RL training, typically:
            - "input_ids"
            - "attention_mask"
            - "pixel_values"
            - "action_tokens"
          (See `compute_fisher_information_from_rollout` for exact expectations.)
        num_samples: Number of samples from the batch to use.
        device: Device to compute on.

    Returns:
        Dict mapping LoRA parameter name → Fisher information tensor (on CPU).

    Interpretation:
        - Large values mean “this LoRA weight is important for this task”
        - Plotting the top-k entries gives you a sense of *where* capacity is used.
    """
    model.to(device)
    fisher_dict = compute_fisher_information_from_rollout(
        model=model,
        rollout_batch=rollout_batch,
        num_samples=num_samples,
        device=device,
    )
    return fisher_dict


def summarize_parameter_importance(
    importance_dict: TensorDict,
    top_k: int = 20,
) -> List[Tuple[str, float]]:
    """
    Summarize per-parameter importance (e.g., Fisher or gradient norm).

    Args:
        importance_dict: Dict name → tensor of importance scores
                         (e.g., Fisher values or squared gradients).
        top_k: How many parameters to keep in the summary.

    Returns:
        List of (name, scalar_importance) sorted by decreasing importance.

    Interpretation:
        - Use this to see which LoRA matrices (or layers) are most critical.
        - Often you'll see a small subset dominate, confirming capacity bottlenecks.
    """
    scores: List[Tuple[str, float]] = []
    for name, t in importance_dict.items():
        if not isinstance(t, torch.Tensor):
            continue
        # Aggregate importance over all elements (L2 norm is a reasonable choice)
        score = float(t.float().norm(p=2).item())
        scores.append((name, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def plot_parameter_importance_bar(
    summary: List[Tuple[str, float]],
    title: str,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 5),
) -> None:
    """
    Plot a simple bar chart for top-k parameter importances.

    Args:
        summary: Output of `summarize_parameter_importance`.
        title: Plot title.
        save_path: If provided, save figure to this path instead of showing.
        figsize: Figure size.

    Interpretation:
        - Each bar corresponds to a parameter tensor (e.g., a LoRA matrix).
        - Use name parsing (e.g., by layer index) to see which layers soak up
          most importance for a given task.
    """
    if not summary:
        raise ValueError("Empty summary: nothing to plot.")

    names, scores = zip(*summary)
    x = np.arange(len(names))

    plt.figure(figsize=figsize)
    plt.bar(x, scores)
    plt.xticks(x, [n.replace("lora_", "") for n in names], rotation=45, ha="right")
    plt.ylabel("Importance (L2 norm)")
    plt.title(title)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


# ---------------------------------------------------------------------------
# 3. Convenience loader for SimpleCNN checkpoints
# ---------------------------------------------------------------------------

def load_simple_cnn_from_checkpoint(
    checkpoint_path: str,
    device: str = "cuda",
) -> SimpleCNNPolicy:
    """
    Load a `SimpleCNNPolicy` from a training checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint saved by `simple_cnn_train.py`.
        device: Device to load the model on.

    Returns:
        An instance of `SimpleCNNPolicy` with weights loaded.

    Notes:
        - The training script saves all hyperparameters needed to rebuild the model.
        - This loader reconstructs the model with the same dims and norm_stats.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    action_dim = ckpt.get("action_dim", 7)
    num_action_chunks = ckpt.get("num_action_chunks", 8)
    norm_stats = ckpt.get("norm_stats", None)
    unnorm_key = ckpt.get("unnorm_key", None)
    vocab_size = ckpt.get("vocab_size", 32000)
    n_action_bins = ckpt.get("n_action_bins", 256)
    num_tasks = ckpt.get("num_tasks", None)

    model = SimpleCNNPolicy(
        action_dim=action_dim,
        num_action_chunks=num_action_chunks,
        num_tasks=num_tasks,
        vocab_size=vocab_size,
        n_action_bins=n_action_bins,
        norm_stats=norm_stats,
        unnorm_key=unnorm_key,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# 4. Minimal CLI demo (end-to-end, using the SimpleCNN base model)
# ---------------------------------------------------------------------------

class OnlineRolloutDataset(torch.utils.data.Dataset):
    """
    Dataset that collects rollout data by running the model in the LIBERO environment.
    
    This performs actual online rollouts (no synthetic data, no HDF5 files needed).
    """
    
    def __init__(
        self,
        model: nn.Module,
        task_id: int,
        task_id_map: Dict[str, int],
        num_episodes: int = 5,
        max_steps_per_episode: int = 100,
        image_size: int = 224,
        num_action_chunks: int = 8,
        seed: int = 42,
    ):
        """
        Args:
            model: Model to use for rollouts.
            task_id: Task ID to collect rollouts for.
            task_id_map: Mapping from task description to task ID.
            num_episodes: Number of episodes to collect.
            max_steps_per_episode: Maximum steps per episode.
            image_size: Image size for preprocessing.
            num_action_chunks: Number of action chunks.
            seed: Random seed.
        """
        self.model = model
        self.task_id = task_id
        self.task_id_map = task_id_map
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.image_size = image_size
        self.num_action_chunks = num_action_chunks
        self.seed = seed
        
        # Image preprocessing
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Collect rollouts
        self.samples = self._collect_rollouts()
    
    def _collect_rollouts(self):
        """Collect rollout data by running the model in the environment."""
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        
        # Get task suite
        task_suite_name = "libero_spatial"  # Default, can be made configurable
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[task_suite_name]()
        
        # Get task
        task = task_suite.get_task(self.task_id)
        task_description = task.language
        
        # Get task ID for model
        task_id_for_model = self.task_id_map.get(task_description, self.task_id)
        
        # Create environment
        bddl_file = os.path.join(
            get_libero_path("bddl_files"),
            task.problem_folder,
            task.bddl_file,
        )
        
        env_args = {
            "bddl_file_name": bddl_file,
            "camera_heights": 128,
            "camera_widths": 128,
        }
        
        device = next(self.model.parameters()).device
        self.model.eval()
        
        all_samples = []
        
        print(f"Collecting {self.num_episodes} episodes for task_id={self.task_id} ({task_description})...")
        
        for episode_idx in range(self.num_episodes):
            env = OffScreenRenderEnv(**env_args)
            env.seed(self.seed + episode_idx)
            
            # Reset environment (gymnasium returns (obs, info) tuple)
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                obs, info = reset_result
            else:
                obs = reset_result
                info = {}
            
            # Try to set initial state if available
            try:
                init_states_path = os.path.join(
                    get_libero_path("init_states"),
                    task.problem_folder,
                    task.init_states_file,
                )
                if os.path.exists(init_states_path):
                    init_states = torch.load(init_states_path, map_location="cpu")
                    if isinstance(init_states, torch.Tensor):
                        init_states = init_states.numpy()
                    if len(init_states) > 0:
                        state_idx = episode_idx % len(init_states)
                        env.set_init_state(init_states[state_idx])
                        reset_result = env.reset()
                        if isinstance(reset_result, tuple):
                            obs, info = reset_result
                        else:
                            obs = reset_result
                            info = {}
            except Exception as e:
                # If initial states fail, just continue with random reset
                pass
            
            episode_samples = []
            episode_idx_for_sample = episode_idx  # Track which episode this sample belongs to
            
            for step in range(self.max_steps_per_episode):
                # Extract image from observation using the same utility as the training code
                from rlinf.envs.libero.utils import get_libero_image
                
                image = None
                try:
                    # Try the standard get_libero_image function first
                    image = get_libero_image(obs)
                except (KeyError, TypeError) as e:
                    # Fallback: try different observation formats
                    if isinstance(obs, dict):
                        # Try common keys in order of likelihood
                        for key in ["agentview_image", "agentview_rgb", "image", "rgb"]:
                            if key in obs:
                                image = obs[key]
                                break
                        
                        # If still None, try nested structure
                        if image is None and "images_and_states" in obs:
                            img_dict = obs["images_and_states"]
                            if isinstance(img_dict, dict):
                                image = img_dict.get("full_image", None)
                            elif isinstance(img_dict, (list, tuple)) and len(img_dict) > 0:
                                # Might be a list/tuple of images
                                image = img_dict[0] if isinstance(img_dict[0], np.ndarray) else None
                    elif isinstance(obs, np.ndarray):
                        # Observation might be the image directly
                        image = obs
                    else:
                        # Last resort: print what we got for debugging
                        if step == 0 and episode_idx == 0:
                            print(f"  Debug: Unexpected observation type: {type(obs)}")
                            if isinstance(obs, dict):
                                print(f"  Debug: Observation keys: {list(obs.keys())}")
                
                if image is None:
                    if step == 0 and episode_idx == 0:
                        print(f"  Warning: Could not extract image from observation. Type: {type(obs)}")
                        if isinstance(obs, dict):
                            print(f"  Available keys: {list(obs.keys())}")
                    break
                
                # Convert to numpy if needed
                if torch.is_tensor(image):
                    image = image.cpu().numpy()
                
                # Ensure it's a numpy array
                if not isinstance(image, np.ndarray):
                    print(f"  Warning: Image is not numpy array (type: {type(image)}), skipping...")
                    break
                
                # Handle different image formats
                if len(image.shape) == 2:
                    # Grayscale, convert to RGB
                    image = np.stack([image] * 3, axis=-1)
                elif len(image.shape) == 3 and image.shape[2] == 4:
                    # RGBA, convert to RGB
                    image = image[:, :, :3]
                elif len(image.shape) == 3 and image.shape[2] != 3:
                    print(f"  Warning: Unexpected image shape {image.shape}, skipping...")
                    break
                
                # Ensure uint8 format
                if image.dtype != np.uint8:
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
                
                # Convert to tensor and preprocess
                pixel_values = self.transform(image).unsqueeze(0).to(device)  # [1, C, H, W]
                
                # Get task ID tensor
                task_id_tensor = torch.tensor(task_id_for_model, dtype=torch.long).unsqueeze(0).to(device)
                
                # Predict action (need logprobs for GRPO)
                with torch.no_grad():
                    output = self.model(
                        pixel_values=pixel_values,
                        task_ids=task_id_tensor,
                        return_logprobs=True,  # Need logprobs for GRPO
                        return_values=False,
                    )
                    actions = output["actions"][0]  # [num_chunks, action_dim]
                    logprobs = output.get("logprobs", None)  # [num_chunks, action_dim]
                    if logprobs is not None:
                        logprobs = logprobs[0]  # [action_dim] or [num_chunks, action_dim]
                
                # CRITICAL: Apply gripper transformation (same as evaluator)
                # This must match exactly what happens in simple_cnn_evaluator.py lines 286-291
                action_np = actions[0].cpu().numpy().copy()  # Take first chunk, convert to numpy
                action_np[-1] = 2 * action_np[-1] - 1  # Normalize [0,1] -> [-1,+1]
                action_np[-1] = np.sign(action_np[-1]) * -1.0  # Invert and binarize
                
                # Step environment (gymnasium returns (obs, reward, terminated, truncated, info))
                step_result = env.step(action_np)
                if len(step_result) == 5:
                    # gymnasium format: (obs, reward, terminated, truncated, info)
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                elif len(step_result) == 4:
                    # gym format: (obs, reward, done, info)
                    obs, reward, done, info = step_result
                else:
                    raise ValueError(f"Unexpected step result format: {len(step_result)} elements")
                
                # Success is determined by the done flag (same as evaluator)
                # In the evaluator, success = batch_dones[i], which is set by done flag
                success = bool(done)
                
                # Store sample with reward and success, and episode index
                episode_samples.append({
                    "pixel_values": pixel_values[0].cpu(),  # [C, H, W]
                    "task_id": torch.tensor(task_id_for_model, dtype=torch.long),
                    "actions": actions.cpu(),  # [num_chunks, action_dim]
                    "logprobs": logprobs.cpu() if logprobs is not None else None,
                    "reward": float(reward),
                    "done": bool(done),
                    "success": bool(success),
                    "episode_idx": episode_idx,  # Track which episode this belongs to
                    "step_in_episode": step,  # Track step number for debugging
                })
                
                if done:
                    # Episode completed successfully
                    break
            
            # If we hit max_steps without done=True, mark the last sample as episode end
            # and mark as failure (timeout)
            if len(episode_samples) > 0 and not episode_samples[-1].get("done", False):
                episode_samples[-1]["done"] = True
                episode_samples[-1]["success"] = False  # Timeout = failure
                episode_samples[-1]["timeout"] = True
            
            all_samples.extend(episode_samples)
            env.close()
            
            if (episode_idx + 1) % max(1, self.num_episodes // 5) == 0:
                print(f"  Collected {episode_idx + 1}/{self.num_episodes} episodes ({len(all_samples)} samples so far)")
        
        print(f"Collected {len(all_samples)} total samples from {self.num_episodes} episodes")
        return all_samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class SyntheticRolloutDataset(torch.utils.data.Dataset):
    """
    Synthetic dataset that generates rollout data on-the-fly using the model.
    
    This generates:
      - Random images (or fixed seed for reproducibility)
      - Actions from model's own predictions (or random actions)
    
    This allows gradient alignment analysis without needing external data or
    environment setup.
    """
    
    def __init__(
        self,
        model: nn.Module,
        task_id: int,
        num_samples: int = 100,
        batch_size: int = 32,
        image_size: int = 224,
        action_dim: int = 7,
        num_action_chunks: int = 8,
        seed: int = 42,
        use_model_predictions: bool = True,
    ):
        """
        Args:
            model: Model to use for generating actions (if use_model_predictions=True).
            task_id: Task ID to use for all samples.
            num_samples: Number of synthetic samples to generate.
            batch_size: Batch size (for compatibility with DataLoader).
            image_size: Image size (H=W).
            action_dim: Action dimension.
            num_action_chunks: Number of action chunks.
            seed: Random seed for reproducibility.
            use_model_predictions: If True, use model's predictions as targets.
                                  If False, use random actions.
        """
        self.model = model
        self.task_id = task_id
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.image_size = image_size
        self.action_dim = action_dim
        self.num_action_chunks = num_action_chunks
        self.seed = seed
        self.use_model_predictions = use_model_predictions
        
        # Set seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Pre-generate all samples
        self._generate_samples()
    
    def _generate_samples(self):
        """Pre-generate all synthetic samples."""
        self.samples = []
        
        device = next(self.model.parameters()).device
        self.model.eval()
        
        for i in range(self.num_samples):
            # Generate random image [C, H, W] in [0, 1] range
            # We'll normalize it later to match ImageNet stats
            image = torch.rand(3, self.image_size, self.image_size)
            
            # Normalize to ImageNet stats (same as SimpleCNN preprocessing)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = (image - mean) / std
            
            # Generate task ID tensor
            task_id_tensor = torch.tensor(self.task_id, dtype=torch.long)
            
            if self.use_model_predictions:
                # Use model's predictions as targets
                with torch.no_grad():
                    output = self.model(
                        pixel_values=image.unsqueeze(0).to(device),
                        task_ids=task_id_tensor.unsqueeze(0).to(device),
                        return_logprobs=False,
                        return_values=False,
                    )
                    # Get actions from model [1, num_chunks, action_dim]
                    actions = output["actions"][0]  # [num_chunks, action_dim]
            else:
                # Use random actions in [-1, 1] range
                actions = torch.rand(self.num_action_chunks, self.action_dim) * 2 - 1
            
            self.samples.append({
                "pixel_values": image,
                "task_id": task_id_tensor,
                "actions": actions,
            })
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.samples[idx]


def _build_online_rollout_dataloader(
    model: nn.Module,
    task_id: int,
    task_id_map: Dict[str, int],
    num_episodes: int = 5,
    max_steps_per_episode: int = 100,
    batch_size: int = 32,
    seed: int = 42,
):
    """
    Build a DataLoader by collecting online rollouts from the LIBERO environment.
    
    Args:
        model: Model to use for rollouts.
        task_id: Task ID to collect rollouts for.
        task_id_map: Mapping from task description to task ID.
        num_episodes: Number of episodes to collect.
        max_steps_per_episode: Maximum steps per episode.
        batch_size: Batch size.
        seed: Random seed.
    
    Returns:
        DataLoader yielding rollout batches.
    """
    from torch.utils.data import DataLoader
    
    # Get model config
    image_size = getattr(model, "image_size", 224)
    num_action_chunks = getattr(model, "num_action_chunks", 8)
    
    dataset = OnlineRolloutDataset(
        model=model,
        task_id=task_id,
        task_id_map=task_id_map,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps_per_episode,
        image_size=image_size,
        num_action_chunks=num_action_chunks,
        seed=seed,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle collected rollouts
        num_workers=0,  # No multiprocessing needed
        pin_memory=True,
    )
    return loader


def _build_synthetic_dataloader(
    model: nn.Module,
    task_id: int,
    num_samples: int = 100,
    batch_size: int = 32,
    seed: int = 42,
    use_model_predictions: bool = True,
):
    """
    Build a DataLoader that generates synthetic rollout data on-the-fly.
    
    Args:
        model: Model to use for generating actions.
        task_id: Task ID for all samples.
        num_samples: Number of synthetic samples.
        batch_size: Batch size.
        seed: Random seed.
        use_model_predictions: If True, use model predictions as targets.
    
    Returns:
        DataLoader yielding synthetic batches.
    """
    from torch.utils.data import DataLoader
    
    # Get model config
    image_size = getattr(model, "image_size", 224)
    action_dim = getattr(model, "action_dim", 7)
    num_action_chunks = getattr(model, "num_action_chunks", 8)
    
    dataset = SyntheticRolloutDataset(
        model=model,
        task_id=task_id,
        num_samples=num_samples,
        batch_size=batch_size,
        image_size=image_size,
        action_dim=action_dim,
        num_action_chunks=num_action_chunks,
        seed=seed,
        use_model_predictions=use_model_predictions,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Deterministic for reproducibility
        num_workers=0,  # No multiprocessing needed for synthetic data
        pin_memory=True,
    )
    return loader


def main():
    """
    End-to-end, directly runnable script for gradient diagnostics on SimpleCNN.

    What this does:
      1. Loads the base SimpleCNN policy from a checkpoint (default: ./best_checkpoint.pt).
      2. Generates synthetic rollout data on-the-fly using the model:
         - Random images (with fixed seed for reproducibility)
         - Actions from model's own predictions (or random actions)
      3. Constructs one DataLoader per requested task id.
      4. Computes average gradients for each task and reports cosine similarity.

    This requires NO external data - everything is generated from the model itself.

    Example (from project root):
        python -m rlinf.analysis.gradient_diagnostics \\
            --checkpoint ./best_checkpoint.pt \\
            --task_ids 0 1

    Interpretation:
      - A similarity near +1 means the two tasks want to update the CNN in similar directions.
      - A similarity near  0 means they are largely independent.
      - A similarity near -1 means they strongly interfere (training on one will undo the other).
    """
    parser = argparse.ArgumentParser(description="Gradient diagnostics for SimpleCNN")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best_checkpoint.pt",
        help="Path to SimpleCNN base checkpoint (default: ./best_checkpoint.pt).",
    )
    parser.add_argument(
        "--task_ids",
        type=int,
        nargs="+",
        default=None,
        help="List of integer task IDs to analyze (e.g., 0 1 or 0 1 2). "
             "If not provided, defaults to all 10 tasks (0-9).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run diagnostics on.",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=2,
        help="Number of batches per task to use when averaging gradients.",
    )
    parser.add_argument(
        "--num_samples_per_task",
        type=int,
        default=100,
        help="Number of synthetic samples to generate per task.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for synthetic data generation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic data generation (for reproducibility).",
    )
    parser.add_argument(
        "--use_online_rollouts",
        action="store_true",
        default=True,
        help="Use online environment rollouts (default: True). "
             "If False, use synthetic data instead.",
    )
    parser.add_argument(
        "--num_episodes_per_task",
        type=int,
        default=10,
        help="Number of episodes to collect per task (for online rollouts).",
    )
    parser.add_argument(
        "--max_steps_per_episode",
        type=int,
        default=512,
        help="Maximum steps per episode (for online rollouts).",
    )
    parser.add_argument(
        "--use_model_predictions",
        action="store_true",
        default=True,
        help="Use model's own predictions as targets (for synthetic data only). "
             "If False, use random actions as targets.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save rollout data and gradients. If not provided, data is not saved.",
    )
    parser.add_argument(
        "--use_grpo_loss",
        action="store_true",
        default=True,
        help="Use GRPO loss based on success/failure (default: True). "
             "If False, use supervised learning loss.",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=10,
        help="Group size for GRPO advantage normalization (default: 10, matches num_episodes_per_task).",
    )
    parser.add_argument(
        "--clip_ratio_low",
        type=float,
        default=0.2,
        help="Lower clipping ratio for PPO in GRPO loss (default: 0.2).",
    )
    parser.add_argument(
        "--clip_ratio_high",
        type=float,
        default=0.2,
        help="Upper clipping ratio for PPO in GRPO loss (default: 0.2).",
    )
    parser.add_argument(
        "--compute_subspace_overlap",
        action="store_true",
        default=False,
        help="Also compute gradient subspace overlap between tasks.",
    )
    parser.add_argument(
        "--subspace_k",
        type=int,
        default=5,
        help="Number of principal directions (k) for each task subspace.",
    )
    parser.add_argument(
        "--subspace_max_batches",
        type=int,
        default=None,
        help="Optional cap on number of gradient batches per task for subspace estimation.",
    )
    parser.add_argument(
        "--compute_cnn_fisher",
        action="store_true",
        default=False,
        help="Compute diagonal Fisher information per task from rollouts (CNN policy).",
    )
    parser.add_argument(
        "--cnn_fisher_max_batches",
        type=int,
        default=None,
        help="Optional cap on number of minibatches per task used for CNN Fisher estimation.",
    )
    parser.add_argument(
        "--fisher_interference_task",
        type=int,
        default=0,
        help="Task id whose gradient will be compared against Fisher(rest-of-tasks). Default: 0.",
    )
    parser.add_argument(
        "--load_rollout_data",
        type=str,
        default=None,
        help="Directory containing saved rollout data (rollout_data_task_*.pt files). "
             "If provided, skips rollout collection and loads data from this directory.",
    )

    args = parser.parse_args()
    
    # Default to all 10 tasks if not specified
    if args.task_ids is None:
        args.task_ids = list(range(10))
        print(f"No task_ids specified, defaulting to all 10 tasks: {args.task_ids}")

    device = args.device

    # 1) Load base SimpleCNN model
    print(f"Loading SimpleCNN from checkpoint: {args.checkpoint}")
    model = load_simple_cnn_from_checkpoint(args.checkpoint, device=device)
    print(f"Model loaded on {device}")

    # Get task_id_map from checkpoint (needed for online rollouts)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    task_id_map = ckpt.get("task_id_map", {})
    if not task_id_map:
        # Fallback: create a simple mapping from task IDs
        task_id_map = {f"task_{tid}": tid for tid in args.task_ids}

    # 2) Build per-task dataloaders (online rollouts or synthetic)
    task_ids = args.task_ids
    if len(task_ids) < 2:
        raise ValueError("Please provide at least two task_ids to compare.")

    # Create output directory if specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"\nOutput directory: {args.output_dir}")

    # Check if we should load saved rollout data
    if args.load_rollout_data:
        print(f"\nLoading rollout data from: {args.load_rollout_data}")
        rollout_data_by_task = {}
        task_stats = {}
        
        for tid in task_ids:
            data_path = os.path.join(args.load_rollout_data, f"rollout_data_task_{tid}.pt")
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Rollout data not found for task {tid}: {data_path}")
            
            # Load saved data (weights_only=False because we saved numpy arrays)
            saved_data = torch.load(data_path, map_location="cpu", weights_only=False)
            
            # Convert back to tensor format (if saved as numpy)
            converted_samples = []
            for sample in saved_data:
                converted_sample = {
                    "pixel_values": torch.tensor(sample["pixel_values"]) if isinstance(sample["pixel_values"], np.ndarray) else sample["pixel_values"],
                    "task_id": torch.tensor(sample["task_id"], dtype=torch.long) if not isinstance(sample["task_id"], torch.Tensor) else sample["task_id"],
                    "actions": torch.tensor(sample["actions"]) if isinstance(sample["actions"], np.ndarray) else sample["actions"],
                    "reward": sample["reward"],
                    "done": sample["done"],
                    "success": sample["success"],
                }
                # Include episode_idx and step_in_episode if available
                if "episode_idx" in sample:
                    converted_sample["episode_idx"] = sample["episode_idx"]
                if "step_in_episode" in sample:
                    converted_sample["step_in_episode"] = sample["step_in_episode"]
                if "timeout" in sample:
                    converted_sample["timeout"] = sample["timeout"]
                if "logprobs" in sample and sample["logprobs"] is not None:
                    logp = sample["logprobs"]
                    converted_sample["logprobs"] = torch.tensor(logp) if isinstance(logp, np.ndarray) else logp
                converted_samples.append(converted_sample)
            
            rollout_data_by_task[tid] = converted_samples
            
            # Compute statistics - group by episode_idx if available, otherwise use done flags
            episodes_dict = {}
            for sample in converted_samples:
                episode_idx = sample.get("episode_idx", None)
                if episode_idx is not None:
                    if episode_idx not in episodes_dict:
                        episodes_dict[episode_idx] = []
                    episodes_dict[episode_idx].append(sample)
                else:
                    # Fallback: use done flags for backward compatibility
                    if not hasattr(OnlineRolloutDataset, '_current_episode'):
                        OnlineRolloutDataset._current_episode = []
                    OnlineRolloutDataset._current_episode.append(sample)
                    if sample.get("done", False):
                        episodes_dict[len(episodes_dict)] = OnlineRolloutDataset._current_episode
                        OnlineRolloutDataset._current_episode = []
            
            # Handle remaining samples in fallback case
            if hasattr(OnlineRolloutDataset, '_current_episode') and OnlineRolloutDataset._current_episode:
                episodes_dict[len(episodes_dict)] = OnlineRolloutDataset._current_episode
                delattr(OnlineRolloutDataset, '_current_episode')
            
            # Convert to list sorted by episode index
            episodes = [episodes_dict[i] for i in sorted(episodes_dict.keys())]
            
            num_episodes = len(episodes)
            successful_episodes = sum(1 for ep in episodes if any(s.get("success", False) for s in ep))
            success_rate = successful_episodes / num_episodes if num_episodes > 0 else 0.0
            total_samples = len(converted_samples)
            avg_episode_length = total_samples / num_episodes if num_episodes > 0 else 0.0
            
            task_stats[tid] = {
                "num_episodes": num_episodes,
                "successful_episodes": successful_episodes,
                "success_rate": success_rate,
                "total_samples": total_samples,
                "avg_episode_length": avg_episode_length,
            }
            
            print(f"  Loaded task {tid}: {num_episodes} episodes, {successful_episodes} successful ({success_rate*100:.1f}%), "
                  f"{total_samples} total samples, {avg_episode_length:.1f} avg steps/episode")
        
        # Print summary table
        print("\n" + "="*80)
        print("Task Success Rate Summary (from loaded data):")
        print("="*80)
        print(f"{'Task ID':<10} {'Episodes':<12} {'Successful':<12} {'Success Rate':<15} {'Total Samples':<15} {'Avg Steps/Ep':<15}")
        print("-"*80)
        for tid in sorted(task_ids):
            stats = task_stats[tid]
            print(f"{tid:<10} {stats['num_episodes']:<12} {stats['successful_episodes']:<12} "
                  f"{stats['success_rate']*100:>6.1f}%        {stats['total_samples']:<15} {stats['avg_episode_length']:>6.1f}")
        print("="*80)
    
    elif args.use_online_rollouts:
        print(f"\nCollecting online rollout data for tasks: {task_ids}")
        print(f"  - {args.num_episodes_per_task} episodes per task")
        print(f"  - Max {args.max_steps_per_episode} steps per episode")
        print(f"  - Seed: {args.seed} (for reproducibility)")
        print(f"  - Using GRPO loss: {args.use_grpo_loss}")
        
        # Collect rollout data for each task
        rollout_data_by_task = {}
        task_stats = {}  # Store stats for each task
        
        for tid in task_ids:
            print(f"\nCollecting rollouts for task_id={tid}...")
            dataset = OnlineRolloutDataset(
                model=model,
                task_id=tid,
                task_id_map=task_id_map,
                num_episodes=args.num_episodes_per_task,
                max_steps_per_episode=args.max_steps_per_episode,
                image_size=getattr(model, "image_size", 224),
                num_action_chunks=getattr(model, "num_action_chunks", 8),
                seed=args.seed + tid,
            )
            rollout_data_by_task[tid] = dataset.samples
            
            # Compute success rate and episode statistics
            episodes = []
            current_episode = []
            for sample in dataset.samples:
                current_episode.append(sample)
                if sample.get("done", False):
                    episodes.append(current_episode)
                    current_episode = []
            if current_episode:
                episodes.append(current_episode)
            
            num_episodes = len(episodes)
            successful_episodes = sum(1 for ep in episodes if any(s.get("success", False) for s in ep))
            success_rate = successful_episodes / num_episodes if num_episodes > 0 else 0.0
            total_samples = len(dataset.samples)
            avg_episode_length = total_samples / num_episodes if num_episodes > 0 else 0.0
            
            task_stats[tid] = {
                "num_episodes": num_episodes,
                "successful_episodes": successful_episodes,
                "success_rate": success_rate,
                "total_samples": total_samples,
                "avg_episode_length": avg_episode_length,
            }
            
            print(f"  Task {tid}: {num_episodes} episodes, {successful_episodes} successful ({success_rate*100:.1f}%), "
                  f"{total_samples} total samples, {avg_episode_length:.1f} avg steps/episode")
            
            # Save rollout data if output_dir is provided
            if args.output_dir:
                data_path = os.path.join(args.output_dir, f"rollout_data_task_{tid}.pt")
                # Convert to serializable format (tensors to numpy where needed)
                serializable_data = []
                for sample in dataset.samples:
                    serializable_sample = {
                        "pixel_values": sample["pixel_values"].numpy() if torch.is_tensor(sample["pixel_values"]) else sample["pixel_values"],
                        "task_id": sample["task_id"].item() if torch.is_tensor(sample["task_id"]) else sample["task_id"],
                        "actions": sample["actions"].numpy() if torch.is_tensor(sample["actions"]) else sample["actions"],
                        "reward": sample["reward"],
                        "done": sample["done"],
                        "success": sample["success"],
                    }
                    # Include episode_idx and step_in_episode for proper episode grouping
                    if "episode_idx" in sample:
                        serializable_sample["episode_idx"] = sample["episode_idx"]
                    if "step_in_episode" in sample:
                        serializable_sample["step_in_episode"] = sample["step_in_episode"]
                    if "timeout" in sample:
                        serializable_sample["timeout"] = sample["timeout"]
                    if sample.get("logprobs") is not None:
                        logp = sample["logprobs"]
                        serializable_sample["logprobs"] = logp.numpy() if torch.is_tensor(logp) else logp
                    serializable_data.append(serializable_sample)
                torch.save(serializable_data, data_path)
                print(f"  Saved rollout data to {data_path}")
        
        # Print summary table of success rates
        print("\n" + "="*80)
        print("Task Success Rate Summary:")
        print("="*80)
        print(f"{'Task ID':<10} {'Episodes':<12} {'Successful':<12} {'Success Rate':<15} {'Total Samples':<15} {'Avg Steps/Ep':<15}")
        print("-"*80)
        for tid in sorted(task_ids):
            stats = task_stats[tid]
            print(f"{tid:<10} {stats['num_episodes']:<12} {stats['successful_episodes']:<12} "
                  f"{stats['success_rate']*100:>6.1f}%        {stats['total_samples']:<15} {stats['avg_episode_length']:>6.1f}")
        print("="*80)
    else:
        raise ValueError("GRPO loss requires online rollouts. Please use --use_online_rollouts (default).")

    # 3) Compute gradients once per task using GRPO loss
    print(f"\nComputing gradients using {'GRPO' if args.use_grpo_loss else 'supervised'} loss...")
    grads = {}
    
    for tid in task_ids:
        print(f"  Computing gradient for task_id={tid}...")
        # CRITICAL: Reload model checkpoint for each task to ensure we compute gradients
        # from the same starting point. Otherwise, gradients from previous tasks might
        # affect the model state.
        model = load_simple_cnn_from_checkpoint(args.checkpoint, device=device)
        model.eval()  # Set to eval mode first, then train() will be called inside compute_grpo_gradient_for_task
        
        if args.use_grpo_loss:
            grads[tid] = compute_grpo_gradient_for_task(
                model=model,
                rollout_data=rollout_data_by_task[tid],
                param_mode="all",
                device=device,
                group_size=args.group_size,
                clip_ratio_low=args.clip_ratio_low,
                clip_ratio_high=args.clip_ratio_high,
                entropy_bonus=0.0,  # Can be made configurable
            )
            
            # Debug: Print gradient statistics
            flat_grad = _flatten_param_dict(grads[tid])
            print(f"    Gradient stats: mean={flat_grad.mean().item():.6f}, "
                  f"std={flat_grad.std().item():.6f}, "
                  f"min={flat_grad.min().item():.6f}, "
                  f"max={flat_grad.max().item():.6f}, "
                  f"norm={flat_grad.norm().item():.6f}")
        else:
            # Fallback to supervised loss (for comparison)
            from torch.utils.data import DataLoader
            dataset = OnlineRolloutDataset(
                model=model,
                task_id=tid,
                task_id_map=task_id_map,
                num_episodes=args.num_episodes_per_task,
                max_steps_per_episode=args.max_steps_per_episode,
                image_size=getattr(model, "image_size", 224),
                num_action_chunks=getattr(model, "num_action_chunks", 8),
                seed=args.seed + tid,
            )
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
            criterion = nn.CrossEntropyLoss()
            grads[tid] = compute_average_gradient_for_task(
                model=model,
                dataloader=loader,
                loss_fn=criterion,
                param_mode="all",
                device=device,
                max_batches=args.max_batches,
            )
        
        # Save gradients if output_dir is provided
        if args.output_dir:
            grad_path = os.path.join(args.output_dir, f"gradients_task_{tid}.pt")
            # Convert gradients to serializable format
            serializable_grads = {
                name: grad.numpy() if torch.is_tensor(grad) else grad
                for name, grad in grads[tid].items()
            }
            torch.save(serializable_grads, grad_path)
            print(f"    Saved gradients to {grad_path}")

    # 4) Report pairwise cosine similarities (each task vs all remaining tasks)
    print("\n" + "="*80)
    print("Pairwise gradient cosine similarities:")
    print("="*80)
    print("\nFormat: task_i vs task_j = similarity")
    print("(Each row shows one task compared to all remaining tasks)\n")
    
    sorted_tids = sorted(task_ids)
    similarity_matrix = {}
    
    for i, t1 in enumerate(sorted_tids):
        similarities = []
        for t2 in sorted_tids:
            if t1 == t2:
                sim = 1.0  # Self-similarity
            else:
                sim = gradient_cosine_similarity_between_tasks(grads[t1], grads[t2])
            similarities.append(sim)
            similarity_matrix[(t1, t2)] = sim
        
        # Print row: task t1 vs all other tasks
        other_tasks = [t2 for t2 in sorted_tids if t2 != t1]
        sim_str = ", ".join([f"task_{t2}: {similarities[sorted_tids.index(t2)]:.4f}" for t2 in other_tasks])
        print(f"  task_{t1} vs [{sim_str}]")
    
    print("="*80)
    
    # Also print as a matrix for easier reading
    print("\nSimilarity matrix (rows = task_i, cols = task_j):")
    print("      " + " ".join([f"t{j:2d}" for j in sorted_tids]))
    for i, t1 in enumerate(sorted_tids):
        row_str = " ".join([f"{similarity_matrix[(t1, t2)]:5.3f}" for t2 in sorted_tids])
        print(f"  t{t1:2d}  {row_str}")
    
    print("\nInterpretation:")
    print("  +1.0 = highly aligned (tasks want similar parameter updates)")
    print("   0.0 = orthogonal (independent updates)")
    print("  -1.0 = highly conflicting (one task undoes the other)")
    
    # Save similarity matrix if output_dir is provided
    if args.output_dir:
        matrix_path = os.path.join(args.output_dir, "similarity_matrix.pt")
        torch.save({
            "task_ids": sorted_tids,
            "similarity_matrix": similarity_matrix,
            "similarity_matrix_array": np.array([[similarity_matrix[(t1, t2)] for t2 in sorted_tids] for t1 in sorted_tids]),
        }, matrix_path)
        print(f"\nSaved similarity matrix to {matrix_path}")

    # 5) Gradient subspace overlap (optional)
    if args.compute_subspace_overlap:
        print("\n" + "="*80)
        print(f"Computing gradient subspace overlap (k={args.subspace_k})")
        print("="*80)

        subspaces: Dict[int, torch.Tensor] = {}
        for tid in sorted(task_ids):
            print(f"  Building subspace for task_id={tid}...")
            # Reload model so all tasks share the same starting weights
            subspace_model = load_simple_cnn_from_checkpoint(args.checkpoint, device=device)
            subspace_model.eval()

            grad_matrix = compute_task_gradient_matrix_grpo(
                model=subspace_model,
                rollout_data=rollout_data_by_task[tid],
                param_mode="all",
                device=device,
                batch_size=args.batch_size,
                max_batches=args.subspace_max_batches,
            )
            U = compute_subspace_basis(grad_matrix, k=args.subspace_k)
            subspaces[tid] = U

        # Pairwise subspace overlaps
        print("\nSubspace overlap matrix (rows = task_i, cols = task_j):")
        print("      " + " ".join([f"t{j:2d}" for j in sorted_tids]))
        for t1 in sorted_tids:
            row_vals = []
            for t2 in sorted_tids:
                if t1 == t2:
                    ov = 1.0
                else:
                    ov = subspace_overlap(subspaces[t1], subspaces[t2], normalize=True)
                row_vals.append(f"{ov:5.3f}")
            print(f"  t{t1:2d}  " + " ".join(row_vals))

    # 6) Fisher-style parameter importance + interference metric (optional)
    if args.compute_cnn_fisher:
        print("\n" + "="*80)
        print("Computing CNN diagonal Fisher per task from rollouts")
        print("="*80)

        fisher_by_task: Dict[int, TensorDict] = {}
        for tid in sorted(task_ids):
            print(f"  Computing Fisher for task_id={tid}...")
            fisher_model = load_simple_cnn_from_checkpoint(args.checkpoint, device=device)
            fisher_model.eval()
            fisher_by_task[tid] = compute_cnn_fisher_diagonal_for_task_from_rollouts(
                model=fisher_model,
                rollout_data=rollout_data_by_task[tid],
                param_mode="all",
                device=device,
                batch_size=args.batch_size,
                max_batches=args.cnn_fisher_max_batches,
            )

            if args.output_dir:
                fisher_path = os.path.join(args.output_dir, f"fisher_task_{tid}.pt")
                serializable_fisher = {
                    name: t.numpy() if torch.is_tensor(t) else t
                    for name, t in fisher_by_task[tid].items()
                }
                torch.save(serializable_fisher, fisher_path)
                print(f"    Saved Fisher to {fisher_path}")

        # Compute interference metric: grad(task_k) vs Fisher(rest)
        k = args.fisher_interference_task
        if k not in grads:
            raise ValueError(
                f"--fisher_interference_task={k} not in computed grads. "
                f"Available task_ids: {sorted(task_ids)}"
            )
        if k not in fisher_by_task:
            raise ValueError(
                f"--fisher_interference_task={k} not in computed fisher. "
                f"Available task_ids: {sorted(task_ids)}"
            )

        # Fisher(rest) = sum_{t != k} Fisher(t)
        fisher_rest: TensorDict = {name: torch.zeros_like(t) for name, t in fisher_by_task[k].items()}
        for tid in sorted(task_ids):
            if tid == k:
                continue
            for name, t in fisher_by_task[tid].items():
                fisher_rest[name] += t

        metrics = fisher_weighted_gradient_interference(grads[k], fisher_rest)
        print("\n" + "="*80)
        print(f"Interference metric: gradient(task={k}) vs Fisher(rest-of-tasks)")
        print("="*80)
        for kk, vv in metrics.items():
            print(f"  {kk}: {vv:.6e}" if isinstance(vv, float) else f"  {kk}: {vv}")

        if args.output_dir:
            out_path = os.path.join(args.output_dir, f"fisher_interference_task_{k}.pt")
            torch.save(
                {
                    "task_id": k,
                    "metrics": metrics,
                },
                out_path,
            )
            print(f"\nSaved interference metrics to {out_path}")


if __name__ == "__main__":
    main()

