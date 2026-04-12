"""
Elastic Weight Consolidation (EWC) for Continual Learning with LoRA.

This module implements EWC regularization on top of LoRA adapters to prevent
catastrophic forgetting when training on sequential tasks.
"""

import math
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


def get_lora_parameters(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extract LoRA parameters from model.
    
    Args:
        model: Model with LoRA adapters (may be wrapped in FSDP)
        
    Returns:
        Dictionary mapping parameter names to tensors (on CPU)
    """
    lora_params = {}
    
    # Handle FSDP wrapping - get the underlying model
    if hasattr(model, 'module'):
        model_to_check = model.module
    else:
        model_to_check = model
    
    # Get all parameters and filter for LoRA
    for name, param in model_to_check.named_parameters():
        if 'lora' in name.lower() and param.requires_grad:
            # Clone to CPU to avoid device issues
            lora_params[name] = param.data.detach().cpu().clone()
    
    return lora_params


def snapshot_params_for_delta(model: nn.Module, lora_only: bool) -> Dict[str, torch.Tensor]:
    """
    FP32 clone of trainable parameters before optimizer.step for measuring update size.

    Uses the same FSDP unwrapping as get_lora_parameters. For FSDP, tensors are local shards;
    pair with local_param_delta_squared_sum and all_reduce(SUM) before sqrt for global L2.
    """
    if hasattr(model, "module"):
        model_to_check = model.module
    else:
        model_to_check = model

    snapshots: Dict[str, torch.Tensor] = {}
    for name, param in model_to_check.named_parameters():
        if not param.requires_grad:
            continue
        if lora_only and "lora" not in name.lower():
            continue
        snapshots[name] = param.detach().float().clone()
    return snapshots


def local_param_delta_squared_sum(
    model: nn.Module,
    snapshots: Dict[str, torch.Tensor],
    lora_only: bool,
) -> float:
    """
    Sum of squared elements of (θ - θ_snapshot) for parameters on this rank.
    Global Frobenius/L2 norm: sqrt(all_reduce_sum(local sums)).
    """
    if not snapshots:
        return float("nan")

    if hasattr(model, "module"):
        model_to_check = model.module
    else:
        model_to_check = model

    delta_sq = 0.0
    for name, param in model_to_check.named_parameters():
        if not param.requires_grad:
            continue
        if lora_only and "lora" not in name.lower():
            continue
        if name not in snapshots:
            continue
        d = param.detach().float() - snapshots[name]
        delta_sq += d.pow(2).sum().item()
    return delta_sq


def global_param_delta_l2(
    model: nn.Module,
    snapshots: Dict[str, torch.Tensor],
    lora_only: bool,
    device: torch.device,
) -> float:
    """||θ' - θ||_2 over trainable (LoRA-only or all) parameters, FSDP-safe."""
    loc_sq = local_param_delta_squared_sum(model, snapshots, lora_only)
    if not math.isfinite(loc_sq):
        return float("nan")
    t = torch.tensor([loc_sq], device=device, dtype=torch.float64)
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(torch.sqrt(t).item())


def compute_fisher_information_from_rollout(
    model: nn.Module,
    rollout_batch: Dict[str, torch.Tensor],
    num_samples: int = 100,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Compute Fisher Information Matrix for LoRA parameters using rollout batch data.
    
    Fisher Information F_i = E[(∇_θ log p(y|x, θ))^2]
    Approximated using empirical estimate over samples from rollout batch.
    
    Args:
        model: Model with LoRA adapters
        rollout_batch: Dictionary containing rollout data with keys:
            - input_ids: [batch_size, seq_len]
            - attention_mask: [batch_size, seq_len]
            - pixel_values: [batch_size, ...]
            - action_tokens: [batch_size, action_len] (ground truth actions)
        num_samples: Number of samples to use for estimation
        device: Device to compute on
        
    Returns:
        Dictionary mapping parameter names to Fisher information values (on CPU)
    """
    model.eval()
    fisher_dict = {}
    
    # Get LoRA parameters
    lora_params = {}
    for name, param in model.named_parameters():
        if 'lora' in name.lower() and param.requires_grad:
            lora_params[name] = param
            fisher_dict[name] = torch.zeros_like(param.data, device='cpu')
    
    if len(lora_params) == 0:
        raise ValueError("No LoRA parameters found for Fisher information computation")
    
    # Sample from rollout batch
    batch_size = rollout_batch["input_ids"].shape[0]
    num_samples = min(num_samples, batch_size)
    
    # Randomly sample indices
    indices = torch.randperm(batch_size, device=device)[:num_samples]
    
    # Extract samples
    sample_input_ids = rollout_batch["input_ids"][indices].to(device)
    sample_attention_mask = rollout_batch.get("attention_mask", None)
    if sample_attention_mask is not None:
        sample_attention_mask = sample_attention_mask[indices].to(device)
    sample_pixel_values = rollout_batch.get("pixel_values", None)
    if sample_pixel_values is not None:
        sample_pixel_values = sample_pixel_values[indices].to(device)
    
    # Get action tokens (ground truth) for computing log-likelihood
    sample_action_tokens = rollout_batch.get("action_tokens", None)
    if sample_action_tokens is not None:
        sample_action_tokens = sample_action_tokens[indices].to(device)
    
    # Compute Fisher information
    model.zero_grad()
    
    # Use custom_forward from model_utils to match training behavior
    from rlinf.models.embodiment.model_utils import custom_forward
    
    # Get action_token_len from rollout batch or model
    action_token_len = rollout_batch.get("action_tokens", None)
    if action_token_len is not None:
        action_token_len = action_token_len.shape[-1]  # Get action length from shape
    else:
        # Fallback: try to get from model
        if hasattr(model, 'action_dim') and hasattr(model, 'num_action_chunks'):
            action_token_len = model.action_dim * model.num_action_chunks
        else:
            # Try to get from wrapped model
            model_to_check = model.module if hasattr(model, 'module') else model
            if hasattr(model_to_check, 'action_dim') and hasattr(model_to_check, 'num_action_chunks'):
                action_token_len = model_to_check.action_dim * model_to_check.num_action_chunks
            else:
                raise ValueError("Cannot determine action_token_len for Fisher computation")
    
    # Forward pass using custom_forward (matches training)
    output_dict = custom_forward(
        model,
        input_ids=sample_input_ids,
        attention_mask=sample_attention_mask,
        pixel_values=sample_pixel_values,
        output_hidden_states=False,
        action_token_len=action_token_len,
        value_model=False,
        value_head_mode=None,
        logits_processor=None,  # We don't need processing for Fisher computation
        temperature=1.0,
        top_k=-1,
        logits_processor_args=None,
        has_bc_batch=False,
    )
    
    # Get raw logits from output_dict
    raw_logits = output_dict.get("raw_logits", None)
    if raw_logits is None:
        # Fallback to intermediate_logits
        raw_logits = output_dict.get("intermediate_logits", None)
    
    if raw_logits is None:
        raise ValueError("Could not find logits in model output for Fisher computation")
    
    # Compute log-likelihood of ground truth actions
    if sample_action_tokens is not None:
        # raw_logits: [batch_size, action_len, vocab_size]
        # sample_action_tokens: [batch_size, action_len]
        
        # Reshape for cross-entropy
        raw_logits_flat = raw_logits.reshape(-1, raw_logits.shape[-1])  # [batch_size * action_len, vocab_size]
        action_tokens_flat = sample_action_tokens.reshape(-1)  # [batch_size * action_len]
        
        # Compute log-likelihood
        log_probs = F.log_softmax(raw_logits_flat, dim=-1)
        log_likelihood = log_probs.gather(1, action_tokens_flat.unsqueeze(1)).squeeze(1)
        
        # Negative log-likelihood as loss (for gradient computation)
        loss = -log_likelihood.mean()
    else:
        # Fallback: use log-probability of the model's predictions
        log_probs = F.log_softmax(raw_logits, dim=-1)
        # Sample from the distribution
        probs = F.softmax(raw_logits, dim=-1)
        sampled_tokens = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(probs.shape[:-1])
        log_likelihood = log_probs.gather(-1, sampled_tokens.unsqueeze(-1)).squeeze(-1)
        loss = -log_likelihood.mean()
    
    # Backward pass to get gradients
    loss.backward()
    
    # Accumulate squared gradients (Fisher information estimate)
    for name, param in lora_params.items():
        if param.grad is not None:
            # Move to CPU and accumulate
            grad_sq = (param.grad.data ** 2).cpu()
            fisher_dict[name] += grad_sq
    
    # Average over samples
    for name in fisher_dict:
        fisher_dict[name] /= num_samples
    
    # Clean up gradients
    model.zero_grad()
    model.train()
    
    return fisher_dict


def compute_ewc_loss(
    model: nn.Module,
    fisher_dict: Dict[str, torch.Tensor],
    old_params: Dict[str, torch.Tensor],
    lambda_ewc: float = 1.0,
) -> torch.Tensor:
    """
    Compute EWC regularization loss.
    
    L_EWC = λ * Σ_i F_i * (θ_i - θ_i*)^2
    where F_i is Fisher info, θ_i* is old parameter value, λ is importance weight.
    
    Args:
        model: Current model with LoRA adapters
        fisher_dict: Dictionary of Fisher information values (on CPU)
        old_params: Dictionary of previous task's optimal parameters (on CPU)
        lambda_ewc: EWC regularization weight
        
    Returns:
        EWC loss tensor (on same device as model)
    """
    device = next(model.parameters()).device
    ewc_loss = torch.tensor(0.0, device=device)
    
    # Get current LoRA parameters - use same logic as get_lora_parameters for name consistency
    current_params = {}
    # Handle FSDP wrapping - get the underlying model (same as get_lora_parameters)
    if hasattr(model, 'module'):
        model_to_check = model.module
    else:
        model_to_check = model
    
    for name, param in model_to_check.named_parameters():
        if 'lora' in name.lower() and param.requires_grad:
            current_params[name] = param
    
    # Track matching parameters for debugging
    matched_params = 0
    skipped_params = 0
    skipped_reasons = []
    
    # Compute EWC loss
    for name in current_params:
        if name in fisher_dict and name in old_params:
            fisher = fisher_dict[name].to(device)
            old_param = old_params[name].to(device)
            current_param = current_params[name]
            
            # Ensure shapes match (important for FSDP)
            if (current_param.shape == old_param.shape and 
                current_param.shape == fisher.shape):
                # Additional validation: check for NaN/inf after moving to device
                if torch.isnan(fisher).any() or torch.isinf(fisher).any():
                    skipped_reasons.append(f"{name}: Fisher has NaN/Inf after device transfer")
                    skipped_params += 1
                    continue
                if torch.isnan(old_param).any() or torch.isinf(old_param).any():
                    skipped_reasons.append(f"{name}: old_param has NaN/Inf after device transfer")
                    skipped_params += 1
                    continue
                if torch.isnan(current_param).any() or torch.isinf(current_param).any():
                    skipped_reasons.append(f"{name}: current_param has NaN/Inf")
                    skipped_params += 1
                    continue
                
                diff = current_param - old_param
                contribution = (fisher * (diff ** 2)).sum()
                
                # Check if contribution is NaN/inf
                if torch.isnan(contribution) or torch.isinf(contribution):
                    skipped_reasons.append(f"{name}: EWC contribution is NaN/Inf")
                    skipped_params += 1
                    continue
                
                ewc_loss += contribution
                matched_params += 1
            else:
                skipped_params += 1
                skipped_reasons.append(
                    f"{name}: shape mismatch (current={current_param.shape}, "
                    f"old={old_param.shape}, fisher={fisher.shape})"
                )
        else:
            skipped_params += 1
            missing = []
            if name not in fisher_dict:
                missing.append("fisher_dict")
            if name not in old_params:
                missing.append("old_params")
            skipped_reasons.append(f"{name}: missing in {', '.join(missing)}")
    
    # Debug output (only print on rank 0 if available)
    import os
    rank = int(os.environ.get("LOCAL_RANK", 0))
    if rank == 0:
        # Always print matching stats for first few calls to help debug
        print(f"[EWC] Parameter matching: matched={matched_params}, skipped={skipped_params}, "
              f"current={len(current_params)}, fisher={len(fisher_dict)}, old={len(old_params)}")
        
        if matched_params == 0 and len(current_params) > 0:
            print(f"[EWC] WARNING: No parameters matched! EWC will have no effect.")
            # Print sample keys to help diagnose naming issues
            sample_current = list(current_params.keys())[:2]
            sample_fisher = list(fisher_dict.keys())[:2]
            sample_old = list(old_params.keys())[:2]
            print(f"[EWC] Sample current param names: {sample_current}")
            print(f"[EWC] Sample fisher dict names: {sample_fisher}")
            print(f"[EWC] Sample old_params names: {sample_old}")
            
        if len(skipped_reasons) > 0 and matched_params < len(current_params):
            print(f"[EWC] First 5 skip reasons: {skipped_reasons[:5]}")
    
    # Final validation: check if ewc_loss is NaN/inf
    if torch.isnan(ewc_loss) or torch.isinf(ewc_loss):
        import os
        rank = int(os.environ.get("LOCAL_RANK", 0))
        if rank == 0:
            print(f"ERROR: EWC loss is NaN/Inf! Matched params: {matched_params}, "
                  f"Skipped params: {skipped_params}")
            if len(skipped_reasons) > 0:
                print(f"Skipped reasons: {skipped_reasons[:5]}")
        # Return zero loss instead of NaN to prevent training crash
        ewc_loss = torch.tensor(0.0, device=device)
    
    return lambda_ewc * ewc_loss


def save_ewc_data(
    fisher_dict: Dict[str, torch.Tensor],
    save_path: str,
):
    """
    Save EWC Fisher information to disk.
    
    Note: We only save Fisher information, not old_params. The old_params reference
    is captured from the loaded checkpoint weights at the start of training, which
    avoids FSDP sharding issues and ensures we regularize toward the exact loaded weights.
    
    Args:
        fisher_dict: Dictionary of accumulated Fisher information values (from all tasks)
        save_path: Path to save the EWC data
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'fisher_dict': fisher_dict,
    }, save_path)
    print(f"Saved EWC Fisher data to {save_path}")


def load_ewc_data(load_path: str) -> Dict[str, torch.Tensor]:
    """
    Load EWC Fisher information from disk.
    
    Note: old_params is no longer saved/loaded. The reference weights for EWC
    regularization are captured from the loaded checkpoint at the start of training.
    
    Args:
        load_path: Path to load the EWC data from
        
    Returns:
        fisher_dict: Dictionary of Fisher information values
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"EWC data not found at {load_path}")
    
    data = torch.load(load_path, map_location='cpu')
    fisher_dict = data['fisher_dict']
    print(f"Loaded EWC Fisher data from {load_path}")
    
    return fisher_dict
