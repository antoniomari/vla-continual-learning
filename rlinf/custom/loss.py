import torch
import torch.nn.functional as F


def behavior_cloning_loss(**kwargs):
    predicted_actions = kwargs["action_tokens"]
    expert_actions = kwargs["expert_action_tokens"]
    bc_coeff = kwargs["bc_coeff"]

    bc_loss = F.mse_loss(predicted_actions, expert_actions, reduction="mean")
    weighted_loss = bc_coeff * bc_loss

    metrics = {
        "bc/loss": bc_loss.detach().item(),
        "bc/weighted_loss": weighted_loss.detach().item(),
        "bc/coeff": bc_coeff,
        "bc/mean_abs_error": (predicted_actions - expert_actions).abs().mean().item(),
    }

    return weighted_loss, metrics


def behavior_cloning_loss_with_reference_logits(**kwargs):
    """BC loss computed as MSE between current model logits and reference model logits.

    This is similar to KL penalty but uses MSE on logits instead of logprobs,
    and is applied on expert trajectories (BC batch) rather than RL trajectories.

    Args:
        current_logits: Logits from current model on expert trajectory [B, act, vocab_range]
        reference_logits: Logits from reference model on expert trajectory [B, act, vocab_range]
        bc_coeff: Coefficient to weight the BC loss

    Returns:
        weighted_loss: Weighted BC loss (scalar tensor)
        metrics: Dictionary of metrics for logging
    """
    current_logits = kwargs["current_logits"]
    reference_logits = kwargs["reference_logits"]
    bc_coeff = kwargs["bc_coeff"]

    # Ensure reference logits are detached to prevent gradient flow
    reference_logits = reference_logits.detach()

    # Compute MSE loss between logits
    bc_loss = F.mse_loss(current_logits, reference_logits, reduction="mean")
    weighted_loss = bc_coeff * bc_loss

    # Compute additional metrics
    logits_diff = current_logits - reference_logits
    mean_abs_error = logits_diff.abs().mean().item()
    max_abs_error = logits_diff.abs().max().item()

    metrics = {
        "bc/loss": bc_loss.detach().item(),
        "bc/weighted_loss": weighted_loss.detach().item(),
        "bc/coeff": bc_coeff,
        "bc/mean_abs_error": mean_abs_error,
        "bc/max_abs_error": max_abs_error,
        "bc/logits_mse": bc_loss.detach().item(),
    }

    return weighted_loss, metrics
