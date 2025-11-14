import torch
import torch.nn.functional as F

def behavior_cloning_loss(**kwargs):
    predicted_actions = kwargs["action_tokens"]
    expert_actions = kwargs["expert_action_tokens"]
    bc_coeff = kwargs["bc_coeff"]

    bc_loss = F.mse_loss(predicted_actions, expert_actions, reduction='mean')
    weighted_loss = bc_coeff * bc_loss
    
    metrics = {
        "bc/loss": bc_loss.detach().item(),
        "bc/weighted_loss": weighted_loss.detach().item(),
        "bc/coeff": bc_coeff,
        "bc/mean_abs_error": (predicted_actions - expert_actions).abs().mean().item(),
    }
    
    return weighted_loss, metrics