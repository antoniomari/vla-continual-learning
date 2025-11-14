import torch
import torch.nn.functional as F

def behavior_cloning_loss(**kwargs):
    predicted_actions = kwargs["action_tokens"]
    expert_actions = kwargs["expert_action_tokens"]
    bc_coeff = kwargs["bc_coeff"]
    
    bc_loss = F.mse_loss(predicted_actions, expert_actions, reduction='mean')
    weighted_loss = bc_coeff * bc_loss
    per_dim_mse = ((predicted_actions - expert_actions) ** 2).mean(dim=[0, 1])
    
    metrics = {
        "bc/loss": bc_loss.detach().item(),
        "bc/weighted_loss": weighted_loss.detach().item(),
        "bc/coeff": bc_coeff,
        "bc/mean_abs_error": (predicted_actions - expert_actions).abs().mean().item(),
    }
    
    for i, dim_mse in enumerate(per_dim_mse):
        metrics[f"bc/mse_dim_{i}"] = dim_mse.item()
    
    return weighted_loss, metrics