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

from typing import Optional, Tuple

import torch

from rlinf.algorithms.registry import register_advantage


def _compute_embodied_trajectory_scores(
    rewards: torch.Tensor,
    dones: torch.Tensor,
) -> torch.Tensor:
    """Return one undiscounted env score per batch item, zeroing after terminal."""
    n_chunk_step, actual_bsz, num_action_chunks = rewards.shape
    flattened_rewards = rewards.transpose(1, 2).reshape(
        n_chunk_step * num_action_chunks, -1
    )
    flattened_dones = dones.transpose(1, 2).reshape(
        (n_chunk_step + 1) * num_action_chunks, -1
    )
    flattened_dones = flattened_dones[-(n_chunk_step * num_action_chunks + 1) :]

    scores = torch.zeros(
        actual_bsz,
        device=rewards.device,
        dtype=rewards.dtype,
    )
    for step in reversed(range(flattened_rewards.shape[0])):
        scores = scores * ~flattened_dones[step + 1]
        scores += flattened_rewards[step]
    return scores


@register_advantage("embodied_gae")
def compute_embodied_gae_advantages_and_returns(
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate advantages and returns for Proximal Policy Optimization (PPO).
    NOTE: currently this function does not support auto-reset.

    This function implements Generalized Advantage Estimation (GAE) to compute
    advantages and returns for PPO training. The advantages are normalized
    using mean and standard deviation for stable training.

    Args:
        rewards (torch.Tensor): Reward tensor of shape [num-chunk, bsz, chunk-size]
        values (torch.Tensor): Value predictions of shape [num-chunk + 1, bsz, chunk-size]
        dones (torch.Tensor): Done flag tensor of shape [num-chunk + 1, bsz, chunk-size]
        gamma (float): Discount factor
        gae_lambda (float): GAE lambda parameter
        normalize_advantages (bool): Whether to normalize advantages

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (advantages, returns) tensors
    """

    rewards = kwargs["rewards"]
    values = kwargs["values"]
    dones = kwargs["dones"]
    gamma = kwargs.get("gamma", 1.0)
    gae_lambda = kwargs.get("gae_lambda", 1.0)
    normalize_advantages = kwargs.get("normalize_advantages", True)
    normalize_returns = kwargs.get("normalize_returns", False)

    num_chunk, bsz, chunk_size = rewards.shape
    flattened_rewards = rewards.transpose(1, 2).reshape(num_chunk * chunk_size, -1)
    flattened_values = values.transpose(1, 2).reshape((num_chunk + 1) * chunk_size, -1)
    flattened_values = flattened_values[
        : num_chunk * chunk_size + 1
    ]  # [n_steps+1, bsz]
    flattened_dones = dones.transpose(1, 2).reshape((num_chunk + 1) * chunk_size, -1)[
        -(num_chunk * chunk_size + 1) :
    ]

    flattened_returns = torch.zeros_like(flattened_rewards)

    # GAE(γ, λ) backward pass; `flattened_returns[step]` stores A_t + V_t for bootstrapping path.
    gae = 0
    for step in reversed(range(flattened_rewards.shape[0])):
        vt1 = flattened_values[step + 1]
        vt = flattened_values[step]

        delta = (
            flattened_rewards[step] + gamma * vt1 * (~flattened_dones[step + 1]) - vt
        )
        gae = delta + gamma * gae_lambda * (~flattened_dones[step + 1]) * gae
        flattened_returns[step] = gae + vt

    # Advantage = return estimate minus value at same timestep.
    flattened_advantages = flattened_returns - flattened_values[:-1]

    if normalize_advantages:
        mean_advantages = flattened_advantages.mean()
        std_advantages = flattened_advantages.std(correction=0)
        flattened_advantages = (flattened_advantages - mean_advantages) / (
            std_advantages + 1e-5
        )
    if normalize_returns:
        mean_returns = flattened_returns.mean()
        std_retuns = flattened_returns.std(correction=0)
        flattened_returns = (flattened_returns - mean_returns) / (std_retuns + 1e-5)

    advantages = flattened_advantages.reshape(num_chunk, chunk_size, -1).transpose(1, 2)
    returns = flattened_returns.reshape(num_chunk, chunk_size, -1).transpose(1, 2)

    return advantages, returns


@register_advantage("embodied_grpo")
def compute_embodied_grpo_advantages(
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Group Relative Policy Optimization (GRPO) advantages for embodied RL.

    Layout: `rewards` / `dones` are [n_chunk_step, batch, num_action_chunks].
    For each parallel env index in `batch`, we first compute a scalar **trajectory
    return** (discounted sum is not used here; it is undiscounted cumulative reward
    with zeroing after terminal—see loop). Then we **subtract the group mean** and
    **divide by group std** over GRPO groups of size `group_size`, so advantages
    are **relative within groups** (same task / prompt family), not absolute.

    Groups are laid out as `rollout_epoch * num_group_envs` rows × `group_size`
    columns after reshaping `scores`. Returns second tensor = advantages (used as
    surrogate "returns" for logging paths that expect both).
    """

    rewards = kwargs["rewards"]
    dones = kwargs["dones"]
    num_group_envs = kwargs.get("num_group_envs", 1)
    group_size = kwargs.get("group_size", 2)
    normalize_advantages = kwargs.get("normalize_advantages", True)
    loss_mask = kwargs.get("loss_mask", None)
    rollout_epoch = kwargs.get("rollout_epoch", 1)
    epsilon = kwargs.get("epsilon", 1e-6)

    n_chunk_step, actual_bsz, num_action_chunks = rewards.shape
    # Time-major grid: each row is one fine timestep (chunk step × action substep).
    flattened_rewards = rewards.transpose(1, 2).reshape(
        n_chunk_step * num_action_chunks, -1
    )

    flattened_dones = dones.transpose(1, 2).reshape(
        (n_chunk_step + 1) * num_action_chunks, -1
    )
    flattened_dones = flattened_dones[-(n_chunk_step * num_action_chunks + 1) :]

    flattened_loss_mask = None
    if loss_mask is not None:
        flattened_loss_mask = loss_mask.transpose(1, 2).reshape(
            n_chunk_step * num_action_chunks, -1
        )

    # Per-env scalar score = sum of rewards until episode end (backward pass).
    # `scores * ~dones` clears contribution after terminal; no gamma here.
    n_steps = flattened_rewards.shape[0]
    scores = _compute_embodied_trajectory_scores(rewards, dones)

    # Group-relative standardization: each group has `group_size` trajectories.
    if normalize_advantages:
        scores = scores.reshape(rollout_epoch * num_group_envs, group_size)
        mean, std = scores.mean(dim=-1, keepdim=True), scores.std(dim=-1, keepdim=True)
        flattened_advantages = (scores - mean) / (std + epsilon)
        flattened_advantages = flattened_advantages.reshape(1, -1)
    else:
        flattened_advantages = scores.reshape(1, -1)

    # Broadcast scalar advantage per trajectory to every fine timestep for token loss.
    if flattened_loss_mask is not None:
        flattened_advantages = (
            flattened_advantages.tile([n_steps, 1]) * flattened_loss_mask
        )
    else:
        flattened_advantages = flattened_advantages.tile([n_steps, 1])

    advantages = flattened_advantages.reshape(
        n_chunk_step, num_action_chunks, actual_bsz
    ).transpose(1, 2)
    return advantages, advantages


@register_advantage("embodied_opd")
def compute_embodied_opd_advantages(
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """OPD reward r_t = log pi_teacher(a|s) - log pi_student(a|s) at rollout student logprobs.

    By default (when ``normalize_advantages`` is True, from algorithm config), the
    selected ``opd_reward_normalization`` mode rescales these margins before training.
    ``group_zscore`` is GRPO-like: it reduces each action chunk to one scalar, normalizes
    that scalar within groups of size ``algorithm.group_size``, then broadcasts it back
    to all tokens in the chunk. Other modes preserve token/action-level feedback.

    When ``normalize_advantages`` is False, raw per-token margins are returned.

    ``advantages`` / ``returns`` use the same tensor layout as ``prev_logprobs``.
    """

    # shape: (n_chunk, bsz, num_action_tokens) = (64, 32, 7x8) -> 7x8 is cause action_dim=7 and num_action_chunk=8
    # (assuming to use token_level option)
    # so in total the batch will use 64x32x7x8 = 114k logprobs 
    teacher_logprobs = kwargs["teacher_logprobs"]
    student_logprobs = kwargs["student_logprobs"]
    loss_mask = kwargs.get("loss_mask", None)
    normalize_advantages = kwargs.get("normalize_advantages", True)
    norm_mode = kwargs.get("opd_reward_normalization", "group_zscore")
    tanh_tau = float(kwargs.get("opd_reward_tanh_tau", 5.0))
    clip_c = float(kwargs.get("opd_reward_clip_c", 1.0))
    group_size = kwargs.get("group_size", 1)
    loss_type = kwargs.get("loss_type", "embodied_opd")
    success_gate_teacher_lambda = float(
        kwargs.get("opd_success_gate_teacher_lambda", 1.0)
    )
    success_gate_threshold = float(
        kwargs.get("opd_success_gate_reward_threshold", 0.0)
    )
    single_action_dim = kwargs.get("single_action_dim", None)
    epsilon = kwargs.get("epsilon", 1e-6)

    if loss_type in (
        "embodied_opd_success_gate",
        "embodied_opd_grpo_plus_success_gate",
    ):
        teacher_kwargs = dict(kwargs)
        teacher_kwargs["loss_type"] = "embodied_opd"
        teacher_advantages, _ = compute_embodied_opd_advantages(**teacher_kwargs)
        env_normalize_advantages = kwargs.get(
            "opd_success_gate_env_normalize_advantages",
            kwargs.get("normalize_advantages", True),
        )
        env_advantages, _ = compute_embodied_grpo_advantages(
            rewards=kwargs["rewards"],
            dones=kwargs["dones"],
            normalize_advantages=env_normalize_advantages,
            num_group_envs=kwargs.get("num_group_envs", 1),
            group_size=group_size,
            rollout_epoch=kwargs.get("rollout_epoch", 1),
            loss_mask=None,
            epsilon=epsilon,
        )
        # GRPO env advantages are typically per action-chunk (e.g., 8), while OPD
        # teacher advantages can be token/action-dim flattened (e.g., 8*7=56).
        # Align shapes explicitly before mixing them in success-gated OPD.
        if env_advantages.shape != teacher_advantages.shape:
            if env_advantages.shape[:2] != teacher_advantages.shape[:2]:
                raise ValueError(
                    "embodied_opd_success_gate shape mismatch on [n_chunk, bsz]: "
                    f"env_advantages={tuple(env_advantages.shape)} vs "
                    f"teacher_advantages={tuple(teacher_advantages.shape)}"
                )
            env_last = int(env_advantages.shape[-1])
            teacher_last = int(teacher_advantages.shape[-1])
            if env_last <= 0 or teacher_last % env_last != 0:
                raise ValueError(
                    "embodied_opd_success_gate cannot align token dimension: "
                    f"env_advantages last dim={env_last}, "
                    f"teacher_advantages last dim={teacher_last}"
                )
            repeat_factor = teacher_last // env_last
            env_advantages = env_advantages.repeat_interleave(
                repeat_factor, dim=-1
            )
        env_advantages = env_advantages.to(
            dtype=teacher_advantages.dtype,
            device=teacher_advantages.device,
        )

        success_scores = _compute_embodied_trajectory_scores(
            kwargs["rewards"],
            kwargs["dones"],
        )
        success_gate = (success_scores > success_gate_threshold).to(
            dtype=teacher_advantages.dtype,
            device=teacher_advantages.device,
        )
        success_gate = success_gate.view(1, -1, 1).expand_as(teacher_advantages)

        if loss_type == "embodied_opd_success_gate":
            adv_out = (
                success_gate * env_advantages
                + (1.0 - success_gate)
                * success_gate_teacher_lambda
                * teacher_advantages
            )
        else:
            # Always keep normalized env/GRPO credit assignment; only gate the
            # additional teacher-imitation OPD signal on failed trajectories.
            adv_out = (
                env_advantages
                + (1.0 - success_gate)
                * success_gate_teacher_lambda
                * teacher_advantages
            )
        if loss_mask is not None:
            adv_out = adv_out * loss_mask
        return adv_out, adv_out

    # Advantages are computed as VLA-OPD reward initially
    advantages = teacher_logprobs - student_logprobs
    if loss_mask is not None:
        advantages = advantages * loss_mask

    if not normalize_advantages:
        return advantages, advantages

    if norm_mode not in (
        "group_zscore",
        "token_zscore",
        "action_dim_zscore",
        "mad_abs",
        "batch_zscore",
        "tanh_squash",
        "clip",
        "positive_clip",
        "teacher_prob",
    ):
        raise ValueError(
            f"Unsupported OPD reward normalization mode: {norm_mode}. "
            "Use 'group_zscore', 'token_zscore', 'action_dim_zscore', 'mad_abs', "
            "'batch_zscore', 'tanh_squash', 'clip', 'positive_clip', or 'teacher_prob'."
        )

    # Sign-preserving global scaling for REINFORCE-like OPD:
    # R_scaled = R_raw / (mean(|R_raw|) + eps), computed on current device rollout batch.
    # No mean subtraction, so positive/negative signs are preserved.
    if norm_mode == "mad_abs":
        if loss_mask is not None:
            m = loss_mask.float()
            denom = m.sum().clamp(min=1.0)
            scale = (advantages.abs() * m).sum() / denom
        else:
            scale = advantages.abs().mean()
        adv_out = advantages / (scale + epsilon)
        if loss_mask is not None:
            adv_out = adv_out * loss_mask
        return adv_out, adv_out

    # Whole-batch z-score on the local rollout tensor (mask-aware when provided).
    # This centers rewards and normalizes by std over all valid entries.
    if norm_mode == "batch_zscore":
        if loss_mask is not None:
            m = loss_mask.float()
            denom = m.sum().clamp(min=1.0)
            mean = (advantages * m).sum() / denom
            var = (((advantages - mean) * m) ** 2).sum() / denom
        else:
            mean = advantages.mean()
            var = ((advantages - mean) ** 2).mean()
        std = torch.sqrt(var + epsilon)
        adv_out = (advantages - mean) / std
        if loss_mask is not None:
            adv_out = adv_out * loss_mask
        return adv_out, adv_out

    # Per-token-position z-score. This keeps dense OPD feedback distinct for each
    # action token position instead of mixing the whole action chunk into one scalar.
    if norm_mode == "token_zscore":
        if loss_mask is not None:
            m = loss_mask.float()
            denom = m.sum(dim=(0, 1), keepdim=True).clamp(min=1.0)
            mean = (advantages * m).sum(dim=(0, 1), keepdim=True) / denom
            var = (((advantages - mean) * m) ** 2).sum(dim=(0, 1), keepdim=True) / denom
        else:
            mean = advantages.mean(dim=(0, 1), keepdim=True)
            var = ((advantages - mean) ** 2).mean(dim=(0, 1), keepdim=True)
        adv_out = (advantages - mean) / torch.sqrt(var + epsilon)
        if loss_mask is not None:
            adv_out = adv_out * loss_mask
        return adv_out, adv_out

    # Per-action-dimension z-score. For flattened chunks laid out as
    # [num_action_chunks * action_dim], pool the same semantic action dimension
    # across chunk horizon, rollout time, and batch, but keep dimensions separate.
    if norm_mode == "action_dim_zscore":
        if single_action_dim is None:
            raise ValueError(
                "opd_reward_normalization=action_dim_zscore requires single_action_dim"
            )
        single_action_dim = int(single_action_dim)
        if single_action_dim <= 0 or advantages.shape[-1] % single_action_dim != 0:
            raise ValueError(
                "embodied_opd action_dim_zscore expected token dimension divisible by "
                f"single_action_dim={single_action_dim}, got {advantages.shape[-1]}"
            )
        reshaped = advantages.reshape(*advantages.shape[:-1], -1, single_action_dim)
        if loss_mask is not None:
            m = loss_mask.reshape(*loss_mask.shape[:-1], -1, single_action_dim).float()
            denom = m.sum(dim=(0, 1, 2), keepdim=True).clamp(min=1.0)
            mean = (reshaped * m).sum(dim=(0, 1, 2), keepdim=True) / denom
            var = (((reshaped - mean) * m) ** 2).sum(dim=(0, 1, 2), keepdim=True) / denom
        else:
            mean = reshaped.mean(dim=(0, 1, 2), keepdim=True)
            var = ((reshaped - mean) ** 2).mean(dim=(0, 1, 2), keepdim=True)
        adv_out = ((reshaped - mean) / torch.sqrt(var + epsilon)).reshape_as(advantages)
        if loss_mask is not None:
            adv_out = adv_out * loss_mask
        return adv_out, adv_out

    # Sign-preserving squashing to cap outlier rewards in (-1, 1).
    if norm_mode == "tanh_squash":
        if tanh_tau <= 0:
            raise ValueError(
                f"opd_reward_tanh_tau must be > 0 for tanh_squash, got {tanh_tau}"
            )
        adv_out = torch.tanh(advantages / tanh_tau)
        if loss_mask is not None:
            adv_out = adv_out * loss_mask
        return adv_out, adv_out

    # Hard-clip rewards to [-c, c] to bound gradient contributions.
    if norm_mode == "clip":
        if clip_c <= 0:
            raise ValueError(
                f"opd_reward_clip_c must be > 0 for clip mode, got {clip_c}"
            )
        adv_out = torch.clamp(advantages, min=-clip_c, max=clip_c)
        if loss_mask is not None:
            adv_out = adv_out * loss_mask
        return adv_out, adv_out

    # Positive-only clipped margins. This preserves token-level dense feedback while
    # turning OPD into conservative sampled imitation: reinforce teacher-preferred
    # sampled actions, but do not actively push away actions with negative margins.
    if norm_mode == "positive_clip":
        if clip_c <= 0:
            raise ValueError(
                f"opd_reward_clip_c must be > 0 for positive_clip mode, got {clip_c}"
            )
        adv_out = torch.clamp(advantages, min=0.0, max=clip_c)
        if loss_mask is not None:
            adv_out = adv_out * loss_mask
        return adv_out, adv_out

    # Sampled teacher-confidence weighting. This is a lightweight on-policy
    # distillation approximation for the current sampled actions, not a full
    # distribution KL: high teacher probability increases the weight, low teacher
    # probability makes the update small. The mean normalization keeps scale stable.
    if norm_mode == "teacher_prob":
        teacher_weight = torch.exp(torch.clamp(teacher_logprobs, min=-20.0, max=0.0))
        if loss_mask is not None:
            m = loss_mask.float()
            denom = m.sum().clamp(min=1.0)
            scale = (teacher_weight * m).sum() / denom
        else:
            scale = teacher_weight.mean()
        adv_out = teacher_weight / (scale + epsilon)
        if loss_mask is not None:
            adv_out = adv_out * loss_mask
        return adv_out, adv_out

    # Scalar trajectory score = sum of valid token margins per env slot (flatten time×tokens).
    # n_chunk: 512/8 = 64, bsz: (group_size * num_group_envs * rollout_epoch) = 32 in the latest runs,
    n_chunk, bsz, _token_flat = advantages.shape
    flat = advantages.reshape(n_chunk * bsz, -1)
    if loss_mask is not None:
        m = loss_mask.reshape(n_chunk * bsz, -1).float()
        scores = (flat * m).sum(dim=-1)
        denom = m.sum(dim=-1).clamp(min=1.0)
        scores = scores / denom
    else:
        scores = flat.mean(dim=-1)

    if bsz % group_size != 0:
        raise ValueError(
            f"embodied_opd group norm: batch {bsz} not divisible by group_size {group_size}"
        )
    n_groups = bsz // group_size
    scores_g = scores.reshape(n_chunk, n_groups, group_size)
    mean = scores_g.mean(dim=-1, keepdim=True)
    std = scores_g.std(dim=-1, keepdim=True)
    scores_norm = (scores_g - mean) / (std + epsilon)
    scores_norm = scores_norm.reshape(n_chunk, bsz)

    adv_out = scores_norm.unsqueeze(-1).expand_as(advantages)
    if loss_mask is not None:
        adv_out = adv_out * loss_mask
    return adv_out, adv_out


@register_advantage("math_grpo")
def compute_math_grpo_advantages(**kwargs):
    reward_scores = kwargs["reward_scores"]
    mask = kwargs["mask"]
    num_responses = kwargs["num_responses"]

    grouped_rewards = reward_scores.view(-1, num_responses)
    # compute median
    grouped_reward_mean = grouped_rewards.mean(dim=1).repeat_interleave(
        num_responses, dim=0
    )
    grouped_reward_std = grouped_rewards.std(dim=1).repeat_interleave(
        num_responses, dim=0
    )

    advantages = reward_scores - grouped_reward_mean
    advantages = advantages / (grouped_reward_std + 1e-6)
    device = mask.device
    advantages = advantages.to(device)

    advantages = (torch.zeros_like(mask) + advantages.view(-1, 1)) * mask

    return advantages, None


if __name__ == "__main__":
    # test compute_ppo_advantages_and_returns
    torch.manual_seed(0)
    rewards = torch.randn(4, 2, 3)
    values = torch.randn(5, 2, 3)
    dones = torch.zeros(5, 2, 3).bool()
    dones[-1] = 1
    advantages, returns = compute_embodied_gae_advantages_and_returns(
        rewards=rewards, values=values, dones=dones, gamma=0.99, gae_lambda=0.95
    )
    print(advantages.mean())
    print(returns.mean())

    # test compute_grpo_advantages_and_returns
    torch.manual_seed(0)
    rewards = torch.randn(4, 4, 3)
    dones = torch.zeros(5, 4, 3).bool()
    loss_mask = torch.rand_like(rewards) > 0.5
    dones[-1] = 1
    advantages, _ = compute_embodied_grpo_advantages(
        rewards=rewards,
        dones=dones,
        loss_mask=loss_mask,
        num_group_envs=2,
        group_size=2,
        normalize_advantages=False,
    )
    print(advantages)
