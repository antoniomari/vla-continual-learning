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

import torch


def huber_loss(error: torch.Tensor, delta: float) -> torch.Tensor:
    return torch.where(
        error.abs() < delta, 0.5 * error**2, delta * (error.abs() - 0.5 * delta)
    )


def kl_penalty(
    logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty
) -> torch.FloatTensor:
    """
    Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        # For numerical stability
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def preprocess_loss_inputs(**kwargs) -> dict:
    logprob_type = kwargs.get("logprob_type", None)
    entropy_type = kwargs.get("entropy_type", None)
    single_action_dim = kwargs.get("single_action_dim", None)

    logprobs = kwargs["logprobs"]
    old_logprobs = kwargs["old_logprobs"]
    advantages = kwargs["advantages"]
    entropy = kwargs.get("entropy", None)
    loss_mask = kwargs.get("loss_mask", None)
    loss_mask_sum = kwargs.get("loss_mask_sum", None)

    bsz = logprobs.shape[0]
    raw_lp_shape = logprobs.shape

    if logprob_type == "token_level":
        logprobs = logprobs.reshape(bsz, -1, single_action_dim)
        old_logprobs = old_logprobs.reshape(bsz, -1, single_action_dim)
        # Per-token advantages (e.g. embodied_opd: teacher_lp - student_lp) match flat
        # logprob layout [bsz, n_tokens]. Per-chunk scalar advantages use [bsz, n_chunks].
        if advantages.shape == raw_lp_shape:
            advantages = advantages.reshape(bsz, -1, single_action_dim)
        else:
            advantages = advantages.unsqueeze(-1)
        if loss_mask is not None:
            if loss_mask.shape == raw_lp_shape:
                loss_mask = loss_mask.reshape(bsz, -1, single_action_dim)
            else:
                loss_mask = loss_mask.unsqueeze(-1)
        if loss_mask_sum is not None:
            if loss_mask_sum.shape == raw_lp_shape:
                loss_mask_sum = loss_mask_sum.reshape(bsz, -1, single_action_dim)
            else:
                loss_mask_sum = loss_mask_sum.unsqueeze(-1)

    elif logprob_type == "action_level":
        logprobs = logprobs.reshape(bsz, -1, single_action_dim).sum(dim=-1)
        old_logprobs = old_logprobs.reshape(bsz, -1, single_action_dim).sum(dim=-1)

    elif logprob_type == "chunk_level":
        logprobs = logprobs.sum(dim=-1)
        old_logprobs = old_logprobs.sum(dim=-1)
        advantages = advantages.sum(dim=-1)

    if entropy is not None:
        if entropy_type == "token_level" and logprob_type == "token_level":
            # Reshape entropy to match logprobs shape: (bsz, -1, single_action_dim)
            entropy = entropy.reshape(bsz, -1, single_action_dim)
        elif entropy_type == "action_level":
            entropy = entropy.reshape(bsz, -1, single_action_dim).sum(dim=-1)
        elif entropy_type == "chunk_level":
            entropy = entropy.sum(dim=-1)

    kwargs.update(
        {
            "logprobs": logprobs,
            "old_logprobs": old_logprobs,
            "advantages": advantages,
            "entropy": entropy,
            "loss_mask": loss_mask,
            "loss_mask_sum": loss_mask_sum,
        }
    )

    return kwargs


def preprocess_advantages_inputs(**kwargs) -> dict:
    """Align reward/done tensors with the advantage routine before dispatch.

    For `reward_type == "chunk_level"`, each env step carries a vector of per-substep
    rewards/dones (last dim = num_action_chunks). GRPO/GAE here expect one scalar
    reward (and terminal flag) per chunk step: collapse the chunk dimension by
    summing rewards and taking the last substep's done flag.
    """
    reward_type = kwargs.get("reward_type", None)
    if reward_type == "chunk_level":
        rewards = kwargs["rewards"]
        dones = kwargs["dones"]
        kwargs["rewards"] = rewards.sum(dim=-1, keepdim=True)
        kwargs["dones"] = dones[..., -1:]
    return kwargs
