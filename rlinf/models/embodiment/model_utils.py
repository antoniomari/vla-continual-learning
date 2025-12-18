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

from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers.generation import TopKLogitsWarper


def default_logits_processor(logits, action_tokens, vocab_size, n_action_bins):
    logits = logits.permute(0, 2, 1)  # [B, vocab-size, action-dim]

    logits[:, : vocab_size - n_action_bins] = -torch.inf
    logits[:, vocab_size:] = -torch.inf

    logprobs = compute_logprobs_from_logits(logits=logits, target=action_tokens)

    entropy = compute_entropy_from_logits(logits)

    ret = {"logprobs": logprobs, "entropy": entropy}

    return ret


def compute_logprobs_from_logits(logits, target):
    logprobs = -F.cross_entropy(
        logits, target=target, reduction="none"
    )  # [B, action-dim]
    return logprobs


def compute_entropy_from_logits(logits, epsilon=1e-10):
    """
    Compute entropy by logits.

    Args:
        logits: [B, vocab-size, seq-len]
    Returns:
        entropy: [B, seq-len]
    """
    all_probs = F.softmax(logits, dim=1)  # [B, vocab-size, seq-len]
    all_log_probs = torch.log(all_probs + epsilon)
    entropy = -torch.sum(all_probs * all_log_probs, dim=1)  # [B, seq-len]
    return entropy


def _normalize_actions(model, actions, norm_key=None):
    actions = actions.cpu().numpy()
    from prismatic.vla.constants import (
        ACTION_PROPRIO_NORMALIZATION_TYPE,
        NormalizationType,
    )

    """Normalize actions to [-1, 1] using dataset statistics"""
    action_norm_stats = model.get_action_stats(norm_key)

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

    action_dim = actions.shape[-1]
    repeat_factor = action_dim // action_high.shape[0]

    action_high = action_high.repeat(repeat_factor)
    action_low = action_low.repeat(repeat_factor)
    mask = mask * repeat_factor

    normalized_actions = np.where(
        mask,
        2.0 * (actions - action_low) / (action_high - action_low + 1e-8) - 1.0,
        actions,
    )

    return normalized_actions


def compute_action_tokens_from_actions(model, actions):
    """
    Inverse of the action tokens to continuous actions


    chunk_action_tokens = idxs.reshape(-1, model.action_dim)
    predicted_action_token_ids = chunk_action_tokens.cpu().numpy()
    discretized_actions = model.vocab_size - predicted_action_token_ids
    discretized_actions = np.clip(
        discretized_actions - 1, a_min=0, a_max=model.bin_centers.shape[0] - 1
    )
    # normalized_actions = model.bin_centers[discretized_actions]
    normalized_actions = np.asarray(
        [model.bin_centers[da] for da in discretized_actions]
    )  # [B, dim]
    normalized_actions = normalized_actions.reshape(-1, model.action_dim)

    # Unnormalize predicted actions
    actions = model._unnormalize_actions(normalized_actions, model.unnorm_key)
    actions = actions.reshape(idxs.shape)
    """

    B, T, D = actions.shape
    assert D == model.action_dim

    normalized_actions = _normalize_actions(model, actions, norm_key=model.unnorm_key)
    normalized_actions = normalized_actions.reshape(-1, D)
    bin_centers = model.bin_centers

    discretized_actions = []
    for dim in range(D):
        vals = normalized_actions[:, dim][:, None]  # (B*T, 1)
        dists = np.abs(vals - bin_centers[None, :])  # (B*T, n_bins)
        nearest_bins = np.argmin(dists, axis=1)  # (B*T,)
        discretized_actions.append(nearest_bins)

    discretized_actions = np.stack(discretized_actions, axis=1)  # (B*T, D)

    token_ids = model.vocab_size - 1 - discretized_actions
    token_ids = np.clip(
        token_ids,
        model.vocab_size - model.config.n_action_bins,
        model.vocab_size - 1,
    )

    token_ids = token_ids.reshape(B, T * D)
    return token_ids


def actor_forward(
    model,
    rl_batch,
    bc_batch=None,
    output_hidden_states=True,
    action_token_len=None,
    value_model=False,
    value_head_mode: str = "a",
    logits_processor=default_logits_processor,
    temperature: int = 1.0,
    top_k: int = -1,
    logits_processor_args: Optional[dict] = None,
    do_sample=False,  # unused
    return_bc_logits=False,
    logits_type="processed",
):
    """Forward pass for actor model with optional BC batch.

    Args:
        model: The model to use
        ...
        return_bc_logits: If True, return logits from BC forward pass

    Returns:
        output_dict: Dictionary with logprobs, entropy, values, etc.
        actions: Actions from BC forward (numpy array) or None
        bc_logits: Optional logits from BC forward (if return_bc_logits=True)
    """
    assert rl_batch["input_ids"].shape[0] == bc_batch["input_ids"].shape[0]
    has_bc_batch = bc_batch is not None
    if has_bc_batch:
        batch = {}
        for k in ["input_ids", "attention_mask", "pixel_values"]:
            batch[k] = torch.cat([rl_batch[k], bc_batch[k]], dim=0)
    else:
        batch = rl_batch

    output_dict = custom_forward(
        model,
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        pixel_values=batch["pixel_values"],
        output_hidden_states=output_hidden_states,
        action_token_len=action_token_len,
        value_model=value_model,
        value_head_mode=value_head_mode,
        temperature=temperature,
        top_k=top_k,
        logits_processor=logits_processor,
        logits_processor_args=logits_processor_args,
        has_bc_batch=has_bc_batch,
    )

    if has_bc_batch:
        bs = rl_batch["input_ids"].shape[0]
        rl_output_dict = {}
        for key, value in output_dict.items():
            rl_output_dict[key] = value[:bs]
        raw_bc_logits = output_dict["raw_logits"][bs:]
        processed_bc_logits = output_dict["processed_logits"][bs:]

        rl_output_dict["intermediate_logits"] = output_dict["intermediate_logits"][bs:]
        output_dict = rl_output_dict

    if return_bc_logits:
        if logits_type == "processed":
            return output_dict, processed_bc_logits
        elif logits_type == "raw":
            return output_dict, raw_bc_logits
    else:
        return output_dict


def custom_forward(
    model,
    input_ids,
    attention_mask,
    pixel_values,
    output_hidden_states=True,
    action_token_len=None,
    value_model=False,
    value_head_mode: str = "a",
    logits_processor=default_logits_processor,
    temperature: int = 1.0,
    top_k: int = -1,
    logits_processor_args: Optional[dict] = None,
    has_bc_batch: bool = False,
):
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        output_hidden_states=output_hidden_states,
    )

    logits_tensor = outputs.logits[
        :, -action_token_len - 1 : -1
    ]  # [B, action_dim, vocab_size]
    raw_logits = logits_tensor.clone()

    processed_logits_tensor = logits_tensor / temperature
    top_k = min(top_k, processed_logits_tensor.size(-1))  # Safety check
    if top_k > 0:
        logits_warper = TopKLogitsWarper(
            top_k
        )  # since here is logprob instead of logits, we use 0 instead of -inf
        processed_logits_tensor = logits_warper(None, processed_logits_tensor)

    # to handle bc batching case
    bs = input_ids.shape[0]
    if has_bc_batch:
        bs //= 2

    output_dict = logits_processor(
        processed_logits_tensor[:bs], **logits_processor_args
    )
    output_dict["raw_logits"] = raw_logits
    output_dict["intermediate_logits"] = processed_logits_tensor
    valid_start = model.vocab_size - model.config.n_action_bins
    valid_end = model.vocab_size
    processed_bc_logits = raw_logits[
        ..., valid_start:valid_end
    ]  # [B, act, n_action_bins]
    output_dict["processed_logits"] = processed_bc_logits

    if value_model:
        # NOTE: Here we subtract 1 because the input tokens do not include the EOS token.
        last_hidden_state = outputs.hidden_states[-1]  # [B, L, hidden_dim]
        if value_head_mode == "a0":
            hidden_features = last_hidden_state[
                :, -action_token_len - 1
            ]  # [batch_size, hidden_dim]
            values = model.value_head(hidden_features)  # [batch_size, 1]
        else:
            raise ValueError(f"Unknown value head mode: {value_head_mode}")
    else:
        values = None

    if values is not None:
        output_dict.update({"values": values})

    return output_dict


def prepare_observations_for_vla(
    simulator_type: str,
    model_name: str,
    raw_obs: dict,
    use_proprio: bool,
    max_length: int,
    processor: Any,
    precision: torch.dtype,
    device: torch.device = torch.device("cuda:0"),
):
    task_descriptions = [
        f"In: What action should the robot take to {t.lower()}?\nOut: "
        for t in raw_obs["task_descriptions"]
    ]

    if simulator_type == "libero":
        image_tensor = torch.stack(
            [
                value.clone().to(device).permute(2, 0, 1)
                for value in raw_obs["images_and_states"]["full_image"]
            ]
        )
    elif simulator_type == "maniskill":
        images = raw_obs["images"]
        image_tensor = images.to(device=device, dtype=precision)
    elif simulator_type == "robotwin":
        images = raw_obs["images"]
        image_tensor = images.to(device=device, dtype=precision)
    else:
        raise NotImplementedError

    proprio_states = None
    if use_proprio:
        proprio_keys = [
            key for key in raw_obs["images_and_states"] if "image" not in key
        ]
        proprio_states = {
            key: torch.stack(
                [val.to(device) for val in raw_obs["images_and_states"][key]]
            )
            for key in proprio_keys
        }

    # Add num_images dimension
    if image_tensor.ndim == 4:
        image_tensor = image_tensor.unsqueeze(1)
    assert image_tensor.ndim == 5

    if model_name == "openvla":
        processed_obs = processor(
            text=task_descriptions,
            images=image_tensor,
            padding="max_length",
            max_length=max_length,
        )
    elif model_name == "openvla_oft":
        images = {"images": image_tensor}
        processed_obs = processor(
            text=task_descriptions,
            images=images,
            proprio_states=proprio_states,
            padding="max_length",
            max_length=max_length,
        )

    processed_obs = processed_obs.to(device=device, dtype=precision)
    for key, value in processed_obs.items():
        processed_obs[key] = value.contiguous()

    return processed_obs


def prepare_observations(
    simulator_type: str,
    model_name: str,
    raw_obs: dict,
    use_proprio: bool,
    max_length: int,
    processor: Any,
    precision: torch.dtype,
    device: torch.device = torch.device("cuda:0"),
):
    if model_name == "openvla" or model_name == "openvla_oft":
        return prepare_observations_for_vla(
            simulator_type=simulator_type,
            model_name=model_name,
            raw_obs=raw_obs,
            use_proprio=use_proprio,
            max_length=max_length,
            processor=processor,
            precision=precision,
            device=device,
        )
    else:
        raise NotImplementedError
