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

import torch
import torch.nn.functional as F
from transformers.generation import TopKLogitsWarper
import numpy as np


# def default_logits_processor(logits, action_tokens, vocab_size, n_action_bins):
#     logits = logits.permute(0, 2, 1)  # [B, vocab-size, action-dim]

#     logits[:, : vocab_size - n_action_bins] = -torch.inf
#     logits[:, vocab_size:] = -torch.inf

#     logprobs = compute_logprobs_from_logits(logits=logits, target=action_tokens)

#     entropy = compute_entropy_from_logits(logits)

#     ret = {"logprobs": logprobs, "entropy": entropy}

#     return ret

def default_logits_processor(logits, action_tokens, vocab_size, n_action_bins):
    logits = logits.permute(0, 2, 1)  # [B, vocab-size, action-dim]
    
    # Define valid action token range
    valid_start = vocab_size - n_action_bins
    valid_end = vocab_size
    
    # VALIDATION: Check if action_tokens are in valid range
    out_of_bounds = (action_tokens < valid_start) | (action_tokens >= valid_end)
    if out_of_bounds.any():
        print(f"⚠️ WARNING: {out_of_bounds.sum()}/{action_tokens.numel()} action tokens out of bounds!")
        print(f"  Valid range: [{valid_start}, {valid_end})")
        print(f"  action_tokens range: [{action_tokens.min()}, {action_tokens.max()}]")
        print(f"  Out of bounds indices: {torch.where(out_of_bounds)}")
        
        # Show specific problematic values
        bad_tokens = action_tokens[out_of_bounds]
        print(f"  Bad token values: {bad_tokens[:10]}")  # Show first 10
    
    # Apply masking
    logits[:, :valid_start] = -torch.inf
    logits[:, valid_end:] = -torch.inf
    
    # VALIDATION: Check if valid region has at least some finite values
    valid_region = logits[:, valid_start:valid_end, :]
    all_inf_mask = torch.isinf(valid_region).all(dim=1)  # Check per [B, action-dim]
    if all_inf_mask.any():
        print(f"⚠️ WARNING: {all_inf_mask.sum()} positions have all -inf in valid region!")
        print(f"  This will cause NaN in softmax")
    
    logprobs = compute_logprobs_from_logits(logits=logits, target=action_tokens)
    
    # VALIDATION: Check for NaNs in output
    if torch.isnan(logprobs).any():
        print(f"⚠️ NaN detected in logprobs!")
        nan_mask = torch.isnan(logprobs)
        print(f"  Number of NaNs: {nan_mask.sum()}/{logprobs.numel()}")
        
        # Check corresponding action tokens
        print(f"  Action tokens at NaN positions: {action_tokens[nan_mask][:10]}")
    
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
    do_sample=False,
):
    actions = None
    if bc_batch:
        # RL + BC forward
        output_dict, actions = custom_forward_with_bc(
            model,
            rl_batch=rl_batch,
            bc_batch=bc_batch,
            output_hidden_states=output_hidden_states,
            action_token_len=action_token_len,
            value_model=value_model,
            value_head_mode=value_head_mode,
            logits_processor=logits_processor,
            logits_processor_args=logits_processor_args,
            temperature=temperature,
            top_k=top_k,
            do_sample=do_sample,
        )
        return output_dict, actions

    else:
        # RL-only forward
        output_dict = custom_forward(
            model,
            input_ids=rl_batch["input_ids"],
            attention_mask=rl_batch["attention_mask"],
            pixel_values=rl_batch["pixel_values"],
            output_hidden_states=output_hidden_states,
            action_token_len=action_token_len,
            value_model=value_model,
            value_head_mode=value_head_mode,
            temperature=temperature,
            top_k=top_k,
            logits_processor=logits_processor,
            logits_processor_args=logits_processor_args
        )
    return output_dict, actions

def custom_forward_with_bc(
    model,
    rl_batch,
    bc_batch,
    output_hidden_states=True,
    action_token_len=None,
    value_model=False,
    value_head_mode: str = "a",
    logits_processor=default_logits_processor,
    temperature: int = 1.0,
    top_k: int = -1,
    logits_processor_args: Optional[dict] = None,
    do_sample=False,
):
    ### POTENTIAL BUG: how does concat handle different length input_ids? 
    input_ids = torch.cat([rl_batch["input_ids"], bc_batch["input_ids"]], dim=0)
    attention_mask = torch.cat([rl_batch["attention_mask"], bc_batch["attention_mask"]], dim=0)
    pixel_values = torch.cat([rl_batch["pixel_values"], bc_batch["pixel_values"]], dim=0)

    bs_rl = rl_batch["input_ids"].shape[0]
    bs_bc = bc_batch["input_ids"].shape[0]

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        output_hidden_states=output_hidden_states,
    )
    logits = outputs.logits[:, -action_token_len - 1 : -1]  # [B, action_dim, vocab_size]

    rl_logits = logits[:bs_rl]
    bc_logits = logits[bs_rl:]
    last_hidden_state = outputs.hidden_states[-1]
    rl_hstates = last_hidden_state[:bs_rl]

    # rl processing
    processed_logits_tensor = rl_logits / temperature
    top_k = min(top_k, processed_logits_tensor.size(-1))  # Safety check
    if top_k > 0:
        logits_warper = TopKLogitsWarper(
            top_k
        )  # since here is logprob instead of logits, we use 0 instead of -inf
        processed_logits_tensor = logits_warper(None, processed_logits_tensor)

    output_dict = logits_processor(processed_logits_tensor, **logits_processor_args)

    if value_model:
        # NOTE: Here we subtract 1 because the input tokens do not include the EOS token.
        last_hidden_state = rl_hstates  # [B, L, hidden_dim]
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

    # bc processing
    n_prompt_tokens = bc_batch["input_ids"].shape[-1] - 1
    # Calculate number of patches (including proprio token and/or diffusion timestep embedding if present)
    n_patches = (
        model.vision_backbone.get_num_patches()
        * model.vision_backbone.get_num_images_in_input()
    )

    # Extract hidden states for action tokens
    logits_tensor = bc_logits[
        :,
        n_patches + n_prompt_tokens : n_patches
        + n_prompt_tokens
        + model.action_dim * model.num_action_chunks,
        :,
    ]  # [B, act, vocab_size + 64]

    logits_tensor[..., : model.vocab_size - model.config.n_action_bins] = -torch.inf
    logits_tensor[..., model.vocab_size :] = -torch.inf

    processed_logits_tensor = logits_tensor / temperature
    top_k = min(
        top_k, processed_logits_tensor.size(-1)
    )  # Safety check
    if top_k > 0:
        logits_warper = TopKLogitsWarper(
            top_k
        )  # since here is logprob instead of logits, we use 0 instead of -inf
        processed_logits_tensor = logits_warper(None, processed_logits_tensor)

    processed_logprob_tensor = F.log_softmax(
        processed_logits_tensor, dim=-1
    )  # [B, act, vocab_size + 64]

    if do_sample:
        probs_tensor = torch.exp(
            processed_logprob_tensor
        )  # [B, act, vocab_size + 64]
        probs_flat = probs_tensor.view(
            -1, processed_logprob_tensor.shape[-1]
        )  # [B * act, vocab_size + 64]

        sample_flat = torch.multinomial(
            probs_flat, num_samples=1, replacement=True
        )  # [B * act, 1]
        idxs = sample_flat.view(
            processed_logprob_tensor.shape[0], processed_logprob_tensor.shape[1]
        )  # [B, act]
    else:
        idxs = processed_logprob_tensor.argmax(dim=-1)  # [B, act]

    assert torch.all(
        idxs >= model.vocab_size - model.config.n_action_bins
    ) and torch.all(idxs < model.vocab_size)

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

    return output_dict, actions

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
):
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        output_hidden_states=output_hidden_states,
    )
    logits = output.logits[:, -action_token_len - 1 : -1]  # [B, action_dim, vocab_size]

    processed_logits_tensor = logits / temperature
    top_k = min(top_k, processed_logits_tensor.size(-1))  # Safety check
    if top_k > 0:
        logits_warper = TopKLogitsWarper(
            top_k
        )  # since here is logprob instead of logits, we use 0 instead of -inf
        processed_logits_tensor = logits_warper(None, processed_logits_tensor)

    output_dict = logits_processor(processed_logits_tensor, **logits_processor_args)

    if value_model:
        # NOTE: Here we subtract 1 because the input tokens do not include the EOS token.
        last_hidden_state = output.hidden_states[-1]  # [B, L, hidden_dim]
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
