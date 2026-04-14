# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""On-policy distillation (OPD): load a frozen teacher and score student actions."""

from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf

from rlinf.config import torch_dtype_from_precision
from rlinf.models import get_model
from rlinf.models.embodiment.model_utils import custom_forward


def _actor_model_cfg_without_lora(cfg: DictConfig):
    mcfg = OmegaConf.create(OmegaConf.to_container(cfg.actor.model, resolve=True))
    OmegaConf.set_struct(mcfg, False)
    mcfg.lora_path = None
    mcfg.lora_paths = None
    mcfg.is_lora = False
    return mcfg


def load_opd_teacher_model(cfg: DictConfig, rank: int) -> torch.nn.Module:
    """Load a general teacher: full HF dir, or PEFT adapter dir over a base checkpoint."""
    path = cfg.algorithm.get("opd_teacher_model_path", None)
    if not path:
        raise ValueError(
            "algorithm.opd_teacher_model_path is required when adv_type is embodied_opd"
        )
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    torch_dtype = torch_dtype_from_precision(cfg.actor.model.precision)
    base_path = cfg.algorithm.get(
        "opd_teacher_base_model_path", cfg.actor.checkpoint_load_path
    )

    has_adapter = os.path.isfile(
        os.path.join(path, "adapter_config.json")
    ) or os.path.isfile(os.path.join(path, "adapter_model.bin"))

    ws = dist.get_world_size() if dist.is_initialized() else 1
    mcfg = _actor_model_cfg_without_lora(cfg)

    if has_adapter:
        model = get_model(
            base_path,
            mcfg,
            load_role="opd_teacher_base_for_adapter",
            worker_rank=rank,
            worker_world_size=ws,
        )
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, path, is_trainable=False)
    else:
        model = get_model(
            path,
            mcfg,
            load_role="opd_teacher_standalone",
            worker_rank=rank,
            worker_world_size=ws,
        )

    model = model.to(device=device, dtype=torch_dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if rank == 0:
        print(
            f"[OPD] Loaded teacher (adapter_dir={has_adapter}) path={path} base={base_path if has_adapter else path}",
            flush=True,
        )
    return model


def compute_teacher_logprobs_one_batch(
    teacher_model: torch.nn.Module,
    student_core: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_values: torch.Tensor,
    action_tokens_flat: torch.Tensor,
    cfg: DictConfig,
) -> torch.Tensor:
    """Teacher log p(a|s) for fixed student actions; one microbatch.

    Args:
        action_tokens_flat: [B, action_dim * num_action_chunks], same device as vision/text.
    Returns:
        logprobs with same shape as student ``predict`` / ``prev_logprobs`` for this batch.
    """
    action_token_len = student_core.action_dim * student_core.num_action_chunks
    logits_processor_args = {
        "action_tokens": action_tokens_flat,
        "vocab_size": student_core.vocab_size,
        "n_action_bins": student_core.config.n_action_bins,
    }
    with torch.no_grad():
        od = custom_forward(
            teacher_model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            action_token_len=action_token_len,
            value_model=False,
            temperature=cfg.algorithm.sampling_params.temperature_train,
            top_k=cfg.algorithm.sampling_params.top_k,
            logits_processor_args=logits_processor_args,
            has_bc_batch=False,
        )
    return od["logprobs"].detach()


def compute_teacher_logprobs_for_rollout(
    teacher_model: torch.nn.Module,
    rollout_batch: dict,
    cfg: DictConfig,
    student_model: torch.nn.Module,
) -> torch.Tensor:
    """Teacher log p(a | s) for student actions a in the rollout; same shape as `prev_logprobs`."""
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    input_ids = rollout_batch["input_ids"]
    attention_mask = rollout_batch["attention_mask"]
    pixel_values = rollout_batch["pixel_values"]
    action_tokens = rollout_batch["action_tokens"]
    prev_lp = rollout_batch["prev_logprobs"]

    t_steps, b_env = input_ids.shape[:2]
    flat_n = t_steps * b_env
    mb = int(
        cfg.algorithm.get(
            "opd_teacher_micro_batch_size",
            cfg.actor.micro_batch_size,
        )
    )

    ids_f = input_ids.reshape(flat_n, *input_ids.shape[2:]).to(device)
    attn_f = attention_mask.reshape(flat_n, *attention_mask.shape[2:]).to(device)
    pix_f = pixel_values.reshape(flat_n, *pixel_values.shape[2:]).to(device)
    act_f = action_tokens.reshape(flat_n, *action_tokens.shape[2:]).to(device)
    act_f = act_f.reshape(
        flat_n, student_model.action_dim * student_model.num_action_chunks
    )

    # Stashing each microbatch on CPU forces a device sync every step and leaves the GPU
    # idle (looks like ~0% utilization for a long time). Default: keep tensors on GPU.
    stash_on_cpu = bool(cfg.algorithm.get("opd_teacher_stash_logprobs_on_cpu", False))
    outs: list[torch.Tensor] = []
    _main = (not dist.is_initialized()) or dist.get_rank() == 0
    _t0 = time.perf_counter()
    if _main:
        print(
            f"[OPD] Teacher logprobs: {flat_n} rows, microbatch={mb}, "
            f"stash_on_cpu={stash_on_cpu}",
            flush=True,
        )
    with torch.no_grad():
        for start in range(0, flat_n, mb):
            sl = slice(start, min(start + mb, flat_n))
            chunk = compute_teacher_logprobs_one_batch(
                teacher_model,
                student_model,
                ids_f[sl],
                attn_f[sl],
                pix_f[sl],
                act_f[sl],
                cfg,
            )
            outs.append(chunk.cpu() if stash_on_cpu else chunk)

    teacher_lp = torch.cat(outs, dim=0)
    teacher_lp = teacher_lp.reshape(prev_lp.shape)
    out = teacher_lp.to(dtype=prev_lp.dtype, device=prev_lp.device)
    if _main:
        print(
            f"[OPD] Teacher logprobs done in {time.perf_counter() - _t0:.1f}s",
            flush=True,
        )
    return out

