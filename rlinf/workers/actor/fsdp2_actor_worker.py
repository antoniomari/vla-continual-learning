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

import gc
import json
import math
import os
import time
from contextlib import nullcontext
from itertools import cycle

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from peft import get_peft_model_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader
from tqdm import tqdm

import rlinf.algorithms  # noqa: F401
from rlinf.algorithms.registry import actor_loss, calculate_adv_and_returns
from rlinf.algorithms.utils import preprocess_advantages_inputs, preprocess_loss_inputs
from rlinf.custom.libero_trajectory_dataset import LiberoSFTDataset
from rlinf.custom.loss import (
    behavior_cloning_ce_loss,
    behavior_cloning_loss_with_reference_logits,
)
from rlinf.hybrid_engines.fsdp.fsdp2_model_manager import (
    FSDP2ModelManager,
)
from rlinf.models import get_model
from rlinf.models.embodiment.model_utils import (
    actor_forward,
    compute_action_tokens_from_actions,
    custom_forward,
)
from rlinf.scheduler import Cluster, Worker
from rlinf.workers.actor.opd_teacher import (
    compute_teacher_logprobs_for_rollout,
    load_opd_teacher_model,
)
from rlinf.utils.data_iter_utils import get_iterator_k_split
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import (
    append_to_dict,
    compute_loss_mask,
    compute_rollout_metrics,
    compute_split_num,
    expand_loss_mask_to_match_logprob_tokens,
)
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.utils.runner_utils import cfg_show_progress_bar


class EmbodiedFSDP2Actor(FSDP2ModelManager, Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        super().__init__(cfg.actor)

        self.cfg = cfg
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        self.device = torch.cuda.current_device()
        world_size = self._world_size
        self.device_mesh = init_device_mesh(
            "cuda", mesh_shape=(world_size,), mesh_dim_names=["fsdp"]
        )

        self._env_group_name = cfg.env.group_name
        self._rollout_group_name = cfg.rollout.group_name
        self._component_placement = HybridComponentPlacement(cfg, Cluster())
        self._weight_dst_rank_in_rollout = self._rank
        if self._weight_dst_rank_in_rollout >= self._component_placement.get_world_size(
            "rollout"
        ):
            self._weight_dst_rank_in_rollout = None

        self._obs_queue_name = cfg.env.channel.queue_name
        self._action_queue_name = cfg.rollout.channel.queue_name
        self._replay_buffer_name = cfg.actor.channel.queue_name
        # stage_num: default to 2, use for pipeline rollout process
        self.stage_num = cfg.rollout.pipeline_stage_num

        self.channel = self.connect_channel(cfg.actor.channel.name)
        self.channel.create_queue(
            cfg.actor.channel.queue_name, maxsize=cfg.actor.channel.queue_size
        )

        # initialize sft buffer
        self.use_experience_replay = cfg.algorithm.use_experience_replay
        self._opd_bc_steps = cfg.algorithm.get("opd_bc_steps", 0)
        self._opd_teacher_model = None
        self.use_reference_logits_bc = cfg.algorithm.get(
            "use_reference_logits_bc", False
        )
        self.use_cached_bc_logits = cfg.algorithm.get("use_cached_bc_logits", False)
        self.logits_type = cfg.algorithm.get("logits_type", "processed")
        if self.logits_type not in ["processed", "raw"]:
            raise NotImplementedError(
                f"returning logits type {self.logits_type} is not implemented"
            )

        if self.use_experience_replay or self._opd_bc_steps > 0:
            self._init_sft_replay_buffer(use_cached_logits=self.use_cached_bc_logits)

        self._preallocated_memory = None

        # Track training step for logit comparison checks
        self.training_step_count = 0
        # Enable logit comparison check (for testing).
        exp_name = cfg.runner.logger.get("experiment_name", "")
        self.enable_logit_check = exp_name.endswith("_oom")
        print(
            f"Enable logit comparison check: {self.enable_logit_check}. experiment name: {exp_name}"
        )
        # if not self.enable_logit_check:
        #     exit()

    def _init_sft_replay_buffer(self, use_cached_logits=False):
        dataset_path = os.environ.get("LIBERO_REPO_PATH")
        if self._rank == 0:
            print(f"Initializing SFT dataset on rank {self._rank}")

        demos_per_task = self.cfg.algorithm.get("opd_sft_demos_per_task", 1)
        self.sft_dataset = LiberoSFTDataset(
            cfg=self.cfg,
            root_dir=dataset_path,
            demos_per_task=demos_per_task,
            rank=self._rank,
            world_size=self._world_size,
            use_cached_logits=use_cached_logits,
            logits_type=self.logits_type if self.use_reference_logits_bc else "",
            use_preprocessed=True,
        )

        sft_batch = (
            int(self.cfg.algorithm.get("opd_bc_batch_size", self.cfg.actor.micro_batch_size))
            if self._opd_bc_steps > 0
            else int(self.cfg.actor.micro_batch_size)
        )

        self.sft_dataloader = cycle(
            DataLoader(
                self.sft_dataset,
                batch_size=sft_batch,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
                drop_last=True,
            )
        )

        self.sft_iterator = iter(self.sft_dataloader)
        if self._rank == 0:
            print(f"SFT dataset initialized: {len(self.sft_dataset)} samples")

    def init_worker(self):
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        else:
            if torch.distributed.get_backend() != "nccl":
                # Destroy existing process group and reinitialize with NCCL
                torch.distributed.destroy_process_group()
                torch.distributed.init_process_group(backend="nccl")

        self.setup_model_and_optimizer()

        # This default to using the base model without LoRA weights
        # Store reference model state dict if using reference logits BC loss
        self.ref_policy_state_dict = None
        use_ref_logits_bc = self.cfg.algorithm.get("use_reference_logits_bc", False)
        if (
            use_ref_logits_bc
            and self.use_experience_replay
            and not self.use_cached_bc_logits
        ):
            # Check if we should load from a checkpoint path
            reference_model_path = self.cfg.actor.get("reference_model_path", None)

            # Keep existing logic (we may reuse reference weights for non-LoRA cases)
            if reference_model_path is not None:
                if self._rank == 0:
                    print(
                        f"Loading reference model from checkpoint: {reference_model_path}"
                    )
                ref_state_dict = torch.load(reference_model_path, map_location="cpu")
                if isinstance(ref_state_dict, dict) and "model" in ref_state_dict:
                    ref_state_dict = ref_state_dict["model"]
                self.ref_policy_state_dict = ref_state_dict

        if self.cfg.actor.get("enable_offload", False):
            self.offload_fsdp_optimizer()
            self.offload_fsdp_param_and_grad()
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

    def preallocate_memory(self):
        """Preallocate GPU memory to prevent OOM during rollout."""
        preallocate_gb = self.cfg.actor.get("preallocate", 0)
        if preallocate_gb == 0:
            return
        # Convert GB to bytes
        size_bytes = int(float(preallocate_gb) * 1024 * 1024 * 1024)
        bytes_per_float32 = 4
        num_elements = size_bytes // bytes_per_float32
        if self._rank == 0:
            print(
                f"[INFO] Preallocating {preallocate_gb} GB of GPU memory on each actor worker..."
            )
        try:
            self._preallocated_memory = torch.empty(
                num_elements, dtype=torch.float32, device=self.device
            )
            torch.cuda.synchronize()
            if self._rank == 0:
                print(
                    f"[INFO] Successfully preallocated {preallocate_gb} GB on GPU {self.device}"
                )
        except RuntimeError as e:
            if self._rank == 0:
                print(f"[ERROR] Failed to preallocate memory: {e}")
            raise

    def _deallocate_preallocated_memory(self):
        """Deallocate preallocated memory before training."""
        if self._preallocated_memory is not None:
            if self._rank == 0:
                size_gb = self._preallocated_memory.numel() * 4 / (1024**3)
                print(f"[INFO] Deallocating {size_gb:.2f} GB of preallocated memory...")
            del self._preallocated_memory
            self._preallocated_memory = None
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if self._rank == 0:
                print("[INFO] Preallocated memory deallocated")
        else:
            if self._rank == 0:
                print("[WARNING] No memory has been preallocated. Nothing to free.")

    def model_provider_func(self):
        model = get_model(
            self.cfg.actor.checkpoint_load_path,
            self.cfg.actor.model,
            load_role="actor_train_fsdp_init",
            worker_rank=self._rank,
            worker_world_size=self._world_size,
        )
        if model is not None:
            return model
        return super().model_provider_func()

    def sync_model_to_rollout(self):
        # 1. Offload First! (Force clear GPU to prevent OOM race with Rollout)
        if self.cfg.actor.get("enable_offload", False):
            self.offload_fsdp_optimizer()
            self.offload_fsdp_param_and_grad()
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

        # 2. Get State Dict (Now guaranteed to hit the CPU path because model is on CPU)
        state_dict = self.get_model_state_dict()

        # 3. Send state dict to rollout worker (Rollout can only grab AFTER we offloaded)
        if self._weight_dst_rank_in_rollout is not None:
            self.send(
                state_dict, self._rollout_group_name, self._weight_dst_rank_in_rollout
            )

        del state_dict
        gc.collect()

    async def recv_rollout_batch(self):
        _log = self._rank == 0 and self.cfg.runner.get("log_step_phase_timings", True)
        _t_recv = time.perf_counter()
        send_num = self._component_placement.get_world_size("rollout") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        self.rollout_batch = {}
        recv_list = []
        for i in range(split_num):
            recv_list.append(
                await self.channel.get(
                    queue_name=self._replay_buffer_name, async_op=True
                ).async_wait()
            )

        # shape [num_chunk, bsz, chunk_size], cat dim 1
        for key in recv_list[0].keys():
            if "env_info/" not in key:
                self.rollout_batch[key] = torch.cat(
                    [recv_list[i][key] for i in range(split_num)], dim=1
                )
            else:
                self.rollout_batch[key] = torch.cat(
                    [recv_list[i][key] for i in range(split_num)], dim=0
                )

        self.rollout_batch = self._process_received_rollout_batch(self.rollout_batch)
        if _log:
            has_teacher = "teacher_logprobs" in self.rollout_batch
            print(
                f"[Actor r0] recv_rollout_batch: wall={time.perf_counter() - _t_recv:.2f}s "
                f"split_num={split_num} keys={len(self.rollout_batch)} "
                f"input_ids={tuple(self.rollout_batch['input_ids'].shape)} "
                f"teacher_logprobs_in_batch={has_teacher}",
                flush=True,
            )

    def _process_received_rollout_batch(self, rollout_batch):
        """
        original shape: [rollout_epoch x n_chunk_steps, bsz, num_action_chunks, ...]
        target shape: [n_chunk_steps, rollout_epoch x bsz, num_action_chunks, ...]
        """
        rollout_epoch = self.cfg.algorithm.rollout_epoch
        for key, value in rollout_batch.items():
            new_value = value.reshape(
                rollout_epoch, -1, *value.shape[1:]
            )  # [rollout_epoch, n_chunk_step, bsz, ...]
            new_value = new_value.transpose(
                0, 1
            )  # [n_chunk_step, rollout_epoch, bsz, ...]
            new_value = new_value.reshape(new_value.shape[0], -1, *new_value.shape[3:])
            rollout_batch[key] = new_value

        if (
            not self.cfg.env.train.auto_reset
            and not self.cfg.env.train.ignore_terminations
        ):
            dones = rollout_batch[
                "dones"
            ]  # [n_chunk_step, rollout_epoch x bsz, num_action_chunks]
            loss_mask, loss_mask_sum = compute_loss_mask(dones)

            rollout_batch["loss_mask"] = loss_mask
            rollout_batch["loss_mask_sum"] = loss_mask_sum

        # filter data by rewards
        if self.cfg.algorithm.get("filter_rewards", False):
            rewards = rollout_batch[
                "rewards"
            ]  # [n_chunk_step, batch, num_action_chunks]
            if self.rollout_batch.get("loss_mask", None) is not None:
                rewards = rewards * rollout_batch["loss_mask"]
            n_chunk_step, batch_size, num_action_chunks = rewards.shape

            group_size = self.cfg.algorithm.group_size
            assert batch_size % group_size == 0, (
                f"batch {batch_size} not divisible by group_size {group_size}"
            )
            n_prompts = batch_size // group_size

            # calculate rewards by prompt
            rewards = rewards.transpose(
                0, 1
            )  # [batch, n_chunk_step, num_action_chunks]
            rewards = rewards.reshape(rewards.shape[0], -1)  # [batch, n_step]
            reward_matrix = rewards.reshape(
                n_prompts, group_size, rewards.shape[-1]
            )  # [n_prompts, group_size, n_step]
            reward_matrix = reward_matrix.sum(dim=-1)  # [n_prompts, group_size]
            mean_reward_in_group = reward_matrix.mean(dim=1)  # [n_prompts]

            # mask
            reward_filter_mask = (
                mean_reward_in_group >= self.cfg.algorithm.rewards_lower_bound
            ) & (
                mean_reward_in_group <= self.cfg.algorithm.rewards_upper_bound
            )  # [n_prompts]

            # extend mask dimension
            reward_filter_mask = reward_filter_mask.repeat_interleave(
                group_size
            )  # [batch]
            reward_filter_mask = (
                reward_filter_mask.unsqueeze(0).expand(n_chunk_step, -1).unsqueeze(-1)
            )  # [n_chunk_step, batch, 1]

            # update loss_mask
            if self.rollout_batch.get("loss_mask", None) is not None:
                rollout_batch["loss_mask"] = (
                    reward_filter_mask & self.rollout_batch["loss_mask"]
                )
            else:
                rollout_batch["loss_mask"] = reward_filter_mask

        return rollout_batch

    def compute_logprobs(self):
        self.model.eval()
        self.rollout_batch["logprob"] = self.rollout_batch["prev_logprobs"]

    def compute_advantages_and_returns(self):
        _log = self._rank == 0 and self.cfg.runner.get("log_step_phase_timings", True)
        _t0 = time.perf_counter()
        stage_num = self.cfg.rollout.pipeline_stage_num
        env_world_size = self._component_placement.get_world_size("env")
        actor_world_size = self._component_placement.get_world_size("actor")
        num_group_envs_for_train = (
            self.cfg.algorithm.num_group_envs
            * stage_num
            * env_world_size
            // actor_world_size
        )

        rewards = self.rollout_batch["rewards"]
        dones = self.rollout_batch["dones"]
        loss_mask = self.rollout_batch.get("loss_mask", None)

        if self.cfg.algorithm.adv_type == "embodied_opd":
            loss_mask_sum = self.rollout_batch.get("loss_mask_sum", None)
            if loss_mask is not None:
                loss_mask, loss_mask_sum = expand_loss_mask_to_match_logprob_tokens(
                    loss_mask,
                    loss_mask_sum,
                    self.rollout_batch["prev_logprobs"],
                )
                self.rollout_batch["loss_mask"] = loss_mask
                self.rollout_batch["loss_mask_sum"] = loss_mask_sum
            student_core = (
                self.model.module if hasattr(self.model, "module") else self.model
            )
            pre_lp = self.rollout_batch.get("teacher_logprobs", None)
            _t_teacher = time.perf_counter()
            if pre_lp is not None:
                teacher_lp = pre_lp
                if _log:
                    print(
                        "[Actor r0] OPD compute_adv: using teacher_logprobs from rollout "
                        f"(device={teacher_lp.device})",
                        flush=True,
                    )
            else:
                if _log:
                    print(
                        "[Actor r0] OPD compute_adv: no rollout teacher_logprobs — "
                        "scoring on actor (slow)",
                        flush=True,
                    )
                if self._opd_teacher_model is None:
                    self._opd_teacher_model = load_opd_teacher_model(
                        self.cfg, self._rank
                    )
                teacher_lp = compute_teacher_logprobs_for_rollout(
                    self._opd_teacher_model,
                    self.rollout_batch,
                    self.cfg,
                    student_core,
                )
            if _log:
                print(
                    f"[Actor r0] OPD compute_adv: teacher_logprobs_ready "
                    f"wall={time.perf_counter() - _t_teacher:.2f}s",
                    flush=True,
                )
            kwargs = {
                "adv_type": "embodied_opd",
                "rewards": rewards,
                "dones": dones,
                "teacher_logprobs": teacher_lp,
                "student_logprobs": self.rollout_batch["prev_logprobs"],
                "loss_mask": loss_mask,
                "reward_type": self.cfg.algorithm.reward_type,
            }
        else:
            kwargs = {
                "adv_type": self.cfg.algorithm.adv_type,
                "rewards": rewards,
                "dones": dones,
                "normalize_advantages": self.cfg.algorithm.get(
                    "normalize_advantages", True
                ),
                "values": self.rollout_batch.get("prev_values", None),
                "gamma": self.cfg.algorithm.get("gamma", 1),
                "gae_lambda": self.cfg.algorithm.get("gae_lambda", 1),
                "num_group_envs": num_group_envs_for_train,
                "group_size": self.cfg.algorithm.get("group_size", 8),
                "reward_type": self.cfg.algorithm.reward_type,
                "loss_mask": loss_mask,
                "rollout_epoch": self.cfg.algorithm.get("rollout_epoch", 1),
            }
        kwargs = preprocess_advantages_inputs(**kwargs)
        _t_adv = time.perf_counter()
        advantages, returns = calculate_adv_and_returns(**kwargs)
        if _log:
            print(
                f"[Actor r0] compute_adv: calculate_adv_and_returns "
                f"wall={time.perf_counter() - _t_adv:.2f}s",
                flush=True,
            )

        self.rollout_batch.update({"advantages": advantages, "returns": returns})
        self.rollout_batch.pop("teacher_logprobs", None)
        _t_rm = time.perf_counter()
        rollout_metrics = compute_rollout_metrics(self.rollout_batch)
        if _log:
            print(
                f"[Actor r0] compute_adv: compute_rollout_metrics "
                f"wall={time.perf_counter() - _t_rm:.2f}s "
                f"total_wall={time.perf_counter() - _t0:.2f}s",
                flush=True,
            )
        return rollout_metrics

    def _get_peft_base_model(self):
        """
        Return the underlying PEFT model if present (handle FSDP wrapper), else None.
        """
        try:
            from peft import PeftModel
        except ImportError:
            return None

        model_to_check = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        if isinstance(model_to_check, PeftModel):
            return model_to_check
        return None

    def _compute_reference_bc_logits(self, bc_batch):
        """
        Compute reference logits by disabling LoRA adapters (no weight swapping).
        Falls back to regular forward if no PEFT model is present.
        """
        peft_model = self._get_peft_base_model()
        assert peft_model is not None and hasattr(peft_model, "disable_adapter"), (
            "PEFT model is not present or does not have disable_adapter method"
        )
        adapter_ctx = peft_model.disable_adapter()

        with torch.no_grad():
            with adapter_ctx:
                _, reference_bc_logits = custom_forward(
                    self.model,
                    input_ids=bc_batch["input_ids"],
                    attention_mask=bc_batch["attention_mask"],
                    pixel_values=bc_batch["pixel_values"],
                    temperature=self.cfg.algorithm.sampling_params.temperature_train,
                    top_k=self.cfg.algorithm.sampling_params.top_k,
                    do_sample=False,
                    return_logits=True,
                )
        return reference_bc_logits

    def run_training(self):
        _log = self._rank == 0 and self.cfg.runner.get("log_step_phase_timings", True)
        _rt0 = time.perf_counter()
        self._deallocate_preallocated_memory()

        if self.cfg.actor.get("enable_offload", False):
            self.load_fsdp_param_and_grad(self.device)
            self.load_fsdp_optimizer(self.device)

        self.model.train()
        self.optimizer.zero_grad()
        rollout_size = (
            self.rollout_batch["input_ids"].shape[0]
            * self.rollout_batch["input_ids"].shape[1]
        )
        shuffle_id = torch.randperm(rollout_size)

        for key, value in self.rollout_batch.items():
            if self.cfg.runner.get("log_training_tensor_shapes", False):
                self.log_on_first_rank(f"run training, {key}: {value.shape}")

        with torch.no_grad():
            for key, value in self.rollout_batch.items():
                if key in ["dones", "prev_values"]:
                    value = value[:-1]
                if "env_info" in key:
                    continue
                value = value.reshape(rollout_size, *value.shape[2:])
                self.rollout_batch[key] = value[shuffle_id]

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )

        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        rollout_size = self.rollout_batch["input_ids"].size(0)
        batch_size_per_rank = self.cfg.actor.global_batch_size // self._world_size
        assert rollout_size % batch_size_per_rank == 0, (
            f"{rollout_size} is not divisible by {batch_size_per_rank}"
        )
        rollout_dataloader_iter = get_iterator_k_split(
            self.rollout_batch,
            rollout_size // batch_size_per_rank,
        )
        bc_coeff = self.cfg.algorithm.get("bc_coeff", 0.0)

        metrics = {}
        num_batches = 0
        _outer_total = rollout_size // batch_size_per_rank
        if _log:
            torch.cuda.synchronize()
            print(
                f"[Actor r0] run_training: prep+shuffle_wall={time.perf_counter() - _rt0:.2f}s "
                f"flat_rows={rollout_size} batch_per_rank={batch_size_per_rank} "
                f"outer_batches_this_rank={_outer_total} "
                f"micro_batch={self.cfg.actor.micro_batch_size} "
                f"grad_accum={self.gradient_accumulation}",
                flush=True,
            )
        _rt_loop = time.perf_counter()
        _pbar = cfg_show_progress_bar(self.cfg)
        for _, train_global_batch in tqdm(
            enumerate(rollout_dataloader_iter),
            desc="get loss and metrics",
            disable=not _pbar or self._rank != 0,
        ):
            num_batches += 1
            if _log and num_batches == 1:
                _t_first = time.perf_counter()
            # split batch into micro_batches
            train_global_batch_size = train_global_batch["input_ids"].shape[0]
            assert (
                train_global_batch_size
                == self.cfg.actor.global_batch_size
                // torch.distributed.get_world_size()
            )
            assert train_global_batch_size % self.cfg.actor.micro_batch_size == 0, (
                f"{train_global_batch_size=}, {self.cfg.actor.micro_batch_size}"
            )
            train_micro_batch = get_iterator_k_split(
                train_global_batch,
                train_global_batch_size // self.cfg.actor.micro_batch_size,
            )

            self.optimizer.zero_grad()
            use_ref_logits_bc = self.cfg.algorithm.get("use_reference_logits_bc", False)

            for data_idx, data in enumerate(train_micro_batch):
                for k, v in data.items():
                    data[k] = v.to(f"cuda:{int(os.environ['LOCAL_RANK'])}")

                data = self.model.preprocess_for_train(data)
                action_token_len = self.model.action_dim * self.model.num_action_chunks

                logits_processor_args = {
                    "action_tokens": data["action_tokens"],
                    "vocab_size": self.model.vocab_size,
                    "n_action_bins": self.model.config.n_action_bins,
                }

                sampling_params = OmegaConf.to_container(
                    self.cfg.algorithm.sampling_params, resolve=True
                )

                bc_batch = None
                if self.use_experience_replay:
                    bc_batch = next(self.sft_iterator)
                    for k, v in bc_batch.items():
                        bc_batch[k] = v.to(f"cuda:{int(os.environ['LOCAL_RANK'])}")

                return_bc_logits = use_ref_logits_bc and bc_batch is not None
                forward_result = actor_forward(
                    self.model,
                    rl_batch=data,
                    bc_batch=bc_batch,
                    action_token_len=action_token_len,
                    value_model=True
                    if self.cfg.algorithm.adv_type == "embodied_gae"
                    else False,
                    value_head_mode=self.cfg.actor.model.get("vh_mode", None),
                    temperature=self.cfg.algorithm.sampling_params.temperature_train,
                    top_k=self.cfg.algorithm.sampling_params.top_k,
                    logits_processor_args=logits_processor_args,
                    do_sample=not sampling_params["use_greedy"],
                    return_bc_logits=return_bc_logits,
                    logits_type=self.logits_type,
                )

                if return_bc_logits:
                    output_dict, current_bc_logits = forward_result
                else:
                    output_dict = forward_result

                kwargs = {
                    "loss_type": self.cfg.algorithm.loss_type,
                    "logprob_type": self.cfg.algorithm.logprob_type,
                    "entropy_type": self.cfg.algorithm.entropy_type,
                    "single_action_dim": self.model.action_dim,
                    "logprobs": output_dict["logprobs"],
                    "entropy": output_dict["entropy"],
                    "values": output_dict.get("values", None),
                    "old_logprobs": data["prev_logprobs"],
                    "advantages": data["advantages"],
                    "returns": data["returns"],
                    "prev_values": data["prev_values"],
                    "clip_ratio_high": self.cfg.algorithm.clip_ratio_high,
                    "clip_ratio_low": self.cfg.algorithm.clip_ratio_low,
                    "value_clip": self.cfg.algorithm.get("value_clip", None),
                    "huber_delta": self.cfg.algorithm.get("huber_delta", None),
                    "entropy_bonus": self.cfg.algorithm.entropy_bonus,
                    "loss_mask": data.get("loss_mask", None),
                    "loss_mask_sum": data.get("loss_mask_sum", None),
                    "max_episode_steps": self.cfg.env.train.max_episode_steps,
                }

                kwargs = preprocess_loss_inputs(**kwargs)
                rl_loss, metrics_data = actor_loss(**kwargs)

                bc_loss, bc_metrics_data = 0.0, {}
                if self.use_experience_replay:
                    if use_ref_logits_bc:
                        if self.use_cached_bc_logits:
                            if self.logits_type == "raw":
                                reference_bc_logits = bc_batch["raw_action_logits"]
                            elif self.logits_type == "processed":
                                reference_bc_logits = bc_batch[
                                    "processed_action_logits"
                                ]
                        else:
                            reference_bc_logits = self._compute_reference_bc_logits(
                                bc_batch
                            )

                        if self.enable_logit_check:
                            logit_diff = (current_bc_logits - reference_bc_logits).abs()
                            max_diff = logit_diff.max().item()
                            mean_diff = logit_diff.mean().item()

                            if self._rank == 0:
                                if self.training_step_count == 0:
                                    if max_diff < 1e-5:
                                        print(
                                            f"✓ Step {self.training_step_count}: Reference and current logits match (max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e})"
                                        )
                                    else:
                                        print(
                                            f"✗ Step {self.training_step_count}: WARNING - Reference and current logits differ (max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e})"
                                        )
                                else:
                                    if max_diff > 1e-5:
                                        print(
                                            f"✓ Step {self.training_step_count}: Reference and current logits differ as expected (max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e})"
                                        )
                                    else:
                                        print(
                                            f"✗ Step {self.training_step_count}: WARNING - Reference and current logits still match (max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e})"
                                        )

                        kwargs = {
                            "current_logits": current_bc_logits,
                            "reference_logits": reference_bc_logits,
                            "bc_coeff": bc_coeff,
                        }
                        bc_loss, bc_metrics_data = (
                            behavior_cloning_loss_with_reference_logits(**kwargs)
                        )
                    else:
                        kwargs = {
                            "intermediate_logits": output_dict["intermediate_logits"],
                            "expert_actions_tokens": torch.tensor(
                                compute_action_tokens_from_actions(
                                    self.model, bc_batch["actions"]
                                ),
                                device=f"cuda:{int(os.environ['LOCAL_RANK'])}",
                            ),
                            "bc_coeff": bc_coeff,
                            "vocab_size": self.model.vocab_size,
                            "n_action_bins": self.model.config.n_action_bins,
                        }
                        bc_loss, bc_metrics_data = behavior_cloning_ce_loss(**kwargs)

                loss = rl_loss + bc_loss
                loss /= self.gradient_accumulation
                loss.backward()

                metrics_data["rl/loss"] = rl_loss.detach().item()
                metrics_data.update(bc_metrics_data)
                metrics_data["total/loss"] = loss.detach().item()
                append_to_dict(metrics, metrics_data)

            torch.cuda.empty_cache()

            log_param_delta = self.cfg.algorithm.get("log_param_delta", True)
            is_lora = self.cfg.actor.model.get("is_lora", False)
            log_all_weights = self.cfg.algorithm.get(
                "log_param_delta_all_trainable", False
            )
            param_delta_snap = None
            param_delta_lora_only = True
            if log_param_delta and (is_lora or log_all_weights):
                from rlinf.algorithms.ewc import snapshot_params_for_delta

                param_delta_lora_only = is_lora and not log_all_weights
                param_delta_snap = snapshot_params_for_delta(
                    self.model, lora_only=param_delta_lora_only
                )

            grad_norm = self.model.clip_grad_norm_(
                max_norm=self.cfg.actor.optim.clip_grad
            )
            self.optimizer.step()
            if _log and num_batches == 1:
                torch.cuda.synchronize()
                print(
                    f"[Actor r0] run_training: first_outer_batch_wall="
                    f"{time.perf_counter() - _t_first:.2f}s (incl. sync)",
                    flush=True,
                )

            # Increment training step counter
            self.training_step_count += 1

            self.optimizer.zero_grad()
            data = {
                "actor/grad_norm": grad_norm.detach().item(),
                "actor/lr": self.optimizer.param_groups[0]["lr"],
            }
            if self.cfg.algorithm.adv_type == "embodied_gae":
                data["critic/lr"] = self.optimizer.param_groups[1]["lr"]
            if param_delta_snap:
                from rlinf.algorithms.ewc import global_param_delta_l2

                pd = global_param_delta_l2(
                    self.model,
                    param_delta_snap,
                    param_delta_lora_only,
                    device=torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}"),
                )
                if math.isfinite(pd):
                    data["actor/param_delta_l2"] = pd
            append_to_dict(metrics, data)

        mean_metric_dict = {key: np.mean(value) for key, value in metrics.items()}
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )

        self.optimizer.zero_grad()
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()

        if _log:
            torch.cuda.synchronize()
            print(
                f"[Actor r0] run_training: loop_wall={time.perf_counter() - _rt_loop:.2f}s "
                f"outer_batches={num_batches} total_wall_incl_barrier="
                f"{time.perf_counter() - _rt0:.2f}s",
                flush=True,
            )

        return mean_metric_dict

    def _opd_bc_optimizer_lrs(self, saved_lrs: list) -> list:
        """Per-param-group LRs for OPD teacher BC warmup (restored after warmup)."""
        o = self.cfg.actor.optim
        t_main = float(o.get("opd_teacher_lr", o.lr))
        vh = self.cfg.actor.model.get("vh_mode", "none")
        n = len(saved_lrs)
        if n == 1:
            return [t_main]
        if n == 2 and vh in ("a", "a0", "a6"):
            t_vh = float(o.get("opd_teacher_value_lr", o.value_lr))
            return [t_main, t_vh]
        base = float(o.lr)
        scale = t_main / base if base > 0 else 1.0
        return [s * scale for s in saved_lrs]

    def run_opd_bc_warmup(self, num_steps: int):
        """Behavior-cloning warmup on expert data before OPD RL (teacher snapshot saved separately)."""
        if self.cfg.actor.model.get("model_name") == "simple_cnn":
            raise NotImplementedError(
                "opd_bc_steps / run_opd_bc_warmup is only supported for VLA models"
            )
        self._deallocate_preallocated_memory()
        if self.cfg.actor.get("enable_offload", False):
            self.load_fsdp_param_and_grad(self.device)
            self.load_fsdp_optimizer(self.device)

        self.model.train()
        device = f"cuda:{int(os.environ['LOCAL_RANK'])}"
        saved_lrs = [float(pg["lr"]) for pg in self.optimizer.param_groups]
        teacher_lrs = self._opd_bc_optimizer_lrs(saved_lrs)
        for pg, lr in zip(self.optimizer.param_groups, teacher_lrs):
            pg["lr"] = lr
        bc_bs = int(
            self.cfg.algorithm.get("opd_bc_batch_size", self.cfg.actor.micro_batch_size)
        )
        _bc_gbs = self.cfg.algorithm.get("opd_bc_global_batch_size")
        gbs = (
            int(_bc_gbs)
            if _bc_gbs is not None
            else int(self.cfg.actor.global_batch_size)
        )
        ws = int(self._world_size)
        if gbs % (bc_bs * ws) != 0:
            raise ValueError(
                f"OPD BC: opd_bc_global_batch_size or actor.global_batch_size ({gbs}) "
                f"must be divisible by opd_bc_batch_size ({bc_bs}) * world_size ({ws})"
            )
        grad_acc = gbs // bc_bs // ws
        if grad_acc < 1:
            raise ValueError(
                f"OPD BC: per-step global batch ({gbs}) is smaller than one microbatch "
                f"across ranks ({bc_bs} * {ws} = {bc_bs * ws}); lower opd_bc_batch_size "
                f"or raise opd_bc_global_batch_size"
            )
        per_step_metrics: list = []
        try:
            _pbar = cfg_show_progress_bar(self.cfg)
            for _ in tqdm(
                range(num_steps),
                desc=f"OPD BC warmup (rank={self._rank})",
                disable=not _pbar or self._rank != 0,
            ):
                step_metrics: dict = {}
                self.optimizer.zero_grad()
                for _ in range(grad_acc):
                    bc_batch = next(self.sft_iterator)
                    bc_batch = {k: v.to(device) for k, v in bc_batch.items()}
                    rl_batch = {
                        k: v.clone() if torch.is_tensor(v) else v
                        for k, v in bc_batch.items()
                    }
                    rl_batch["action_tokens"] = torch.tensor(
                        compute_action_tokens_from_actions(
                            self.model, bc_batch["actions"]
                        ),
                        device=device,
                    )
                    rl_batch = self.model.preprocess_for_train(rl_batch)
                    action_token_len = self.model.action_dim * self.model.num_action_chunks
                    sampling_params = OmegaConf.to_container(
                        self.cfg.algorithm.sampling_params, resolve=True
                    )
                    logits_processor_args = {
                        "action_tokens": rl_batch["action_tokens"],
                        "vocab_size": self.model.vocab_size,
                        "n_action_bins": self.model.config.n_action_bins,
                    }
                    output_dict = actor_forward(
                        self.model,
                        rl_batch=rl_batch,
                        bc_batch=bc_batch,
                        action_token_len=action_token_len,
                        value_model=False,
                        value_head_mode=self.cfg.actor.model.get("vh_mode", None),
                        temperature=self.cfg.algorithm.sampling_params.temperature_train,
                        top_k=self.cfg.algorithm.sampling_params.top_k,
                        logits_processor_args=logits_processor_args,
                        do_sample=not sampling_params["use_greedy"],
                        return_bc_logits=False,
                        logits_type=self.logits_type,
                    )
                    expert_tok = torch.tensor(
                        compute_action_tokens_from_actions(
                            self.model, bc_batch["actions"]
                        ),
                        device=device,
                    )
                    # Pure BC: coefficient 1.0 (tune step size via opd_teacher_* LRs).
                    bc_loss, bc_metrics = behavior_cloning_ce_loss(
                        intermediate_logits=output_dict["intermediate_logits"],
                        expert_actions_tokens=expert_tok,
                        bc_coeff=1.0,
                        vocab_size=self.model.vocab_size,
                        n_action_bins=self.model.config.n_action_bins,
                    )
                    (bc_loss / grad_acc).backward()
                    append_to_dict(step_metrics, bc_metrics)

                grad_norm = self.model.clip_grad_norm_(
                    max_norm=self.cfg.actor.optim.clip_grad
                )
                self.optimizer.step()
                append_to_dict(
                    step_metrics,
                    {"actor/grad_norm": grad_norm.detach().item()},
                )
                step_mean = {
                    key: float(np.mean(value)) for key, value in step_metrics.items()
                }
                step_mean = all_reduce_dict(
                    step_mean, op=torch.distributed.ReduceOp.AVG
                )
                per_step_metrics.append(step_mean)
        finally:
            for pg, lr in zip(self.optimizer.param_groups, saved_lrs):
                pg["lr"] = lr

        mean_metric_dict = {}
        if per_step_metrics:
            keys = per_step_metrics[0].keys()
            mean_metric_dict = {
                k: float(np.mean([float(d[k]) for d in per_step_metrics]))
                for k in keys
            }
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return {"per_step": per_step_metrics, "mean": mean_metric_dict}

    def set_opd_teacher_model_path(self, path: str):
        """Ray actors keep a launch-time cfg copy; call this after BC so OPD can load the teacher."""
        with open_dict(self.cfg.algorithm):
            self.cfg.algorithm.opd_teacher_model_path = path
        self._opd_teacher_model = None
        if self._rank == 0:
            print(f"[OPD] Actor cfg opd_teacher_model_path set to {path}", flush=True)
        return {}

    def save_checkpoint(self, save_base_path, step):
        torch.distributed.barrier()
        is_lora = self.cfg.actor.model.get("is_lora", False)
        model = self.model

        optim_state = self.get_optimizer_state_dict()

        if is_lora:
            # For PEFT models with FSDP, use this pattern:
            # All ranks must enter the context, but only rank 0 gets the full state
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(
                model, StateDictType.FULL_STATE_DICT, save_policy
            ):
                # Access the PeftModel from FSDP wrapper
                cpu_state = get_peft_model_state_dict(model, model.state_dict())

                # Now save_pretrained should work because state is gathered
                if self._rank == 0:
                    os.makedirs(save_base_path, exist_ok=True)
                    print(f"Saving model checkpoint to {save_base_path}")
                    torch.save(optim_state, os.path.join(save_base_path, "optim.pt"))

                    torch.save(
                        cpu_state, os.path.join(save_base_path, "adapter_model.bin")
                    )
                    model.peft_config["default"].save_pretrained(save_base_path)
        else:
            model_state = self.get_model_state_dict()
            if self._rank == 0:
                os.makedirs(save_base_path, exist_ok=True)
                torch.save(model_state, os.path.join(save_base_path, "model.pt"))
                torch.save(optim_state, os.path.join(save_base_path, "optim.pt"))

        torch.distributed.barrier()
