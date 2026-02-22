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
import os
import time
from contextlib import nullcontext
from itertools import cycle

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from peft import get_peft_model_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullOptimStateDictConfig, FullStateDictConfig, StateDictType
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
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import (
    FSDPModelManager,
)
from rlinf.models import get_model
from rlinf.models.embodiment.model_utils import (
    actor_forward,
    compute_action_tokens_from_actions,
    custom_forward,
)
from rlinf.scheduler import Cluster, Worker
from rlinf.utils.data_iter_utils import get_iterator_k_split
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import (
    append_to_dict,
    compute_loss_mask,
    compute_rollout_metrics as compute_rollout_metrics_simple,
    compute_split_num,
)
from rlinf.utils.placement import HybridComponentPlacement


class EmbodiedFSDPActor(FSDPModelManager, Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        super().__init__(cfg.actor)

        self.cfg = cfg
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        self.device = torch.cuda.current_device()
        world_size = self._world_size

        # Skip init_device_mesh for simple_cnn - it triggers init_process_group()
        # during Ray actor creation, before all workers are ready, causing TCPStore failures.
        # device_mesh is not used anywhere else in the codebase so this is safe.
        is_simple_cnn = cfg.actor.model.get("model_name") == "simple_cnn"
        if is_simple_cnn:
            self.device_mesh = None
        else:
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
        self.use_reference_logits_bc = cfg.algorithm.get(
            "use_reference_logits_bc", False
        )
        self.use_cached_bc_logits = cfg.algorithm.get("use_cached_bc_logits", False)
        self.logits_type = cfg.algorithm.get("logits_type", "processed")
        if self.logits_type not in ["processed", "raw"]:
            raise NotImplementedError(
                f"returning logits type {self.logits_type} is not implemented"
            )

        if self.use_experience_replay:
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
        
        # Enable garbage outputs for fast debugging (bypasses model inference)
        self.use_garbage_outputs = cfg.actor.get("use_garbage_outputs", False)
        if self.use_garbage_outputs and self._rank == 0:
            print(
                "WARNING: use_garbage_outputs is enabled. Actor will generate fake outputs "
                "instead of running model inference. This is for debugging only!"
            )
        
        # EWC (Elastic Weight Consolidation) setup
        self.use_ewc = cfg.algorithm.get("use_ewc", False)
        self.ewc_fisher_dict = None
        self.ewc_old_params = None
        if self.use_ewc:
            self._load_ewc_data_if_available()

    def _init_sft_replay_buffer(self, use_cached_logits=False):
        dataset_path = os.environ.get("LIBERO_REPO_PATH")
        if self._rank == 0:
            print(f"Initializing SFT dataset on rank {self._rank}")

        self.sft_dataset = LiberoSFTDataset(
            cfg=self.cfg,
            root_dir=dataset_path,
            demos_per_task=1,
            rank=self._rank,
            world_size=self._world_size,
            use_cached_logits=use_cached_logits,
            logits_type=self.logits_type if self.use_reference_logits_bc else "",
            use_preprocessed=True,
        )

        self.sft_dataloader = cycle(
            DataLoader(
                self.sft_dataset,
                batch_size=self.cfg.actor.micro_batch_size,
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

        # For EWC: capture the loaded model's LoRA parameters as old_params reference
        # This is done AFTER setup_model_and_optimizer() which loads the LoRA checkpoint.
        # Using the loaded weights directly avoids FSDP sharding issues with saved old_params.
        if self.use_ewc and self.ewc_fisher_dict is not None:
            from rlinf.algorithms.ewc import get_lora_parameters
            loaded_params = get_lora_parameters(self.model)
            if self._rank == 0:
                print(f"[EWC] Capturing {len(loaded_params)} loaded LoRA parameters as old_params reference")
                # Print sample for verification
                sample_names = list(loaded_params.keys())[:3]
                for name in sample_names:
                    print(f"  {name}: {loaded_params[name].shape}, "
                          f"mean={loaded_params[name].mean().item():.6e}")
            
            # Use loaded weights instead of whatever was in ewc_data.pt
            # This ensures we regularize toward the actual loaded checkpoint weights
            self.ewc_old_params = loaded_params

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
            self.offload_fsdp_param_and_grad()
            self.offload_fsdp_optimizer()
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
        model = get_model(self.cfg.actor.checkpoint_load_path, self.cfg.actor.model)
        if model is not None:
            return model
        return super().model_provider_func()

    def sync_model_to_rollout(self):
        if next(self.model.parameters()).is_cpu:
            self.load_fsdp_param_and_grad(self.device)
            self.load_fsdp_optimizer(self.device)

        state_dict = self.get_model_state_dict()
        if self._weight_dst_rank_in_rollout is not None:
            self.send(
                state_dict, self._rollout_group_name, self._weight_dst_rank_in_rollout
            )
        if self.cfg.actor.get("enable_offload", False):
            self.offload_fsdp_param_and_grad()
            self.offload_fsdp_optimizer()
            torch.cuda.synchronize()
            del state_dict
            gc.collect()
            torch.cuda.empty_cache()

    async def recv_rollout_batch(self):
        send_num = self._component_placement.get_world_size("rollout") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        self.rollout_batch = {}
        recv_list = []
        for _ in range(split_num):
            recv_list.append(
                await self.channel.get(
                    queue_name=self._replay_buffer_name, async_op=True
                ).async_wait()
            )

        # Collect all keys from all batches (not just the first one).
        # This is important for multi-task where different batches may have
        # different env_info keys.
        all_keys = set()
        for recv_batch in recv_list:
            if recv_batch is None:
                raise RuntimeError(
                    f"Actor rank {self._rank}: received None rollout batch. "
                    "A rollout worker likely crashed (OOM?); check raylet logs."
                )
            all_keys.update(recv_batch.keys())

        # Concatenate along the correct dimension for each key.
        for key in all_keys:
            batches_with_key = [i for i in range(split_num) if key in recv_list[i]]
            if not batches_with_key:
                continue

            tensors = [recv_list[i][key] for i in batches_with_key]
            if "env_info/" not in key:
                self.rollout_batch[key] = torch.cat(tensors, dim=1)
            else:
                self.rollout_batch[key] = torch.cat(tensors, dim=0)

        self.rollout_batch = self._process_received_rollout_batch(self.rollout_batch)

    def _process_received_rollout_batch(self, rollout_batch):
        """
        original shape: [rollout_epoch x n_chunk_steps, bsz, num_action_chunks, ...]
        target shape: [n_chunk_steps, rollout_epoch x bsz, num_action_chunks, ...]
        """
        rollout_epoch = self.cfg.algorithm.rollout_epoch
        for key, value in rollout_batch.items():
            if "env_info/" in key:
                continue
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
        """
        Compute advantages and returns on the actor side and aggregate
        rollout-level metrics in a distributed-safe way.
        """
        stage_num = self.cfg.rollout.pipeline_stage_num
        env_world_size = self._component_placement.get_world_size("env")
        actor_world_size = self._component_placement.get_world_size("actor")
        num_group_envs_for_train = (
            self.cfg.algorithm.num_group_envs * stage_num * env_world_size
        ) // actor_world_size

        rewards = self.rollout_batch["rewards"]
        dones = self.rollout_batch["dones"]
        prev_values = self.rollout_batch.get("prev_values", None)
        loss_mask = self.rollout_batch.get("loss_mask", None)

        kwargs = {
            "adv_type": self.cfg.algorithm.adv_type,
            "rewards": rewards,
            "dones": dones,
            "normalize_advantages": self.cfg.algorithm.get(
                "normalize_advantages", True
            ),
            "values": prev_values,
            "gamma": self.cfg.algorithm.get("gamma", 1),
            "gae_lambda": self.cfg.algorithm.get("gae_lambda", 1),
            "num_group_envs": num_group_envs_for_train,
            "group_size": self.cfg.algorithm.get("group_size", 8),
            "reward_type": self.cfg.algorithm.reward_type,
            "loss_mask": loss_mask,
            "rollout_epoch": self.cfg.algorithm.get("rollout_epoch", 1),
        }

        kwargs = preprocess_advantages_inputs(**kwargs)
        advantages, returns = calculate_adv_and_returns(**kwargs)

        self.rollout_batch.update({"advantages": advantages, "returns": returns})

        # This uses the distributed-safe implementation in metric_utils.
        rollout_metrics = compute_rollout_metrics_simple(self.rollout_batch)
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

    def _generate_garbage_outputs(self, batch_size, action_token_len, value_model=False, return_bc_logits=False, bc_batch_size=None):
        """
        Generate garbage/fake outputs with correct shapes for debugging training.
        This bypasses model inference to speed up training iteration during debugging.
        
        Args:
            batch_size: Batch size (B)
            action_token_len: Total action token length (action_dim * num_action_chunks)
            value_model: Whether to include values in output
            return_bc_logits: Whether to return BC logits
            bc_batch_size: Batch size for BC logits (if return_bc_logits is True)
        
        Returns:
            output_dict: Dictionary with fake outputs matching actor_forward structure
            bc_logits: Optional fake BC logits (if return_bc_logits is True)
        """
        device = f"cuda:{int(os.environ['LOCAL_RANK'])}"
        vocab_size = self.model.vocab_size
        n_action_bins = self.model.config.n_action_bins
        
        # Generate random logprobs: [B, action_token_len]
        logprobs = torch.randn(batch_size, action_token_len, device=device, requires_grad=True)
        
        # Generate random entropy: [B, action_token_len]
        entropy = torch.rand(batch_size, action_token_len, device=device, requires_grad=True)
        
        # Generate random raw logits: [B, action_token_len, vocab_size]
        raw_logits = torch.randn(batch_size, action_token_len, vocab_size, device=device, requires_grad=True)
        
        # Generate random intermediate logits: [B, action_token_len, vocab_size]
        intermediate_logits = torch.randn(batch_size, action_token_len, vocab_size, device=device, requires_grad=True)
        
        # Generate random processed logits: [B, action_token_len, n_action_bins]
        processed_logits = torch.randn(batch_size, action_token_len, n_action_bins, device=device, requires_grad=True)
        
        output_dict = {
            "logprobs": logprobs,
            "entropy": entropy,
            "raw_logits": raw_logits,
            "intermediate_logits": intermediate_logits,
            "processed_logits": processed_logits,
        }
        
        # Add values if needed
        if value_model:
            values = torch.randn(batch_size, 1, device=device, requires_grad=True)
            output_dict["values"] = values
        
        if return_bc_logits:
            if bc_batch_size is None:
                bc_batch_size = batch_size
            if self.logits_type == "processed":
                bc_logits = torch.randn(bc_batch_size, action_token_len, n_action_bins, device=device, requires_grad=True)
            else:  # raw
                bc_logits = torch.randn(bc_batch_size, action_token_len, vocab_size, device=device, requires_grad=True)
            return output_dict, bc_logits
        else:
            return output_dict

    def run_training(self, is_last_step: bool = False):
        """
        Run training on rollout batch.
        
        Args:
            is_last_step: If True, accumulate Fisher information during backward pass for EWC
        """
        self._deallocate_preallocated_memory()

        if self.cfg.actor.get("enable_offload", False):
            self.load_fsdp_param_and_grad(self.device)
            self.load_fsdp_optimizer(self.device)

        self.model.train()

        # Freeze BatchNorm layers for CNN to avoid train/eval mismatch.
        # During rollout the model is in eval() mode (BN uses running stats),
        # so training must also use running stats to keep logprobs consistent.
        is_cnn_model = self.cfg.actor.model.get("model_name") == "simple_cnn"
        if is_cnn_model:
            _model = self.model.module if hasattr(self.model, "module") else self.model
            for m in _model.modules():
                if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                    m.eval()

        self.optimizer.zero_grad()
        
        # Initialize Fisher accumulation if this is the last step and EWC is enabled
        if is_last_step and self.cfg.algorithm.get("use_ewc", False):
            self._init_fisher_accumulation()
        
        rollout_size = (
            self.rollout_batch["input_ids"].shape[0]
            * self.rollout_batch["input_ids"].shape[1]
        )

        shuffle_id = torch.randperm(rollout_size)

        for key, value in self.rollout_batch.items():
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
        for _, train_global_batch in tqdm(
            enumerate(rollout_dataloader_iter),
            desc=f"get loss and metrics (rank={self._rank})",
        ):
            num_batches += 1

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

            is_cnn = self.cfg.actor.model.get("model_name") == "simple_cnn"

            for data_idx, data in enumerate(train_micro_batch):
                for k, v in data.items():
                    data[k] = v.to(f"cuda:{int(os.environ['LOCAL_RANK'])}")

                data = self.model.preprocess_for_train(data)
                action_token_len = self.model.action_dim * self.model.num_action_chunks

                if is_cnn:
                    # ── CNN-specific forward ──────────────────────────
                    # Call self.model(...) (not .train_forward) so FSDP's forward()
                    # hooks are triggered and gradient all-reduce works correctly.
                    task_ids = data["input_ids"].squeeze(-1).long()
                    output_dict = self.model(
                        pixel_values=data["pixel_values"],
                        task_ids=task_ids,
                        action_tokens=data["action_tokens"],
                    )
                    # No BC batch for CNN
                    bc_batch = None
                    return_bc_logits = False
                else:
                    # ── VLA forward ───────────────────────────────────
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
                    
                    if self.use_garbage_outputs:
                        # Generate garbage outputs for fast debugging (bypasses model inference)
                        batch_size = data["input_ids"].shape[0]
                        value_model = self.cfg.algorithm.adv_type == "embodied_gae"
                        bc_batch_size = bc_batch["input_ids"].shape[0] if bc_batch is not None else None
                        forward_result = self._generate_garbage_outputs(
                            batch_size=batch_size,
                            action_token_len=action_token_len,
                            value_model=value_model,
                            return_bc_logits=return_bc_logits,
                            bc_batch_size=bc_batch_size,
                        )
                    else:
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

                # Add EWC loss if enabled
                device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
                ewc_loss = torch.tensor(0.0, device=device)
                if self.use_ewc and self.ewc_fisher_dict is not None and self.ewc_old_params is not None:
                    from rlinf.algorithms.ewc import compute_ewc_loss
                    ewc_loss = compute_ewc_loss(
                        self.model,
                        self.ewc_fisher_dict,
                        self.ewc_old_params,
                        lambda_ewc=self.cfg.algorithm.get("ewc_lambda", 1000000.0),
                    )
                    metrics_data["ewc/loss"] = ewc_loss.detach().item()
                    if self._rank == 0:
                        print(f"EWC loss: {ewc_loss.detach().item()}", flush=True)
                
                loss = rl_loss + bc_loss + ewc_loss
                loss /= self.gradient_accumulation
                loss.backward()
                
                # Accumulate Fisher information if this is the last step
                if is_last_step and self.cfg.algorithm.get("use_ewc", False):
                    self._accumulate_fisher_from_gradients()

                metrics_data["rl/loss"] = rl_loss.detach().item()
                metrics_data.update(bc_metrics_data)
                metrics_data["total/loss"] = loss.detach().item()
                append_to_dict(metrics, metrics_data)

            torch.cuda.empty_cache()

            grad_norm = self.model.clip_grad_norm_(
                max_norm=self.cfg.actor.optim.clip_grad
            )
            self.optimizer.step()

            # Increment training step counter
            self.training_step_count += 1

            self.optimizer.zero_grad()
            data = {
                "actor/grad_norm": grad_norm.detach().item(),
                "actor/lr": self.optimizer.param_groups[0]["lr"],
            }
            if self.cfg.algorithm.adv_type == "embodied_gae":
                data["critic/lr"] = self.optimizer.param_groups[1]["lr"]
            append_to_dict(metrics, data)

        mean_metric_dict = {key: np.mean(value) for key, value in metrics.items()}
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )

        self.optimizer.zero_grad()
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        
        # Finalize Fisher information if this was the last step
        if is_last_step and self.cfg.algorithm.get("use_ewc", False):
            self._finalize_fisher_accumulation(num_batches)

        return mean_metric_dict

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
    
    def _load_ewc_data_if_available(self):
        """Load EWC Fisher information from previous task if available.
        
        NOTE: old_params is NOT loaded from file. It will be captured from the 
        loaded checkpoint weights in init_worker() after setup_model_and_optimizer().
        """
        ewc_path = self.cfg.algorithm.get("previous_task_ewc_path", None)
        if ewc_path and os.path.exists(ewc_path):
            from rlinf.algorithms.ewc import load_ewc_data
            try:
                self.ewc_fisher_dict = load_ewc_data(ewc_path)
                self.ewc_old_params = None  # Will be set in init_worker from loaded checkpoint
                
                # Maximum Fisher value (should match max_grad_norm^2)
                max_fisher_value = 100.0 ** 2  # 10000
                
                # Validate and clean loaded Fisher data for NaN/inf
                if self.ewc_fisher_dict is not None:
                    cleaned_count = 0
                    for name in list(self.ewc_fisher_dict.keys()):
                        fisher_val = self.ewc_fisher_dict[name]
                        if torch.isnan(fisher_val).any() or torch.isinf(fisher_val).any():
                            if self._rank == 0:
                                print(f"✗ WARNING: Fisher information contains NaN/Inf for '{name}'. "
                                      f"Replacing with max value {max_fisher_value} (indicating high importance).")
                            # Replace NaN/Inf with max_fisher_value (not zero!)
                            self.ewc_fisher_dict[name] = torch.nan_to_num(
                                fisher_val, 
                                nan=max_fisher_value, 
                                posinf=max_fisher_value, 
                                neginf=0.0
                            )
                            cleaned_count += 1
                        
                        # Clip to max value
                        self.ewc_fisher_dict[name] = torch.clamp(
                            self.ewc_fisher_dict[name], max=max_fisher_value
                        )
                    
                    if self._rank == 0 and cleaned_count > 0:
                        print(f"  Cleaned {cleaned_count} Fisher parameters with NaN/Inf values "
                              f"(replaced with max value {max_fisher_value})")
                
                if self._rank == 0:
                    print(f"✓ Loaded EWC Fisher data from {ewc_path}")
                    print(f"  Fisher dict: {len(self.ewc_fisher_dict)} parameters")
                    print(f"  (old_params will be captured from loaded checkpoint in init_worker)")
            except Exception as e:
                if self._rank == 0:
                    print(f"✗ Failed to load EWC data from {ewc_path}: {e}")
                self.ewc_fisher_dict = None
                self.ewc_old_params = None
        else:
            self.ewc_fisher_dict = None
            self.ewc_old_params = None
            if self._rank == 0 and self.use_ewc:
                print("No previous EWC data found - training first task without EWC regularization")
    
    def _init_fisher_accumulation(self):
        """Initialize Fisher information accumulation for last step."""
        # Use get_lora_parameters to get full parameter shapes (handles FSDP)
        from rlinf.algorithms.ewc import get_lora_parameters
        lora_params = get_lora_parameters(self.model)
        
        self.fisher_accumulator = {}
        self.fisher_param_names = set(lora_params.keys())
        for name, param_tensor in lora_params.items():
            # Initialize with zeros matching the full parameter shape
            self.fisher_accumulator[name] = torch.zeros_like(param_tensor, device='cpu')
        
        if self._rank == 0:
            print(f"Initialized Fisher accumulation for {len(self.fisher_accumulator)} LoRA parameters")

    def _accumulate_fisher_from_gradients(self):
        """Accumulate squared gradients for Fisher information computation."""
        if not hasattr(self, 'fisher_accumulator'):
            return
        
        # Maximum gradient norm to prevent overflow when squaring
        # If gradient norm exceeds this, clip before squaring
        max_grad_norm = 100.0
        max_fisher_value = max_grad_norm ** 2  # Maximum Fisher value (100^2 = 10000)
        
        # Access gradients through the model
        model_to_check = self.model.module if hasattr(self.model, 'module') else self.model
        
        for name, param in model_to_check.named_parameters():
            if 'lora' in name.lower() and param.requires_grad and param.grad is not None:
                if name in self.fisher_param_names:
                    # Get gradient and clip to prevent overflow when squaring
                    grad = param.grad.data
                    
                    # Clip gradients to prevent overflow when squaring
                    grad_norm = grad.norm()
                    if grad_norm > max_grad_norm:
                        grad = grad * (max_grad_norm / grad_norm)
                    
                    # Square the gradient
                    grad_sq = (grad ** 2).cpu()
                    
                    # Additional safety: clip any values that somehow became inf
                    grad_sq = torch.clamp(grad_sq, max=max_fisher_value)
                    
                    # Check if shape matches (might be sharded)
                    if name in self.fisher_accumulator:
                        if grad_sq.shape == self.fisher_accumulator[name].shape:
                            # Shapes match - accumulate directly
                            self.fisher_accumulator[name] += grad_sq
                            # Clip accumulator to prevent accumulation of inf
                            self.fisher_accumulator[name] = torch.clamp(
                                self.fisher_accumulator[name], max=max_fisher_value
                            )
                        else:
                            # Shape mismatch - skip for now
                            pass

    def _finalize_fisher_accumulation(self, num_batches: int):
        """Finalize Fisher information by averaging over batches."""
        if not hasattr(self, 'fisher_accumulator'):
            return
        
        if num_batches == 0:
            if self._rank == 0:
                print("WARNING: num_batches is 0, cannot finalize Fisher accumulation")
            return
        
        # Maximum Fisher value (should match max_grad_norm^2 from accumulation)
        max_fisher_value = 100.0 ** 2  # 10000
        
        # Average over batches
        for name in self.fisher_accumulator:
            # Check for inf/nan before averaging and replace with max value
            if torch.isnan(self.fisher_accumulator[name]).any() or torch.isinf(self.fisher_accumulator[name]).any():
                if self._rank == 0:
                    print(f"WARNING: Fisher accumulator for '{name}' contains NaN/Inf before averaging. "
                          f"Replacing with max value {max_fisher_value}.")
                # Replace NaN/Inf with max_fisher_value (not zero!)
                self.fisher_accumulator[name] = torch.nan_to_num(
                    self.fisher_accumulator[name], 
                    nan=max_fisher_value, 
                    posinf=max_fisher_value, 
                    neginf=0.0  # Negative inf should be 0 (shouldn't happen for squared values)
                )
            
            self.fisher_accumulator[name] /= num_batches
            
            # Clip after averaging to ensure no values exceed max
            self.fisher_accumulator[name] = torch.clamp(
                self.fisher_accumulator[name], max=max_fisher_value
            )
        
        # With FSDP, parameters are sharded across ranks.
        if torch.distributed.is_initialized() and self._world_size > 1:
            for name in self.fisher_accumulator:
                fisher_tensor = self.fisher_accumulator[name].to(self.device)
                
                # Check for inf/nan before all-reduce and replace with max value
                if torch.isnan(fisher_tensor).any() or torch.isinf(fisher_tensor).any():
                    if self._rank == 0:
                        print(f"WARNING: Fisher tensor for '{name}' contains NaN/Inf before all-reduce. "
                              f"Replacing with max value {max_fisher_value}.")
                    fisher_tensor = torch.nan_to_num(
                        fisher_tensor, 
                        nan=max_fisher_value, 
                        posinf=max_fisher_value, 
                        neginf=0.0
                    )
                
                torch.distributed.all_reduce(fisher_tensor, op=torch.distributed.ReduceOp.SUM)
                
                # Check for inf/nan after all-reduce
                if torch.isnan(fisher_tensor).any() or torch.isinf(fisher_tensor).any():
                    if self._rank == 0:
                        print(f"WARNING: Fisher tensor for '{name}' contains NaN/Inf after all-reduce. "
                              f"Replacing with max value {max_fisher_value}.")
                    fisher_tensor = torch.nan_to_num(
                        fisher_tensor, 
                        nan=max_fisher_value, 
                        posinf=max_fisher_value, 
                        neginf=0.0
                    )
                
                # Clip to max value
                fisher_tensor = torch.clamp(fisher_tensor, max=max_fisher_value)
                
                self.fisher_accumulator[name] = fisher_tensor.cpu()
        
        if self._rank == 0:
            print(f"Finalized Fisher accumulation over {num_batches} batches")
            print(f"  Fisher computed for {len(self.fisher_accumulator)} parameters")

    def compute_and_save_ewc_data(self, save_path: str):
        """
        Save EWC data using Fisher accumulated during the last training step.
        This is called from the runner after training completes.
        """
        if not self.use_ewc:
            return

        from rlinf.algorithms.ewc import save_ewc_data

        if not hasattr(self, "fisher_accumulator") or self.fisher_accumulator is None:
            if self._rank == 0:
                raise ValueError(
                    "EWC is enabled but Fisher information was not computed during training"
                )
            return

        # Maximum Fisher value (should match max_grad_norm^2)
        max_fisher_value = 100.0 ** 2  # 10000

        # Current task Fisher (already averaged over batches and all-reduced)
        current_fisher = self.fisher_accumulator
        
        # Validate and clean current Fisher for NaN/inf
        for name in list(current_fisher.keys()):
            if torch.isnan(current_fisher[name]).any() or torch.isinf(current_fisher[name]).any():
                if self._rank == 0:
                    print(f"WARNING: Current Fisher for '{name}' contains NaN/Inf. "
                          f"Replacing with max value {max_fisher_value}.")
                current_fisher[name] = torch.nan_to_num(
                    current_fisher[name], 
                    nan=max_fisher_value, 
                    posinf=max_fisher_value, 
                    neginf=0.0
                )
            # Clip to max value
            current_fisher[name] = torch.clamp(current_fisher[name], max=max_fisher_value)

        # Online accumulation: merge current Fisher into accumulated Fisher from previous tasks
        if self.ewc_fisher_dict is not None:
            # First, validate and clean previous Fisher
            for name in list(self.ewc_fisher_dict.keys()):
                if torch.isnan(self.ewc_fisher_dict[name]).any() or torch.isinf(self.ewc_fisher_dict[name]).any():
                    if self._rank == 0:
                        print(f"WARNING: Previous Fisher for '{name}' contains NaN/Inf. "
                              f"Replacing with max value {max_fisher_value}.")
                    self.ewc_fisher_dict[name] = torch.nan_to_num(
                        self.ewc_fisher_dict[name], 
                        nan=max_fisher_value, 
                        posinf=max_fisher_value, 
                        neginf=0.0
                    )
                # Clip to max value
                self.ewc_fisher_dict[name] = torch.clamp(
                    self.ewc_fisher_dict[name], max=max_fisher_value
                )
            
            accumulated_fisher = {}
            for name in current_fisher:
                if name in self.ewc_fisher_dict:
                    if current_fisher[name].shape == self.ewc_fisher_dict[name].shape:
                        # Merge Fisher values
                        merged = self.ewc_fisher_dict[name] + current_fisher[name]
                        
                        # Clip merged value to prevent overflow
                        merged = torch.clamp(merged, max=max_fisher_value)
                        
                        # Check for inf/nan after merging (shouldn't happen after clamping)
                        if torch.isnan(merged).any() or torch.isinf(merged).any():
                            if self._rank == 0:
                                print(f"WARNING: Merged Fisher for '{name}' contains NaN/Inf. "
                                      "Using current Fisher only.")
                            accumulated_fisher[name] = current_fisher[name]
                        else:
                            accumulated_fisher[name] = merged
                    else:
                        # Shape mismatch: fall back to current Fisher
                        if self._rank == 0:
                            print(f"WARNING: Shape mismatch for '{name}': "
                                  f"previous={self.ewc_fisher_dict[name].shape}, "
                                  f"current={current_fisher[name].shape}. Using current Fisher.")
                        accumulated_fisher[name] = current_fisher[name]
                else:
                    accumulated_fisher[name] = current_fisher[name]

            # Include any parameters that were only present in previous tasks
            for name in self.ewc_fisher_dict:
                if name not in accumulated_fisher:
                    # Validate before including
                    if torch.isnan(self.ewc_fisher_dict[name]).any() or torch.isinf(self.ewc_fisher_dict[name]).any():
                        if self._rank == 0:
                            print(f"WARNING: Previous-only Fisher for '{name}' contains NaN/Inf. "
                                  f"Replacing with max value {max_fisher_value}.")
                        self.ewc_fisher_dict[name] = torch.nan_to_num(
                            self.ewc_fisher_dict[name], 
                            nan=max_fisher_value, 
                            posinf=max_fisher_value, 
                            neginf=0.0
                        )
                    accumulated_fisher[name] = torch.clamp(
                        self.ewc_fisher_dict[name], max=max_fisher_value
                    )

            fisher_to_save = accumulated_fisher
        else:
            # First task: no previous Fisher to accumulate
            fisher_to_save = current_fisher

        # Final validation of Fisher to save
        for name in list(fisher_to_save.keys()):
            if torch.isnan(fisher_to_save[name]).any() or torch.isinf(fisher_to_save[name]).any():
                if self._rank == 0:
                    print(f"ERROR: Final Fisher for '{name}' still contains NaN/Inf after cleaning! "
                          f"Replacing with max value {max_fisher_value}.")
                fisher_to_save[name] = torch.nan_to_num(
                    fisher_to_save[name], 
                    nan=max_fisher_value, 
                    posinf=max_fisher_value, 
                    neginf=0.0
                )
            # Final clip
            fisher_to_save[name] = torch.clamp(fisher_to_save[name], max=max_fisher_value)

        # Only rank 0 actually writes to disk
        # NOTE: We only save Fisher information, not old_params.
        # old_params is captured from the loaded checkpoint at the start of training
        # in init_worker(), which avoids FSDP sharding issues.
        if self._rank == 0:
            print(f"[EWC] Saving {len(fisher_to_save)} Fisher parameters")
            save_ewc_data(fisher_to_save, save_path)
            print(f"✓ EWC data saved to {save_path}")

