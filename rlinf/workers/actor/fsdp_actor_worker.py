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
import os

import json
import numpy as np
import torch
from omegaconf import DictConfig
from torch.distributed.device_mesh import init_device_mesh
from tqdm import tqdm

import rlinf.algorithms  # noqa: F401
from rlinf.algorithms.registry import actor_loss, calculate_adv_and_returns
from rlinf.algorithms.utils import preprocess_advantages_inputs, preprocess_loss_inputs
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import (
    FSDPModelManager,
)
from rlinf.models import get_model
from rlinf.models.embodiment.model_utils import actor_forward 
from rlinf.scheduler import Cluster, Worker
from rlinf.utils.data_iter_utils import get_iterator_k_split
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import (
    append_to_dict,
    compute_loss_mask,
    compute_rollout_metrics,
    compute_split_num,
)
from rlinf.utils.placement import HybridComponentPlacement
from peft import get_peft_model_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from rlinf.models import get_model_config_and_processor
from rlinf.custom.loss import behavior_cloning_loss
from rlinf.custom.libero_trajectory_dataset import LiberoSFTDataset
from torch.utils.data import DataLoader
from itertools import cycle
from omegaconf import OmegaConf

class EmbodiedFSDPActor(FSDPModelManager, Worker):
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
        if self.use_experience_replay:
            self._init_sft_replay_buffer()

    def _init_sft_replay_buffer(self):
        dataset_path = os.environ.get("LIBERO_REPO_PATH")
        if self._rank == 0:
            print(f"Initializing SFT dataset on rank {self._rank}")

        self.sft_dataset = LiberoSFTDataset(
            cfg=self.cfg,
            root_dir=dataset_path, 
            demos_per_task=1,
            rank=self._rank,
            world_size=self._world_size
        )

        self.sft_dataloader = cycle(DataLoader(
            self.sft_dataset,
            batch_size=self.cfg.actor.micro_batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        ))

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
        if self.cfg.actor.get("enable_offload", False):
            self.offload_fsdp_param_and_grad()
            self.offload_fsdp_optimizer()
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

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
        stage_num = self.cfg.rollout.pipeline_stage_num
        env_world_size = self._component_placement.get_world_size("env")
        actor_world_size = self._component_placement.get_world_size("actor")
        num_group_envs_for_train = (
            self.cfg.algorithm.num_group_envs
            * stage_num
            * env_world_size
            // actor_world_size
        )

        kwargs = {
            "adv_type": self.cfg.algorithm.adv_type,
            "rewards": self.rollout_batch["rewards"],
            "dones": self.rollout_batch["dones"],
            "normalize_advantages": self.cfg.algorithm.get(
                "normalize_advantages", True
            ),
            "values": self.rollout_batch.get("prev_values", None),
            "gamma": self.cfg.algorithm.get("gamma", 1),
            "gae_lambda": self.cfg.algorithm.get("gae_lambda", 1),
            "num_group_envs": num_group_envs_for_train,
            "group_size": self.cfg.algorithm.get("group_size", 8),
            "reward_type": self.cfg.algorithm.reward_type,
            "loss_mask": self.rollout_batch.get("loss_mask", None),
            "rollout_epoch": self.cfg.algorithm.get("rollout_epoch", 1),
        }
        kwargs = preprocess_advantages_inputs(**kwargs)
        advantages, returns = calculate_adv_and_returns(**kwargs)

        self.rollout_batch.update({"advantages": advantages, "returns": returns})
        rollout_metrics = compute_rollout_metrics(self.rollout_batch)
        return rollout_metrics

    def run_training(self):
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
        bc_coeff = self.cfg.actor.get("bc_coeff", 0.0)

        metrics = {}
        for _, train_global_batch in tqdm(
            enumerate(rollout_dataloader_iter), desc="get loss and metrics"
        ):
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

                bc_batch = None
                if self.use_experience_replay:
                    bc_batch = next(self.sft_iterator)
                    for k, v in bc_batch.items():
                        bc_batch[k] = v.to(f"cuda:{int(os.environ['LOCAL_RANK'])}")

                    bc_batch = self.model.preprocess_for_train(bc_batch)

                    sampling_params = OmegaConf.to_container(
                        self.cfg.algorithm.sampling_params, resolve=True
                    )

                output_dict, actions = actor_forward(
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
                )

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
                    kwargs = {
                        "action_tokens": actions,
                        "expert_action_tokens": bc_batch["action_tokens"],
                        "bc_coeff": bc_coeff
                    }

                    bc_loss, bc_metrics_data = behavior_cloning_loss(**kwargs)

                loss = rl_loss + bc_loss

                loss /= self.gradient_accumulation
                loss.backward()

                metrics_data["rl/loss"] = rl_loss.detach().item()
                metrics_data.update(bc_metrics_data)
                metrics_data["total/loss"] = loss.detach().item()
                append_to_dict(metrics, metrics_data)

            torch.cuda.empty_cache()

            grad_norm = self.model.clip_grad_norm_(
                max_norm=self.cfg.actor.optim.clip_grad
            )
            self.optimizer.step()

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
                model,
                StateDictType.FULL_STATE_DICT,
                save_policy
            ):
                # Access the PeftModel from FSDP wrapper
                cpu_state = get_peft_model_state_dict(model, model.state_dict())
                
                # Now save_pretrained should work because state is gathered
                if self._rank == 0:
                    os.makedirs(save_base_path, exist_ok=True)
                    torch.save(optim_state, os.path.join(save_base_path, "optim.pt"))
                    
                    torch.save(cpu_state, os.path.join(save_base_path, "adapter_model.bin"))
                    model.peft_config["default"].save_pretrained(save_base_path)
        else:
            model_state = self.get_model_state_dict()
            if self._rank == 0:
                os.makedirs(save_base_path, exist_ok=True)
                torch.save(model_state, os.path.join(save_base_path, "model.pt"))
                torch.save(optim_state, os.path.join(save_base_path, "optim.pt"))
        
        torch.distributed.barrier()