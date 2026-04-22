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
import time
from collections import defaultdict

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

from rlinf.config import torch_dtype_from_precision
from rlinf.models import get_model, get_model_config_and_processor
from rlinf.models.embodiment.model_utils import (
    default_logits_processor,
    prepare_observations,
)
from rlinf.scheduler import Cluster, Worker
from rlinf.utils.metric_utils import compute_split_num
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.utils.runner_utils import cfg_show_progress_bar
from rlinf.workers.actor.opd_teacher import (
    compute_teacher_logprobs_one_batch,
    load_opd_teacher_model,
)


def create_rollout_batch(data):
    ret_data = {}
    for key, value in data.items():
        if "env_info/" not in key:
            ret_data[key] = torch.stack(value, dim=0).contiguous().cpu()
        else:
            ret_data[key] = torch.cat(value, dim=0).contiguous().cpu()
    return ret_data


def _check_actor_memory(device_index, threshold_gb=4.0):
    """
    Check if 'EmbodiedFSDPActor' process on the given device is using less than threshold_gb.
    Returns True if safe (memory low or process not found), False if memory high.
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)

        actor_mem_bytes = 0
        found_actor = False
        for p in procs:
                # p.pid is available, but name might require looking up via psutil or nvmlSystemGetProcessName
                # nvmlSystemGetProcessName is available in newer pynvml/drivers
                name = pynvml.nvmlSystemGetProcessName(p.pid)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')

                # We look for the Ray process name pattern
                if "EmbodiedFSDPActor" in name:
                    actor_mem_bytes += p.usedGpuMemory
                    found_actor = True

        pynvml.nvmlShutdown()

        if not found_actor:
            return True # Actor not on this GPU or not found, assume safe

        actor_mem_gb = actor_mem_bytes / (1024**3)
        print(f"Actor memory usage: {actor_mem_gb}, threshold: {threshold_gb}", flush=True)
        return actor_mem_gb < threshold_gb

    except Exception as e:
        print(f"Warning: Failed to check actor memory: {e}")
        return True # Fail open to avoid deadlock if NVML issues


class MultiStepRolloutWorker(Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        self._env_group_name = cfg.env.group_name
        self._actor_group_name = cfg.actor.group_name
        self.device = torch.cuda.current_device()

        self.model_config, self.input_processor = get_model_config_and_processor(
            cfg.actor
        )
        self.precision = torch_dtype_from_precision(cfg.actor.model.precision)

        self._obs_queue_name = cfg.env.channel.queue_name
        self._action_queue_name = cfg.rollout.channel.queue_name
        self._replay_buffer_name = cfg.actor.channel.queue_name
        # stage_num: default to 2, use for pipeline rollout process
        self.stage_num = cfg.rollout.pipeline_stage_num

        self._component_placement = HybridComponentPlacement(cfg, Cluster())
        self.channel = self.connect_channel(cfg.rollout.channel.name)
        for i in range(self._component_placement.get_world_size("rollout")):
            self.channel.create_queue(
                f"{self._action_queue_name}_{i}", maxsize=cfg.rollout.channel.queue_size
            )

        self.use_proprio = self.cfg.actor.model.get("use_proprio", False)
        self._opd_teacher_model = None
        self._debug_sft_rollout_checks = bool(
            self.cfg.algorithm.get(
                "debug_sft_rollout_checks",
                self.cfg.algorithm.get("adv_type", None) == "embodied_opd",
            )
        )
        self._debug_rollout_dump_done = False
        self._debug_dir = os.path.join(
            self.cfg.runner.logger.log_path, "debug_sft_rollout_checks"
        )
        self._debug_rollout_img_path = os.path.join(
            self._debug_dir, "teacher_eval_first_rollout_obs.png"
        )

        # Debug logging setup
        self.enable_action_logging = cfg.rollout.get("enable_action_logging", False)
        if self.enable_action_logging:
            self.action_log_dir = cfg.rollout.get(
                "action_log_dir",
                os.path.join(cfg.runner.logger.log_path, "action_logs"),
            )
            os.makedirs(self.action_log_dir, exist_ok=True)
            self.action_log_data = defaultdict(list)
            if self._rank == 0:
                print(
                    f"[DEBUG] Action logging enabled. Saving to: {self.action_log_dir}"
                )

    def init_worker(self):
        self.hf_model = get_model(
            self.cfg.rollout.model_dir,
            self.cfg.actor.model,
            load_role="rollout_inference",
            worker_rank=self._rank,
            worker_world_size=self._world_size,
        )
        self.hf_model.setup_params(self.model_config, self.cfg)
        self.hf_model.to(self.precision)
        self.hf_model.eval()
        self.setup_sample_params()
        if self.cfg.rollout.get("enable_offload", False):
            self.offload_model()

    def _should_precompute_teacher_in_rollout(self) -> bool:
        return self.cfg.algorithm.get("adv_type") == "embodied_opd" and bool(
            self.cfg.algorithm.get("opd_precompute_teacher_in_rollout", True)
        )

    def _ensure_opd_teacher_for_rollout(self) -> None:
        if not self._should_precompute_teacher_in_rollout():
            return
        path = self.cfg.algorithm.get("opd_teacher_model_path", None)
        if not path:
            raise ValueError(
                "opd_precompute_teacher_in_rollout requires algorithm.opd_teacher_model_path "
                "(set after BC via runner, or in config)."
            )
        if self._opd_teacher_model is None:
            self._opd_teacher_model = load_opd_teacher_model(self.cfg, self._rank)

    def set_opd_teacher_model_path(self, path: str):
        """Mirror actor: BC writes teacher path so rollout can load the same checkpoint."""
        with open_dict(self.cfg.algorithm):
            self.cfg.algorithm.opd_teacher_model_path = path
        self._opd_teacher_model = None
        if self._rank == 0:
            print(
                f"[OPD] Rollout opd_teacher_model_path set to {path}",
                flush=True,
            )
        return {}

    def set_algorithm_adv_type(self, adv_type: str):
        """Switch rollout-side algorithm mode (controls OPD teacher precompute path)."""
        with open_dict(self.cfg.algorithm):
            self.cfg.algorithm.adv_type = adv_type
        if adv_type != "embodied_opd":
            self._opd_teacher_model = None
        if self._rank == 0:
            print(f"[OPD] Rollout algorithm adv_type set to {adv_type}", flush=True)
        return {}

    def setup_sample_params(self):
        # length parameters for rollout
        self._length_params = OmegaConf.to_container(
            self.cfg.algorithm.length_params, resolve=True
        )
        # sampling parameters for rollout
        self._sampling_params = OmegaConf.to_container(
            self.cfg.algorithm.sampling_params, resolve=True
        )
        self._train_sampling_params = {
            "do_sample": not self._sampling_params["use_greedy"],
            "temperature": self._sampling_params["temperature_train"],
            "top_k": self._sampling_params["top_k"],
            "top_p": self._sampling_params["top_p"],
            "max_new_tokens": self._length_params["max_new_token"],
            "use_cache": True,
        }

        self._eval_sampling_params = {
            "do_sample": not self._sampling_params["use_greedy"],
            "temperature": self._sampling_params["temperature_eval"],
            "top_k": self._sampling_params["top_k"],
            "top_p": self._sampling_params["top_p"],
            "max_new_tokens": self._length_params["max_new_token"],
        }

    def _maybe_dump_first_teacher_eval_sample(self, env_batch, chunk_actions):
        if (
            not self._debug_sft_rollout_checks
            or self._debug_rollout_dump_done
            or self._rank != 0
        ):
            return
        try:
            os.makedirs(self._debug_dir, exist_ok=True)
            raw_task = env_batch["obs"]["task_descriptions"][0]
            formatted_prompt = (
                f"In: What action should the robot take to {str(raw_task).lower()}?\nOut: "
            )
            raw_img = env_batch["obs"]["images_and_states"]["full_image"][0]
            if torch.is_tensor(raw_img):
                img_np = raw_img.detach().cpu().numpy()
            else:
                img_np = np.asarray(raw_img)
            img_np = np.asarray(img_np)

            acts_np = np.asarray(chunk_actions)
            print(
                "[DBG ROLLOUT] first teacher-eval sample: "
                f"obs_shape={img_np.shape}, obs_dtype={img_np.dtype}, "
                f"obs_min={float(img_np.min()):.3f}, obs_max={float(img_np.max()):.3f}, "
                f"actions_shape={acts_np.shape}, actions_dtype={acts_np.dtype}, "
                f"actions_min={float(acts_np.min()):.6f}, actions_max={float(acts_np.max()):.6f}",
                flush=True,
            )
            print(
                f"[DBG ROLLOUT] first raw task_description='{raw_task}'",
                flush=True,
            )
            print(
                f"[DBG ROLLOUT] first formatted prompt='{formatted_prompt}'",
                flush=True,
            )
            print(
                f"[DBG ROLLOUT] first rollout action chunk[0,0]={np.array2string(acts_np[0, 0], precision=6)}",
                flush=True,
            )

            try:
                from PIL import Image

                Image.fromarray(img_np.astype(np.uint8)).save(self._debug_rollout_img_path)
                print(
                    f"[DBG ROLLOUT] saved first rollout obs image to: {self._debug_rollout_img_path}",
                    flush=True,
                )
            except Exception as e:
                print(f"[DBG ROLLOUT] failed to save rollout obs image: {e}", flush=True)
        finally:
            self._debug_rollout_dump_done = True

    def predict(self, processed_obs, mode="train"):
        action_token_len = self.hf_model.action_dim * self.hf_model.num_action_chunks

        sample_kwargs = (
            self._train_sampling_params
            if mode == "train"
            else self._eval_sampling_params
        )

        with torch.no_grad():
            actions, action_tokens, action_logits, last_hidden_state = (
                self.hf_model.predict_action_batch(
                    input_ids=processed_obs["input_ids"],
                    attention_mask=processed_obs["attention_mask"],
                    pixel_values=processed_obs["pixel_values"],
                    **sample_kwargs,
                )
            )

        chunk_logprobs = default_logits_processor(
            action_logits,
            action_tokens,
            self.hf_model.vocab_size,
            self.hf_model.config.n_action_bins,
        )["logprobs"]

        chunk_values = None
        if self.cfg.algorithm.require_values:
            if self.cfg.actor.model.vh_mode == "a0":
                hidden_features = last_hidden_state[
                    :, -action_token_len
                ]  # [batch_size, hidden_dim]
                with torch.no_grad():
                    chunk_values = self.hf_model.value_head(
                        hidden_features
                    )  # [batch_size, 1]

        if chunk_values is None:
            chunk_values = torch.zeros_like(chunk_logprobs[..., :1])

        chunk_actions = actions.reshape(
            -1, self.hf_model.num_action_chunks, self.hf_model.action_dim
        )
        chunk_action_tokens = action_tokens.reshape(
            -1, self.hf_model.num_action_chunks, self.hf_model.action_dim
        )

        return chunk_actions, chunk_action_tokens, chunk_logprobs, chunk_values

    def log_actions_and_tokens(
        self, chunk_actions, chunk_action_tokens, step, stage_id, rollout_epoch
    ):
        """
        Log actions and action tokens to disk for debugging.

        Args:
            chunk_actions: Tensor of shape [batch, num_chunks, action_dim] with continuous actions
            chunk_action_tokens: Tensor of shape [batch, num_chunks, action_dim] with token IDs
            step: Current step in rollout
            stage_id: Stage ID in pipeline
            rollout_epoch: Current rollout epoch
        """
        if not self.enable_action_logging:
            return

        # Convert to numpy and store
        actions_np = chunk_actions  # [batch, num_chunks, action_dim]
        tokens_np = chunk_action_tokens.cpu().numpy()  # [batch, num_chunks, action_dim]

        # Store in memory (will save to disk at end of generate)
        self.action_log_data["actions"].append(actions_np)
        self.action_log_data["action_tokens"].append(tokens_np)
        self.action_log_data["step"].append(step)
        self.action_log_data["stage_id"].append(stage_id)
        self.action_log_data["rollout_epoch"].append(rollout_epoch)

    def save_action_logs(self, global_step):
        """Save accumulated action logs to disk."""
        if not self.enable_action_logging or not self.action_log_data["actions"]:
            return

        # Concatenate all logged data
        all_actions = np.concatenate(self.action_log_data["actions"], axis=0)
        all_tokens = np.concatenate(self.action_log_data["action_tokens"], axis=0)
        all_steps = np.array(self.action_log_data["step"])
        all_stage_ids = np.array(self.action_log_data["stage_id"])
        all_epochs = np.array(self.action_log_data["rollout_epoch"])

        # Save to npz file
        save_path = os.path.join(
            self.action_log_dir, f"rank_{self._rank}_step_{global_step}.npz"
        )

        np.savez_compressed(
            save_path,
            actions=all_actions,
            action_tokens=all_tokens,
            steps=all_steps,
            stage_ids=all_stage_ids,
            rollout_epochs=all_epochs,
            vocab_size=self.hf_model.vocab_size,
            n_action_bins=self.hf_model.config.n_action_bins,
            action_dim=self.hf_model.action_dim,
            num_action_chunks=self.hf_model.num_action_chunks,
        )

        if self._rank == 0:
            print(f"[DEBUG] Saved action logs to: {save_path}")
            print(f"  Total samples: {all_actions.shape[0]}")
            print(f"  Actions shape: {all_actions.shape}")
            print(f"  Tokens shape: {all_tokens.shape}")
            print(f"  Action range: [{all_actions.min():.4f}, {all_actions.max():.4f}]")
            print(f"  Token range: [{all_tokens.min()}, {all_tokens.max()}]")

        # Clear logged data
        self.action_log_data.clear()

    def update_env_batch(self, i, env_batch):
        # first step for env_batch
        if env_batch["rews"] is None:
            self.buffer_list[i]["dones"].append(env_batch["dones"].contiguous().cpu())
            return

        self.buffer_list[i]["rewards"].append(env_batch["rews"].cpu().contiguous())
        self.buffer_list[i]["dones"].append(
            env_batch["dones"].bool().cpu().contiguous()
        )

        if self.cfg.env.train.auto_reset or self.cfg.env.train.ignore_terminations:
            env_info_list = env_batch["meta"]
            for key, value in env_info_list.items():
                self.buffer_list[i][f"env_info/{key}"].append(value)

        # Note: currently this is not correct for chunk-size>1 with partial reset
        if env_batch["dones"].any() and self.cfg.env.train.auto_reset:
            if self.cfg.algorithm.require_values:
                dones = env_batch["dones"]
                # if self.require_values:
                final_obs = env_batch["infos"]["final_observation"]
                with torch.no_grad():
                    processed_obs = prepare_observations(
                        simulator_type=self.cfg.env.train.simulator_type,
                        model_name=self.cfg.actor.model.model_name,
                        raw_obs=final_obs,
                        use_proprio=self.use_proprio,
                        max_length=self.hf_model.max_prompt_length,
                        processor=self.input_processor,
                        precision=self.precision,
                    )
                    _, _, _, _final_values = self.predict(processed_obs)
                final_values = torch.zeros_like(_final_values[:, 0])  # [bsz, ]
                last_step_dones = dones[:, -1]  # [bsz, ]

                final_values[last_step_dones] = _final_values[:, 0][last_step_dones]

                self.buffer_list[i]["rewards"][-1][:, -1] += (
                    self.cfg.algorithm.gamma * final_values.cpu()
                )

    async def generate(self, global_step=0):
        _log = self._rank == 0 and self.cfg.runner.get("log_step_phase_timings", True)
        _t_gen = time.perf_counter()
        if self.cfg.rollout.get("enable_offload", False):
            self.reload_model(
                load_opd_teacher=self._should_precompute_teacher_in_rollout()
            )
        self._ensure_opd_teacher_for_rollout()
        self.buffer_list = []
        for i in range(self.stage_num):
            self.buffer_list.append(defaultdict(list))

        for rollout_epoch in range(self.cfg.algorithm.rollout_epoch):
            self._logger.info(f"Now epoch is={rollout_epoch}")

            # n_chunk_steps == how many times (per rollout_epoch pass) we perform a policy forward pass
            # it is equal to max_episode_steps / num_action_chunks
            for step in tqdm(
                range(self.cfg.algorithm.n_chunk_steps),
                desc=f"Rollout ID {self._rank} Epoch {rollout_epoch} in Generate Step",
                disable=not cfg_show_progress_bar(self.cfg),
            ):
                # Stage_num is number of parallel pipeline stages in one rollout collection
                # there are stage_num simulators
                for i in range(self.stage_num):
                    env_batch = await self.recv_env_batch()
                    self.update_env_batch(i, env_batch)
                    processed_obs = prepare_observations(
                        simulator_type=self.cfg.env.train.simulator_type,
                        model_name=self.cfg.actor.model.model_name,
                        raw_obs=env_batch["obs"],
                        use_proprio=self.use_proprio,
                        max_length=self.hf_model.max_prompt_length,
                        processor=self.input_processor,
                        precision=self.precision,
                    )
                    chunk_actions, chunk_action_token, chunk_logprobs, chunk_values = (
                        self.predict(processed_obs)
                    )
                    await self.send_chunk_actions(chunk_actions)

                    # Log actions and tokens for debugging
                    self.log_actions_and_tokens(
                        chunk_actions, chunk_action_token, step, i, rollout_epoch
                    )

                    self.buffer_list[i]["input_ids"].append(
                        processed_obs["input_ids"].cpu().contiguous()
                    )
                    self.buffer_list[i]["pixel_values"].append(
                        processed_obs["pixel_values"].cpu().contiguous()
                    )
                    self.buffer_list[i]["attention_mask"].append(
                        processed_obs["attention_mask"].bool().cpu().contiguous()
                    )
                    self.buffer_list[i]["action_tokens"].append(
                        chunk_action_token.cpu().contiguous()
                    )
                    self.buffer_list[i]["prev_logprobs"].append(
                        chunk_logprobs.cpu().contiguous()
                    )
                    if self._should_precompute_teacher_in_rollout():
                        t_lp = compute_teacher_logprobs_one_batch(
                            self._opd_teacher_model,
                            self.hf_model,
                            processed_obs["input_ids"],
                            processed_obs["attention_mask"],
                            processed_obs["pixel_values"],
                            chunk_action_token.reshape(
                                chunk_action_token.shape[0], -1
                            ),
                            self.cfg,
                        )
                        self.buffer_list[i]["teacher_logprobs"].append(
                            t_lp.cpu().contiguous()
                        )
                    self.buffer_list[i]["prev_values"].append(
                        chunk_values.cpu().contiguous()
                    )

            for i in range(self.stage_num):
                env_batch = await self.recv_env_batch()
                self.update_env_batch(i, env_batch)
                processed_obs = prepare_observations(
                    simulator_type=self.cfg.env.train.simulator_type,
                    model_name=self.cfg.actor.model.model_name,
                    raw_obs=env_batch["obs"],
                    use_proprio=self.use_proprio,
                    max_length=self.hf_model.max_prompt_length,
                    processor=self.input_processor,
                    precision=self.precision,
                )
                _, _, _, final_chunk_values = self.predict(processed_obs)
                self.buffer_list[i]["prev_values"].append(
                    final_chunk_values.cpu().contiguous()
                )

                if (
                    not self.cfg.env.train.auto_reset
                    and not self.cfg.env.train.ignore_terminations
                ):
                    infos = env_batch["infos"]
                    if "episode" in infos:
                        for key, value in infos["episode"].items():
                            self.buffer_list[i][f"env_info/{key}"].append(value.cpu())

        # Save action logs to disk
        self.save_action_logs(global_step)

        for i in range(self.stage_num):
            await self.send_rollout_batch(i)
            self.buffer_list[i].clear()

        if _log:
            print(
                f"[Rollout r0] generate: total_wall={time.perf_counter() - _t_gen:.2f}s "
                f"precompute_teacher={self._should_precompute_teacher_in_rollout()} "
                f"stages={self.stage_num} rollout_epoch={self.cfg.algorithm.rollout_epoch} "
                f"n_chunk_steps={self.cfg.algorithm.n_chunk_steps}",
                flush=True,
            )

        gc.collect()
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:
            pass

        if self.cfg.rollout.get("enable_offload", False):
            self.offload_model()

    async def evaluate(self):
        if self.cfg.rollout.get("enable_offload", False):
            self.reload_model(load_opd_teacher=False)
        eval_info = defaultdict(list)

        for step in tqdm(
            range(self.cfg.algorithm.n_eval_chunk_steps),
            desc="Rollout in Eval Step",
            disable=not cfg_show_progress_bar(self.cfg),
        ):
            for i in range(self.stage_num):
                env_batch = await self.recv_env_batch()
                processed_obs = prepare_observations(
                    simulator_type=self.cfg.env.eval.simulator_type,
                    model_name=self.cfg.actor.model.model_name,
                    raw_obs=env_batch["obs"],
                    use_proprio=self.use_proprio,
                    max_length=self.hf_model.max_prompt_length,
                    processor=self.input_processor,
                    precision=self.precision,
                )
                chunk_actions, _, _, _ = self.predict(
                    processed_obs,
                    mode="eval",
                )
                self._maybe_dump_first_teacher_eval_sample(env_batch, chunk_actions)
                await self.send_chunk_actions(chunk_actions)

                if "meta" in env_batch:
                    env_info_list = env_batch["meta"]
                    for key, value in env_info_list.items():
                        eval_info[f"env_info/{key}"].append(value)

        env_batch = await self.recv_env_batch()
        if "meta" in env_batch:
            env_info_list = env_batch["meta"]
            for key, value in env_info_list.items():
                eval_info[f"env_info/{key}"].append(value)
        eval_metrics = create_rollout_batch(eval_info)
        if self.cfg.rollout.get("enable_offload", False):
            self.offload_model()
        return eval_metrics

    def offload_model(self):
        self.hf_model = self.hf_model.to("cpu")
        if self._opd_teacher_model is not None:
            self._opd_teacher_model = self._opd_teacher_model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    def reload_model(self, load_opd_teacher: bool = False):
        """Restore rollout student to GPU; optionally OPD teacher (skipped during eval)."""
        self.hf_model = self.hf_model.to(self.device)
        if load_opd_teacher and self._opd_teacher_model is not None:
            self._opd_teacher_model = self._opd_teacher_model.to(self.device)

    def _wait_for_actor_offload(self, threshold_gb=2.0):
        """Wait until Actor process on this GPU uses less than threshold_gb memory."""
        if self._rank == 0:
            print(f"[Rollout] Waiting for Actor to offload GPU memory (Threshold: {threshold_gb}GB)...")

        while True:
            is_safe = _check_actor_memory(self.device, threshold_gb)
            if is_safe:
                break

            time.sleep(1.0)

    def sync_model_from_actor(self):
        if self.cfg.actor.model.get('use_fsdp2', False):
            print("Waiting for actor to offload memory...", self._rank)
            self._wait_for_actor_offload(threshold_gb=5.0)

        param_state_dict = self.recv(self._actor_group_name, src_rank=self._rank)
        self.hf_model.load_state_dict(param_state_dict)
        del param_state_dict
        gc.collect()
        torch.cuda.empty_cache()

    def clear_cuda_runtime_state(self, keep_teacher_path: bool = True):
        """
        Aggressively clear rollout-side CUDA/IPC state between phases.
        Keep only checkpoint paths in config; force lazy reload later.
        """
        try:
            self.offload_model()
        except Exception:
            pass

        # Keep the saved teacher checkpoint path, but drop in-memory teacher module.
        if keep_teacher_path:
            self._opd_teacher_model = None

        gc.collect()
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
        except Exception:
            pass
        return {}

    async def recv_env_batch(self):
        env_batch = await self.channel.get(
            queue_name=f"{self._obs_queue_name}_{self._rank}", async_op=True
        ).async_wait()
        return env_batch

    async def send_chunk_actions(self, chunk_actions):
        await self.channel.put(
            item=chunk_actions,
            queue_name=f"{self._action_queue_name}_{self._rank}",
            async_op=True,
        ).async_wait()

    async def send_rollout_batch(self, stage_id):
        # send rollout_batch to actor
        send_num = self._component_placement.get_world_size("rollout") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(recv_num, send_num)
        rollout_batch = create_rollout_batch(self.buffer_list[stage_id])
        for i in range(split_num):
            rollout_batch_i = {}
            for key in rollout_batch.keys():
                if "env_info/" not in key:
                    rollout_batch_i[key] = torch.chunk(
                        rollout_batch[key], split_num, dim=1
                    )[i].contiguous()
                else:
                    rollout_batch_i[key] = torch.chunk(
                        rollout_batch[key], split_num, dim=0
                    )[i].contiguous()
            await self.channel.put(
                item=rollout_batch_i, queue_name=self._replay_buffer_name, async_op=True
            ).async_wait()
