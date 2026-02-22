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

"""
CNN Rollout Worker for simple CNN policy.

This worker handles both training rollouts (generate()) and evaluation (evaluate())
for simple CNN policies, producing data in the same format as MultiStepRolloutWorker
so the actor's run_training can consume it with minimal branching.
"""

import gc
import os
from collections import defaultdict

import numpy as np
import torch
from omegaconf import DictConfig
from torchvision import transforms
from tqdm import tqdm

from rlinf.custom.random_action_rollout_worker import create_rollout_batch
from rlinf.models.simple_cnn_policy import SimpleCNNPolicy
from rlinf.scheduler import Cluster, Worker
from rlinf.utils.metric_utils import compute_split_num
from rlinf.utils.placement import HybridComponentPlacement


class CNNRolloutWorker(Worker):
    """
    Rollout worker for simple CNN policy.

    Produces rollout batches in the same format as MultiStepRolloutWorker
    so that the actor's run_training() can consume them with minimal branching.

    Rollout batch keys produced by generate():
        - input_ids: [step, bsz, 1]          (task IDs, used for shape compat)
        - attention_mask: [step, bsz, 1]      (dummy, all ones)
        - pixel_values: [step, bsz, C, H, W]  (preprocessed images)
        - action_tokens: [step, bsz, n_chunks, action_dim]  (bin indices 0-255)
        - prev_logprobs: [step, bsz, n_chunks * action_dim]  (per-token logprobs)
        - prev_values: [step+1, bsz, 1]       (zeros, no value function)
        - rewards: [step, bsz, n_chunks]       (from env)
        - dones: [step+1, bsz, n_chunks]       (from env)
    """

    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        self._env_group_name = cfg.env.group_name
        self._actor_group_name = cfg.actor.group_name
        self.device = torch.cuda.current_device()

        self._obs_queue_name = cfg.env.channel.queue_name
        self._action_queue_name = cfg.rollout.channel.queue_name
        self._replay_buffer_name = cfg.actor.channel.queue_name

        # Stage_num: default to 2, use for pipeline rollout process
        self.stage_num = cfg.rollout.pipeline_stage_num

        self._component_placement = HybridComponentPlacement(cfg, Cluster())
        self.channel = self.connect_channel(cfg.rollout.channel.name)
        for i in range(self._component_placement.get_world_size("rollout")):
            self.channel.create_queue(
                f"{self._action_queue_name}_{i}", maxsize=cfg.rollout.channel.queue_size
            )

        # Action space parameters
        self.action_dim = cfg.actor.model.get("action_dim", 7)
        self.num_action_chunks = cfg.actor.model.get("num_action_chunks", 8)
        image_size_raw = cfg.actor.model.get("image_size", 224)
        if isinstance(image_size_raw, (list, tuple)):
            self.image_size = int(image_size_raw[0])
        else:
            self.image_size = int(image_size_raw)

        # Model will be loaded in init_worker
        self.model = None
        self.task_id_map = None

        # Image preprocessing (same as SimpleCNNEvaluator)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def init_worker(self):
        """Initialize model from checkpoint."""
        checkpoint_path = self.cfg.rollout.get("checkpoint_path", None)
        if not checkpoint_path:
            checkpoint_path = self.cfg.actor.get("checkpoint_load_path", None)

        if not checkpoint_path:
            raise ValueError(
                "CNN rollout worker requires checkpoint_path in rollout.checkpoint_path "
                "or actor.checkpoint_load_path"
            )

        checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{self.device}")

        # Handle two checkpoint formats:
        # 1. Supervised training: dict with "model_state_dict" key
        # 2. RL training (FSDP): dict IS the state_dict directly
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model_state_dict = checkpoint["model_state_dict"]
            task_id_map = checkpoint.get("task_id_map", {})
            num_tasks = checkpoint.get("num_tasks", len(task_id_map))
            norm_stats = checkpoint.get("norm_stats", {})
            unnorm_key = checkpoint.get("unnorm_key", None)
            vocab_size = checkpoint.get("vocab_size", 32000)
            n_action_bins = checkpoint.get("n_action_bins", 256)
        else:
            # RL checkpoint: state dict directly
            model_state_dict = checkpoint
            task_id_map = self.cfg.actor.model.get("task_id_map", {})
            num_tasks = self.cfg.actor.model.get("num_tasks", 10)
            norm_stats = {}
            unnorm_key = self.cfg.actor.model.get("unnorm_key", "libero_spatial_no_noops")
            vocab_size = self.cfg.actor.model.get("vocab_size", 32000)
            n_action_bins = self.cfg.actor.model.get("n_action_bins", 256)

        self.task_id_map = task_id_map

        # Compute norm_stats if not in checkpoint
        if not norm_stats:
            from rlinf.custom.simple_cnn_utils import compute_action_statistics
            dataset_path = os.environ.get("LIBERO_REPO_PATH", "")
            if dataset_path:
                dataset_path = os.path.join(
                    dataset_path, "libero", "datasets_with_logits",
                    "libero_spatial_simplevla_trajall"
                )
                if os.path.exists(dataset_path):
                    norm_stats = compute_action_statistics(
                        dataset_path, unnorm_key=unnorm_key
                    )
                else:
                    raise ValueError(
                        f"norm_stats not in checkpoint and dataset {dataset_path} not found"
                    )
            else:
                raise ValueError(
                    "norm_stats not in checkpoint and LIBERO_REPO_PATH not set"
                )

        # Validate unnorm_key
        if unnorm_key is None:
            if len(norm_stats) == 1:
                unnorm_key = next(iter(norm_stats.keys()))
            else:
                raise ValueError(
                    f"unnorm_key is None and norm_stats has multiple keys: {list(norm_stats.keys())}"
                )

        if unnorm_key not in norm_stats:
            alt_key = f"{unnorm_key}_no_noops"
            if alt_key in norm_stats:
                unnorm_key = alt_key
            else:
                raise ValueError(
                    f"unnorm_key '{unnorm_key}' not found in norm_stats: {list(norm_stats.keys())}"
                )

        # Create model
        self.model = SimpleCNNPolicy(
            action_dim=self.action_dim,
            num_action_chunks=self.num_action_chunks,
            image_size=self.image_size,
            num_tasks=num_tasks,
            use_task_embedding=True,
            vocab_size=vocab_size,
            n_action_bins=n_action_bins,
            norm_stats=norm_stats,
            unnorm_key=unnorm_key,
        ).to(self.device)

        # Load weights
        self.model.load_state_dict(model_state_dict)
        self.model.eval()

        self._logger.info(
            f"CNNRolloutWorker {self._rank} initialized with model from {checkpoint_path}"
        )
        self._logger.info(f"Task ID map: {len(self.task_id_map)} tasks")

        if self.cfg.rollout.get("enable_offload", False):
            self.offload_model()

    # ─── Observation Processing ──────────────────────────────────────────

    def _get_task_id_from_description(self, task_description: str) -> int:
        """Get task ID from task description."""
        if task_description in self.task_id_map:
            return self.task_id_map[task_description]
        for desc, tid in self.task_id_map.items():
            if task_description.lower() in desc.lower() or desc.lower() in task_description.lower():
                return tid
        return 0

    def _preprocess_images(self, images):
        """Preprocess images for CNN model. Returns [B, C, H, W] tensor on device."""
        processed = []
        for img in images:
            if isinstance(img, torch.Tensor):
                if img.dim() == 3:
                    img_np = img.cpu().numpy()
                    if img_np.max() > 1.0:
                        img_np = img_np.astype(np.uint8)
                    else:
                        img_np = (img_np * 255.0).astype(np.uint8)
                    processed.append(self.transform(img_np))
                else:
                    processed.append(img)
            else:
                processed.append(self.transform(img))
        return torch.stack(processed).to(self.device)

    def _process_obs(self, raw_obs):
        """Process raw observations into CNN inputs.

        Returns:
            pixel_values: [B, C, H, W]
            task_ids: [B]
        """
        images = raw_obs["images_and_states"]["full_image"]
        pixel_values = self._preprocess_images(images)
        task_descriptions = raw_obs["task_descriptions"]
        task_ids = torch.tensor(
            [self._get_task_id_from_description(d) for d in task_descriptions],
            dtype=torch.long, device=self.device,
        )
        return pixel_values, task_ids

    # ─── Prediction ──────────────────────────────────────────────────────

    def predict(self, raw_obs, mode="train"):
        """Predict actions from raw observations.

        Returns (matching MultiStepRolloutWorker interface):
            chunk_actions: [B, num_action_chunks, action_dim] continuous actions
            chunk_action_tokens: [B, num_action_chunks, action_dim] bin indices (0-255)
            chunk_logprobs: [B, num_action_chunks * action_dim] per-token logprobs
            chunk_values: [B, 1] (zeros)
            pixel_values: [B, C, H, W] preprocessed images (reuse to avoid double processing)
            task_ids: [B] task IDs
        """
        pixel_values, task_ids = self._process_obs(raw_obs)
        batch_size = pixel_values.shape[0]

        with torch.no_grad():
            output = self.model(
                pixel_values=pixel_values,
                task_ids=task_ids,
                return_logprobs=True,
                return_values=False,
            )

        actions = output["actions"]          # [B, n_chunks, action_dim]
        bin_indices = output["bin_indices"]   # [B, n_chunks, action_dim]
        logprobs = output.get("logprobs", torch.zeros_like(actions))  # [B, n_chunks, action_dim]

        # Flatten logprobs to [B, n_chunks * action_dim] for compatibility
        logprobs_flat = logprobs.reshape(batch_size, -1)

        chunk_values = torch.zeros(batch_size, 1, device=self.device)
        return actions, bin_indices, logprobs_flat, chunk_values, pixel_values, task_ids

    # ─── Training: generate() ────────────────────────────────────────────

    def update_env_batch(self, i, env_batch):
        """Process env info (rewards, dones) - same as MultiStepRolloutWorker."""
        if env_batch is None:
            self._logger.error(
                f"Received None env_batch in update_env_batch (stage={i}). "
                "A worker was likely killed by OOM; see raylet logs."
            )
            return

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

    async def generate(self):
        """Generate training rollouts - mirrors MultiStepRolloutWorker.generate().

        Produces rollout batches with the SAME keys as VLA rollouts so the actor
        can consume them with minimal branching.
        """
        if self.cfg.rollout.get("enable_offload", False):
            self.reload_model()

        self.buffer_list = []
        for i in range(self.stage_num):
            self.buffer_list.append(defaultdict(list))

        for rollout_epoch in range(self.cfg.algorithm.rollout_epoch):
            self._logger.info(f"CNN Rollout epoch={rollout_epoch}")
            for step in tqdm(
                range(self.cfg.algorithm.n_chunk_steps),
                desc=f"CNN Rollout {self._rank} Epoch {rollout_epoch}",
            ):
                for i in range(self.stage_num):
                    env_batch = await self.recv_env_batch()
                    if env_batch is None:
                        raise RuntimeError(
                            f"CNN Rollout rank {self._rank}: recv_env_batch returned None "
                            f"at epoch={rollout_epoch} step={step}. "
                            "A worker was likely killed by OOM; check raylet logs."
                        )
                    self.update_env_batch(i, env_batch)

                    # Predict actions (also returns pixel_values/task_ids
                    # to avoid a second _process_obs call)
                    chunk_actions, chunk_action_tokens, chunk_logprobs, chunk_values, pixel_values, task_ids = (
                        self.predict(env_batch["obs"], mode="train")
                    )

                    # Send actions to env (as numpy, since env uses np.concatenate)
                    chunk_actions_np = chunk_actions.float().cpu().numpy()
                    await self.send_chunk_actions(chunk_actions_np)

                    # Store in buffer with VLA-compatible keys
                    batch_size = chunk_actions.shape[0]

                    # pixel_values: [B, C, H, W] — store as bf16 to halve memory
                    self.buffer_list[i]["pixel_values"].append(
                        pixel_values.to(torch.bfloat16).cpu().contiguous()
                    )

                    # input_ids: [B, 1] - task IDs (for shape compatibility with VLA)
                    self.buffer_list[i]["input_ids"].append(
                        task_ids.unsqueeze(-1).cpu().contiguous()  # [B, 1]
                    )

                    # attention_mask: [B, 1] - dummy
                    self.buffer_list[i]["attention_mask"].append(
                        torch.ones(batch_size, 1, dtype=torch.bool).contiguous()
                    )

                    # action_tokens: [B, n_chunks, action_dim] - bin indices
                    self.buffer_list[i]["action_tokens"].append(
                        chunk_action_tokens.cpu().contiguous()
                    )

                    # prev_logprobs: [B, n_chunks * action_dim]
                    self.buffer_list[i]["prev_logprobs"].append(
                        chunk_logprobs.cpu().contiguous()
                    )

                    # prev_values: [B, 1]
                    self.buffer_list[i]["prev_values"].append(
                        chunk_values.cpu().contiguous()
                    )

            # Final step: get final values (zeros for CNN)
            for i in range(self.stage_num):
                env_batch = await self.recv_env_batch()
                if env_batch is None:
                    raise RuntimeError(
                        f"CNN Rollout rank {self._rank}: recv_env_batch returned None "
                        f"at final step of epoch={rollout_epoch}. "
                        "A worker was likely killed by OOM; check raylet logs."
                    )
                self.update_env_batch(i, env_batch)
                batch_size = len(env_batch["obs"]["task_descriptions"])
                final_values = torch.zeros(batch_size, 1)
                self.buffer_list[i]["prev_values"].append(final_values.contiguous())

                if (
                    not self.cfg.env.train.auto_reset
                    and not self.cfg.env.train.ignore_terminations
                ):
                    infos = env_batch["infos"]
                    if "episode" in infos:
                        for key, value in infos["episode"].items():
                            self.buffer_list[i][f"env_info/{key}"].append(value.cpu())

        # Send rollout batches to actor
        for i in range(self.stage_num):
            await self.send_rollout_batch(i)

        # Free buffer data immediately to reduce memory pressure.
        # The pixel_values alone can consume ~9+ GB per rollout worker;
        # keeping it alive through the training phase causes OOM kills.
        del self.buffer_list
        self.buffer_list = None
        gc.collect()

        if self.cfg.rollout.get("enable_offload", False):
            self.offload_model()

    # ─── Evaluation ──────────────────────────────────────────────────────

    async def evaluate(self):
        """Evaluation loop - compatible with existing infrastructure."""
        if self.cfg.rollout.get("enable_offload", False):
            self.reload_model()

        eval_info = defaultdict(list)

        for step in tqdm(
            range(self.cfg.algorithm.n_eval_chunk_steps),
            desc=f"CNN Rollout {self._rank} Eval",
        ):
            for i in range(self.stage_num):
                env_batch = await self.recv_env_batch()
                chunk_actions, _, _, _, _, _ = self.predict(env_batch["obs"], mode="eval")
                chunk_actions_np = chunk_actions.float().cpu().numpy()
                await self.send_chunk_actions(chunk_actions_np)

                if "meta" in env_batch:
                    for key, value in env_batch["meta"].items():
                        eval_info[f"env_info/{key}"].append(value)

        # Final step
        env_batch = await self.recv_env_batch()
        if "meta" in env_batch:
            for key, value in env_batch["meta"].items():
                eval_info[f"env_info/{key}"].append(value)

        eval_metrics = create_rollout_batch(eval_info)

        if self.cfg.rollout.get("enable_offload", False):
            self.offload_model()

        return eval_metrics

    # ─── Weight sync / communication ─────────────────────────────────────

    def sync_model_from_actor(self):
        """Receive updated model weights from actor."""
        param_state_dict = self.recv(self._actor_group_name, src_rank=self._rank)
        self.model.load_state_dict(param_state_dict)
        del param_state_dict
        gc.collect()
        torch.cuda.empty_cache()

    async def send_rollout_batch(self, stage_id):
        """Send rollout batch to actor - same logic as MultiStepRolloutWorker."""
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
                item=rollout_batch_i,
                queue_name=self._replay_buffer_name,
                async_op=True,
            ).async_wait()

    async def recv_env_batch(self):
        """Receive batch from environment."""
        env_batch = await self.channel.get(
            queue_name=f"{self._obs_queue_name}_{self._rank}", async_op=True
        ).async_wait()
        return env_batch

    async def send_chunk_actions(self, chunk_actions):
        """Send actions to environment."""
        await self.channel.put(
            item=chunk_actions,
            queue_name=f"{self._action_queue_name}_{self._rank}",
            async_op=True,
        ).async_wait()

    def offload_model(self):
        """Offload model to CPU."""
        if self.model is not None:
            self.model = self.model.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()

    def reload_model(self):
        """Reload model to GPU."""
        if self.model is not None:
            self.model = self.model.to(self.device)
