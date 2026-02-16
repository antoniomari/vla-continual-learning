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

This worker integrates with the existing evaluation infrastructure,
allowing CNN policies to be evaluated using the same eval_embodiment.sh script.
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
from rlinf.envs.libero.utils import get_libero_image
from rlinf.models.simple_cnn_policy import SimpleCNNPolicy
from rlinf.scheduler import Cluster, Worker
from rlinf.utils.placement import HybridComponentPlacement


class CNNRolloutWorker(Worker):
    """
    Rollout worker for simple CNN policy.
    
    Integrates with existing evaluation infrastructure - can be used with
    eval_embodiment.sh by setting rollout.random_action=False and providing
    appropriate config.
    """
    
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        
        self.cfg = cfg
        self._env_group_name = cfg.env.group_name
        self.device = torch.cuda.current_device()
        
        self._obs_queue_name = cfg.env.channel.queue_name
        self._action_queue_name = cfg.rollout.channel.queue_name
        
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
        self.num_action_chunks = cfg.actor.model.get("num_action_chunks", 1)
        image_size_raw = cfg.actor.model.get("image_size", 224)
        # Handle both integer and list formats (VLA configs use [H, W], CNN uses int)
        if isinstance(image_size_raw, (list, tuple)):
            self.image_size = int(image_size_raw[0])  # Use first element for square images
        else:
            self.image_size = int(image_size_raw)
        
        # Model will be loaded in init_worker
        self.model = None
        self.task_id_map = None
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def init_worker(self):
        """Initialize model from checkpoint."""
        # Load checkpoint
        checkpoint_path = self.cfg.rollout.get("checkpoint_path", None)
        if checkpoint_path is None:
            # Try to get from actor config
            checkpoint_path = self.cfg.actor.get("checkpoint_load_path", None)
        
        if checkpoint_path is None:
            raise ValueError(
                "CNN rollout worker requires checkpoint_path in rollout.checkpoint_path "
                "or actor.checkpoint_load_path"
            )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{self.device}")
        
        # Get model config
        task_id_map = checkpoint.get("task_id_map", {})
        num_tasks = checkpoint.get("num_tasks", len(task_id_map))
        norm_stats = checkpoint.get("norm_stats", {})
        unnorm_key = checkpoint.get("unnorm_key", None)
        vocab_size = checkpoint.get("vocab_size", 32000)
        n_action_bins = checkpoint.get("n_action_bins", 256)
        self.task_id_map = task_id_map
        
        # Validate that norm_stats are present
        if not norm_stats or len(norm_stats) == 0:
            raise ValueError(
                f"Checkpoint at {checkpoint_path} does not contain norm_stats or norm_stats is empty! "
                f"This is required for action unnormalization. "
                f"Checkpoint keys: {list(checkpoint.keys())}"
            )
        
        # Validate that unnorm_key is present and matches a key in norm_stats
        if unnorm_key is None:
            # Try to infer from norm_stats
            if len(norm_stats) == 1:
                unnorm_key = next(iter(norm_stats.keys()))
            else:
                raise ValueError(
                    f"unnorm_key is None in checkpoint and norm_stats has multiple keys: {list(norm_stats.keys())}. "
                    f"Please specify unnorm_key in the checkpoint or use a dataset with a single normalization key."
                )
        
        if unnorm_key not in norm_stats:
            # Try with _no_noops suffix
            alt_key = f"{unnorm_key}_no_noops"
            if alt_key in norm_stats:
                unnorm_key = alt_key
            else:
                raise ValueError(
                    f"unnorm_key '{unnorm_key}' not found in norm_stats! "
                    f"Available keys: {list(norm_stats.keys())}"
                )
        
        # Create model with action tokenization support
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
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        
        self._logger.info(
            f"CNNRolloutWorker {self._rank} initialized with model from {checkpoint_path}"
        )
        self._logger.info(f"Task ID map: {len(self.task_id_map)} tasks")
    
    def _get_task_id_from_description(self, task_description: str) -> int:
        """Get task ID from task description."""
        if task_description in self.task_id_map:
            return self.task_id_map[task_description]
        
        # Try partial match
        for desc, tid in self.task_id_map.items():
            if task_description.lower() in desc.lower() or desc.lower() in task_description.lower():
                return tid
        
        # Fallback to 0
        return 0
    
    def _preprocess_images(self, images):
        """Preprocess images for CNN model."""
        # images is a list of tensors [H, W, 3] from env
        processed_images = []
        conversion_changed = False
        
        for img in images:
            if isinstance(img, torch.Tensor):
                # Convert to numpy if needed, or handle tensor directly
                if img.dim() == 3:
                    # [H, W, 3] tensor - convert to numpy for PIL
                    img_np_old = None
                    if img.max() > 1.0:
                        # OLD (BUGGY): Divide by 255.0 then convert to uint8 (truncates!)
                        img_np_old = (img.cpu().numpy() / 255.0).astype(np.uint8)
                        # NEW (CORRECT): Convert directly to uint8 [0, 255]
                        img_np = img.cpu().numpy().astype(np.uint8)
                    else:
                        # Tensor is in [0, 1] range, convert to [0, 255] uint8
                        img_np = (img.cpu().numpy() * 255.0).astype(np.uint8)
                    
                    # Check if conversion changed anything (for debugging)
                    if img_np_old is not None:
                        if not np.array_equal(img_np, img_np_old):
                            conversion_changed = True
                            if not hasattr(self, '_conversion_warning_shown'):
                                self._logger.warning(
                                    f"[CNNRolloutWorker] Image conversion fix detected difference! "
                                    f"Old method: max={img_np_old.max()}, mean={img_np_old.mean():.2f}, "
                                    f"New method: max={img_np.max()}, mean={img_np.mean():.2f}. "
                                    f"This fix ensures correct preprocessing matching SimpleCNNEvaluator."
                                )
                                self._conversion_warning_shown = True
                    
                    img_processed = self.transform(img_np)
                else:
                    # Already in [C, H, W] format
                    img_processed = img
            else:
                # Assume numpy array [H, W, 3]
                img_processed = self.transform(img)
            processed_images.append(img_processed)
        
        return torch.stack(processed_images).to(self.device)  # [B, C, H, W]
    
    def predict(self, raw_obs, mode="eval"):
        """
        Predict actions from raw observations.
        
        Args:
            raw_obs: Raw observation dict with:
                - images_and_states: dict with "full_image" key
                - task_descriptions: list of task description strings
            mode: "train" or "eval" (unused for CNN, kept for compatibility)
        
        Returns:
            chunk_actions: [B, num_action_chunks, action_dim]
            chunk_action_tokens: [B, num_action_chunks, action_dim] (same as actions)
            chunk_logprobs: [B, num_action_chunks, action_dim] (dummy for compatibility)
            chunk_values: [B, 1] (dummy for compatibility)
        """
        batch_size = len(raw_obs["task_descriptions"])
        
        # Extract images
        images = raw_obs["images_and_states"]["full_image"]  # List of [H, W, 3] tensors
        pixel_values = self._preprocess_images(images)  # [B, C, H, W]
        
        # Get task IDs
        task_descriptions = raw_obs["task_descriptions"]
        task_ids = torch.tensor(
            [self._get_task_id_from_description(desc) for desc in task_descriptions],
            dtype=torch.long,
            device=self.device,
        )
        
        # Predict actions (model outputs discrete tokens and decodes to continuous actions)
        with torch.no_grad():
            output = self.model(
                pixel_values=pixel_values,
                task_ids=task_ids,
                return_logprobs=True,  # Get logprobs for compatibility
                return_values=False,
            )
            actions = output["actions"]  # [B, num_action_chunks, action_dim] (decoded continuous)
            bin_indices = output["bin_indices"]  # [B, num_action_chunks, action_dim] (bin indices 0-255)
            logprobs = output.get("logprobs")  # [B, num_action_chunks, action_dim]
        
        # For compatibility with existing interface
        chunk_actions = actions
        chunk_action_tokens = bin_indices  # Use bin_indices as "tokens" for compatibility
        chunk_logprobs = logprobs if logprobs is not None else torch.zeros_like(actions)
        chunk_values = torch.zeros(batch_size, 1, device=self.device)  # Dummy
        
        return chunk_actions, chunk_action_tokens, chunk_logprobs, chunk_values
    
    async def evaluate(self):
        """Evaluation loop - compatible with existing infrastructure."""
        if self.cfg.rollout.get("enable_offload", False):
            self.reload_model()
        
        eval_info = defaultdict(list)
        
        for step in tqdm(
            range(self.cfg.algorithm.n_eval_chunk_steps),
            desc=f"CNN Rollout Worker {self._rank} in Eval Step",
        ):
            for i in range(self.stage_num):
                env_batch = await self.recv_env_batch()
                
                # Predict actions using CNN policy
                chunk_actions, _, _, _ = self.predict(
                    env_batch["obs"],
                    mode="eval",
                )
                chunk_actions = chunk_actions.float().cpu().contiguous()
                
                await self.send_chunk_actions(chunk_actions)
                
                if "meta" in env_batch:
                    env_info_list = env_batch["meta"]
                    for key, value in env_info_list.items():
                        eval_info[f"env_info/{key}"].append(value)
        
        # Final step
        env_batch = await self.recv_env_batch()
        if "meta" in env_batch:
            env_info_list = env_batch["meta"]
            for key, value in env_info_list.items():
                eval_info[f"env_info/{key}"].append(value)
        
        eval_metrics = create_rollout_batch(eval_info)
        
        if self.cfg.rollout.get("enable_offload", False):
            self.offload_model()
        
        return eval_metrics
    
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
