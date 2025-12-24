import gc
from collections import defaultdict

import torch
from omegaconf import DictConfig
from tqdm import tqdm

from rlinf.scheduler import Cluster, Worker
from rlinf.utils.placement import HybridComponentPlacement


def create_rollout_batch(data):
    ret_data = {}
    for key, value in data.items():
        if "env_info/" not in key:
            ret_data[key] = torch.stack(value, dim=0).contiguous().cpu()
        else:
            ret_data[key] = torch.cat(value, dim=0).contiguous().cpu()
    return ret_data


class RandomActionRolloutWorker(Worker):
    """Rollout worker that uses hardcoded repeating actions instead of using a model."""

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

        # Create base action patterns (one per dimension)
        base_actions = [
            [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ]

        # Build action sequence with proper chunking
        # Shape: [num_sequence_steps, num_action_chunks, action_dim]
        action_list = []
        for base_action in base_actions:
            # Repeat the same action across all chunks
            chunks = [base_action for _ in range(self.num_action_chunks)]
            action_list.append(chunks)

        self.action_sequence = torch.tensor(
            action_list, device=self.device, dtype=torch.float32
        )

        self.current_action_idx = 0

    def init_worker(self):
        """Initialize worker - no model needed for hardcoded actions."""
        self._logger.info(
            f"RandomActionRolloutWorker {self._rank} initialized with hardcoded action sequence"
        )
        self._logger.info(
            f"Action sequence shape: {self.action_sequence.shape} "
            f"(num_steps={len(self.action_sequence)}, chunks={self.num_action_chunks}, dim={self.action_dim})"
        )

    def get_next_action(self, batch_size):
        """Get the next action from the sequence, repeated across batch dimension."""
        # Get current action: [num_action_chunks, action_dim]
        action = self.action_sequence[self.current_action_idx]

        # Repeat across batch dimension: [batch_size, num_action_chunks, action_dim]
        chunk_actions = action.unsqueeze(0).repeat(batch_size, 1, 1)

        # Move to next action in sequence
        self.current_action_idx = (self.current_action_idx + 1) % len(
            self.action_sequence
        )

        # Return with correct shape for environment
        # Shape should be [batch_size, num_action_chunks, action_dim]
        return chunk_actions.contiguous()

    async def evaluate(self):
        """Evaluation loop with hardcoded actions."""
        eval_info = defaultdict(list)

        for step in tqdm(
            range(self.cfg.algorithm.n_eval_chunk_steps),
            desc=f"Hardcoded Action Rollout Worker {self._rank} in Eval Step",
        ):
            for i in range(self.stage_num):
                env_batch = await self.recv_env_batch()

                batch_size = env_batch["obs"]["images_and_states"]["full_image"].shape[
                    0
                ]

                # Get next hardcoded action
                chunk_actions = self.get_next_action(batch_size)
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
