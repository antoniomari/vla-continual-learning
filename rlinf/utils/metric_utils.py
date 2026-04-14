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

import math
from typing import Dict

import torch
import torch.distributed


def compute_split_num(num, split_num):
    return math.lcm(num, split_num) // split_num


# def compute_evaluate_metrics(eval_metrics_list):
#     """
#     List of evaluate metrics, list length stands for rollout process
#     """
#     all_eval_metrics = {}
#     env_info_keys = eval_metrics_list[0].keys()

#     for env_info_key in env_info_keys:
#         all_eval_metrics[env_info_key] = [
#             eval_metrics[env_info_key] for eval_metrics in eval_metrics_list
#         ]

#     for key in all_eval_metrics:
#         all_eval_metrics[key] = (
#             torch.concat(all_eval_metrics[key]).float().mean().numpy()
#         )

#     return all_eval_metrics

def compute_evaluate_metrics(eval_metrics_list):
    all_eval_metrics = {}

    if not eval_metrics_list:
        raise RuntimeError(
            "compute_evaluate_metrics: empty eval_metrics_list (rollout returned no metrics). "
            "If Ray logged 'Failed to register worker to Raylet: EOF', a worker likely crashed "
            "during startup—check stderr under $RAY_TMPDIR or /tmp/ray/session_latest/logs/."
        )
    
    # Identify task-specific success keys
    task_success_keys = [
        key for key in eval_metrics_list[0].keys()
        if "task_" in key and key.endswith("_success")
    ]
    
    # Compute task-specific success rates
    task_stats = {}  # {task_id: {'successes': count, 'total': count}}
    
    prefix = "env_info"
    for eval_metrics in eval_metrics_list:
        task_ids = eval_metrics[f"{prefix}/task_id"]
        
        for task_key in task_success_keys:
            task_id = int(task_key.split("_")[2]) # env, info/task, {id}, success
            
            if task_id not in task_stats:
                task_stats[task_id] = {'successes': 0, 'total': 0}
            
            # Get which completed episodes ran this task
            task_mask = (task_ids == task_id)
            n_episodes_this_task = task_mask.sum().item()
            
            if n_episodes_this_task > 0:
                # Get success values for episodes that ran this task
                successes = eval_metrics[task_key][task_mask]
                n_successes = successes.float().sum().item()
                
                task_stats[task_id]['successes'] += n_successes
                task_stats[task_id]['total'] += n_episodes_this_task
    
    # Compute final success rates per task
    for task_id, stats in task_stats.items():
        task_key = f"{prefix}/task_{task_id}_success"
        if stats['total'] > 0:
            all_eval_metrics[task_key] = stats['successes'] / stats['total']
        else:
            all_eval_metrics[task_key] = -1.0

        all_eval_metrics[f"{task_key}_total"] = stats['total']
    
    # Collect and compute non-task-specific metrics
    non_task_keys = [
        key for key in eval_metrics_list[0].keys()
        if key not in task_success_keys and key != f"{prefix}/task_id"
    ]
    
    for key in non_task_keys:
        collected_values = [
            eval_metrics[key] for eval_metrics in eval_metrics_list
            if key in eval_metrics
        ]
        
        if collected_values:
            all_eval_metrics[key] = (
                torch.concat(collected_values).float().mean().item()
            )

    return all_eval_metrics


def compute_rollout_metrics(data_buffer: Dict) -> Dict:
    """
    Compute rollout metrics in a data-parallel setting.

    Key constraints:
    - All ranks must participate in the same sequence of collective ops.
    - Some ranks may be missing certain keys (e.g., task-specific env_info),
      especially in multi-task settings.

    This implementation:
    - Ensures all ranks participate in the all_reduce calls for rewards,
      advantages, and returns if any rank has them.
    - Gathers the union of env_info keys across ranks and processes them
      in a consistent, sorted order, using zeros on ranks that lack a key.
    """
    import torch.distributed as dist

    rollout_metrics: Dict[str, float] = {}

    # ---- rewards / advantages / returns ----
    required_keys = ["rewards", "advantages", "returns"]
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    if dist.is_initialized() and world_size > 1:
        # For each key, determine if any rank has it
        local_has_keys = {key: key in data_buffer for key in required_keys}
        all_has_keys_list = [None] * world_size
        dist.all_gather_object(all_has_keys_list, local_has_keys)

        keys_to_process = {
            key: any(rank_has.get(key, False) for rank_has in all_has_keys_list)
            for key in required_keys
        }
    else:
        # Single process: just use local presence
        keys_to_process = {key: key in data_buffer for key in required_keys}

    # Rewards
    if keys_to_process.get("rewards", False):
        if "rewards" in data_buffer:
            rewards = data_buffer["rewards"].clone()
            mean_rewards = torch.mean(rewards).to(torch.cuda.current_device())
        else:
            # Rank without rewards still participates in all_reduce
            mean_rewards = torch.tensor(
                0.0, device=torch.cuda.current_device(), dtype=torch.float32
            )

        if dist.is_initialized() and world_size > 1:
            dist.all_reduce(mean_rewards, op=dist.ReduceOp.AVG)

        rollout_metrics["rewards"] = mean_rewards.item()

    # Advantages
    if keys_to_process.get("advantages", False):
        if "advantages" in data_buffer:
            advantages = data_buffer["advantages"]
            mean_adv = torch.mean(advantages).to(torch.cuda.current_device())
            max_adv = torch.max(advantages).detach().item()
            min_adv = torch.min(advantages).detach().item()
        else:
            # Rank without advantages still participates in all_reduce
            mean_adv = torch.tensor(
                0.0, device=torch.cuda.current_device(), dtype=torch.float32
            )
            max_adv = float("-inf")
            min_adv = float("inf")

        if dist.is_initialized() and world_size > 1:
            dist.all_reduce(mean_adv, op=dist.ReduceOp.AVG)

        reduce_adv_tensor = torch.as_tensor(
            [-min_adv, max_adv],
            device=torch.cuda.current_device(),
            dtype=torch.float32,
        )
        if dist.is_initialized() and world_size > 1:
            dist.all_reduce(reduce_adv_tensor, op=dist.ReduceOp.MAX)
        min_adv, max_adv = reduce_adv_tensor.tolist()

        rollout_metrics.update(
            {
                "advantages_mean": mean_adv.item(),
                "advantages_max": max_adv,
                "advantages_min": -min_adv,
            }
        )

    # Returns
    if keys_to_process.get("returns", False):
        if "returns" in data_buffer:
            returns = data_buffer["returns"]
            mean_ret = torch.mean(returns).to(torch.cuda.current_device())
            max_ret = torch.max(returns).detach().item()
            min_ret = torch.min(returns).detach().item()
        else:
            mean_ret = torch.tensor(
                0.0, device=torch.cuda.current_device(), dtype=torch.float32
            )
            max_ret = float("-inf")
            min_ret = float("inf")

        if dist.is_initialized() and world_size > 1:
            dist.all_reduce(mean_ret, op=dist.ReduceOp.AVG)

        reduce_ret_tensor = torch.as_tensor(
            [-min_ret, max_ret],
            device=torch.cuda.current_device(),
            dtype=torch.float32,
        )
        if dist.is_initialized() and world_size > 1:
            dist.all_reduce(reduce_ret_tensor, op=dist.ReduceOp.MAX)
        min_ret, max_ret = reduce_ret_tensor.tolist()

        rollout_metrics.update(
            {
                "returns_mean": mean_ret.item(),
                "returns_max": max_ret,
                "returns_min": -min_ret,
            }
        )

    # ---- env_info/* keys ----
    # Collect union of env_info keys across all ranks, and ensure every rank
    # calls all_reduce for each key in the same order.
    local_env_info_keys = sorted(
        key for key in data_buffer.keys() if key.startswith("env_info/")
    )

    if dist.is_initialized() and world_size > 1:
        keys_list = [None] * world_size
        dist.all_gather_object(keys_list, local_env_info_keys)
        all_env_info_keys = sorted({k for keys in keys_list for k in keys})
    else:
        all_env_info_keys = local_env_info_keys

    for env_info_key in all_env_info_keys:
        if env_info_key in data_buffer:
            value = data_buffer.pop(env_info_key)
            value = value.float().mean().to(torch.cuda.current_device())
        else:
            value = torch.tensor(
                0.0, device=torch.cuda.current_device(), dtype=torch.float32
            )

        if dist.is_initialized() and world_size > 1:
            dist.all_reduce(value, op=dist.ReduceOp.AVG)

        rollout_metrics[env_info_key] = value.item()

    return rollout_metrics


def append_to_dict(data, new_data):
    for key, val in new_data.items():
        if key not in data:
            data[key] = []
        data[key].append(val)


def compute_loss_mask(dones):
    _, actual_bsz, num_action_chunks = dones.shape
    n_chunk_step = dones.shape[0] - 1
    flattened_dones = dones.transpose(1, 2).reshape(
        -1, actual_bsz
    )  # [n_chunk_step + 1, rollout_epoch x bsz]
    flattened_dones = flattened_dones[
        -(n_chunk_step * num_action_chunks + 1) :
    ]  # [n_steps+1, actual-bsz]
    flattened_loss_mask = (flattened_dones.cumsum(dim=0) == 0)[
        :-1
    ]  # [n_steps, actual-bsz]

    loss_mask = flattened_loss_mask.reshape(n_chunk_step, num_action_chunks, actual_bsz)
    loss_mask = loss_mask.transpose(
        1, 2
    )  # [n_chunk_step, actual_bsz, num_action_chunks]

    loss_mask_sum = loss_mask.sum(dim=(0, 2), keepdim=True)  # [1, bsz, 1]
    loss_mask_sum = loss_mask_sum.expand_as(loss_mask)

    return loss_mask, loss_mask_sum


def expand_loss_mask_to_match_logprob_tokens(loss_mask, loss_mask_sum, logprobs):
    """Broadcast chunk-level mask to token-level logprobs for OPD / token-level policy.

    ``compute_loss_mask`` yields ``[..., num_action_chunks]`` while rollout ``prev_logprobs``
    are ``[..., action_dim * num_action_chunks]`` (one logprob per discretized action token).
    Repeat each chunk's mask across ``action_dim`` so advantages and actor loss multiply
    element-wise with logprobs.
    """
    if loss_mask is None:
        return loss_mask, loss_mask_sum
    n_chunk = loss_mask.shape[-1]
    n_tok = logprobs.shape[-1]
    if n_chunk == n_tok:
        return loss_mask, loss_mask_sum
    if n_tok % n_chunk != 0:
        raise ValueError(
            f"loss_mask last dim ({n_chunk}) must divide logprobs last dim ({n_tok}); "
            f"loss_mask.shape={tuple(loss_mask.shape)}, logprobs.shape={tuple(logprobs.shape)}"
        )
    loss_mask = loss_mask.repeat_interleave(n_tok // n_chunk, dim=-1)
    loss_mask_sum = loss_mask.sum(dim=(0, 2), keepdim=True).expand_as(loss_mask)
    return loss_mask, loss_mask_sum
