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
Parallel evaluation system for simple CNN policy on LIBERO.

This module provides:
- Parallel environment evaluation using SubprocVectorEnv
- Result saving (JSON, CSV)
- Video/GIF saving for rollouts
- Reusable for RL training later
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import imageio
import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs.venv import SubprocVectorEnv, DummyVectorEnv
from tqdm import tqdm

from rlinf.envs.libero.utils import get_libero_image
from torchvision import transforms


class SimpleCNNEvaluator:
    """
    Parallel evaluator for simple CNN policy.
    
    Designed to be reusable for RL training later.
    """
    
    def __init__(
        self,
        model,
        task_id_map: Dict[str, int],
        device: torch.device = torch.device("cuda"),
        num_parallel_envs: int = 20,
        image_size: int = 224,
        save_dir: Optional[str] = None,
        save_videos: bool = True,
        video_fps: int = 10,
    ):
        """
        Args:
            model: The policy model (SimpleCNNPolicy or compatible)
            task_id_map: Mapping from task description to task ID
            device: Device to run model on
            num_parallel_envs: Number of parallel environments
            image_size: Image size for preprocessing
            save_dir: Directory to save results and videos
            save_videos: Whether to save rollout videos
            video_fps: FPS for saved videos
        """
        self.model = model
        self.task_id_map = task_id_map
        self.device = device
        self.num_parallel_envs = num_parallel_envs
        self.image_size = image_size
        self.save_dir = save_dir
        self.save_videos = save_videos
        self.video_fps = video_fps
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Create save directory
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            if self.save_videos:
                os.makedirs(os.path.join(self.save_dir, "videos"), exist_ok=True)
    
    def preprocess_image(self, obs_image: np.ndarray) -> torch.Tensor:
        """Preprocess observation image."""
        image = self.transform(obs_image)  # [C, H, W]
        return image.unsqueeze(0)  # [1, C, H, W]
    
    def get_task_id_for_model(self, task_description: str, task_id: int) -> int:
        """Get task ID for model from task description."""
        if task_description in self.task_id_map:
            return self.task_id_map[task_description]
        
        # Try partial match
        for desc, tid in self.task_id_map.items():
            if task_description.lower() in desc.lower() or desc.lower() in task_description.lower():
                return tid
        
        # Fallback
        if task_id < len(self.task_id_map):
            return task_id
        return 0
    
    def predict_actions_batch(
        self,
        pixel_values: torch.Tensor,
        task_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict actions for a batch of observations.
        
        This interface is designed to be compatible with RL training later.
        """
        with torch.no_grad():
            output = self.model(
                pixel_values=pixel_values,
                task_ids=task_ids,
                return_logprobs=False,
                return_values=False,
            )
            return output["actions"][:, 0]  # [B, action_dim] - take first chunk
    
    def evaluate_task_parallel(
        self,
        task_suite,
        task_id: int,
        num_trials: int = 10,
        max_steps: int = 512,
    ) -> Dict:
        """
        Evaluate a single task using parallel environments.
        
        Args:
            task_suite: LIBERO task suite
            task_id: Task ID to evaluate
            num_trials: Number of evaluation trials
            max_steps: Maximum steps per episode
        
        Returns:
            Dictionary with evaluation results
        """
        task = task_suite.get_task(task_id)
        task_description = task.language
        task_id_for_model = self.get_task_id_for_model(task_description, task_id)
        
        # Create environment
        bddl_file = os.path.join(
            get_libero_path("bddl_files"),
            task.problem_folder,
            task.bddl_file,
        )
        
        # Use parallel environments
        env_num = min(self.num_parallel_envs, num_trials)
        env_args = {
            "bddl_file_name": bddl_file,
            "camera_heights": 128,
            "camera_widths": 128,
        }
        
        # Create vectorized environment
        if env_num == 1:
            env = DummyVectorEnv([lambda: OffScreenRenderEnv(**env_args)])
        else:
            env = SubprocVectorEnv([lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)])
        
        # Load initial states if available
        try:
            init_states_path = os.path.join(
                get_libero_path("init_states"),
                task.problem_folder,
                task.init_states_file,
            )
            if os.path.exists(init_states_path):
                init_states = torch.load(init_states_path)
                # Shuffle initial states once for randomization across trials
                if isinstance(init_states, torch.Tensor):
                    indices = torch.randperm(len(init_states))
                    init_states = init_states[indices]
                else:
                    indices = np.random.permutation(len(init_states))
                    init_states = init_states[indices]
            else:
                init_states = None
        except:
            init_states = None
        
        # Run evaluation in batches
        all_successes = []
        all_episode_lengths = []
        all_videos = []
        
        num_batches = (num_trials + env_num - 1) // env_num
        
        for batch_idx in range(num_batches):
            batch_size = min(env_num, num_trials - batch_idx * env_num)
            if batch_size <= 0:
                break
            
            # Reset environments
            # SubprocVectorEnv.reset() returns numpy array (object dtype) of observation dicts
            obs_array = env.reset()
            # Convert to list of dicts for easier handling
            if isinstance(obs_array, np.ndarray):
                if obs_array.dtype == object:
                    # Object array: each element is a dict
                    obs_list = [obs_array[i] for i in range(batch_size)]
                else:
                    # Shouldn't happen for dict observations, but handle it
                    obs_list = [obs_array[i] for i in range(batch_size)]
            elif isinstance(obs_array, tuple):
                # (obs, info) format
                obs_array = obs_array[0]
                if isinstance(obs_array, np.ndarray) and obs_array.dtype == object:
                    obs_list = [obs_array[i] for i in range(batch_size)]
                else:
                    obs_list = [obs_array[i] for i in range(batch_size)]
            else:
                obs_list = [obs_array] if batch_size == 1 else list(obs_array)
            
            # Set initial states if available
            if init_states is not None:
                # Use different offset for each batch to ensure different initial states
                start_idx = (batch_idx * env_num) % len(init_states)
                batch_indices = (np.arange(batch_size) + start_idx) % len(init_states)
                batch_init_states = init_states[batch_indices]
                # Convert to numpy if needed
                if isinstance(batch_init_states, torch.Tensor):
                    batch_init_states = batch_init_states.cpu().numpy()
                # set_init_state returns updated observations as numpy array
                obs_array = env.set_init_state(batch_init_states)
                # Convert to list of dicts
                if isinstance(obs_array, np.ndarray) and obs_array.dtype == object:
                    obs_list = [obs_array[i] for i in range(batch_size)]
                else:
                    obs_list = [obs_array[i] for i in range(batch_size)]
                # Simulate physics
                for _ in range(5):
                    step_result = env.step(np.zeros((batch_size, 7)))
                    obs_array = step_result[0]
                    # Convert to list of dicts
                    if isinstance(obs_array, np.ndarray) and obs_array.dtype == object:
                        obs_list = [obs_array[i] for i in range(batch_size)]
                    else:
                        obs_list = [obs_array[i] for i in range(batch_size)]
            
            # Storage for this batch
            batch_dones = np.zeros(batch_size, dtype=bool)
            batch_steps = np.zeros(batch_size, dtype=int)
            batch_videos = [[] for _ in range(batch_size)]
            
            # Run episodes
            for step in range(max_steps):
                # Collect images for videos
                if self.save_videos:
                    for i, obs in enumerate(obs_list):
                        if not batch_dones[i]:
                            image = get_libero_image(obs)
                            batch_videos[i].append(image)
                
                # Preprocess observations
                images = [get_libero_image(obs) for obs in obs_list]
                pixel_values = torch.stack([
                    self.preprocess_image(img).squeeze(0) for img in images
                ]).to(self.device)  # [B, C, H, W]
                
                task_ids_tensor = torch.full(
                    (batch_size,), task_id_for_model, dtype=torch.long, device=self.device
                )
                
                # Predict actions
                actions = self.predict_actions_batch(pixel_values, task_ids_tensor)
                actions_np = actions.cpu().numpy()  # [B, action_dim]
                
                # CRITICAL: Apply gripper transformation (same as prepare_actions_for_libero and replay_actions)
                # This must match exactly what happens in replay_actions (inspect_dataset.py lines 414-416)
                # and prepare_actions_for_libero (action_utils.py lines 71-72)
                actions_np = actions_np.copy()
                actions_np[:, -1] = 2 * actions_np[:, -1] - 1  # Normalize [0,1] -> [-1,+1]
                actions_np[:, -1] = np.sign(actions_np[:, -1]) * -1.0  # Invert and binarize
                
                # Step environments
                step_result = env.step(actions_np)
                obs_array, rewards, dones, infos = step_result[:4]
                
                # Convert observation array to list of dicts
                if isinstance(obs_array, np.ndarray) and obs_array.dtype == object:
                    obs_list = [obs_array[i] for i in range(batch_size)]
                elif isinstance(obs_array, np.ndarray):
                    obs_list = [obs_array[i] for i in range(batch_size)]
                else:
                    obs_list = [obs_array] if batch_size == 1 else list(obs_array)
                
                # Ensure dones is array
                if not isinstance(dones, np.ndarray):
                    dones = np.array(dones)
                
                # Update tracking
                batch_dones = batch_dones | dones
                batch_steps += 1
                
                # Check if all done
                if batch_dones.all():
                    break
            
            # Collect results
            for i in range(batch_size):
                success = batch_dones[i]
                all_successes.append(success)
                all_episode_lengths.append(batch_steps[i])
                
                # Save video if requested
                if self.save_videos and len(batch_videos[i]) > 0:
                    video_path = os.path.join(
                        self.save_dir,
                        "videos",
                        f"task_{task_id}_trial_{batch_idx * env_num + i}_{'success' if success else 'fail'}.gif"
                    )
                    imageio.mimsave(
                        video_path,
                        batch_videos[i],
                        fps=self.video_fps,
                    )
                    all_videos.append(video_path)
        
        env.close()
        
        # Compute statistics
        success_rate = np.mean(all_successes)
        avg_episode_length = np.mean(all_episode_lengths)
        
        result = {
            "task_id": task_id,
            "task_description": task_description,
            "success_rate": float(success_rate),
            "num_successes": int(np.sum(all_successes)),
            "num_trials": len(all_successes),
            "avg_episode_length": float(avg_episode_length),
            "video_paths": all_videos if self.save_videos else [],
        }
        
        return result
    
    def evaluate_suite(
        self,
        task_suite_name: str,
        num_trials_per_task: int = 10,
        max_steps: int = 512,
        task_ids: Optional[List[int]] = None,
    ) -> Dict:
        """
        Evaluate entire task suite.
        
        Args:
            task_suite_name: Name of task suite (e.g., "libero_spatial")
            num_trials_per_task: Number of trials per task
            max_steps: Maximum steps per episode
            task_ids: Optional list of task IDs to evaluate (default: all)
        
        Returns:
            Dictionary with all evaluation results
        """
        # Get task suite
        benchmark_dict = benchmark.get_benchmark_dict()
        if task_suite_name not in benchmark_dict:
            raise ValueError(f"Unknown task suite: {task_suite_name}")
        
        task_suite = benchmark_dict[task_suite_name]()
        num_tasks = task_suite.n_tasks
        
        if task_ids is None:
            task_ids = list(range(num_tasks))
        
        # Evaluate each task
        all_results = []
        for task_id in tqdm(task_ids, desc="Evaluating tasks"):
            result = self.evaluate_task_parallel(
                task_suite,
                task_id,
                num_trials=num_trials_per_task,
                max_steps=max_steps,
            )
            all_results.append(result)
            
            print(
                f"Task {result['task_id']}: {result['success_rate']:.2%} "
                f"({result['num_successes']}/{result['num_trials']})"
            )
        
        # Compute overall statistics
        total_successes = sum(r["num_successes"] for r in all_results)
        total_trials = sum(r["num_trials"] for r in all_results)
        avg_success_rate = total_successes / total_trials if total_trials > 0 else 0.0
        
        summary = {
            "task_suite": task_suite_name,
            "num_tasks": len(all_results),
            "total_trials": total_trials,
            "total_successes": total_successes,
            "avg_success_rate": float(avg_success_rate),
            "per_task_results": all_results,
        }
        
        # Save results
        if self.save_dir:
            # Save JSON
            json_path = os.path.join(self.save_dir, "evaluation_results.json")
            with open(json_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"Results saved to {json_path}")
            
            # Save CSV summary
            csv_path = os.path.join(self.save_dir, "evaluation_summary.csv")
            import csv
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["task_id", "task_description", "success_rate", "num_successes", "num_trials", "avg_episode_length"])
                for r in all_results:
                    writer.writerow([
                        r["task_id"],
                        r["task_description"],
                        f"{r['success_rate']:.4f}",
                        r["num_successes"],
                        r["num_trials"],
                        f"{r['avg_episode_length']:.2f}",
                    ])
            print(f"Summary saved to {csv_path}")
        
        return summary
