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
Utility functions for simple CNN policy training.
"""

import os
from typing import Dict, Any

import h5py
import numpy as np


def compute_action_statistics(root_dir: str, unnorm_key: str = "libero_spatial_no_noops") -> Dict[str, Any]:
    """
    Compute action normalization statistics from LIBERO dataset.
    
    Args:
        root_dir: Root directory containing HDF5 files
        unnorm_key: Key to use for normalization stats
    
    Returns:
        Dictionary with normalization statistics in format:
        {
            unnorm_key: {
                "action": {
                    "q01": [...],
                    "q99": [...],
                    "min": [...],
                    "max": [...],
                    "mean": [...],
                    "std": [...],
                }
            }
        }
    """
    task_files = [
        os.path.join(root_dir, f)
        for f in os.listdir(root_dir)
        if f.endswith(".hdf5")
    ]
    
    all_actions = []
    
    print(f"Computing action statistics from {len(task_files)} files...")
    for path in task_files:
        with h5py.File(path, "r") as f:
            demo_names = sorted(list(f["data"].keys()))
            for name in demo_names:
                if "actions" in f["data"][name]:
                    actions = np.array(f["data"][name]["actions"])
                    all_actions.append(actions)
    
    if len(all_actions) == 0:
        raise ValueError(f"No actions found in dataset at {root_dir}")
    
    all_actions = np.concatenate(all_actions, axis=0)  # [N, action_dim]
    
    # Compute statistics
    action_stats = {
        "q01": np.quantile(all_actions, 0.01, axis=0).tolist(),
        "q99": np.quantile(all_actions, 0.99, axis=0).tolist(),
        "min": all_actions.min(axis=0).tolist(),
        "max": all_actions.max(axis=0).tolist(),
        "mean": all_actions.mean(axis=0).tolist(),
        "std": all_actions.std(axis=0).tolist(),
    }
    
    norm_stats = {
        unnorm_key: {
            "action": action_stats
        }
    }
    
    print(f"Computed action statistics:")
    print(f"  Action dim: {all_actions.shape[1]}")
    print(f"  Total samples: {len(all_actions)}")
    print(f"  Q01: {action_stats['q01']}")
    print(f"  Q99: {action_stats['q99']}")
    
    return norm_stats
