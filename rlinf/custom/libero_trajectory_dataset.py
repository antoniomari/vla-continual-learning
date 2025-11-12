import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class LiberoSFTDataset(Dataset):
    def __init__(self, root_dir, task_suite_name, demos_per_task=1, rank=0, world_size=1):
        self.root_dir = os.path.join(root_dir, "libero", "datasets", task_suite_name)
        self.task_files = [os.path.join(self.root_dir, f) for f in os.listdir(self.root_dir) if f.endswith(".hdf5")]
        self.demos_per_task = demos_per_task

        self.trajectories = []
        for path in self.task_files:
            with h5py.File(path, "r") as f:
                demo_names = sorted(list(f["data"].keys()))[:demos_per_task]
                for name in demo_names:
                    traj_len = len(f["data"][name]["actions"])
                    self.trajectories.append((path, name, traj_len))

        self.sample_indices = []
        for path, demo_name, traj_len in self.trajectories:
            for t in range(traj_len):
                self.sample_indices.append((path, demo_name, t))

        self.sample_indices = self.sample_indices[rank::world_size]
    
    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        path, demo_name, timestep = self.sample_indices[idx]
        
        with h5py.File(path, "r") as f:
            demo = f["data"][demo_name]
            obs = np.array(demo["obs"]["agentview_rgb"][timestep])
            action = np.array(demo["actions"][timestep])
        
        return {
            'obs': torch.from_numpy(obs).float(),
            'action': torch.from_numpy(action).float(),
        }

if __name__ == "__main__":
    EMBODIED_PATH = os.path.dirname(os.path.abspath(__file__))
    REPO_PATH = os.path.dirname(os.path.dirname(EMBODIED_PATH))
    LIBERO_REPO_PATH = os.path.join(REPO_PATH, "LIBERO")
    sft_dataset = LiberoSFTDataset(
        LIBERO_REPO_PATH, 
        task_suite_name="libero_spatial",
        demos_per_task=1,
        rank=0,
        world_size=4
    )

    sft_dataloader = DataLoader(
        sft_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    sft_iterator = iter(sft_dataloader)
    batch = next(sft_iterator)
    obs, action = batch["obs"], batch["action"]
    print(f"obs shape: {obs.shape}, action shape: {action.shape}")