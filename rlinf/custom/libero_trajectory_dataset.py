import hydra
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from rlinf.models import get_model_config_and_processor
from rlinf.config import torch_dtype_from_precision
from rlinf.models.embodiment.model_utils import prepare_observations

class LiberoSFTDataset(Dataset):
    def __init__(self, cfg, root_dir, demos_per_task=1, rank=0, world_size=1):
        self.cfg = cfg
        task_suite_name = cfg.env.train.task_suite_name
        self.root_dir = os.path.join(root_dir, "libero", "datasets", task_suite_name)
        self.task_files = [os.path.join(self.root_dir, f) for f in os.listdir(self.root_dir) if f.endswith(".hdf5")]
        self.demos_per_task = demos_per_task
        self.num_action_chunks = cfg.actor.model.num_action_chunks

        _, self.input_processor = get_model_config_and_processor(cfg.actor)
        self.precision = torch_dtype_from_precision(cfg.actor.model.precision)

        self.trajectories = []
        for path in self.task_files:
            filename = os.path.basename(path)
            task_desc = self._extract_task_description(filename)
            with h5py.File(path, "r") as f:
                demo_names = sorted(list(f["data"].keys()))[:demos_per_task]
                for name in demo_names:
                    traj_len = len(f["data"][name]["actions"])
                    self.trajectories.append((path, name, traj_len))

        self.sample_indices = []
        for path, demo_name, traj_len in self.trajectories:
            valid_len = traj_len - self.num_action_chunks + 1

            if valid_len > 0:
                for t in range(valid_len):
                    self.sample_indices.append((path, demo_name, t, task_desc))

        self.sample_indices = self.sample_indices[rank::world_size]

    def _extract_task_description(self, filename):
        name = filename.replace(".hdf5", "")
        parts = name.split("_")
        return " ".join(parts)
    
    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        path, demo_name, timestep, task_desc = self.sample_indices[idx]
        
        with h5py.File(path, "r") as f:
            demo = f["data"][demo_name]
            obs = np.array(demo["obs"]["agentview_rgb"][timestep])
            actions = np.array(demo["actions"][timestep:timestep + self.num_action_chunks])
        
        assert actions.shape[0] == self.num_action_chunks, (
            f"Expected {self.num_action_chunks} actions, got {actions.shape[0]}"
        )

        raw_obs_batch = {
            "task_descriptions": [task_desc],
            "images_and_states": {
                "full_image": [torch.from_numpy(obs).float()]
            }
        }
        
        processed_obs = prepare_observations(
            simulator_type=self.cfg.env.train.simulator_type,
            model_name=self.cfg.actor.model.model_name,
            raw_obs=raw_obs_batch,
            use_proprio=self.cfg.actor.model.use_proprio,
            max_length=self.cfg.runner.max_prompt_length,
            processor=self.input_processor,
            precision=self.precision,
        ) 
        processed_obs = {k: v.squeeze(0) for k, v in processed_obs.items()}
        action_chunks = torch.from_numpy(actions).float()
        return {**processed_obs, "action_tokens": action_chunks}


@hydra.main(
    version_base="1.1", config_path="../../examples/embodiment/config", config_name="libero_spatial_grpo_openvlaoft"
)
def main(cfg):
    from omegaconf import OmegaConf
    cfg = OmegaConf.create(cfg)

    EMBODIED_PATH = os.path.dirname(os.path.abspath(__file__))
    REPO_PATH = os.path.dirname(os.path.dirname(EMBODIED_PATH))
    LIBERO_REPO_PATH = os.path.join(REPO_PATH, "LIBERO")
    sft_dataset = LiberoSFTDataset(
        cfg=cfg,
        root_dir=LIBERO_REPO_PATH, 
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
    batch = next(iter(sft_iterator))

    # obs, action = batch["obs"], batch["action"]
    # print(f"obs shape: {obs.shape}, action shape: {action.shape}")

if __name__ == "__main__":
    import os
    EMBODIED_PATH = os.path.dirname(os.path.abspath(__file__))
    os.environ["EMBODIED_PATH"] = EMBODIED_PATH
    REPO_PATH = os.path.dirname(os.path.dirname(EMBODIED_PATH))
    os.environ["REPO_PATH"] = REPO_PATH
    main()