import gc
import os

import h5py
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from rlinf.config import torch_dtype_from_precision
from rlinf.models import get_model, get_model_config_and_processor
from rlinf.models.embodiment.model_utils import (
    default_logits_processor,
    prepare_observations,
)
from rlinf.scheduler import Cluster, Worker
from rlinf.utils.placement import HybridComponentPlacement


class LogitsPrecomputeWorker(Worker):
    """Worker to precompute and save logits to HDF5 files using Ray."""

    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        self.device = torch.cuda.current_device()

        # Setup model config and processor
        self.model_config, self.input_processor = get_model_config_and_processor(
            cfg.actor
        )
        self.precision = torch_dtype_from_precision(cfg.actor.model.precision)
        self.use_proprio = self.cfg.actor.model.get("use_proprio", False)

        # Setup paths
        self.root_dir = cfg.logits_precompute.get("root_dir")
        self.output_dir = cfg.logits_precompute.get("output_dir", None)

        task_suite_name = cfg.env.train.task_suite_name
        self.dataset_path = os.path.join(
            self.root_dir, "libero", "datasets", task_suite_name
        )

        if self.output_dir is None:
            self.output_dir = os.path.join(
                self.root_dir, "libero", "datasets_with_logits", task_suite_name
            )

        # Get task files and shard them across workers
        self._component_placement = HybridComponentPlacement(cfg, Cluster())
        world_size = self._component_placement.get_world_size("rollout")

        all_task_files = sorted(
            [
                os.path.join(self.dataset_path, f)
                for f in os.listdir(self.dataset_path)
                if f.endswith(".hdf5")
            ]
        )

        # Shard files across workers
        self.task_files = [
            f for i, f in enumerate(all_task_files) if i % world_size == self._rank
        ]

        self._logger.info(
            f"Worker {self._rank}: Processing {len(self.task_files)} files"
        )

    def init_worker(self):
        """Initialize the model on this worker."""
        self._logger.info(f"Worker {self._rank}: Loading model...")
        self.hf_model = get_model(self.cfg.rollout.model_dir, self.cfg.actor.model)
        self.hf_model.setup_params(self.model_config, self.cfg)
        self.hf_model.to(self.precision)
        self.hf_model.to(self.device)
        self.hf_model.eval()
        self._logger.info(f"Worker {self._rank}: Model loaded on {self.device}")

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

    def _extract_task_description(self, filename):
        """Extract task description from filename."""
        name = os.path.basename(filename).replace(".hdf5", "")
        parts = name.split("_")
        return " ".join(parts)

    def compute_logits(self, processed_obs):
        """Run forward pass and return logits."""
        action_token_len = self.hf_model.action_dim * self.hf_model.num_action_chunks

        with torch.no_grad():
            actions, action_tokens, action_logits, last_hidden_state = (
                self.hf_model.predict_action_batch(
                    input_ids=processed_obs["input_ids"],
                    attention_mask=processed_obs["attention_mask"],
                    pixel_values=processed_obs["pixel_values"],
                    do_sample=False,
                    max_new_tokens=action_token_len,
                    use_cache=True,
                )
            )

            # Process logits
            chunk_logprobs = default_logits_processor(
                action_logits,
                action_tokens,
                self.hf_model.vocab_size,
                self.hf_model.config.n_action_bins,
            )["logprobs"]

        return {
            "action_logits": action_logits.cpu().numpy(),
            "action_tokens": action_tokens.cpu().numpy(),
            "logprobs": chunk_logprobs.cpu().numpy(),
            "actions": actions,
        }

    def process_file(self, input_path: str, demos_per_task: int = None):
        """Process a single HDF5 file and save logits."""
        filename = os.path.basename(input_path)
        output_path = os.path.join(self.output_dir, filename)
        task_desc = self._extract_task_description(filename)

        self._logger.info(f"Worker {self._rank}: Processing {filename}")
        self._logger.info(f"Worker {self._rank}: Task: {task_desc}")

        # Open input file
        with h5py.File(input_path, "r") as f_in:
            demo_names = sorted(list(f_in["data"].keys()))
            if demos_per_task:
                demo_names = demo_names[:demos_per_task]

            with h5py.File(output_path, "w") as f_out:
                if "data" in f_in:
                    f_out.create_group("data")

                # Process each demonstration
                for demo_name in tqdm(
                    demo_names,
                    desc=f"Worker {self._rank}: Demos in {filename}",
                    disable=(self._rank != 0),  # Only show progress on rank 0
                ):
                    demo_in = f_in["data"][demo_name]
                    demo_out = f_out["data"].create_group(demo_name)

                    for key in demo_in.keys():
                        if key == "obs":
                            obs_out = demo_out.create_group("obs")
                            for obs_key in demo_in["obs"].keys():
                                obs_out.create_dataset(
                                    obs_key, data=demo_in["obs"][obs_key][:]
                                )
                        else:
                            demo_out.create_dataset(key, data=demo_in[key][:])

                    # Compute and save logits for each timestep
                    traj_len = len(demo_in["actions"])

                    all_action_logits = []
                    all_action_tokens = []
                    all_logprobs = []
                    all_predicted_actions = []

                    for t in range(traj_len):
                        obs = demo_in["obs"]["agentview_rgb"][t]
                        raw_obs_batch = {
                            "task_descriptions": [task_desc],
                            "images_and_states": {
                                "full_image": [torch.from_numpy(obs[:]).float()]
                            },
                        }

                        processed_obs = prepare_observations(
                            simulator_type=self.cfg.env.train.simulator_type,
                            model_name=self.cfg.actor.model.model_name,
                            raw_obs=raw_obs_batch,
                            use_proprio=self.use_proprio,
                            max_length=self.hf_model.max_prompt_length,
                            processor=self.input_processor,
                            precision=self.precision,
                        )

                        processed_obs = {
                            k: v.to(self.device) for k, v in processed_obs.items()
                        }

                        logits_dict = self.compute_logits(processed_obs)

                        all_action_logits.append(logits_dict["action_logits"])
                        all_action_tokens.append(logits_dict["action_tokens"])
                        all_logprobs.append(logits_dict["logprobs"])
                        all_predicted_actions.append(logits_dict["actions"])

                    demo_out.create_dataset(
                        "action_logits", data=np.concatenate(all_action_logits, axis=0)
                    )
                    demo_out.create_dataset(
                        "action_tokens", data=np.concatenate(all_action_tokens, axis=0)
                    )
                    demo_out.create_dataset(
                        "logprobs", data=np.concatenate(all_logprobs, axis=0)
                    )
                    demo_out.create_dataset(
                        "predicted_actions",
                        data=np.concatenate(all_predicted_actions, axis=0),
                    )

        self._logger.info(f"Worker {self._rank}: Saved to {output_path}")

    def process_all_files(self):
        """Process all HDF5 files assigned to this worker."""
        demos_per_task = self.cfg.logits_precompute.get("demos_per_task", None)

        for task_file in self.task_files:
            self.process_file(task_file, demos_per_task)

            gc.collect()
            torch.cuda.empty_cache()

        self._logger.info(f"Worker {self._rank}: Completed processing all files")
