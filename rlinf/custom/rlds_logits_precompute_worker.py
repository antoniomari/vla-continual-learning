import gc
import os
from collections import defaultdict

import h5py
import numpy as np
import tensorflow as tf
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from rlinf.config import torch_dtype_from_precision
from rlinf.models import get_model, get_model_config_and_processor
from rlinf.models.embodiment.model_utils import (
    bc_custom_forward,
    prepare_observations,
)
from rlinf.scheduler import Cluster, Worker
from rlinf.utils.placement import HybridComponentPlacement


class RLDSLogitsPrecomputeWorker(Worker):
    """Worker to precompute logits from RLDS dataset and save to HDF5."""

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
        task_suite_name = cfg.env.train.task_suite_name
        self.root_dir = cfg.logits_precompute.get("root_dir")
        self.rlds_path = cfg.logits_precompute.get(
            "rlds_path",
            os.path.join(
                self.root_dir,
                "libero",
                "datasets",
                "modified_libero_rlds/libero_spatial_no_noops/1.0.0",
            ),
        )
        self.output_dir = cfg.logits_precompute.get("output_dir", None)
        if self.output_dir is None:
            self.output_dir = os.path.join(
                self.root_dir,
                "libero",
                "datasets_with_logits",
                f"{task_suite_name}_simplevla",
            )

        self.demos_per_task = cfg.logits_precompute.get("demos_per_task", None)

        # Get world size for sharding
        self._component_placement = HybridComponentPlacement(cfg, Cluster())
        self.world_size = self._component_placement.get_world_size("rollout")

        # Load and organize episodes by task
        self._load_and_organize_episodes()

        self._logger.info(
            f"Worker {self._rank}: Processing {len(self.assigned_tasks)} tasks"
        )

    def _load_and_organize_episodes(self):
        """Load RLDS dataset and organize episodes by task, then shard by task."""
        self._logger.info(
            f"Worker {self._rank}: Loading RLDS dataset from {self.rlds_path}"
        )

        # Load all TFRecord files
        file_pattern = f"{self.rlds_path}/*.tfrecord-*"
        files = tf.io.gfile.glob(file_pattern)

        if not files:
            raise ValueError(f"No TFRecord files found at {file_pattern}")

        raw_dataset = tf.data.TFRecordDataset(files)

        # Organize episodes by task
        task_episodes = defaultdict(list)

        for raw_record in raw_dataset:
            episode = self._parse_episode(raw_record)
            if episode is not None:
                task_name = episode["task_name"]
                task_episodes[task_name].append(episode)

        # Sort tasks for consistent ordering
        all_task_names = sorted(task_episodes.keys())

        # Limit demos per task if specified
        if self.demos_per_task:
            for task_name in all_task_names:
                task_episodes[task_name] = task_episodes[task_name][
                    : self.demos_per_task
                ]

        # Shard tasks across workers (round-robin by task, not by episode)
        self.assigned_tasks = {}
        for i, task_name in enumerate(all_task_names):
            if i % self.world_size == self._rank:
                self.assigned_tasks[task_name] = task_episodes[task_name]

        total_episodes = sum(len(eps) for eps in self.assigned_tasks.values())
        self._logger.info(
            f"Worker {self._rank}: Assigned {len(self.assigned_tasks)} tasks "
            f"with {total_episodes} total episodes"
        )

    def _parse_episode(self, raw_record):
        """Parse a single episode from TFRecord."""
        try:
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())

            features = example.features.feature

            # Extract task name
            language_bytes = features["steps/language_instruction"].bytes_list.value
            task_name = language_bytes[0].decode("utf-8")

            # Get number of steps
            num_steps = len(features["steps/is_first"].int64_list.value)

            # Extract actions
            actions_flat = list(features["steps/action"].float_list.value)
            actions = np.array(actions_flat, dtype=np.float32).reshape(num_steps, 7)

            # Extract images
            image_bytes = list(features["steps/observation/image"].bytes_list.value)

            # Optional wrist images
            wrist_image_bytes = None
            if "steps/observation/wrist_image" in features:
                wrist_image_bytes = list(
                    features["steps/observation/wrist_image"].bytes_list.value
                )

            return {
                "task_name": task_name,
                "num_steps": num_steps,
                "actions": actions,
                "image_bytes": image_bytes,
                "wrist_image_bytes": wrist_image_bytes,
            }
        except Exception as e:
            self._logger.error(f"Error parsing episode: {e}")
            return None

    def _decode_image(self, image_bytes):
        """Decode image from bytes."""
        image = tf.image.decode_image(image_bytes, channels=3)
        image = image.numpy().astype(np.uint8)
        return image

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

    def compute_logits(self, processed_obs):
        """Run forward pass and return logits."""
        with torch.no_grad():
            actions, raw_logits = bc_custom_forward(
                model=self.hf_model,
                input_ids=processed_obs["input_ids"],
                attention_mask=processed_obs["attention_mask"],
                pixel_values=processed_obs["pixel_values"],
                do_sample=False,
                return_logits=True,
                logits_type="raw",
            )

        valid_start = self.hf_model.vocab_size - self.hf_model.config.n_action_bins
        valid_end = self.hf_model.vocab_size
        processed_logits_tensor = raw_logits[..., valid_start:valid_end]

        return {
            "raw_action_logits": raw_logits.cpu().numpy(),
            "processed_action_logits": processed_logits_tensor.cpu().numpy(),
            "actions": actions,
        }

    def process_task(self, task_name: str, episodes: list):
        """Process all episodes for a single task and save to HDF5."""
        # Create output filename from task name
        safe_task_name = task_name.replace(" ", "_").replace("/", "_")
        output_path = os.path.join(self.output_dir, f"{safe_task_name}.hdf5")

        self._logger.info(
            f"Worker {self._rank}: Processing task '{task_name}' "
            f"with {len(episodes)} episodes"
        )

        with h5py.File(output_path, "w") as f_out:
            data_group = f_out.create_group("data")

            # Process each episode
            for demo_idx, episode in enumerate(
                tqdm(
                    episodes,
                    desc=f"Worker {self._rank}: {task_name}",
                    disable=(self._rank != 0),
                )
            ):
                demo_name = f"demo_{demo_idx}"
                demo_group = data_group.create_group(demo_name)

                # Save actions
                demo_group.create_dataset("actions", data=episode["actions"])

                # Save observations
                obs_group = demo_group.create_group("obs")

                # Decode and save images
                images = np.array(
                    [
                        self._decode_image(img_bytes)
                        for img_bytes in episode["image_bytes"]
                    ]
                )
                obs_group.create_dataset("agentview_rgb", data=images)

                # Save wrist images if available
                if episode["wrist_image_bytes"] is not None:
                    wrist_images = np.array(
                        [
                            self._decode_image(img_bytes)
                            for img_bytes in episode["wrist_image_bytes"]
                        ]
                    )
                    obs_group.create_dataset("wrist_image", data=wrist_images)

                # Compute logits for each timestep
                num_steps = episode["num_steps"]
                all_processed_action_logits = []
                all_predicted_actions = []

                for t in range(num_steps):
                    obs = images[t]

                    raw_obs_batch = {
                        "task_descriptions": [task_name],
                        "images_and_states": {
                            "full_image": [torch.from_numpy(obs).float()]
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

                    all_processed_action_logits.append(
                        logits_dict["processed_action_logits"]
                    )
                    all_predicted_actions.append(logits_dict["actions"])

                # Save logits
                demo_group.create_dataset(
                    "processed_action_logits",
                    data=np.concatenate(all_processed_action_logits, axis=0),
                )
                demo_group.create_dataset(
                    "predicted_actions",
                    data=np.concatenate(all_predicted_actions, axis=0),
                )

        self._logger.info(f"Worker {self._rank}: Saved to {output_path}")

    def process_all_tasks(self):
        """Process all tasks assigned to this worker."""
        for task_name, episodes in self.assigned_tasks.items():
            self.process_task(task_name, episodes)

            # Clean up memory
            gc.collect()
            torch.cuda.empty_cache()

        self._logger.info(f"Worker {self._rank}: Completed processing all tasks")
