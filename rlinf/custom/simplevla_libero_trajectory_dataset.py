import os
from collections import defaultdict

import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import Dataset

from rlinf.config import torch_dtype_from_precision
from rlinf.models import get_model_config_and_processor
from rlinf.models.embodiment.model_utils import prepare_observations


class SimpleVLALiberoSFTDataset(Dataset):
    def __init__(
        self,
        cfg,
        rlds_path,
        demos_per_task=1,
        rank=0,
        world_size=1,
    ):
        self.cfg = cfg
        self.rlds_path = rlds_path
        self.demos_per_task = demos_per_task
        self.rank = rank
        self.world_size = world_size
        self.num_action_chunks = cfg.actor.model.num_action_chunks

        _, self.input_processor = get_model_config_and_processor(cfg.actor)
        self.precision = torch_dtype_from_precision(cfg.actor.model.precision)

        # Load dataset
        self._load_dataset()

        # Shard across processes
        self.sample_indices = self.sample_indices[rank::world_size]

        print(f"[Rank {rank}] Dataset loaded: {len(self.sample_indices)} samples")

    def _load_dataset(self):
        """Load and parse RLDS dataset."""
        print(f"Loading RLDS dataset from {self.rlds_path}")

        # Load all TFRecord files
        file_pattern = f"{self.rlds_path}/*.tfrecord-*"
        files = tf.io.gfile.glob(file_pattern)

        if not files:
            raise ValueError(f"No TFRecord files found at {file_pattern}")

        raw_dataset = tf.data.TFRecordDataset(files)

        task_episodes = defaultdict(list)
        episode_count = 0

        for raw_record in raw_dataset:
            episode = self._parse_episode(raw_record)
            if episode is not None:
                task_name = episode["task_name"]
                task_episodes[task_name].append(episode)
                episode_count += 1

        self.task_names = sorted(task_episodes.keys())

        self.episodes = []
        for task_name in self.task_names:
            demos = task_episodes[task_name][: self.demos_per_task]
            self.episodes.extend(demos)

        print(f"Using {len(self.episodes)} episodes ({self.demos_per_task} per task)")

        self.sample_indices = []
        for episode_idx, episode in enumerate(self.episodes):
            num_steps = episode["num_steps"]
            valid_len = num_steps - self.num_action_chunks + 1

            if valid_len > 0:
                for timestep in range(valid_len):
                    self.sample_indices.append((episode_idx, timestep))

        print(f"Created {len(self.sample_indices)} training samples")

    def _parse_episode(self, raw_record):
        try:
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())

            features = example.features.feature

            language_bytes = features["steps/language_instruction"].bytes_list.value
            task_name = language_bytes[0].decode("utf-8")

            num_steps = len(features["steps/is_first"].int64_list.value)

            actions_flat = list(features["steps/action"].float_list.value)
            actions = np.array(actions_flat, dtype=np.float32).reshape(num_steps, 7)

            image_bytes = list(features["steps/observation/image"].bytes_list.value)

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
            print(f"Error parsing episode: {e}")
            return None

    def _decode_image(self, image_bytes):
        """Decode image from bytes."""
        image = tf.image.decode_image(image_bytes, channels=3)

        # Convert to numpy [H, W, C]
        image = image.numpy().astype(np.uint8)

        return image

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        episode_idx, timestep = self.sample_indices[idx]
        episode = self.episodes[episode_idx]

        image_bytes = episode["image_bytes"][timestep]
        image = self._decode_image(image_bytes)  # [H, W, 3] uint8

        actions = episode["actions"][timestep : timestep + self.num_action_chunks]
        assert actions.shape[0] == self.num_action_chunks, (
            f"Expected {self.num_action_chunks} actions, got {actions.shape[0]}"
        )

        task_desc = episode["task_name"]
        raw_obs_batch = {
            "task_descriptions": [task_desc],
            "images_and_states": {"full_image": [torch.from_numpy(image).float()]},
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
        output = {
            **processed_obs,
            "actions": action_chunks,
        }

        return output


if __name__ == "__main__":
    import os
    import time
    from itertools import cycle

    import hydra
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    EMBODIED_PATH = os.path.dirname(os.path.abspath(__file__))
    os.environ["EMBODIED_PATH"] = EMBODIED_PATH
    REPO_PATH = os.path.dirname(os.path.dirname(EMBODIED_PATH))
    os.environ["REPO_PATH"] = REPO_PATH

    @hydra.main(
        version_base="1.1",
        config_path="../../examples/embodiment/config",
        config_name="libero_spatial_grpo_openvlaoft_bcrl_logit",
    )
    def main(cfg):
        cfg = OmegaConf.create(cfg)
        rlds_path = "./LIBERO/libero/datasets/modified_libero_rlds/libero_spatial_no_noops/1.0.0"
        sft_dataset = SimpleVLALiberoSFTDataset(
            cfg=cfg,
            rlds_path=rlds_path,
            demos_per_task=1,
            rank=0,
            world_size=1,
        )

        print(f"\nDataset size: {len(sft_dataset)}")
        print("\nTesting single sample...")
        sample = sft_dataset[0]
        print("Sample keys:", sample.keys())
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")

        print("\nTesting DataLoader...")
        sft_dataloader = cycle(
            DataLoader(
                sft_dataset,
                batch_size=16,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
            )
        )
        sft_iterator = iter(sft_dataloader)

        print("Warming up...")
        for _ in range(5):
            batch = next(sft_iterator)

        print("Timing batches...")
        times = []
        for _ in range(30):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            batch = next(sft_iterator)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)

        print(f"\nBatch timing (30 batches):")
        print(f"  Times: {[f'{t:.4f}' for t in times[:10]]}...")
        print(f"  Average: {sum(times) / len(times):.4f}s")
        print(f"  Min: {min(times):.4f}s")
        print(f"  Max: {max(times):.4f}s")

        print(f"\nFinal batch shapes:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")

    main()
