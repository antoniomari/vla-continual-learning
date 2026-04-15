import os

import h5py
import hydra
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from rlinf.config import torch_dtype_from_precision
from rlinf.models import get_model_config_and_processor
from rlinf.models.embodiment.model_utils import prepare_observations


def worker_init_fn(worker_id):
    """Reset HDF5 file handles in forked worker processes."""
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        dataset.file_handles = {}


def make_collate_fn(cfg, input_processor, precision):
    """Factory that returns a collate_fn which batches raw data and runs
    the HuggingFace processor ONCE per micro-batch instead of per sample."""

    simulator_type = cfg.env.train.simulator_type
    model_name = cfg.actor.model.model_name
    use_proprio = cfg.actor.model.use_proprio
    max_length = cfg.runner.max_prompt_length

    def collate_fn(samples):
        # samples: list of dicts with 'obs_rgb', 'task_desc', 'actions', optionally logits

        # List of [H,W,C] uint8 numpy -> list of float tensors
        images = [torch.from_numpy(s["obs_rgb"]).float() for s in samples]
        task_descs = [s["task_desc"] for s in samples]

        # Build the raw_obs batch the same way the env would
        raw_obs_batch = {
            "task_descriptions": task_descs,
            "images_and_states": {"full_image": images},
        }

        # Run processor ONCE for the entire batch
        processed_obs = prepare_observations(
            simulator_type=simulator_type,
            model_name=model_name,
            raw_obs=raw_obs_batch,
            use_proprio=use_proprio,
            max_length=max_length,
            processor=input_processor,
            precision=precision,
            device=torch.device("cpu"),
        )

        # Stack actions
        actions = torch.stack([torch.from_numpy(s["actions"]).float() for s in samples])

        output = {
            **processed_obs,
            "actions": actions,
        }

        # Handle optional logits
        if "raw_action_logits" in samples[0] and samples[0]["raw_action_logits"] is not None:
            output["raw_action_logits"] = torch.stack(
                [torch.from_numpy(s["raw_action_logits"]).float() for s in samples]
            )
        elif "processed_action_logits" in samples[0] and samples[0]["processed_action_logits"] is not None:
            output["processed_action_logits"] = torch.stack(
                [torch.from_numpy(s["processed_action_logits"]).float() for s in samples]
            )

        return output

    return collate_fn


class LiberoSFTDataset(Dataset):
    def __init__(
        self,
        cfg,
        root_dir,
        demos_per_task=1,
        rank=0,
        world_size=1,
        use_preprocessed=True,
        use_cached_logits=False,
        logits_type="",
    ):
        self.cfg = cfg
        self.logits_type = logits_type
        # Path layout:
        # - BC / OPD without cached logits: libero/datasets/<suite> (standard LIBERO demo download).
        # - DER / reference logits: libero/datasets_with_logits/<suite>_simplevla (precomputed logits).
        # use_preprocessed is kept for API compatibility; path choice follows use_cached_logits.
        suite = cfg.env.train.task_suite_name
        if use_cached_logits:
            task_suite_name = f"{suite}_simplevla"
            dataset_dir = "datasets_with_logits"
        else:
            task_suite_name = suite
            dataset_dir = "datasets"
        self.root_dir = os.path.join(root_dir, "libero", dataset_dir, task_suite_name)
        self.file_handles = {}
        print(f"root dir: {self.root_dir}")
        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(
                f"LiberoSFTDataset: directory does not exist: {self.root_dir}\n"
                f"  Install demos under LIBERO (see LIBERO/benchmark_scripts/download_libero_datasets.py), "
                f"e.g. libero/datasets/{suite}/\n"
                f"  For logits-augmented HDF5, precompute/cache and use use_cached_logits=True "
                f"(libero/datasets_with_logits/{suite}_simplevla/)."
            )
        self.task_files = [
            os.path.join(self.root_dir, f)
            for f in os.listdir(self.root_dir)
            if f.endswith(".hdf5")
        ]

        # Add task files sorted by task id
        self.task_files = sorted(self.task_files)

        # Optional: restrict BC/SFT data to selected train task ids (sequential setting).
        # Apply only for OPD runs so other training scripts remain unchanged.
        # This uses LIBERO benchmark ordering (same source used by LiberoEnv).
        is_opd = cfg.algorithm.get("adv_type", None) == "embodied_opd"
        fixed_task_ids = cfg.env.train.get("fixed_task_ids", None)
        if is_opd and fixed_task_ids is not None:
            fixed_task_ids = [int(x) for x in fixed_task_ids]
            if len(fixed_task_ids) > 0:
                from libero.libero.benchmark import get_benchmark

                benchmark = get_benchmark(suite)()
                num_tasks = benchmark.get_num_tasks()
                invalid_ids = [tid for tid in fixed_task_ids if tid < 0 or tid >= num_tasks]
                if invalid_ids:
                    raise ValueError(
                        f"LiberoSFTDataset: invalid fixed_task_ids={invalid_ids} for suite "
                        f"{suite} with num_tasks={num_tasks}"
                    )

                allowed_files = {
                    f"{benchmark.get_task(tid).name}_demo.hdf5" for tid in fixed_task_ids
                }
                self.task_files = [
                    p for p in self.task_files if os.path.basename(p) in allowed_files
                ]
                if len(self.task_files) == 0:
                    raise ValueError(
                        "LiberoSFTDataset: task filter removed all files. "
                        f"suite={suite}, fixed_task_ids={fixed_task_ids}, "
                        f"expected files={sorted(allowed_files)} in {self.root_dir}"
                    )
                selected_files = [os.path.basename(p) for p in self.task_files]
                print(
                    f"LiberoSFTDataset: filtered to {len(self.task_files)} task file(s) "
                    f"for fixed_task_ids={fixed_task_ids}; files={selected_files}"
                )
        self.demos_per_task = demos_per_task
        self.num_action_chunks = cfg.actor.model.num_action_chunks

        self.precision = torch_dtype_from_precision(cfg.actor.model.precision)

        self.trajectories = []
        for path in self.task_files:
            filename = os.path.basename(path)
            task_desc = self._extract_task_description(filename)
            with h5py.File(path, "r") as f:
                all_demo_names = sorted(list(f["data"].keys()))
                # demos_per_task <= 0 or None: use all demonstrations for the task (OPD / full-data BC).
                if demos_per_task is None or demos_per_task <= 0:
                    demo_names = all_demo_names
                else:
                    demo_names = all_demo_names[:demos_per_task]
                for name in demo_names:
                    traj_len = len(f["data"][name]["actions"])
                    self.trajectories.append((path, name, traj_len, task_desc))

        self.sample_indices = []
        for path, demo_name, traj_len, task_desc in self.trajectories:
            valid_len = traj_len - self.num_action_chunks + 1
            if valid_len > 0:
                for t in range(valid_len):
                    self.sample_indices.append((path, demo_name, t, task_desc))

        self.sample_indices = self.sample_indices[rank::world_size]

        print(f"LiberoSFTDataset: {len(self.sample_indices)} samples (rank {rank}/{world_size})")

    def _extract_task_description(self, filename):
        name = filename.replace(".hdf5", "")
        parts = name.split("_")
        if parts[-1] == "demo":
            parts = parts[:-1]
        return " ".join(parts)

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        path, demo_name, timestep, task_desc = self.sample_indices[idx]

        if path not in self.file_handles:
            self.file_handles[path] = h5py.File(path, "r")
        f = self.file_handles[path]

        demo = f["data"][demo_name]

        # Return RAW numpy — no processor call here
        obs = np.array(demo["obs"]["agentview_rgb"][timestep])
        actions = np.array(
            demo["actions"][timestep : timestep + self.num_action_chunks]
        )

        assert actions.shape[0] == self.num_action_chunks, (
            f"Expected {self.num_action_chunks} actions, got {actions.shape[0]}"
        )

        output = {
            "obs_rgb": obs,         # [H, W, C] uint8 numpy
            "task_desc": task_desc,  # str
            "actions": actions,      # [C, D] float numpy
        }

        # Optional logits (still raw numpy)
        if self.logits_type == "raw" and "raw_action_logits" in demo.keys():
            output["raw_action_logits"] = np.array(demo["raw_action_logits"][timestep])
        else:
            output["raw_action_logits"] = None

        if self.logits_type == "processed" and "processed_action_logits" in demo.keys():
            output["processed_action_logits"] = np.array(
                demo["processed_action_logits"][timestep]
            )
        else:
            output["processed_action_logits"] = None

        return output

    def __del__(self):
        handles = getattr(self, "file_handles", None)
        if not handles:
            return
        for fh in list(handles.values()):
            try:
                fh.close()
            except Exception:
                pass


@hydra.main(
    version_base="1.1",
    config_path="../../examples/embodiment/config",
    config_name="libero_spatial_grpo_openvlaoft_bcrl_logit",
)
def main(cfg):
    from itertools import cycle
    import time
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
        world_size=4,
        use_preprocessed=True,
        use_cached_logits=True,
        logits_type="processed",
    )

    # Create processor once, pass to collate_fn
    _, input_processor = get_model_config_and_processor(cfg.actor)
    precision = torch_dtype_from_precision(cfg.actor.model.precision)
    collate_fn = make_collate_fn(cfg, input_processor, precision)

    sft_dataloader = cycle(
        DataLoader(
            sft_dataset,
            batch_size=16,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=4,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn,
            persistent_workers=True,
        )
    )
    sft_iterator = iter(sft_dataloader)

    times = []
    for i in range(30):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        batch = next(sft_iterator)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        if i > 0:
            print(f"Step {i}: {t1-t0:.4f}s")

    print(f"\nAvg (excluding warmup): {np.mean(times[1:]):.4f}s")
    print(f"input_ids shape: {batch['input_ids'].shape}")
    print(f"attention_mask shape: {batch['attention_mask'].shape}")
    print(f"pixel_values shape: {batch['pixel_values'].shape}")
    print(f"actions shape: {batch['actions'].shape}")


if __name__ == "__main__":
    EMBODIED_PATH = os.path.dirname(os.path.abspath(__file__))
    os.environ["EMBODIED_PATH"] = EMBODIED_PATH
    REPO_PATH = os.path.dirname(os.path.dirname(EMBODIED_PATH))
    os.environ["REPO_PATH"] = REPO_PATH
    main()
