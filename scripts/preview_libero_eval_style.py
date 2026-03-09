#!/usr/bin/env python3
"""
Generate LIBERO preview images using the exact same env initialization as eval.

This script mirrors the LiberoEnv reset flow:
  1. reset() - triggers robosuite's full reset including visualize(vis_settings) that hides the gripper line
  2. set_init_state() - restores the desired mujoco state
  3. N zero-action steps (eval uses 10; we use 100 for physics settling)
  4. Take observation

No custom gripper-hiding hacks: the gripper line is hidden because reset() runs first.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from PIL import Image

# Add paths and env for libero (must match eval_embodiment.sh)
repo_root = Path(__file__).resolve().parents[1]
libero_repo = repo_root / "LIBERO"
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(libero_repo))
os.environ.setdefault("LIBERO_CONFIG_PATH", str(libero_repo))

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv


def load_eval_config(config_path: str) -> dict:
    """Load eval env config (init_params, etc.) from YAML."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("init_params", {})



def main(
    task_suite: str = "libero_spatial_cam2",
    output_dir: Optional[str] = None,
    config_path: Optional[str] = None,
    num_init_states_per_task: int = 1,
    num_settle_steps: int = 100,
):
    """
    Generate preview images using eval-style initialization.

    Args:
        task_suite: LIBERO benchmark name (libero_spatial_cam2, libero_spatial_light2, libero_spatial_pos2)
        output_dir: Directory to save PNGs. If None, uses {task_suite}_previews_eval_style
        config_path: Path to eval config YAML. If None, uses default path for task_suite.
        num_init_states_per_task: Init states to capture per task
        num_settle_steps: Zero-action steps after set_init_state (eval uses 10; 100 ≈ 5s settling)
    """
    # Auto-configure output dir from task_suite if not specified
    if output_dir is None:
        output_dir = f"{task_suite}_previews_eval_style"
    os.makedirs(output_dir, exist_ok=True)

    # Resolve config path
    if config_path is None:
        config_path = (
            repo_root
            / "examples"
            / "embodiment"
            / "config"
            / "env"
            / "eval"
            / f"{task_suite}.yaml"
        )
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    init_params = load_eval_config(config_path)
    camera_overrides_per_task = init_params.pop("camera_overrides_per_task", None)
    light_overrides_per_task = init_params.pop("light_overrides_per_task", None)
    robot_base_pos_override_per_task = init_params.pop(
        "robot_base_pos_override_per_task", None
    )
    base_env_args = {
        k: v
        for k, v in init_params.items()
        if k
        not in (
            "camera_overrides_per_task",
            "light_overrides_per_task",
            "robot_base_pos_override_per_task",
        )
    }

    benchmark = get_benchmark(task_suite)()
    num_tasks = benchmark.get_num_tasks()

    for task_idx in range(num_tasks):
        task = benchmark.get_task(task_idx)
        task_bddl_file = os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        )

        env_args = {
            **base_env_args,
            "bddl_file_name": task_bddl_file,
        }
        if camera_overrides_per_task and 0 <= task_idx < len(camera_overrides_per_task):
            env_args["camera_overrides"] = camera_overrides_per_task[task_idx]
        if light_overrides_per_task and 0 <= task_idx < len(light_overrides_per_task):
            env_args["light_overrides"] = light_overrides_per_task[task_idx]
        if robot_base_pos_override_per_task and 0 <= task_idx < len(
            robot_base_pos_override_per_task
        ):
            env_args["robot_base_pos_override"] = robot_base_pos_override_per_task[
                task_idx
            ]

        env = OffScreenRenderEnv(**env_args)
        env.seed(0)

        for state_idx in range(num_init_states_per_task):
            init_states = benchmark.get_task_init_states(task_idx)
            state_idx_clamped = min(state_idx, len(init_states) - 1)
            mujoco_state = init_states[state_idx_clamped]

            # --- Exact eval sequence (LiberoEnv._reconfigure + reset) ---
            # 1. reset() first - runs robosuite reset which calls visualize(vis_settings) and hides gripper
            env.reset()

            # 2. set_init_state() - restores the specific mujoco state
            env.set_init_state(mujoco_state)

            # 3. Zero-action steps (eval uses 10; we use num_settle_steps for physics settling)
            zero_action = np.zeros(7, dtype=np.float64)
            for _ in range(num_settle_steps):
                env.step(zero_action)

            # 4. Get observation (same key as LiberoEnv uses)
            obs = env.env._get_observations()

            # Match RLinf preprocessing: rotate 180 degrees
            img = obs["agentview_image"]
            img = img[::-1, ::-1].copy()

            img_pil = Image.fromarray(img.astype(np.uint8))
            fname = f"task{task_idx:02d}_state{state_idx_clamped:02d}.png"
            img_pil.save(os.path.join(output_dir, fname))

        env.close()

    print(f"Saved previews to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate LIBERO preview images using eval-style env init"
    )
    parser.add_argument(
        "--task-suite",
        default="libero_spatial_cam2",
        choices=["libero_spatial_cam2", "libero_spatial_light2", "libero_spatial_pos2"],
        help="LIBERO task suite (cam2=camera, light2=lighting, pos2=robot position)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: {task_suite}_previews_eval_style)",
    )
    parser.add_argument(
        "--config-path",
        default=None,
        help="Path to eval config YAML (default: env/eval/<task_suite>.yaml)",
    )
    parser.add_argument(
        "--num-states",
        type=int,
        default=1,
        help="Init states per task to capture",
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=100,
        help="Zero-action steps for physics settling (eval uses 10)",
    )
    args = parser.parse_args()
    main(
        task_suite=args.task_suite,
        output_dir=args.output_dir,
        config_path=args.config_path,
        num_init_states_per_task=args.num_states,
        num_settle_steps=args.settle_steps,
    )
