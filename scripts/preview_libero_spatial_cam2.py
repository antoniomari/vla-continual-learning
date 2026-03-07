import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Allow importing preview_libero_utils when run as script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv

from preview_libero_utils import settle_physics, hide_gripper_orientation_site


# Per-task camera overrides used for the `libero_spatial_cam2` suite.
# Keep this in sync with examples/embodiment/config/env/eval/libero_spatial_cam2.yaml.
# Actual default (from libero_tabletop_manipulation.py): 
#   pos=[0.65861317, 0.0, 1.61035002], quat=[0.63801772, 0.30484972, 0.30484984, 0.63801772]
CAMERA_OVERRIDES_PER_TASK = [
    {"agentview": {"pos": [0.65861317, 0.00, 1.61035002], "quat": [0.63801772, 0.30484972, 0.30484984, 0.63801772]}},  # task 0: actual default
    {"agentview": {"pos": [0.6586, 0.05, 1.6104], "quat": [0.638, 0.305, 0.304, 0.638]}},
    {"agentview": {"pos": [0.6586, -0.05, 1.6104], "quat": [0.638, 0.304, 0.305, 0.638]}},
    {"agentview": {"pos": [0.6786, 0.00, 1.6304], "quat": [0.637, 0.305, 0.305, 0.637]}},
    {"agentview": {"pos": [0.659, 0.01, 1.611], "quat": [0.638, 0.305, 0.304, 0.638]}},  # task 4: minimal deviation
    {"agentview": {"pos": [0.6786, -0.06, 1.6304], "quat": [0.637, 0.304, 0.306, 0.637]}},
    {"agentview": {"pos": [0.6986, 0.00, 1.5904], "quat": [0.639, 0.304, 0.304, 0.639]}},
    {"agentview": {"pos": [0.659, 0.00, 1.611], "quat": [0.638, 0.304, 0.305, 0.638]}},  # task 7: minimal deviation
    {"agentview": {"pos": [0.658, 0.00, 1.610], "quat": [0.638, 0.305, 0.304, 0.638]}},  # task 8: minimal deviation
    {"agentview": {"pos": [0.6786, 0.00, 1.6104], "quat": [0.639, 0.305, 0.305, 0.639]}},
]


def get_camera_overrides_for_task(task_idx: int):
    return CAMERA_OVERRIDES_PER_TASK[task_idx]


def main(output_dir: str = "libero_spatial_cam2_previews", num_init_states_per_task: int = 1):
    os.makedirs(output_dir, exist_ok=True)

    benchmark = get_benchmark("libero_spatial_cam2")()
    num_tasks = benchmark.get_num_tasks()

    for task_idx in range(num_tasks):
        task = benchmark.get_task(task_idx)
        task_bddl_file = os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        )

        for state_idx in range(num_init_states_per_task):
            env_args = {
                "bddl_file_name": task_bddl_file,
                "camera_heights": 256,
                "camera_widths": 256,
                "camera_overrides": get_camera_overrides_for_task(task_idx),
            }
            env = OffScreenRenderEnv(**env_args)

            # Use LIBERO's first N init states for this task
            init_states = benchmark.get_task_init_states(task_idx)
            state_idx_clamped = min(state_idx, len(init_states) - 1)
            mujoco_state = init_states[state_idx_clamped]
            obs = env.set_init_state(mujoco_state)

            # Let objects settle (physics run for ~5 seconds)
            settle_physics(env)
            hide_gripper_orientation_site(env)
            obs = env.env._get_observations()

            # Match RLinf preprocessing: rotate 180 degrees
            img = obs["agentview_image"]
            img = img[::-1, ::-1].copy()

            img_pil = Image.fromarray(img.astype(np.uint8))
            fname = f"task{task_idx:02d}_state{state_idx_clamped:02d}.png"
            img_pil.save(os.path.join(output_dir, fname))

            env.close()


if __name__ == "__main__":
    main()

