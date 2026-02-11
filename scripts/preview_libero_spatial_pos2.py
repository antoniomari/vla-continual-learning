import os

import numpy as np
from PIL import Image

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv


# Per-task robot base position overrides used for the `libero_spatial_pos2` suite.
# Keep this in sync with examples/embodiment/config/env/eval/libero_spatial_pos2.yaml.
# Each entry is [x, y, z] offset from the default robot position.
# Default position for table arena is approximately (-0.66, 0, 0) for mounted panda.
# Offsets are in meters (reduced range for subtle variations).
ROBOT_BASE_POS_OVERRIDE_PER_TASK = [
    # task 0: Default position (no offset)
    [0.0, 0.0, 0.0],
    # task 1: Move backward slightly (negative x) - reduced movement
    [-0.02, 0.0, 0.0],
    # task 2: Move backward slightly (negative x)
    [-0.03, 0.0, 0.0],
    # task 3: Move right slightly (positive y)
    [0.0, 0.03, 0.0],
    # task 4: Move left slightly (negative y)
    [0.0, -0.03, 0.0],
    # task 5: Move forward and right slightly
    [0.02, 0.02, 0.0],
    # task 6: Move backward and left slightly
    [-0.02, -0.02, 0.0],
    # task 7: Move forward and left slightly
    [0.02, -0.02, 0.0],
    # task 8: Move backward and right slightly
    [-0.02, 0.02, 0.0],
    # task 9: Move backward slightly (negative x) - reduced movement
    [-0.015, 0.0, 0.0],
]


def get_robot_base_pos_override_for_task(task_idx: int):
    return ROBOT_BASE_POS_OVERRIDE_PER_TASK[task_idx]


def main(output_dir: str = "libero_spatial_pos2_previews", num_init_states_per_task: int = 1):
    os.makedirs(output_dir, exist_ok=True)

    benchmark = get_benchmark("libero_spatial_pos2")()
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
                "robot_base_pos_override": get_robot_base_pos_override_for_task(task_idx),
            }
            env = OffScreenRenderEnv(**env_args)

            # Use LIBERO's first N init states for this task
            init_states = benchmark.get_task_init_states(task_idx)
            state_idx_clamped = min(state_idx, len(init_states) - 1)
            mujoco_state = init_states[state_idx_clamped]
            obs = env.set_init_state(mujoco_state)

            # Match RLinf preprocessing: rotate 180 degrees
            img = obs["agentview_image"]
            img = img[::-1, ::-1].copy()

            img_pil = Image.fromarray(img.astype(np.uint8))
            fname = f"task{task_idx:02d}_state{state_idx_clamped:02d}.png"
            img_pil.save(os.path.join(output_dir, fname))

            env.close()


if __name__ == "__main__":
    main()
