import os

import numpy as np
from PIL import Image

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv


# Per-task light overrides used for the `libero_spatial_light2` suite.
# Keep this in sync with examples/embodiment/config/env/eval/libero_spatial_light2.yaml.
# Default values (from libero_tabletop_base_style.xml):
#   light1: pos=[1, 1, 4.0], diffuse=[0.8, 0.8, 0.8], specular=[0.3, 0.3, 0.3], dir=[0, -0.15, -1]
#   light2: pos=[-3, -3, 4.0], diffuse=[0.8, 0.8, 0.8], specular=[0.3, 0.3, 0.3], dir=[0, -0.15, -1]
LIGHT_OVERRIDES_PER_TASK = [
    # task 0 (use default)
    {
        "light1": {
            "pos": [1, 1, 4.0],
            "dir": [0, -0.15, -1],
            "diffuse": [0.8, 0.8, 0.8],
            "specular": [0.3, 0.3, 0.3],
        },
        "light2": {
            "pos": [-3, -3, 4.0],
            "dir": [0, -0.15, -1],
            "diffuse": [0.8, 0.8, 0.8],
            "specular": [0.3, 0.3, 0.3],
        },
    },
    # task 1: Brighter lighting
    {
        "light1": {
            "pos": [1, 1, 4.0],
            "dir": [0, -0.15, -1],
            "diffuse": [1.0, 1.0, 1.0],
            "specular": [0.5, 0.5, 0.5],
        },
        "light2": {
            "pos": [-3, -3, 4.0],
            "dir": [0, -0.15, -1],
            "diffuse": [1.0, 1.0, 1.0],
            "specular": [0.5, 0.5, 0.5],
        },
    },
    # task 2: Dimmer lighting
    {
        "light1": {
            "pos": [1, 1, 4.0],
            "dir": [0, -0.15, -1],
            "diffuse": [0.5, 0.5, 0.5],
            "specular": [0.1, 0.1, 0.1],
        },
        "light2": {
            "pos": [-3, -3, 4.0],
            "dir": [0, -0.15, -1],
            "diffuse": [0.5, 0.5, 0.5],
            "specular": [0.1, 0.1, 0.1],
        },
    },
    # task 3: Warmer lighting (slightly yellow tint)
    {
        "light1": {
            "pos": [1, 1, 4.0],
            "dir": [0, -0.15, -1],
            "diffuse": [0.9, 0.85, 0.75],
            "specular": [0.4, 0.35, 0.25],
        },
        "light2": {
            "pos": [-3, -3, 4.0],
            "dir": [0, -0.15, -1],
            "diffuse": [0.9, 0.85, 0.75],
            "specular": [0.4, 0.35, 0.25],
        },
    },
    # task 4: Cooler lighting (slightly blue tint)
    {
        "light1": {
            "pos": [1, 1, 4.0],
            "dir": [0, -0.15, -1],
            "diffuse": [0.75, 0.8, 0.9],
            "specular": [0.25, 0.3, 0.4],
        },
        "light2": {
            "pos": [-3, -3, 4.0],
            "dir": [0, -0.15, -1],
            "diffuse": [0.75, 0.8, 0.9],
            "specular": [0.25, 0.3, 0.4],
        },
    },
    # task 5: Different light positions (closer)
    {
        "light1": {
            "pos": [0.5, 0.5, 3.5],
            "dir": [0, -0.15, -1],
            "diffuse": [0.8, 0.8, 0.8],
            "specular": [0.3, 0.3, 0.3],
        },
        "light2": {
            "pos": [-2.5, -2.5, 3.5],
            "dir": [0, -0.15, -1],
            "diffuse": [0.8, 0.8, 0.8],
            "specular": [0.3, 0.3, 0.3],
        },
    },
    # task 6: Different light positions (farther)
    {
        "light1": {
            "pos": [1.5, 1.5, 4.5],
            "dir": [0, -0.15, -1],
            "diffuse": [0.8, 0.8, 0.8],
            "specular": [0.3, 0.3, 0.3],
        },
        "light2": {
            "pos": [-3.5, -3.5, 4.5],
            "dir": [0, -0.15, -1],
            "diffuse": [0.8, 0.8, 0.8],
            "specular": [0.3, 0.3, 0.3],
        },
    },
    # task 7: Different light directions (more overhead)
    {
        "light1": {
            "pos": [1, 1, 4.0],
            "dir": [0, 0, -1],
            "diffuse": [0.8, 0.8, 0.8],
            "specular": [0.3, 0.3, 0.3],
        },
        "light2": {
            "pos": [-3, -3, 4.0],
            "dir": [0, 0, -1],
            "diffuse": [0.8, 0.8, 0.8],
            "specular": [0.3, 0.3, 0.3],
        },
    },
    # task 8: Different light directions (more angled)
    {
        "light1": {
            "pos": [1, 1, 4.0],
            "dir": [0.2, -0.2, -1],
            "diffuse": [0.8, 0.8, 0.8],
            "specular": [0.3, 0.3, 0.3],
        },
        "light2": {
            "pos": [-3, -3, 4.0],
            "dir": [-0.2, 0.2, -1],
            "diffuse": [0.8, 0.8, 0.8],
            "specular": [0.3, 0.3, 0.3],
        },
    },
    # task 9: Asymmetric lighting (one bright, one dim)
    {
        "light1": {
            "pos": [1, 1, 4.0],
            "dir": [0, -0.15, -1],
            "diffuse": [1.2, 1.2, 1.2],
            "specular": [0.5, 0.5, 0.5],
        },
        "light2": {
            "pos": [-3, -3, 4.0],
            "dir": [0, -0.15, -1],
            "diffuse": [0.4, 0.4, 0.4],
            "specular": [0.1, 0.1, 0.1],
        },
    },
]


def get_light_overrides_for_task(task_idx: int):
    return LIGHT_OVERRIDES_PER_TASK[task_idx]


def main(output_dir: str = "libero_spatial_light2_previews", num_init_states_per_task: int = 1):
    os.makedirs(output_dir, exist_ok=True)

    benchmark = get_benchmark("libero_spatial_light2")()
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
                "light_overrides": get_light_overrides_for_task(task_idx),
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
