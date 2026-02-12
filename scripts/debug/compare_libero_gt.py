import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

EMBODIED_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_PATH = os.path.dirname(EMBODIED_PATH)
LIBERO_REPO_PATH = os.path.join(REPO_PATH, "LIBERO")

# dataset_a = "libero_spatial_simplevla_task0_finetune"
dataset_a = "libero_spatial_simplevla"
path_a = f"{LIBERO_REPO_PATH}/libero/datasets_with_logits/{dataset_a}"
task_files_a = sorted([f for f in os.listdir(path_a) if f.endswith(".hdf5")])

NUM_ACTION_CHUNKS = 8
ACTION_DIM = 7
ACTION_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]

for task_file in task_files_a:
    print(f"\n{'=' * 70}")
    print(f"Checking {task_file}")
    print("=" * 70)
    file_a = os.path.join(path_a, task_file)

    with h5py.File(file_a, "r") as fa:
        demo_a = fa["data"]["demo_0"]

        # ---- Load actions ----
        ground_truth_actions = demo_a["actions"][:]

        print(f"Ground truth actions shape: {ground_truth_actions.shape}")

        # ---- Visualization ----
        fig, axes = plt.subplots(4, 2, figsize=(16, 12))
        fig.suptitle(
            f"{task_file} - Ground Truth Actions\n"
            f"(Total timesteps: {ground_truth_actions.shape[0]})",
            fontsize=14,
        )

        timesteps = np.arange(ground_truth_actions.shape[0])

        for dim in range(ACTION_DIM):
            ax = axes[dim // 2, dim % 2]

            # Plot ground truth
            ax.plot(
                timesteps,
                ground_truth_actions[:, dim],
                "--",
                label="Ground Truth",
                linewidth=2,
                alpha=0.8,
                color="C1",
            )

            # Difference stats (just for display in title)
            diff_dim = np.zeros_like(ground_truth_actions[:, dim])

            ax.set_xlabel("Timestep")
            ax.set_ylabel("Value")
            ax.set_title(f"{ACTION_NAMES[dim]}")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Hide the last subplot if ACTION_DIM is odd
        if ACTION_DIM % 2 == 1:
            axes[-1, -1].axis("off")

        plt.tight_layout()

        # Save figure
        output_dir = os.path.join(path_a, "ground_truth_plots")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir, f"{task_file.replace('.hdf5', '')}_ground_truth.png"
        )
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\n💾 Saved ground truth action plot to: {output_path}")
        plt.close()

print(f"\n{'=' * 70}")
print("✨ Analysis complete!")
print("=" * 70)
