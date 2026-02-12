import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

EMBODIED_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_PATH = os.path.dirname(EMBODIED_PATH)
LIBERO_REPO_PATH = os.path.join(REPO_PATH, "LIBERO")

dataset_a = "libero_spatial_simplevla_task0_finetune"
# dataset_a = "libero_spatial_simplevla_base"
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
        predicted_actions = demo_a["predicted_actions"][:].reshape(-1, 56)
        ground_truth_actions = demo_a["actions"][:]

        print(f"Predicted actions shape: {predicted_actions.shape}")
        print(f"Ground truth actions shape: {ground_truth_actions.shape}")

        # ---- Reshape predicted actions ----
        T = predicted_actions.shape[0]
        expected_flat_dim = NUM_ACTION_CHUNKS * ACTION_DIM

        if predicted_actions.shape[1] != expected_flat_dim:
            print(
                f"❌ Predicted actions have unexpected dimension: {predicted_actions.shape[1]} (expected {expected_flat_dim})"
            )
            continue

        predicted_actions_chunked = predicted_actions.reshape(
            T, NUM_ACTION_CHUNKS, ACTION_DIM
        )  # [T, 8, 7]

        # ---- Subsample predictions every 8 steps ----
        # Only keep predictions at timesteps 0, 8, 16, 24, ...
        num_predictions = len(predicted_actions_chunked) // NUM_ACTION_CHUNKS

        if num_predictions == 0:
            print(f"❌ Not enough timesteps for even one full prediction chunk (T={T})")
            continue

        # Select predictions at intervals of NUM_ACTION_CHUNKS
        prediction_indices = np.arange(
            0, num_predictions * NUM_ACTION_CHUNKS, NUM_ACTION_CHUNKS
        )
        predicted_chunks = predicted_actions_chunked[
            prediction_indices
        ]  # [num_predictions, 8, 7]

        print(f"Number of model predictions: {num_predictions}")
        print(f"Predicted chunks shape: {predicted_chunks.shape}")

        # ---- Create ground truth matching the prediction timeline ----
        # For prediction i at timestep i*8, ground truth is actions[i*8 : i*8+8]
        total_timesteps = num_predictions * NUM_ACTION_CHUNKS
        ground_truth_chunks = ground_truth_actions[
            :total_timesteps
        ]  # [total_timesteps, 7]

        print(f"Total timesteps covered: {total_timesteps}")
        print(f"Ground truth shape: {ground_truth_chunks.shape}")

        # ---- Unfold predictions into timeline ----
        # predicted_chunks: [num_predictions, 8, 7]
        # We want to flatten this to [num_predictions * 8, 7] to match ground truth
        predicted_timeline = predicted_chunks.reshape(
            -1, ACTION_DIM
        )  # [total_timesteps, 7]

        print(f"Predicted timeline shape: {predicted_timeline.shape}")

        # ---- Verify shapes match ----
        if predicted_timeline.shape != ground_truth_chunks.shape:
            print(
                f"❌ Shape mismatch: {predicted_timeline.shape} vs {ground_truth_chunks.shape}"
            )
            continue

        # ---- Compare actions ----
        if np.allclose(predicted_timeline, ground_truth_chunks, atol=1e-6):
            print("⚠️  Actions match within tolerance (atol=1e-6)")
        else:
            diff = np.abs(predicted_timeline - ground_truth_chunks)
            max_diff = diff.max()
            mean_diff = diff.mean()
            print(f"\n❌ Actions differ!")
            print(f"   Max |Δ|: {max_diff:.6f}")
            print(f"   Mean |Δ|: {mean_diff:.6f}")

        # ---- Create visualization ----
        fig, axes = plt.subplots(4, 2, figsize=(16, 12))
        fig.suptitle(
            f"{task_file} - Predicted vs Ground Truth Actions\n"
            f"({num_predictions} model predictions, {total_timesteps} timesteps total)",
            fontsize=14,
        )

        # Create timestep array for x-axis
        timesteps = np.arange(total_timesteps)

        for dim in range(ACTION_DIM):
            ax = axes[dim // 2, dim % 2]

            # Plot ground truth
            ax.plot(
                timesteps,
                ground_truth_chunks[:, dim],
                "--",
                label="Ground Truth",
                linewidth=2,
                alpha=0.8,
                color="C1",
            )

            # Plot predicted actions
            ax.plot(
                timesteps,
                predicted_timeline[:, dim],
                "-",
                label="Predicted",
                linewidth=1.5,
                alpha=0.8,
                color="C0",
            )

            # Add vertical lines to show prediction boundaries
            for pred_idx in range(1, num_predictions):
                ax.axvline(
                    x=pred_idx * NUM_ACTION_CHUNKS,
                    color="gray",
                    linestyle=":",
                    alpha=0.5,
                    linewidth=1,
                )

            # Calculate difference for this dimension
            diff_dim = np.abs(predicted_timeline[:, dim] - ground_truth_chunks[:, dim])

            ax.set_xlabel("Timestep")
            ax.set_ylabel("Value")
            ax.set_title(
                f"{ACTION_NAMES[dim]} (mean diff: {diff_dim.mean():.4f}, max diff: {diff_dim.max():.4f})"
            )
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Hide the last subplot if ACTION_DIM is odd
        if ACTION_DIM % 2 == 1:
            axes[-1, -1].axis("off")

        plt.tight_layout()

        # Save figure
        output_dir = os.path.join(path_a, "comparison_plots")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir, f"{task_file.replace('.hdf5', '')}_action_timeline.png"
        )
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\n💾 Saved action timeline plot to: {output_path}")
        plt.close()


print(f"\n{'=' * 70}")
print("✨ Analysis complete!")
print("=" * 70)
