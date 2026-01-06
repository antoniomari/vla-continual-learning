import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

EMBODIED_PATH = os.path.dirname(os.path.abspath(__file__))
REPO_PATH = os.path.dirname(EMBODIED_PATH)
LIBERO_REPO_PATH = os.path.join(REPO_PATH, "LIBERO")

dataset = "libero_spatial_base"  # Change to your dataset
path = f"{LIBERO_REPO_PATH}/libero/datasets_with_logits/{dataset}"

task_files = sorted([f for f in os.listdir(path) if f.endswith(".hdf5")])

print(f"Found {len(task_files)} tasks\n")

# Create output directory for plots
output_dir = os.path.join(REPO_PATH, "single_state_action_distribution")
os.makedirs(output_dir, exist_ok=True)

# Which timestep to analyze (0 = first state)
TIMESTEP = -50

for task_file in task_files:
    print(f"Analyzing {task_file}")

    file_path = os.path.join(path, task_file)

    with h5py.File(file_path, "r") as f:
        demo = f["data"]["demo_0"]

        # Load data - only first action chunk
        logits = demo["processed_action_logits"][:][:, :7, :]  # [T, 7, n_bins]
        predicted_actions = demo["predicted_actions"][:][:, :7]  # [T, 7]
        ground_truth_actions = demo["actions"][:]  # [T, 7]

        T, action_dim, n_bins = logits.shape
        print(f"  Total timesteps: {T}, analyzing timestep {TIMESTEP}")
        print(
            f"  Shape: logits={logits.shape}, pred={predicted_actions.shape}, gt={ground_truth_actions.shape}"
        )

        # Extract single timestep
        logits_t = logits[TIMESTEP]  # [7, n_bins]
        pred_action_t = predicted_actions[TIMESTEP]  # [7]
        gt_action_t = ground_truth_actions[TIMESTEP]  # [7]

        # Convert logits to probabilities for this timestep
        probs_t = np.exp(logits_t) / np.exp(logits_t).sum(
            axis=1, keepdims=True
        )  # [7, n_bins]

        # Create figure with 7 subplots (one per action dimension)
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()

        fig.suptitle(
            f"{task_file.replace('.hdf5', '')} - Timestep {TIMESTEP} Action Probability Distributions",
            fontsize=16,
            fontweight="bold",
        )

        dim_names = ["X", "Y", "Z", "Roll", "Pitch", "Yaw", "Gripper"]

        for dim in range(action_dim):
            ax = axes[dim]

            probs_dim = probs_t[dim]  # [n_bins]
            x = np.arange(n_bins)

            # Color by probability magnitude
            colors = plt.cm.viridis(probs_dim / (probs_dim.max() + 1e-8))
            bars = ax.bar(x, probs_dim, color=colors, alpha=0.8, width=1.0)

            # Mark the predicted bin (argmax)
            # pred_bin = np.argmax(probs_dim)
            # ax.axvline(
            #     x=pred_bin,
            #     color="red",
            #     linestyle="--",
            #     linewidth=2.5,
            #     alpha=0.8,
            #     label=f"Pred bin={pred_bin}",
            # )

            # Calculate statistics
            max_prob = probs_dim.max()
            entropy = stats.entropy(probs_dim)
            effective_bins = np.sum(probs_dim > 0.01)

            # Color indicator for collapse
            if max_prob > 0.95 or effective_bins < 5:
                ax.set_facecolor("#ffcccc")  # Light red background
                status = "⚠️ COLLAPSED"
            elif entropy < 1.0:
                ax.set_facecolor("#fff4cc")  # Light yellow background
                status = "⚠️ PEAKED"
            else:
                status = "✅ DIVERSE"

            ax.set_xlabel("Action Bin", fontsize=10)
            ax.set_ylabel("Probability", fontsize=10)
            ax.set_title(
                f"{dim_names[dim]} {status}\n"
                f"Max={max_prob:.3f} | Ent={entropy:.2f} | Eff bins={effective_bins}\n"
                f"GT action={gt_action_t[dim]:.3f} | Pred action={pred_action_t[dim]:.3f}",
                fontsize=10,
                fontweight="bold",
            )
            ax.grid(True, alpha=0.3, axis="y")
            ax.set_xlim([0, n_bins])
            ax.legend(fontsize=9)

            # Print statistics
            print(f"    Dim {dim} ({dim_names[dim]}): {status}")
            print(f"      Max prob: {max_prob:.3f}")
            print(f"      Entropy: {entropy:.2f}")
            print(f"      Effective bins: {effective_bins}")
            # print(
            #     f"      Predicted bin: {pred_bin} → action value: {pred_action_t[dim]:.3f}"
            # )
            print(f"      Ground truth action: {gt_action_t[dim]:.3f}")
            print(
                f"      Prediction error: {pred_action_t[dim] - gt_action_t[dim]:.3f}"
            )

        # Hide the 8th subplot (we only have 7 dimensions)
        axes[7].axis("off")

        plt.tight_layout()

        # Save figure
        output_file = os.path.join(
            output_dir, task_file.replace(".hdf5", f"_t{TIMESTEP}_distributions.png")
        )
        plt.savefig(output_file, dpi=200, bbox_inches="tight")
        print(f"  📊 Saved to {output_file}")
        plt.close()

    print("--------------------------------------\n")

print(f"\n✅ All plots saved to {output_dir}")

# Print interpretation guide
print("\n" + "=" * 80)
print("INTERPRETATION GUIDE")
print("=" * 80)
print(f"""
Analyzing single timestep (t={TIMESTEP}) from each task.

Each subplot shows the probability distribution over {n_bins} action bins for one dimension:

BACKGROUND COLORS:
  • RED = Collapsed distribution (Max prob > 0.95 OR Effective bins < 5)
  • YELLOW = Peaked distribution (Entropy < 1.0)
  • WHITE = Diverse distribution (healthy)

STATISTICS:
  • Max prob: Probability of the most likely bin (1.0 = always same prediction)
  • Entropy: Measure of distribution spread (higher = more diverse)
  • Effective bins: Number of bins with probability > 1%
  • Red dashed line: The predicted bin (argmax of distribution)

INTERPRETATION:
  ✅ Diverse (white): Model uses many bins, good uncertainty representation
  ⚠️ Peaked (yellow): Model is confident but uses some bins
  ⚠️ Collapsed (red): Model always predicts same bin = mode collapse

  • Compare predicted vs ground truth action values to see accuracy
  • Low entropy + high accuracy = confident and correct
  • Low entropy + low accuracy = overconfident and wrong
""")
