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
output_dir = os.path.join(REPO_PATH, "action_distribution_analysis")
os.makedirs(output_dir, exist_ok=True)

# Aggregate statistics across all tasks
all_entropies = []
all_max_probs = []
all_effective_bins = []

for task_file in task_files:
    print(f"Analyzing {task_file}")

    file_path = os.path.join(path, task_file)

    with h5py.File(file_path, "r") as f:
        demo = f["data"]["demo_0"]

        # Load data
        logits = demo["processed_action_logits"][:][:, :7, :]  # [T, action_dim, n_bins]
        predicted_actions = demo["predicted_actions"][:][:, :7]  # [T, action_dim]
        ground_truth_actions = demo["actions"][:]  # [T, action_dim]

        T, action_dim, n_bins = logits.shape
        print(
            f"  Shape: {logits.shape} (timesteps={T}, action_dim={action_dim}, bins={n_bins})"
        )
        print(f"  Predicted actions shape: {predicted_actions.shape}")
        print(f"  Ground truth actions shape: {ground_truth_actions.shape}")

        # Check if predicted actions match ground truth
        if np.array_equal(predicted_actions, ground_truth_actions):
            print("  ✅ Predicted actions match ground truth exactly")
        else:
            action_diff = np.abs(predicted_actions - ground_truth_actions)
            action_idxs = np.argwhere(action_diff > 1e-6)
            print(f"  ❌ Actions differ at {len(action_idxs)} positions")
            print(f"    Max action difference: {np.max(action_diff):.6f}")
            print(f"    Mean action difference: {np.mean(action_diff):.6f}")

        # ====== Analyze Distribution Sharpness ======

        # Convert logits to probabilities
        probs = np.exp(logits) / np.exp(logits).sum(
            axis=2, keepdims=True
        )  # [T, action_dim, n_bins]

        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 5 * action_dim))
        gs = fig.add_gridspec(action_dim, 4, hspace=0.3, wspace=0.3)

        fig.suptitle(
            f"{task_file.replace('.hdf5', '')} - Distribution Analysis", fontsize=16
        )

        print(f"  📊 Distribution Sharpness Analysis:")

        for dim in range(action_dim):
            probs_dim = probs[:, dim, :]  # [T, n_bins]

            # Calculate entropy (low entropy = sharp/peaked distribution)
            entropies = stats.entropy(probs_dim, axis=1)  # [T]

            # Calculate max probability (high max_prob = sharp distribution)
            max_probs = np.max(probs_dim, axis=1)  # [T]

            # Calculate "effective number of bins" (bins with >1% probability)
            effective_bins = np.sum(probs_dim > 0.01, axis=1)  # [T]

            # Store for aggregate statistics
            all_entropies.extend(entropies)
            all_max_probs.extend(max_probs)
            all_effective_bins.extend(effective_bins)

            # Statistics
            mean_entropy = np.mean(entropies)
            mean_max_prob = np.mean(max_probs)
            mean_effective_bins = np.mean(effective_bins)

            # Check if distribution is too peaked
            is_collapsed = mean_max_prob > 0.95 or mean_effective_bins < 5
            status = "⚠️  COLLAPSED" if is_collapsed else "✅ Diverse"

            print(f"    Dim {dim}: {status}")
            print(f"      Mean Entropy: {mean_entropy:.4f} (max={np.log(n_bins):.4f})")
            print(f"      Mean Max Prob: {mean_max_prob:.4f}")
            print(f"      Mean Effective Bins: {mean_effective_bins:.1f}/{n_bins}")

            # Plot 1: Entropy over time
            ax1 = fig.add_subplot(gs[dim, 0])
            ax1.plot(entropies, "b-", linewidth=1, alpha=0.7)
            ax1.axhline(
                y=np.log(n_bins),
                color="g",
                linestyle="--",
                label=f"Max entropy={np.log(n_bins):.2f}",
                alpha=0.5,
            )
            ax1.axhline(
                y=mean_entropy,
                color="r",
                linestyle="--",
                label=f"Mean={mean_entropy:.2f}",
                alpha=0.5,
            )
            ax1.set_xlabel("Timestep")
            ax1.set_ylabel("Entropy")
            ax1.set_title(f"Dim {dim}: Entropy over Time")
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3)

            # Plot 2: Max probability over time
            ax2 = fig.add_subplot(gs[dim, 1])
            ax2.plot(max_probs, "r-", linewidth=1, alpha=0.7)
            ax2.axhline(
                y=mean_max_prob,
                color="b",
                linestyle="--",
                label=f"Mean={mean_max_prob:.3f}",
                alpha=0.5,
            )
            ax2.axhline(
                y=0.95,
                color="orange",
                linestyle="--",
                label="Threshold=0.95",
                alpha=0.5,
            )
            ax2.set_xlabel("Timestep")
            ax2.set_ylabel("Max Probability")
            ax2.set_title(f"Dim {dim}: Max Probability over Time")
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim([0, 1.05])

            # Plot 3: Effective number of bins
            ax3 = fig.add_subplot(gs[dim, 2])
            ax3.plot(effective_bins, "g-", linewidth=1, alpha=0.7)
            ax3.axhline(
                y=mean_effective_bins,
                color="b",
                linestyle="--",
                label=f"Mean={mean_effective_bins:.1f}",
                alpha=0.5,
            )
            ax3.axhline(
                y=5, color="orange", linestyle="--", label="Threshold=5", alpha=0.5
            )
            ax3.set_xlabel("Timestep")
            ax3.set_ylabel("Effective Bins (>1%)")
            ax3.set_title(f"Dim {dim}: Effective Number of Bins")
            ax3.legend(fontsize=8)
            ax3.grid(True, alpha=0.3)

            # Plot 4: Average probability distribution
            ax4 = fig.add_subplot(gs[dim, 3])
            avg_probs = probs_dim.mean(axis=0)  # [n_bins]
            x = np.arange(n_bins)

            # Color bars by probability
            colors = plt.cm.RdYlGn_r(avg_probs / avg_probs.max())
            ax4.bar(x, avg_probs, color=colors, alpha=0.7, width=1.0)

            # Highlight top bins
            top_bins = np.argsort(avg_probs)[-5:]
            for bin_idx in top_bins:
                ax4.axvline(
                    x=bin_idx, color="red", linestyle="--", alpha=0.3, linewidth=0.5
                )

            ax4.set_xlabel("Action Bin")
            ax4.set_ylabel("Average Probability")
            ax4.set_title(f"Dim {dim}: Average Distribution\n(Top 5 bins marked)")
            ax4.grid(True, alpha=0.3, axis="y")

            # Add text with concentration info
            top_5_mass = np.sum(np.sort(avg_probs)[-5:])
            ax4.text(
                0.98,
                0.98,
                f"Top 5 bins: {top_5_mass:.1%}\nMax: {avg_probs.max():.3f}",
                transform=ax4.transAxes,
                fontsize=9,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        plt.tight_layout()

        # Save figure
        output_file = os.path.join(
            output_dir, task_file.replace(".hdf5", "_distribution_analysis.png")
        )
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"  📊 Saved plot to {output_file}")
        plt.close()

    print("--------------------------------------\n")

# ====== Aggregate Statistics Across All Tasks ======
print("=" * 80)
print("AGGREGATE STATISTICS ACROSS ALL TASKS")
print("=" * 80)

all_entropies = np.array(all_entropies)
all_max_probs = np.array(all_max_probs)
all_effective_bins = np.array(all_effective_bins)

print(f"\nEntropy Statistics:")
print(f"  Mean: {np.mean(all_entropies):.4f}")
print(f"  Median: {np.median(all_entropies):.4f}")
print(f"  Std: {np.std(all_entropies):.4f}")
print(f"  Min: {np.min(all_entropies):.4f}")
print(f"  Max: {np.max(all_entropies):.4f}")

print(f"\nMax Probability Statistics:")
print(f"  Mean: {np.mean(all_max_probs):.4f}")
print(f"  Median: {np.median(all_max_probs):.4f}")
print(f"  Std: {np.std(all_max_probs):.4f}")
print(f"  % above 0.95: {100 * np.mean(all_max_probs > 0.95):.1f}%")
print(f"  % above 0.99: {100 * np.mean(all_max_probs > 0.99):.1f}%")

print(f"\nEffective Bins Statistics:")
print(f"  Mean: {np.mean(all_effective_bins):.1f}")
print(f"  Median: {np.median(all_effective_bins):.1f}")
print(f"  Std: {np.std(all_effective_bins):.1f}")
print(f"  % below 5 bins: {100 * np.mean(all_effective_bins < 5):.1f}%")
print(f"  % below 10 bins: {100 * np.mean(all_effective_bins < 10):.1f}%")

# Create aggregate histogram
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Histogram 1: Entropy
axes[0].hist(all_entropies, bins=50, alpha=0.7, color="blue", edgecolor="black")
axes[0].axvline(
    x=np.mean(all_entropies),
    color="red",
    linestyle="--",
    label=f"Mean={np.mean(all_entropies):.3f}",
    linewidth=2,
)
axes[0].set_xlabel("Entropy")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Distribution of Entropy Across All Timesteps/Dims/Tasks")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Histogram 2: Max Probability
axes[1].hist(all_max_probs, bins=50, alpha=0.7, color="red", edgecolor="black")
axes[1].axvline(
    x=np.mean(all_max_probs),
    color="blue",
    linestyle="--",
    label=f"Mean={np.mean(all_max_probs):.3f}",
    linewidth=2,
)
axes[1].axvline(
    x=0.95, color="orange", linestyle="--", label="Collapse threshold=0.95", linewidth=2
)
axes[1].set_xlabel("Max Probability")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Distribution of Max Probability Across All Timesteps/Dims/Tasks")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Histogram 3: Effective Bins
axes[2].hist(
    all_effective_bins,
    bins=range(0, 257, 5),
    alpha=0.7,
    color="green",
    edgecolor="black",
)
axes[2].axvline(
    x=np.mean(all_effective_bins),
    color="blue",
    linestyle="--",
    label=f"Mean={np.mean(all_effective_bins):.1f}",
    linewidth=2,
)
axes[2].axvline(
    x=5, color="orange", linestyle="--", label="Collapse threshold=5", linewidth=2
)
axes[2].set_xlabel("Effective Number of Bins (>1% prob)")
axes[2].set_ylabel("Frequency")
axes[2].set_title("Distribution of Effective Bins Across All Timesteps/Dims/Tasks")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
aggregate_file = os.path.join(output_dir, "aggregate_statistics.png")
plt.savefig(aggregate_file, dpi=150, bbox_inches="tight")
print(f"\n📊 Saved aggregate statistics to {aggregate_file}")
plt.close()

# Summary verdict
print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)

if np.mean(all_max_probs) > 0.95:
    print("⚠️  WARNING: Model distributions are HIGHLY PEAKED!")
    print("   The model is collapsing to point distributions.")
    print(
        f"   {100 * np.mean(all_max_probs > 0.95):.1f}% of predictions have >95% confidence."
    )
elif np.mean(all_max_probs) > 0.85:
    print("⚠️  CAUTION: Model distributions are quite sharp.")
    print("   The model may be overfitting or losing diversity.")
elif np.mean(all_effective_bins) < 10:
    print("⚠️  CAUTION: Model is using very few bins effectively.")
    print(
        f"   Average of only {np.mean(all_effective_bins):.1f} bins have >1% probability."
    )
else:
    print("✅ Model distributions appear healthy and diverse.")
    print(f"   Mean max probability: {np.mean(all_max_probs):.3f}")
    print(f"   Mean effective bins: {np.mean(all_effective_bins):.1f}")

print(f"\n✅ All plots saved to {output_dir}")
