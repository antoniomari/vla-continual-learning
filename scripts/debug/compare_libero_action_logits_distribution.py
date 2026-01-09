import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

EMBODIED_PATH = os.path.dirname(os.path.abspath(__file__))
REPO_PATH = os.path.dirname(EMBODIED_PATH)
LIBERO_REPO_PATH = os.path.join(REPO_PATH, "LIBERO")

dataset_a = "libero_spatial4"
dataset_b = "libero_spatial2"

path_a = f"{LIBERO_REPO_PATH}/libero/datasets_with_logits/{dataset_a}"
path_b = f"{LIBERO_REPO_PATH}/libero/datasets_with_logits/{dataset_b}"

task_files_a = sorted([f for f in os.listdir(path_a) if f.endswith(".hdf5")])
task_files_b = sorted([f for f in os.listdir(path_b) if f.endswith(".hdf5")])

print(f"Found {len(task_files_b)} matching tasks\n")

# Create output directory for plots
output_dir = os.path.join(REPO_PATH, "logit_comparison_plots")
os.makedirs(output_dir, exist_ok=True)

for task_file in task_files_b:
    print(f"Checking {task_file}")

    file_a = os.path.join(path_a, task_file)
    file_b = os.path.join(path_b, task_file)

    with h5py.File(file_a, "r") as fa, h5py.File(file_b, "r") as fb:
        demo_a = fa["data"]["demo_0"]
        demo_b = fb["data"]["demo_0"]

        # Load logits
        logits_a = demo_a["processed_action_logits"][:]
        logits_b = demo_b["processed_action_logits"][:]

        # Load predicted actions for comparison
        actions_a = demo_a["predicted_actions"][:]
        actions_b = demo_b["predicted_actions"][:]

        if logits_a.shape != logits_b.shape:
            print("  ❌ Logit shape mismatch:", logits_a.shape, logits_b.shape)
            continue

        # Shape is [timesteps, action_dim, n_action_bins]
        T, action_dim, n_bins = logits_a.shape

        print(
            f"  Shape: {logits_a.shape} (timesteps={T}, action_dim={action_dim}, bins={n_bins})"
        )

        # Check if logits match
        if np.array_equal(logits_a, logits_b):
            print("  ✅ Logits match exactly")
        else:
            diff = np.abs(logits_a - logits_b)
            idxs = np.argwhere(diff != 0)
            print(f"  ❌ Logits differ at {len(idxs)} positions")

            # Show some examples
            MAX_PRINT = 5
            for i, idx in enumerate(idxs[:MAX_PRINT]):
                idx_tuple = tuple(idx)
                print(
                    f"    at {idx_tuple}: "
                    f"logits_a={logits_a[idx_tuple]:.6f}, "
                    f"logits_b={logits_b[idx_tuple]:.6f}, "
                    f"|Δ|={diff[idx_tuple]:.6f}"
                )
            if len(idxs) > MAX_PRINT:
                print(f"    ... and {len(idxs) - MAX_PRINT} more differences")

        # Check if predicted actions match
        if np.array_equal(actions_a, actions_b):
            print("  ✅ Predicted actions match exactly")
        else:
            action_diff = np.abs(actions_a - actions_b)
            action_idxs = np.argwhere(action_diff > 0)
            print(f"  ❌ Actions differ at {len(action_idxs)} positions")
            print(f"    Max action difference: {np.max(action_diff):.6f}")
            print(f"    Mean action difference: {np.mean(action_diff):.6f}")

        # ====== Visualization ======

        # Create figure with subplots for each action dimension
        fig, axes = plt.subplots(action_dim, 3, figsize=(18, 4 * action_dim))
        if action_dim == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle(f"{task_file.replace('.hdf5', '')}", fontsize=16, y=0.995)

        for dim in range(action_dim):
            # Get logits for this dimension across all timesteps
            logits_a_dim = logits_a[:, dim, :]  # [T, n_bins]
            logits_b_dim = logits_b[:, dim, :]  # [T, n_bins]

            # Convert logits to probabilities (softmax)
            probs_a_dim = np.exp(logits_a_dim) / np.exp(logits_a_dim).sum(
                axis=1, keepdims=True
            )
            probs_b_dim = np.exp(logits_b_dim) / np.exp(logits_b_dim).sum(
                axis=1, keepdims=True
            )

            # Average probabilities across timesteps
            avg_probs_a = probs_a_dim.mean(axis=0)
            avg_probs_b = probs_b_dim.mean(axis=0)

            # Plot 1: Average probability distribution
            ax1 = axes[dim, 0]
            x = np.arange(n_bins)
            ax1.plot(x, avg_probs_a, "b-", label=dataset_a, alpha=0.7, linewidth=2)
            ax1.plot(x, avg_probs_b, "r--", label=dataset_b, alpha=0.7, linewidth=2)
            ax1.set_xlabel("Action Bin")
            ax1.set_ylabel("Probability")
            ax1.set_title(f"Dim {dim}: Avg Probability Distribution")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot 2: Difference in probabilities
            ax2 = axes[dim, 1]
            prob_diff = avg_probs_a - avg_probs_b
            ax2.bar(
                x,
                prob_diff,
                color=["red" if d < 0 else "blue" for d in prob_diff],
                alpha=0.7,
            )
            ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
            ax2.set_xlabel("Action Bin")
            ax2.set_ylabel("Probability Difference")
            ax2.set_title(f"Dim {dim}: Δ Probability ({dataset_a} - {dataset_b})")
            ax2.grid(True, alpha=0.3)

            # Plot 3: Per-timestep KL divergence
            ax3 = axes[dim, 2]
            kl_divs = []
            for t in range(T):
                # KL divergence from dataset_b to dataset_a
                kl = stats.entropy(probs_a_dim[t], probs_b_dim[t])
                kl_divs.append(kl)

            ax3.plot(kl_divs, "g-", linewidth=1, alpha=0.7)
            ax3.set_xlabel("Timestep")
            ax3.set_ylabel("KL Divergence")
            ax3.set_title(f"Dim {dim}: KL(A||B) per timestep")
            ax3.grid(True, alpha=0.3)

            # Add statistics text
            mean_kl = np.mean(kl_divs)
            max_kl = np.max(kl_divs)
            ax3.text(
                0.98,
                0.98,
                f"Mean: {mean_kl:.4f}\nMax: {max_kl:.4f}",
                transform=ax3.transAxes,
                fontsize=9,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        plt.tight_layout()

        # Save figure
        output_file = os.path.join(
            output_dir, task_file.replace(".hdf5", "_comparison.png")
        )
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"  📊 Saved plot to {output_file}")
        plt.close()

        # Print summary statistics
        print(f"  📈 Summary Statistics:")
        for dim in range(action_dim):
            logits_a_dim = logits_a[:, dim, :]
            logits_b_dim = logits_b[:, dim, :]
            probs_a_dim = np.exp(logits_a_dim) / np.exp(logits_a_dim).sum(
                axis=1, keepdims=True
            )
            probs_b_dim = np.exp(logits_b_dim) / np.exp(logits_b_dim).sum(
                axis=1, keepdims=True
            )

            kl_divs = [stats.entropy(probs_a_dim[t], probs_b_dim[t]) for t in range(T)]
            mean_kl = np.mean(kl_divs)
            max_kl = np.max(kl_divs)

            print(f"    Dim {dim}: Mean KL={mean_kl:.6f}, Max KL={max_kl:.6f}")

    print("--------------------------------------\n")

print(f"\n✅ All plots saved to {output_dir}")
