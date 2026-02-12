import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

EMBODIED_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_PATH = os.path.dirname(EMBODIED_PATH)
LIBERO_REPO_PATH = os.path.join(REPO_PATH, "LIBERO")

# dataset_a = "libero_spatial_simplevla_task0_finetune"
# dataset_a = "libero_spatial_simplevla_base"
# dataset_a = "libero_spatial_simplevla_bcrl001"
dataset_a = "libero_spatial_simplevla_task0_LoRA"
path_a = f"{LIBERO_REPO_PATH}/libero/datasets_with_logits/{dataset_a}"
task_files_a = sorted([f for f in os.listdir(path_a) if f.endswith(".hdf5")])

NUM_ACTION_CHUNKS = 8
ACTION_DIM = 7
ACTION_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]

# ============================================================
# Token conversion configuration
# ============================================================
VOCAB_SIZE = 32000
N_ACTION_BINS = 256
NORMALIZATION_TYPE = "BOUNDS_Q99"

BINS = np.linspace(-1.0, 1.0, N_ACTION_BINS)
BIN_CENTERS = (BINS[:-1] + BINS[1:]) / 2.0

ACTION_STATS = {
    "max": np.array(
        [
            0.9375,
            0.9053571224212646,
            0.9375,
            0.14249999821186066,
            0.20571428537368774,
            0.08464285731315613,
            1.0,
        ]
    ),
    "min": np.array(
        [
            -0.7901785969734192,
            -0.8303571343421936,
            -0.9375,
            -0.15214285254478455,
            -0.24535714089870453,
            -0.21857142448425293,
            0.0,
        ]
    ),
    "q99": np.array(
        [
            0.9375,
            0.8758928775787354,
            0.8472589308023452,
            0.11423571482300747,
            0.15461785525083535,
            0.06997500024735921,
            1.0,
        ]
    ),
    "q01": np.array(
        [
            -0.7168392646312713,
            -0.6970714360475541,
            -0.9375,
            -0.12107142806053162,
            -0.22056428104639053,
            -0.18339642822742463,
            0.0,
        ]
    ),
    "mask": np.array([True, True, True, True, True, True, False]),
}


def normalize_actions(actions):
    """
    actions: (B, T, D) or (N, D)
    returns normalized actions in [-1, 1]
    """
    actions = np.asarray(actions)

    if NORMALIZATION_TYPE == "BOUNDS":
        mask = ACTION_STATS.get("mask", np.ones_like(ACTION_STATS["min"], dtype=bool))
        action_high, action_low = (
            np.array(ACTION_STATS["max"]),
            np.array(ACTION_STATS["min"]),
        )
    elif NORMALIZATION_TYPE == "BOUNDS_Q99":
        mask = ACTION_STATS.get("mask", np.ones_like(ACTION_STATS["q01"], dtype=bool))
        action_high, action_low = (
            np.array(ACTION_STATS["q99"]),
            np.array(ACTION_STATS["q01"]),
        )
    else:
        raise ValueError("Unsupported normalization type")

    action_dim = actions.shape[-1]
    repeat_factor = action_dim // action_high.shape[0]

    action_high = action_high.repeat(repeat_factor)
    action_low = action_low.repeat(repeat_factor)
    mask = mask * repeat_factor

    normalized_actions = np.where(
        mask,
        2.0 * (actions - action_low) / (action_high - action_low) - 1.0,
        actions,
    )

    return normalized_actions


def compute_action_tokens_from_actions(actions):
    """
    Convert continuous actions to action tokens
    actions: (B, T, D) or (T, D)
    returns: (B, T*D) or (T*D,) action tokens
    """
    actions = np.asarray(actions)

    # Handle both 2D and 3D inputs
    if actions.ndim == 2:
        actions = actions[None, :, :]  # Add batch dimension
        squeeze_output = True
    else:
        squeeze_output = False

    B, T, D = actions.shape

    # Normalize actions
    normalized_actions = normalize_actions(actions)
    normalized_actions = normalized_actions.reshape(-1, D)

    discretized_actions = []
    for dim in range(D):
        vals = normalized_actions[:, dim][:, None]  # (B*T, 1)
        dists = np.abs(vals - BIN_CENTERS[None, :])  # (B*T, n_bins)
        nearest_bins = np.argmin(dists, axis=1)  # (B*T,)
        discretized_actions.append(nearest_bins)

    discretized_actions = np.stack(discretized_actions, axis=1)  # (B*T, D)

    # Convert bin indices to tokens
    token_ids = VOCAB_SIZE - (discretized_actions + 1)
    token_ids = np.clip(
        token_ids,
        VOCAB_SIZE - N_ACTION_BINS,
        VOCAB_SIZE - 1,
    )

    token_ids = token_ids.reshape(B, T * D)

    if squeeze_output:
        token_ids = token_ids[0]  # Remove batch dimension

    return token_ids


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
        num_predictions = len(predicted_actions_chunked) // NUM_ACTION_CHUNKS

        if num_predictions == 0:
            print(f"❌ Not enough timesteps for even one full prediction chunk (T={T})")
            continue

        prediction_indices = np.arange(
            0, num_predictions * NUM_ACTION_CHUNKS, NUM_ACTION_CHUNKS
        )
        predicted_chunks = predicted_actions_chunked[prediction_indices]

        print(f"Number of model predictions: {num_predictions}")
        print(f"Predicted chunks shape: {predicted_chunks.shape}")

        # ---- Create ground truth matching the prediction timeline ----
        total_timesteps = num_predictions * NUM_ACTION_CHUNKS
        ground_truth_chunks = ground_truth_actions[:total_timesteps]

        print(f"Total timesteps covered: {total_timesteps}")
        print(f"Ground truth shape: {ground_truth_chunks.shape}")

        # ---- Unfold predictions into timeline ----
        predicted_timeline = predicted_chunks.reshape(-1, ACTION_DIM)

        print(f"Predicted timeline shape: {predicted_timeline.shape}")

        # ---- Verify shapes match ----
        if predicted_timeline.shape != ground_truth_chunks.shape:
            print(
                f"❌ Shape mismatch: {predicted_timeline.shape} vs {ground_truth_chunks.shape}"
            )
            continue

        # ---- Convert actions to tokens ----
        print("\nConverting actions to tokens...")
        predicted_tokens = compute_action_tokens_from_actions(
            predicted_timeline
        )  # [total_timesteps * 7]
        ground_truth_tokens = compute_action_tokens_from_actions(
            ground_truth_chunks
        )  # [total_timesteps * 7]

        # Reshape to [timesteps, action_dim] for easier plotting
        predicted_tokens = predicted_tokens.reshape(total_timesteps, ACTION_DIM)
        ground_truth_tokens = ground_truth_tokens.reshape(total_timesteps, ACTION_DIM)

        print(f"Predicted tokens shape: {predicted_tokens.shape}")
        print(f"Ground truth tokens shape: {ground_truth_tokens.shape}")

        # ---- Compare tokens ----
        if np.array_equal(predicted_tokens, ground_truth_tokens):
            print("✅ Tokens match exactly!")
        else:
            diff = np.abs(
                predicted_tokens.astype(np.int32) - ground_truth_tokens.astype(np.int32)
            )
            max_diff = diff.max()
            mean_diff = diff.mean()
            num_mismatches = np.sum(predicted_tokens != ground_truth_tokens)
            total_tokens = predicted_tokens.size
            print(f"\n❌ Tokens differ!")
            print(
                f"   Mismatches: {num_mismatches}/{total_tokens} ({100 * num_mismatches / total_tokens:.2f}%)"
            )
            print(f"   Max |Δ|: {max_diff}")
            print(f"   Mean |Δ|: {mean_diff:.4f}")

        # ---- Create visualization ----
        fig, axes = plt.subplots(4, 2, figsize=(16, 12))
        fig.suptitle(
            f"{task_file} - Predicted vs Ground Truth Action Tokens\n"
            f"({num_predictions} model predictions, {total_timesteps} timesteps total)\n"
            f"Token range: [{VOCAB_SIZE - N_ACTION_BINS}, {VOCAB_SIZE - 1}]",
            fontsize=14,
        )

        timesteps = np.arange(total_timesteps)

        for dim in range(ACTION_DIM):
            ax = axes[dim // 2, dim % 2]

            # Plot ground truth tokens
            ax.plot(
                timesteps,
                ground_truth_tokens[:, dim],
                "--",
                label="Ground Truth",
                linewidth=2,
                alpha=0.8,
                color="C1",
            )

            # Plot predicted tokens
            ax.plot(
                timesteps,
                predicted_tokens[:, dim],
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
            diff_dim = np.abs(
                predicted_tokens[:, dim].astype(np.int32)
                - ground_truth_tokens[:, dim].astype(np.int32)
            )
            num_mismatches_dim = np.sum(
                predicted_tokens[:, dim] != ground_truth_tokens[:, dim]
            )

            ax.set_xlabel("Timestep")
            ax.set_ylabel("Token ID")
            ax.set_title(
                f"{ACTION_NAMES[dim]} (mismatches: {num_mismatches_dim}/{total_timesteps}, "
                f"mean diff: {diff_dim.mean():.2f}, max diff: {diff_dim.max()})"
            )
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Set y-axis limits to token range
            ax.set_ylim([VOCAB_SIZE - N_ACTION_BINS - 10, VOCAB_SIZE + 10])

        # Hide the last subplot if ACTION_DIM is odd
        if ACTION_DIM % 2 == 1:
            axes[-1, -1].axis("off")

        plt.tight_layout()

        # Save figure
        output_dir = os.path.join(path_a, "token_comparison_plots")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir, f"{task_file.replace('.hdf5', '')}_token_timeline.png"
        )
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\n💾 Saved token timeline plot to: {output_path}")
        plt.close()


print(f"\n{'=' * 70}")
print("✨ Analysis complete!")
print("=" * 70)
