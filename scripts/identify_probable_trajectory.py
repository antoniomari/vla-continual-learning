import os

import h5py
import numpy as np
from scipy.special import logsumexp

EMBODIED_PATH = os.path.dirname(os.path.abspath(__file__))
REPO_PATH = os.path.dirname(EMBODIED_PATH)
LIBERO_REPO_PATH = os.path.join(REPO_PATH, "LIBERO")

# dataset = "libero_spatial_simplevla_base"  # Change to your dataset
# dataset = "libero_spatial_simplevla_task0_finetune"  # Change to your dataset
# dataset = "libero_spatial_simplevla_task0_LoRA"  # Change to your dataset
dataset = "libero_spatial_simplevla_bcrl001"  # Change to your dataset
path = f"{LIBERO_REPO_PATH}/libero/datasets_with_logits/{dataset}"
task_files = sorted([f for f in os.listdir(path) if f.endswith(".hdf5")])

print(f"Found {len(task_files)} tasks\n")
print("=" * 80)

CHUNK_SIZE = 8  # Number of actions predicted per timestep
ACTION_DIM = 7  # Dimensions per action
VOCAB_SIZE = 32000
N_ACTION_BINS = 256

for task_file in task_files:
    print(f"\n{task_file}")
    print("-" * 80)

    file_path = os.path.join(path, task_file)
    with h5py.File(file_path, "r") as f:
        demo_keys = sorted([key for key in f["data"].keys() if key.startswith("demo_")])
        demo_probabilities = {}

        for demo_key in demo_keys:
            demo = f["data"][demo_key]

            # --- Load logits ---
            logits = demo["processed_action_logits"][:]  # [T_total, 56, 256]
            action_tokens = demo["action_tokens"][:]  # [T_total, 56] absolute tokens

            # Subsample every 8 timesteps
            logits_subsampled = logits[::CHUNK_SIZE]
            action_tokens_subsampled = action_tokens[::CHUNK_SIZE]

            min_len = min(len(logits_subsampled), len(action_tokens_subsampled))
            logits_subsampled = logits_subsampled[:min_len]
            action_tokens_subsampled = action_tokens_subsampled[:min_len]

            T_pred, tokens_per_pred, n_bins = logits_subsampled.shape
            assert tokens_per_pred == CHUNK_SIZE * ACTION_DIM, (
                f"Expected {CHUNK_SIZE * ACTION_DIM} tokens, got {tokens_per_pred}"
            )

            # --- Reshape logits ---
            logits_reshaped = logits_subsampled.reshape(
                T_pred, CHUNK_SIZE, ACTION_DIM, n_bins
            )

            # --- Convert logits to log-probabilities ---
            log_probs = logits_reshaped - logsumexp(
                logits_reshaped, axis=3, keepdims=True
            )

            # --- Adjust action tokens to indices for processed_action_logits ---
            # action_tokens are absolute tokens, convert to bin indices
            adjusted_tokens = action_tokens_subsampled.reshape(
                -1, CHUNK_SIZE, ACTION_DIM
            )
            adjusted_tokens = adjusted_tokens - (
                VOCAB_SIZE - N_ACTION_BINS
            )  # absolute → index

            # --- Accuracy ---
            logits_sliced = logits_subsampled[: adjusted_tokens.shape[0]].reshape(
                -1, CHUNK_SIZE, ACTION_DIM, n_bins
            )
            predicted_bins = np.argmax(logits_sliced, axis=3)
            predicted_bins = np.clip(predicted_bins, 1, N_ACTION_BINS)
            # print(
            #     f"predicted_bins: {predicted_bins[0]}, \n tokens: {adjusted_tokens[0]}"
            # )
            correct = predicted_bins == adjusted_tokens
            accuracy = correct.mean()  # fraction of correctly predicted tokens

            # --- Trajectory probabilities ---
            trajectory_log_probs = log_probs[
                np.arange(T_pred)[:, None, None],
                np.arange(CHUNK_SIZE)[None, :, None],
                np.arange(ACTION_DIM)[None, None, :],
                adjusted_tokens,
            ]
            trajectory_log_probs_flat = trajectory_log_probs.flatten()
            geom_mean_prob = np.exp(np.mean(trajectory_log_probs_flat))

            log_raw_prob = np.sum(trajectory_log_probs_flat)  # This is ln(probability)
            order_of_magnitude = log_raw_prob / np.log(10)

            demo_probabilities[demo_key] = {
                "log_raw_prob": log_raw_prob,
                "order_of_magnitude": order_of_magnitude,
                "geom_mean_prob": geom_mean_prob,
                "accuracy": accuracy,
            }

        # Find the demo with highest geometric mean probability
        best_demo = max(
            demo_probabilities.items(), key=lambda x: x[1]["geom_mean_prob"]
        )

        print("\nTrajectory Probabilities & Accuracy:")
        for demo_key in sorted(
            demo_probabilities.keys(),
            key=lambda k: demo_probabilities[k]["geom_mean_prob"],
            reverse=True,
        ):
            demo_stats = demo_probabilities[demo_key]
            marker = (
                "  ← HIGHEST (likely training demo)" if demo_key == best_demo[0] else ""
            )
            print(
                f"  {demo_key}: "
                f"log_prob={demo_stats['log_raw_prob']:.2f} (10^{demo_stats['order_of_magnitude']:.1f}), "
                f"geom_mean={demo_stats['geom_mean_prob']:.4f}, "
                f"accuracy={demo_stats['accuracy']:.4f}{marker}"
            )

print("\n" + "=" * 80)
