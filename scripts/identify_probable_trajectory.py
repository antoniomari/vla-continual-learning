import os

import h5py
import numpy as np

EMBODIED_PATH = os.path.dirname(os.path.abspath(__file__))
REPO_PATH = os.path.dirname(EMBODIED_PATH)
LIBERO_REPO_PATH = os.path.join(REPO_PATH, "LIBERO")

dataset = "libero_spatial_base"  # Change to your dataset
path = f"{LIBERO_REPO_PATH}/libero/datasets_with_logits/{dataset}"

task_files = sorted([f for f in os.listdir(path) if f.endswith(".hdf5")])

print(f"Found {len(task_files)} tasks\n")
print("=" * 80)

for task_file in task_files:
    print(f"\n{task_file}")
    print("-" * 80)

    file_path = os.path.join(path, task_file)

    with h5py.File(file_path, "r") as f:
        # Get all demo keys
        demo_keys = sorted([key for key in f["data"].keys() if key.startswith("demo_")])

        demo_probabilities = {}

        for demo_key in demo_keys:
            demo = f["data"][demo_key]

            # Load logits - only first action chunk
            logits = demo["processed_action_logits"][:][:, :7, :]  # [T, 7, n_bins]

            T, action_dim, n_bins = logits.shape

            # Convert logits to probabilities
            probs = np.exp(logits) / np.exp(logits).sum(
                axis=2, keepdims=True
            )  # [T, 7, n_bins]

            # Get the probability of the predicted action at each timestep
            predicted_bins = np.argmax(probs, axis=2)  # [T, 7]

            # Extract the probability of each predicted bin
            trajectory_probs = np.zeros((T, action_dim))
            for t in range(T):
                for d in range(action_dim):
                    trajectory_probs[t, d] = probs[t, d, predicted_bins[t, d]]

            # Calculate overall trajectory probability using geometric mean
            # (to avoid underflow, use log space)
            log_prob_sum = np.sum(np.log(trajectory_probs + 1e-10))
            geometric_mean_prob = np.exp(log_prob_sum / (T * action_dim))

            demo_probabilities[demo_key] = geometric_mean_prob

        # Find the demo with highest probability
        best_demo = max(demo_probabilities.items(), key=lambda x: x[1])

        # Print all demos sorted by probability
        for demo_key in sorted(
            demo_probabilities.keys(), key=lambda k: demo_probabilities[k], reverse=True
        ):
            prob = demo_probabilities[demo_key]
            marker = (
                "  ← HIGHEST (likely training demo)" if demo_key == best_demo[0] else ""
            )
            print(f"  {demo_key}: {prob:.8f}{marker}")

print("\n" + "=" * 80)
