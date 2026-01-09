import os

import h5py
import numpy as np

EMBODIED_PATH = os.path.dirname(os.path.abspath(__file__))
REPO_PATH = os.path.dirname(EMBODIED_PATH)
LIBERO_REPO_PATH = os.path.join(REPO_PATH, "LIBERO")

dataset = "libero_spatial_simplevla"  # Change to your dataset
path = f"{LIBERO_REPO_PATH}/libero/datasets_with_logits/{dataset}"
task_files = sorted([f for f in os.listdir(path) if f.endswith(".hdf5")])

print(f"Found {len(task_files)} tasks\n")
print("=" * 80)

CHUNK_SIZE = 8  # Number of actions predicted per timestep
ACTION_DIM = 7  # Dimensions per action

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

            # Load logits - shape is [T_total, 56, n_bins]
            logits = demo["processed_action_logits"][:]  # [T_total, 56, n_bins]

            # Subsample to get only the actual prediction timesteps (every 8th)
            logits_subsampled = logits[::CHUNK_SIZE]  # [T_pred, 56, n_bins]

            T_pred, tokens_per_pred, n_bins = logits_subsampled.shape

            # Verify the shape is correct (56 = 8 chunks * 7 dimensions)
            assert tokens_per_pred == CHUNK_SIZE * ACTION_DIM, (
                f"Expected {CHUNK_SIZE * ACTION_DIM} tokens, got {tokens_per_pred}"
            )

            # Reshape to [T_pred, CHUNK_SIZE, ACTION_DIM, n_bins]
            logits_reshaped = logits_subsampled.reshape(
                T_pred, CHUNK_SIZE, ACTION_DIM, n_bins
            )

            # Convert logits to probabilities
            probs = np.exp(logits_reshaped) / np.exp(logits_reshaped).sum(
                axis=3, keepdims=True
            )  # [T_pred, CHUNK_SIZE, ACTION_DIM, n_bins]

            # Get the predicted bin for each action dimension
            predicted_bins = np.argmax(
                probs, axis=3
            )  # [T_pred, CHUNK_SIZE, ACTION_DIM]

            # Extract the probability of each predicted bin
            trajectory_probs = np.zeros((T_pred, CHUNK_SIZE, ACTION_DIM))
            for t in range(T_pred):
                for c in range(CHUNK_SIZE):
                    for d in range(ACTION_DIM):
                        trajectory_probs[t, c, d] = probs[
                            t, c, d, predicted_bins[t, c, d]
                        ]

            # Calculate trajectory probability by multiplying all probabilities
            # This will be EXTREMELY small, but it's the actual probability
            raw_prob = np.prod(trajectory_probs)

            demo_probabilities[demo_key] = raw_prob

        # Find the demo with highest probability
        best_demo = max(demo_probabilities.items(), key=lambda x: x[1])

        # Print all demos sorted by probability
        print("\nRaw Trajectory Probabilities:")
        for demo_key in sorted(
            demo_probabilities.keys(), key=lambda k: demo_probabilities[k], reverse=True
        ):
            prob = demo_probabilities[demo_key]
            marker = (
                "  ← HIGHEST (likely training demo)" if demo_key == best_demo[0] else ""
            )
            print(f"  {demo_key}: {prob:.2e}{marker}")

print("\n" + "=" * 80)
