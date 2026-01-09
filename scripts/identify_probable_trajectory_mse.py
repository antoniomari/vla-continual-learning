import os

import h5py
import numpy as np

EMBODIED_PATH = os.path.dirname(os.path.abspath(__file__))
REPO_PATH = os.path.dirname(EMBODIED_PATH)
LIBERO_REPO_PATH = os.path.join(REPO_PATH, "LIBERO")

dataset = "libero_spatial_simplevla"
path = f"{LIBERO_REPO_PATH}/libero/datasets_with_logits/{dataset}"
task_files = sorted([f for f in os.listdir(path) if f.endswith(".hdf5")])

print(f"Found {len(task_files)} tasks\n")
print("=" * 80)

CHUNK_SIZE = 8
ACTION_DIM = 7

for task_file in task_files:
    print(f"\n{task_file}")
    print("-" * 80)

    file_path = os.path.join(path, task_file)
    with h5py.File(file_path, "r") as f:
        demo_keys = sorted([key for key in f["data"].keys() if key.startswith("demo_")])

        # ============ FIND GROUND TRUTH ACTIONS ============
        print("\n🔍 Checking for ground truth actions...")

        # Check what keys are available in the first demo
        first_demo = f["data"][demo_keys[0]]
        print(f"Available keys in demo: {list(first_demo.keys())}")

        # Try to find actions
        action_key = None
        for possible_key in ["actions", "action", "obs/action", "gt_actions"]:
            if possible_key in first_demo:
                action_key = possible_key
                print(f"✅ Found actions under key: '{action_key}'")
                break

        if action_key is None:
            print("❌ No ground truth actions found. Cannot do direct comparison.")
            print("   Available keys:", list(first_demo.keys()))
            continue

        # ============ EXTRACT MODEL PREDICTIONS ============
        print(
            "\n📊 Extracting model predictions and computing MSE with ground truth...\n"
        )

        demo_results = {}

        for demo_key in demo_keys:
            demo = f["data"][demo_key]

            # Get ground truth actions
            gt_actions = demo[action_key][:]  # Shape: [T_total, action_dim]

            # Get model predictions from logits
            logits = demo["processed_action_logits"][:]  # [T_total, 56, n_bins]

            # Subsample every CHUNK_SIZE timesteps (when model makes new predictions)
            logits_subsampled = logits[::CHUNK_SIZE]
            gt_actions_subsampled = gt_actions[::CHUNK_SIZE]

            T_pred, tokens_per_pred, n_bins = logits_subsampled.shape

            # Reshape logits
            logits_reshaped = logits_subsampled.reshape(
                T_pred, CHUNK_SIZE, ACTION_DIM, n_bins
            )

            # Convert to probabilities and get predicted bins
            probs = np.exp(logits_reshaped) / np.exp(logits_reshaped).sum(
                axis=3, keepdims=True
            )
            predicted_bins = np.argmax(
                probs, axis=3
            )  # [T_pred, CHUNK_SIZE, ACTION_DIM]

            # We need to convert bins back to continuous actions
            # Check if there's bin information stored
            bin_info_available = False
            if "action_bins" in demo or "bins" in demo:
                bin_info_available = True

            if bin_info_available:
                # Use actual bin edges to convert back to continuous
                if "action_bins" in demo:
                    bin_edges = demo["action_bins"][:]
                else:
                    bin_edges = demo["bins"][:]

                # Convert predicted bins to continuous actions
                predicted_actions = np.zeros((T_pred, CHUNK_SIZE, ACTION_DIM))
                for t in range(T_pred):
                    for c in range(CHUNK_SIZE):
                        for d in range(ACTION_DIM):
                            bin_idx = predicted_bins[t, c, d]
                            # Use bin center
                            if bin_edges.ndim == 2:  # [action_dim, n_bins+1]
                                predicted_actions[t, c, d] = (
                                    bin_edges[d, bin_idx] + bin_edges[d, bin_idx + 1]
                                ) / 2
                            else:  # [n_bins+1] - same bins for all dims
                                predicted_actions[t, c, d] = (
                                    bin_edges[bin_idx] + bin_edges[bin_idx + 1]
                                ) / 2

                # Only compare the first action in each chunk (the one executed)
                predicted_first_actions = predicted_actions[
                    :, 0, :
                ]  # [T_pred, ACTION_DIM]

                # Compute MSE
                mse = np.mean(
                    (predicted_first_actions - gt_actions_subsampled[:T_pred]) ** 2
                )
                mae = np.mean(
                    np.abs(predicted_first_actions - gt_actions_subsampled[:T_pred])
                )

                # Per-dimension errors
                per_dim_mse = np.mean(
                    (predicted_first_actions - gt_actions_subsampled[:T_pred]) ** 2,
                    axis=0,
                )

            else:
                # No bin info - use approximate conversion
                # Assume bins span [-1, 1] uniformly (common for robot actions)
                # print("⚠️  No bin edge info found, assuming uniform bins in [-1, 1]")

                bin_centers = np.linspace(-1, 1, n_bins)

                predicted_actions = np.zeros((T_pred, CHUNK_SIZE, ACTION_DIM))
                for t in range(T_pred):
                    for c in range(CHUNK_SIZE):
                        for d in range(ACTION_DIM):
                            bin_idx = predicted_bins[t, c, d]
                            predicted_actions[t, c, d] = bin_centers[bin_idx]

                predicted_first_actions = predicted_actions[:, 0, :]

                mse = np.mean(
                    (predicted_first_actions - gt_actions_subsampled[:T_pred]) ** 2
                )
                mae = np.mean(
                    np.abs(predicted_first_actions - gt_actions_subsampled[:T_pred])
                )
                per_dim_mse = np.mean(
                    (predicted_first_actions - gt_actions_subsampled[:T_pred]) ** 2,
                    axis=0,
                )

            # Also compute log probability for reference
            trajectory_probs = np.zeros((T_pred, CHUNK_SIZE, ACTION_DIM))
            for t in range(T_pred):
                for c in range(CHUNK_SIZE):
                    for d in range(ACTION_DIM):
                        trajectory_probs[t, c, d] = probs[
                            t, c, d, predicted_bins[t, c, d]
                        ]
            log_prob = np.sum(np.log(trajectory_probs + 1e-10))

            demo_results[demo_key] = {
                "mse": mse,
                "mae": mae,
                "log_prob": log_prob,
                "per_dim_mse": per_dim_mse,
            }

        # ============ RANK BY MSE ============
        print("=" * 70)
        print("🎯 RANKING BY MEAN SQUARED ERROR (lower = better fit):")
        print("=" * 70)

        sorted_by_mse = sorted(demo_results.items(), key=lambda x: x[1]["mse"])

        print(f"\n{'Demo':<12} {'MSE':<12} {'MAE':<12} {'Log Prob':<12}")
        print("-" * 70)

        for i, (demo_key, results) in enumerate(sorted_by_mse):
            marker = "  ⭐ BEST FIT (likely training demo)" if i == 0 else ""
            print(
                f"{demo_key:<12} {results['mse']:<12.6f} {results['mae']:<12.6f} {results['log_prob']:<12.2f}{marker}"
            )

        # Show top 5 in detail
        print("\n" + "=" * 70)
        print("📈 TOP 5 DETAILED BREAKDOWN:")
        print("=" * 70)

        for i, (demo_key, results) in enumerate(sorted_by_mse[:5]):
            print(f"\n{i + 1}. {demo_key}:")
            print(f"   Overall MSE: {results['mse']:.6f}")
            print(f"   Overall MAE: {results['mae']:.6f}")
            print(f"   Log Prob:    {results['log_prob']:.2f}")
            print(f"   Per-dimension MSE: {results['per_dim_mse']}")

        # Check if there's a clear winner
        best_mse = sorted_by_mse[0][1]["mse"]
        second_mse = sorted_by_mse[1][1]["mse"]
        ratio = second_mse / best_mse if best_mse > 0 else float("inf")

        print("\n" + "=" * 70)
        print("🔍 CONFIDENCE ANALYSIS:")
        print("=" * 70)
        print(f"Best MSE:     {best_mse:.6f}")
        print(f"2nd Best MSE: {second_mse:.6f}")
        print(f"Ratio:        {ratio:.2f}x")

        if ratio > 2.0:
            print(f"✅ STRONG SIGNAL: {sorted_by_mse[0][0]} is clearly the best fit!")
            print(f"   It has {ratio:.1f}x lower error than the next best.")
        elif ratio > 1.5:
            print(
                f"⚠️  MODERATE SIGNAL: {sorted_by_mse[0][0]} is likely the training demo,"
            )
            print(f"   but the difference is not dramatic.")
        else:
            print(f"❌ WEAK SIGNAL: Top demos have very similar errors.")
            print(f"   Model may not have been trained on a single demo, or")
            print(f"   multiple demos are very similar.")

print("\n" + "=" * 80)
