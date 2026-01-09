"""
Preprocess LIBERO dataset by applying SimpleVLA transformations:
- Remove no-op actions
- Rotate images by 180 degrees

This creates a new dataset directory with transformed HDF5 files that match
the RLDS format but remain in HDF5 format for easy loading.

Usage:
    python preprocess_libero_dataset.py \
        --task_suite libero_spatial \
        --input_dir ./LIBERO/libero/datasets/libero_spatial \
        --output_dir ./LIBERO/libero/datasets/libero_spatial_transformed
"""

import argparse
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def is_noop(action, prev_action=None, threshold=1e-4):
    """
    Returns whether an action is a no-op action.

    A no-op action satisfies two criteria:
        (1) All action dimensions, except for the last one (gripper action), are near zero.
        (2) The gripper action is equal to the previous timestep's gripper action.
    """
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold

    gripper_action = action[-1]
    prev_gripper_action = prev_action[-1]
    return (
        np.linalg.norm(action[:-1]) < threshold
        and gripper_action == prev_gripper_action
    )


def get_valid_indices(actions):
    """Get indices of non-no-op actions."""
    valid_indices = []
    prev_action = None

    for i, action in enumerate(actions):
        if not is_noop(action, prev_action):
            valid_indices.append(i)
            prev_action = action

    return valid_indices


def process_demo(demo_group, valid_indices):
    """
    Process a single demo by filtering out no-op actions and rotating images.

    Returns a dictionary of processed data arrays.
    """
    processed_data = {}

    # Process observations
    obs_group = demo_group["obs"]

    # Get and rotate images (180 degrees)
    if "agentview_rgb" in obs_group:
        agentview_images = np.array(obs_group["agentview_rgb"])
        agentview_images = agentview_images[valid_indices]
        # Rotate each image 180 degrees
        agentview_images = np.array(
            [np.rot90(img, k=2).copy() for img in agentview_images]
        )
        processed_data["agentview_rgb"] = agentview_images

    if "eye_in_hand_rgb" in obs_group:
        eye_in_hand_images = np.array(obs_group["eye_in_hand_rgb"])
        eye_in_hand_images = eye_in_hand_images[valid_indices]
        # Rotate each image 180 degrees
        eye_in_hand_images = np.array(
            [np.rot90(img, k=2).copy() for img in eye_in_hand_images]
        )
        processed_data["eye_in_hand_rgb"] = eye_in_hand_images

    # Process other observation data
    for key in obs_group.keys():
        if key not in ["agentview_rgb", "eye_in_hand_rgb"]:
            data = np.array(obs_group[key])
            processed_data[key] = data[valid_indices]

    # Process actions
    actions = np.array(demo_group["actions"])
    processed_data["actions"] = actions[valid_indices]

    # Process other top-level data (states, rewards, dones, etc.)
    for key in demo_group.keys():
        if key not in ["obs", "actions"]:
            data = np.array(demo_group[key])
            if len(data) == len(actions):  # Same length as trajectory
                processed_data[key] = data[valid_indices]
            else:  # Different length (e.g., metadata), keep as is
                processed_data[key] = data

    return processed_data


def process_hdf5_file(input_path, output_path):
    """
    Process a single HDF5 file: remove no-ops and rotate images.
    """
    stats = {
        "total_demos": 0,
        "total_timesteps_before": 0,
        "total_timesteps_after": 0,
        "noops_removed": 0,
    }

    with h5py.File(input_path, "r") as input_file:
        with h5py.File(output_path, "w") as output_file:
            # Copy attributes if any
            for attr_name, attr_value in input_file.attrs.items():
                output_file.attrs[attr_name] = attr_value

            # Create data group
            output_data_group = output_file.create_group("data")
            input_data_group = input_file["data"]

            # Process each demo
            for demo_name in sorted(input_data_group.keys()):
                stats["total_demos"] += 1
                demo_group = input_data_group[demo_name]

                # Get actions and find valid indices
                actions = np.array(demo_group["actions"])
                stats["total_timesteps_before"] += len(actions)

                valid_indices = get_valid_indices(actions)
                stats["total_timesteps_after"] += len(valid_indices)
                stats["noops_removed"] += len(actions) - len(valid_indices)

                # Process demo data
                processed_data = process_demo(demo_group, valid_indices)

                # Write processed data to output file
                output_demo_group = output_data_group.create_group(demo_name)

                # Create obs subgroup
                output_obs_group = output_demo_group.create_group("obs")

                # Write observation data
                for key in [
                    "agentview_rgb",
                    "eye_in_hand_rgb",
                    "gripper_states",
                    "joint_states",
                    "ee_states",
                    "ee_pos",
                    "ee_ori",
                ]:
                    if key in processed_data:
                        output_obs_group.create_dataset(key, data=processed_data[key])

                # Write other observation data
                input_obs_group = demo_group["obs"]
                for key in input_obs_group.keys():
                    if key not in output_obs_group and key in processed_data:
                        output_obs_group.create_dataset(key, data=processed_data[key])

                # Write actions
                output_demo_group.create_dataset(
                    "actions", data=processed_data["actions"]
                )

                # Write other top-level data
                for key in demo_group.keys():
                    if key not in ["obs", "actions"] and key in processed_data:
                        output_demo_group.create_dataset(key, data=processed_data[key])

    return stats


def main(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Create output directory
    if output_dir.exists():
        user_input = input(
            f"Output directory already exists at: {output_dir}\n"
            f"Enter 'y' to overwrite, or anything else to exit: "
        )
        if user_input.lower() != "y":
            print("Exiting...")
            return
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all HDF5 files
    hdf5_files = sorted(list(input_dir.glob("*.hdf5")))

    if not hdf5_files:
        print(f"No HDF5 files found in {input_dir}")
        return

    print(f"Found {len(hdf5_files)} HDF5 files to process")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Process each file
    total_stats = {
        "total_demos": 0,
        "total_timesteps_before": 0,
        "total_timesteps_after": 0,
        "noops_removed": 0,
    }

    for input_path in tqdm(hdf5_files, desc="Processing files"):
        output_path = output_dir / input_path.name

        print(f"\nProcessing: {input_path.name}")
        stats = process_hdf5_file(input_path, output_path)

        # Update total stats
        for key in total_stats:
            total_stats[key] += stats[key]

        # Print stats for this file
        print(f"  Demos: {stats['total_demos']}")
        print(
            f"  Timesteps: {stats['total_timesteps_before']} -> {stats['total_timesteps_after']}"
        )
        print(f"  No-ops removed: {stats['noops_removed']}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total demos processed: {total_stats['total_demos']}")
    print(f"Total timesteps before: {total_stats['total_timesteps_before']}")
    print(f"Total timesteps after: {total_stats['total_timesteps_after']}")
    print(f"Total no-ops removed: {total_stats['noops_removed']}")
    print(
        f"Reduction: {total_stats['noops_removed'] / total_stats['total_timesteps_before'] * 100:.2f}%"
    )
    print(f"\nProcessed dataset saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess LIBERO dataset with SimpleVLA transformations"
    )
    parser.add_argument(
        "--task_suite",
        type=str,
        required=True,
        choices=[
            "libero_spatial",
            "libero_object",
            "libero_goal",
            "libero_10",
            "libero_90",
        ],
        help="LIBERO task suite name",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to input HDF5 dataset directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to output directory for processed dataset",
    )

    args = parser.parse_args()
    main(args)
