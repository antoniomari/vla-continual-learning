"""
Explore RLDS dataset structure to understand the data format.

Usage:
    python explore_rlds_dataset.py \
        --rlds_dir ./LIBERO/libero/datasets/modified_libero_rlds/libero_spatial_no_noops/1.0.0
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def explore_rlds_dataset(rlds_dir):
    """Explore the structure of an RLDS dataset."""

    rlds_path = Path(rlds_dir)

    print("=" * 80)
    print("RLDS DATASET EXPLORATION")
    print("=" * 80)
    print(f"Dataset directory: {rlds_dir}\n")

    # Load metadata files
    print("📄 METADATA FILES:")
    print("-" * 80)

    # Load dataset_info.json
    dataset_info_path = rlds_path / "dataset_info.json"
    if dataset_info_path.exists():
        with open(dataset_info_path, "r") as f:
            dataset_info = json.load(f)
        print("\n📋 dataset_info.json:")
        print(json.dumps(dataset_info, indent=2))

    # Load features.json
    features_path = rlds_path / "features.json"
    if features_path.exists():
        with open(features_path, "r") as f:
            features = json.load(f)
        print("\n📋 features.json:")
        print(json.dumps(features, indent=2))

    # List TFRecord files
    tfrecord_files = sorted(list(rlds_path.glob("*.tfrecord-*")))
    print(f"\n📦 Found {len(tfrecord_files)} TFRecord files")
    print(f"   First file: {tfrecord_files[0].name}")
    print(f"   Last file: {tfrecord_files[-1].name}")

    # Load the dataset
    print("\n" + "=" * 80)
    print("LOADING DATASET")
    print("=" * 80)

    # Create a pattern for all tfrecord files
    file_pattern = str(rlds_path / "*.tfrecord-*")

    # Load raw dataset
    raw_dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(file_pattern))

    # Try to parse the dataset
    print("\n🔍 Examining first episode...")

    # Get first example
    for i, raw_record in enumerate(raw_dataset.take(1)):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())

        print("\n📊 EPISODE STRUCTURE:")
        print("-" * 80)

        features = example.features.feature

        print(f"\nFound {len(features)} top-level keys:")
        for key in sorted(features.keys()):
            print(f"  • {key}")

        print("\n📝 DETAILED FEATURE INFORMATION:")
        print("-" * 80)

        for key in sorted(features.keys()):
            feature = features[key]

            # Determine feature type and shape
            if feature.HasField("bytes_list"):
                values = feature.bytes_list.value
                print(f"\n{key}:")
                print(f"  Type: bytes")
                print(f"  Count: {len(values)}")
                if len(values) > 0:
                    # Try to decode first value
                    try:
                        # Might be a serialized tensor
                        first_val = values[0]
                        print(f"  First value length: {len(first_val)} bytes")
                        # Try to decode as string
                        try:
                            decoded = first_val.decode("utf-8")
                            print(f"  First value (string): {decoded[:100]}...")
                        except:
                            print(f"  First value: <binary data>")
                    except Exception as e:
                        print(f"  Could not decode: {e}")

            elif feature.HasField("float_list"):
                values = feature.float_list.value
                print(f"\n{key}:")
                print(f"  Type: float")
                print(f"  Count: {len(values)}")
                if len(values) > 0:
                    print(f"  First few values: {list(values[:5])}")
                    print(f"  Shape hint: array of {len(values)} floats")

            elif feature.HasField("int64_list"):
                values = feature.int64_list.value
                print(f"\n{key}:")
                print(f"  Type: int64")
                print(f"  Count: {len(values)}")
                if len(values) > 0:
                    print(f"  First few values: {list(values[:5])}")
                    print(f"  Shape hint: array of {len(values)} ints")

    # Try to load with tfds if possible
    print("\n" + "=" * 80)
    print("ATTEMPTING TFDS LOAD")
    print("=" * 80)

    try:
        # Try to load as a TFDS dataset
        # This might work if the dataset follows TFDS structure
        builder = tfds.builder_from_directory(rlds_path.parent.parent)
        dataset = builder.as_dataset(split="train")

        print("\n✅ Successfully loaded with TFDS!")

        # Examine structure
        print("\n📊 TFDS DATASET STRUCTURE:")
        print("-" * 80)

        for i, episode in enumerate(dataset.take(1)):
            print("\nEpisode keys:")
            for key in episode.keys():
                value = episode[key]
                print(f"  • {key}: {value}")

                # If it's a nested dataset (steps), explore it
                if hasattr(value, "__iter__") and key == "steps":
                    print(f"\n    Steps structure:")
                    for j, step in enumerate(value.take(1)):
                        print(f"      Step {j} keys:")
                        for step_key in step.keys():
                            step_value = step[step_key]
                            if hasattr(step_value, "shape"):
                                print(
                                    f"        • {step_key}: shape={step_value.shape}, dtype={step_value.dtype}"
                                )
                            else:
                                print(f"        • {step_key}: {type(step_value)}")

        # Count episodes and steps
        print("\n📈 DATASET STATISTICS:")
        print("-" * 80)

        num_episodes = 0
        total_steps = 0
        episode_lengths = []

        for episode in dataset.take(10):  # Sample first 10 episodes
            num_episodes += 1
            if "steps" in episode:
                steps = episode["steps"]
                episode_len = sum(1 for _ in steps)
                episode_lengths.append(episode_len)
                total_steps += episode_len

        if num_episodes > 0:
            print(f"Sample size: {num_episodes} episodes")
            print(f"Total steps: {total_steps}")
            print(f"Average episode length: {total_steps / num_episodes:.1f}")
            print(f"Episode lengths: {episode_lengths}")

    except Exception as e:
        print(f"\n⚠️  Could not load with TFDS: {e}")
        print("   This might be a custom RLDS format")

    print("\n" + "=" * 80)
    print("EXPLORATION COMPLETE")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Explore RLDS dataset structure")
    parser.add_argument(
        "--rlds_dir",
        type=str,
        default="./LIBERO/libero/datasets/modified_libero_rlds/libero_spatial_no_noops/1.0.0",
        help="Path to RLDS dataset version directory (contains .tfrecord files)",
    )

    args = parser.parse_args()
    explore_rlds_dataset(args.rlds_dir)


if __name__ == "__main__":
    main()
