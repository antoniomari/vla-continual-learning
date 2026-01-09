"""
Save RLDS demonstrations as videos to check image orientation.

Usage:
    python save_rlds_demo_video.py \
        --rlds_path /path/to/libero_spatial_no_noops/1.0.0 \
        --output_dir ./rlds_videos \
        --num_demos 5 \
        --fps 10
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def decode_image(image_bytes, rotate=False):
    """Decode JPEG/PNG image from bytes."""
    image = tf.image.decode_image(image_bytes, channels=3)
    image = tf.cast(image, tf.uint8)

    if rotate:
        # Rotate 180 degrees
        image = tf.image.rot90(image, k=2)

    return image.numpy()


def parse_episode(raw_record):
    """Parse a single episode from TFRecord."""
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())

    features = example.features.feature

    # Extract task name
    language_bytes = features["steps/language_instruction"].bytes_list.value
    task_name = language_bytes[0].decode("utf-8")

    # Extract number of steps
    num_steps = len(features["steps/is_first"].int64_list.value)

    # Store image bytes
    image_bytes = features["steps/observation/image"].bytes_list.value
    wrist_image_bytes = features["steps/observation/wrist_image"].bytes_list.value

    # Extract actions for visualization
    actions_flat = list(features["steps/action"].float_list.value)
    actions = np.array(actions_flat).reshape(num_steps, 7)

    return {
        "task_name": task_name,
        "num_steps": num_steps,
        "image_bytes": image_bytes,
        "wrist_image_bytes": wrist_image_bytes,
        "actions": actions,
    }


def save_video(images, output_path, fps=10):
    """Save images as video."""
    if len(images) == 0:
        print("No images to save!")
        return

    # Get dimensions from first image
    height, width = images[0].shape[:2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for img in images:
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out.write(img_bgr)

    out.release()
    print(f"  Saved video to {output_path}")


def create_side_by_side_video(agentview_images, wrist_images, output_path, fps=10):
    """Create side-by-side video with both camera views."""
    if len(agentview_images) == 0:
        print("No images to save!")
        return

    # Get dimensions
    h1, w1 = agentview_images[0].shape[:2]
    h2, w2 = wrist_images[0].shape[:2]

    # Make heights match by resizing wrist image
    if h1 != h2:
        scale = h1 / h2
        w2_new = int(w2 * scale)
        wrist_images = [cv2.resize(img, (w2_new, h1)) for img in wrist_images]
        w2 = w2_new

    # Create side-by-side frames
    width = w1 + w2
    height = h1

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for agent_img, wrist_img in zip(agentview_images, wrist_images):
        # Concatenate horizontally
        combined = np.hstack([agent_img, wrist_img])

        # Convert RGB to BGR for OpenCV
        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        out.write(combined_bgr)

    out.release()
    print(f"  Saved side-by-side video to {output_path}")


def add_text_to_image(image, text_lines, font_scale=0.5, thickness=1):
    """Add text overlay to image."""
    img = image.copy()

    # Add black background for text
    overlay = img.copy()
    cv2.rectangle(overlay, (5, 5), (400, 20 + 25 * len(text_lines)), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)

    # Add text
    y_offset = 25
    for line in text_lines:
        cv2.putText(
            img,
            line,
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
        )
        y_offset += 25

    return img


def main(args):
    rlds_path = Path(args.rlds_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SAVING RLDS DEMOS AS VIDEOS")
    print("=" * 80)
    print(f"RLDS path: {rlds_path}")
    print(f"Output dir: {output_dir}")
    print(f"FPS: {args.fps}")
    print(f"Max demos to save: {args.num_demos}")
    print()

    # Load TFRecord files
    file_pattern = f"{rlds_path}/*.tfrecord-*"
    files = tf.io.gfile.glob(file_pattern)

    if not files:
        print(f"❌ No TFRecord files found at {file_pattern}")
        return

    print(f"Found {len(files)} TFRecord files\n")

    # Create dataset
    raw_dataset = tf.data.TFRecordDataset(files)
    num_episodes = sum(1 for _ in raw_dataset)
    print("Total episodes:", num_episodes)

    # Process episodes
    demo_count = 0

    for raw_record in tqdm(raw_dataset, desc="Processing demos", total=args.num_demos):
        if demo_count >= args.num_demos:
            break

        try:
            episode = parse_episode(raw_record)
            task_name = episode["task_name"]
            task_name_clean = task_name.replace(" ", "_").replace("/", "_")

            print(f"\n📹 Demo {demo_count}: {task_name}")
            print(f"   Steps: {episode['num_steps']}")

            # Decode images WITHOUT rotation
            print("   Decoding images (original orientation)...")
            agentview_images_orig = []
            wrist_images_orig = []

            for i in range(episode["num_steps"]):
                agent_img = decode_image(episode["image_bytes"][i], rotate=False)
                wrist_img = decode_image(episode["wrist_image_bytes"][i], rotate=False)

                # Add text overlay with action info
                action = episode["actions"][i]
                text_lines = [
                    f"Step {i}/{episode['num_steps']}",
                    f"Action: [{action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}, ...]",
                    f"Gripper: {action[6]:.2f}",
                    "ORIGINAL (no rotation)",
                ]

                agent_img = add_text_to_image(agent_img, text_lines)
                wrist_img = add_text_to_image(wrist_img, ["Wrist Camera", "ORIGINAL"])

                agentview_images_orig.append(agent_img)
                wrist_images_orig.append(wrist_img)

            # Decode images WITH 180° rotation
            print("   Decoding images (rotated 180°)...")
            agentview_images_rot = []
            wrist_images_rot = []

            for i in range(episode["num_steps"]):
                agent_img = decode_image(episode["image_bytes"][i], rotate=True)
                wrist_img = decode_image(episode["wrist_image_bytes"][i], rotate=True)

                # Add text overlay
                action = episode["actions"][i]
                text_lines = [
                    f"Step {i}/{episode['num_steps']}",
                    f"Action: [{action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}, ...]",
                    f"Gripper: {action[6]:.2f}",
                    "ROTATED 180 degrees",
                ]

                agent_img = add_text_to_image(agent_img, text_lines)
                wrist_img = add_text_to_image(
                    wrist_img, ["Wrist Camera", "ROTATED 180"]
                )

                agentview_images_rot.append(agent_img)
                wrist_images_rot.append(wrist_img)

            # Save original videos
            orig_output = (
                output_dir / f"demo_{demo_count:03d}_{task_name_clean}_ORIGINAL.mp4"
            )
            create_side_by_side_video(
                agentview_images_orig, wrist_images_orig, orig_output, fps=args.fps
            )

            # Save rotated videos
            rot_output = (
                output_dir / f"demo_{demo_count:03d}_{task_name_clean}_ROTATED.mp4"
            )
            create_side_by_side_video(
                agentview_images_rot, wrist_images_rot, rot_output, fps=args.fps
            )

            demo_count += 1

        except Exception as e:
            print(f"   ❌ Error processing demo: {e}")
            continue

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"Saved {demo_count} demo videos to {output_dir}")
    print("\nCompare the videos:")
    print("  *_ORIGINAL.mp4 - Images as stored in RLDS")
    print("  *_ROTATED.mp4  - Images rotated 180° (what SimpleVLA uses)")
    print("\nLook for:")
    print("  ✓ Robot/objects should be right-side up")
    print("  ✓ Text/labels should be readable")
    print("  ✓ Movement should make sense with actions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Save RLDS demos as videos to check orientation"
    )
    parser.add_argument(
        "--rlds_path",
        type=str,
        required=True,
        help="Path to RLDS dataset directory (contains .tfrecord files)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./rlds_demo_videos",
        help="Output directory for videos",
    )
    parser.add_argument(
        "--num_demos", type=int, default=5, help="Number of demos to save as videos"
    )
    parser.add_argument(
        "--fps", type=int, default=10, help="Frames per second for output videos"
    )

    args = parser.parse_args()
    main(args)
