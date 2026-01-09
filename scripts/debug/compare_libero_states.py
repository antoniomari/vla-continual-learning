#!/usr/bin/env python3
"""
Compare two LIBERO trajectory logs to find where they diverge.

Usage:
    python compare_trajectories.py trajectory_logs/aaronson trajectory_logs/pulisic
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


def load_trajectory(traj_dir):
    """Load trajectory metadata and images"""
    traj_path = Path(traj_dir)

    # Load metadata
    with open(traj_path / "metadata.json", "r") as f:
        metadata = json.load(f)

    return metadata, traj_path


def compare_actions(action1, action2, step, tolerance=1e-6):
    """Compare two actions"""
    if action1 is None and action2 is None:
        return True, None

    if action1 is None or action2 is None:
        return False, f"One action is None: {action1} vs {action2}"

    a1 = np.array(action1)
    a2 = np.array(action2)

    if a1.shape != a2.shape:
        return False, f"Shape mismatch: {a1.shape} vs {a2.shape}"

    diff = np.abs(a1 - a2)
    max_diff = np.max(diff)

    if max_diff > tolerance:
        return (
            False,
            f"Max difference: {max_diff:.10f}, \n\taction1: {action1}, \n\taction2: {action2}, \n\tdiff: {diff}",
        )

    return True, None


def compare_states(state1, state2, step, tolerance=1e-6):
    """Compare two state dictionaries"""
    differences = []

    # Compare time
    if abs(state1.get("time", 0) - state2.get("time", 0)) > tolerance:
        differences.append(f"  Time: {state1.get('time')} vs {state2.get('time')}")

    # Compare qpos
    qpos1 = np.array(state1.get("qpos", []))
    qpos2 = np.array(state2.get("qpos", []))
    if qpos1.shape != qpos2.shape:
        differences.append(f"  qpos shape: {qpos1.shape} vs {qpos2.shape}")
    else:
        qpos_diff = np.abs(qpos1 - qpos2)
        max_qpos_diff = np.max(qpos_diff)
        if max_qpos_diff > tolerance:
            differences.append(f"  qpos max diff: {max_qpos_diff:.10f}")
            # Show first few differences
            diff_indices = np.where(qpos_diff > tolerance)[0][:5]
            for idx in diff_indices:
                differences.append(
                    f"    qpos[{idx}]: {qpos1[idx]:.10f} vs {qpos2[idx]:.10f} (diff: {qpos_diff[idx]:.10f})"
                )

    # Compare qvel
    qvel1 = np.array(state1.get("qvel", []))
    qvel2 = np.array(state2.get("qvel", []))
    if qvel1.shape != qvel2.shape:
        differences.append(f"  qvel shape: {qvel1.shape} vs {qvel2.shape}")
    else:
        qvel_diff = np.abs(qvel1 - qvel2)
        max_qvel_diff = np.max(qvel_diff)
        if max_qvel_diff > tolerance:
            differences.append(f"  qvel max diff: {max_qvel_diff:.10f}")
            diff_indices = np.where(qvel_diff > tolerance)[0][:5]
            for idx in diff_indices:
                differences.append(
                    f"    qvel[{idx}]: {qvel1[idx]:.10f} vs {qvel2[idx]:.10f} (diff: {qvel_diff[idx]:.10f})"
                )

    # Compare EEF position
    if "eef_pos" in state1 and "eef_pos" in state2:
        eef1 = np.array(state1["eef_pos"])
        eef2 = np.array(state2["eef_pos"])
        eef_diff = np.abs(eef1 - eef2)
        max_eef_diff = np.max(eef_diff)
        if max_eef_diff > tolerance:
            differences.append(f"  eef_pos max diff: {max_eef_diff:.10f}")
            differences.append(f"    eef_pos: {eef1} vs {eef2}")

    # Compare objects
    objects1 = state1.get("objects", {})
    objects2 = state2.get("objects", {})

    if set(objects1.keys()) != set(objects2.keys()):
        differences.append(
            f"  Object names differ: {set(objects1.keys())} vs {set(objects2.keys())}"
        )

    for obj_name in objects1.keys():
        if obj_name not in objects2:
            continue

        obj1 = objects1[obj_name]
        obj2 = objects2[obj_name]

        # Compare position
        pos1 = np.array(obj1.get("pos", []))
        pos2 = np.array(obj2.get("pos", []))
        pos_diff = np.abs(pos1 - pos2)
        max_pos_diff = np.max(pos_diff)
        if max_pos_diff > tolerance:
            differences.append(
                f"  Object '{obj_name}' pos max diff: {max_pos_diff:.10f}"
            )
            differences.append(f"    pos: {pos1} vs {pos2}")

        # Compare quaternion
        quat1 = np.array(obj1.get("quat", []))
        quat2 = np.array(obj2.get("quat", []))
        quat_diff = np.abs(quat1 - quat2)
        max_quat_diff = np.max(quat_diff)
        if max_quat_diff > tolerance:
            differences.append(
                f"  Object '{obj_name}' quat max diff: {max_quat_diff:.10f}"
            )
            differences.append(f"    quat: {quat1} vs {quat2}")

    return len(differences) == 0, differences


def compare_images(img_path1, img_path2):
    """Compare two images"""
    if not img_path1.exists() and not img_path2.exists():
        return True, "Both images missing"

    if not img_path1.exists():
        return False, f"Image 1 missing: {img_path1}"

    if not img_path2.exists():
        return False, f"Image 2 missing: {img_path2}"

    img1 = np.array(Image.open(img_path1))
    img2 = np.array(Image.open(img_path2))

    if img1.shape != img2.shape:
        return False, f"Shape mismatch: {img1.shape} vs {img2.shape}"

    # Calculate pixel difference
    diff = np.abs(img1.astype(float) - img2.astype(float))
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    # Consider images different if any pixel differs
    if max_diff > 0:
        num_diff_pixels = np.sum(diff > 0)
        total_pixels = diff.size
        return (
            False,
            f"Max pixel diff: {max_diff:.2f}, Mean: {mean_diff:.4f}, Diff pixels: {num_diff_pixels}/{total_pixels} ({100 * num_diff_pixels / total_pixels:.2f}%)",
        )

    return True, None


def compare_trajectories(
    traj_dir1, traj_dir2, tolerance=1e-6, verbose=False, compare_images=True
):
    """Compare two trajectories step by step"""

    metadata1, path1 = load_trajectory(traj_dir1)
    metadata2, path2 = load_trajectory(traj_dir2)

    steps1 = metadata1["steps"]
    steps2 = metadata2["steps"]

    if len(steps1) != len(steps2):
        return {
            "match": False,
            "first_divergence": 0,
            "reason": f"Different number of steps: {len(steps1)} vs {len(steps2)}",
            "steps_compared": 0,
        }

    min_steps = min(len(steps1), len(steps2))
    first_divergence = None

    for i in range(min_steps):
        step1 = steps1[i]
        step2 = steps2[i]

        step_num = step1["step"]

        # Compare actions
        action_match, action_msg = compare_actions(
            step1.get("action"), step2.get("action"), step_num, tolerance
        )

        # Compare states
        state_match, state_diffs = compare_states(
            step1.get("state", {}), step2.get("state", {}), step_num, tolerance
        )

        # Compare images (optional)
        agentview_match = True
        agentview_msg = "Skipped"
        wrist_match = True
        wrist_msg = "Skipped"

        if compare_images:
            agentview1 = path1 / f"agentview_{step_num:04d}.png"
            agentview2 = path2 / f"agentview_{step_num:04d}.png"
            agentview_match, agentview_msg = compare_images(agentview1, agentview2)

            wrist1 = path1 / f"wrist_{step_num:04d}.png"
            wrist2 = path2 / f"wrist_{step_num:04d}.png"
            wrist_match, wrist_msg = compare_images(wrist1, wrist2)

        # Report any differences
        if not (action_match and state_match and agentview_match and wrist_match):
            if first_divergence is None:
                first_divergence = step_num

                if verbose:
                    print(f"🔴 FIRST DIVERGENCE at step {step_num}")
                    print("=" * 80)

                    if not action_match:
                        print(f"\n❌ Action mismatch:")
                        print(f"  {action_msg}")

                    if not state_match:
                        print(f"\n❌ State mismatch:")
                        for diff in state_diffs:
                            print(diff)

                    if compare_images:
                        if not agentview_match:
                            print(f"\n❌ Agentview image mismatch:")
                            print(f"  {agentview_msg}")

                        if not wrist_match:
                            print(f"\n❌ Wrist image mismatch:")
                            print(f"  {wrist_msg}")

                    print()

                # Collect details for return
                details = {
                    "action_match": action_match,
                    "action_msg": action_msg,
                    "state_match": state_match,
                    "state_diffs": state_diffs,
                    "agentview_match": agentview_match,
                    "agentview_msg": agentview_msg,
                    "wrist_match": wrist_match,
                    "wrist_msg": wrist_msg,
                }

                return {
                    "match": False,
                    "first_divergence": first_divergence,
                    "details": details,
                    "steps_compared": min_steps,
                }

    return {"match": True, "first_divergence": None, "steps_compared": min_steps}


def find_all_trajectories(base_dir):
    """Find all trajectory directories in the base directory"""
    base_path = Path(base_dir)
    trajectories = defaultdict(dict)

    # Find all traj_* directories
    for traj_dir in base_path.rglob("traj_*"):
        if not traj_dir.is_dir():
            continue

        # Parse the path: rank_X/env_Y/traj_Z
        parts = traj_dir.relative_to(base_path).parts

        # Extract rank, env, and traj identifiers
        rank = None
        env = None
        traj = traj_dir.name

        for part in parts:
            if part.startswith("rank_"):
                rank = part
            elif part.startswith("env_"):
                env = part

        if rank and env:
            key = f"{rank}/{env}/{traj}"
            trajectories[key] = traj_dir

    return trajectories


def compare_all_trajectories(
    base_dir1, base_dir2, tolerance=1e-6, verbose=False, compare_images=True
):
    """Compare all trajectories between two base directories"""
    print(f"Scanning directories...")
    print(f"  Directory 1: {base_dir1}")
    print(f"  Directory 2: {base_dir2}")
    print(f"  Compare images: {compare_images}")
    print()

    trajs1 = find_all_trajectories(base_dir1)
    trajs2 = find_all_trajectories(base_dir2)

    print(f"Found {len(trajs1)} trajectories in directory 1")
    print(f"Found {len(trajs2)} trajectories in directory 2")
    print()

    # Find common trajectories
    common_keys = set(trajs1.keys()) & set(trajs2.keys())
    only_in_1 = set(trajs1.keys()) - set(trajs2.keys())
    only_in_2 = set(trajs2.keys()) - set(trajs1.keys())

    if only_in_1:
        print(f"⚠️  {len(only_in_1)} trajectories only in directory 1:")
        for key in sorted(only_in_1)[:5]:
            print(f"  - {key}")
        if len(only_in_1) > 5:
            print(f"  ... and {len(only_in_1) - 5} more")
        print()

    if only_in_2:
        print(f"⚠️  {len(only_in_2)} trajectories only in directory 2:")
        for key in sorted(only_in_2)[:5]:
            print(f"  - {key}")
        if len(only_in_2) > 5:
            print(f"  ... and {len(only_in_2) - 5} more")
        print()

    print(f"Comparing {len(common_keys)} common trajectories...")
    print("=" * 80)
    print()

    results = {"matching": [], "diverging": [], "errors": []}

    for i, key in enumerate(sorted(common_keys)):
        traj1 = trajs1[key]
        traj2 = trajs2[key]

        print(f"[{i + 1}/{len(common_keys)}] {key}...", end=" ")

        try:
            result = compare_trajectories(
                traj1, traj2, tolerance, verbose=False, compare_images=compare_images
            )

            if result["match"]:
                print(f"✅ Match ({result['steps_compared']} steps)")
                results["matching"].append(key)
            else:
                first_div = result.get("first_divergence", "unknown")
                reason = result.get("reason", "")
                print(f"🔴 Diverges at step {first_div} {reason}")
                results["diverging"].append({"key": key, "result": result})
        except Exception as e:
            print(f"❌ Error: {e}")
            results["errors"].append({"key": key, "error": str(e)})

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total compared: {len(common_keys)}")
    print(f"✅ Matching: {len(results['matching'])}")
    print(f"🔴 Diverging: {len(results['diverging'])}")
    print(f"❌ Errors: {len(results['errors'])}")

    if results["diverging"]:
        print()
        print("=" * 80)
        print("DETAILED DIVERGENCE ANALYSIS")
        print("=" * 80)

        for item in results["diverging"]:
            key = item["key"]
            result = item["result"]
            first_div = result.get("first_divergence", "unknown")

            print(f"\n🔴 {key}")
            print(f"   First divergence at step: {first_div}")

            if "details" in result:
                details = result["details"]

                # Check what's different
                diffs = []
                if not details["action_match"]:
                    diffs.append("Actions")
                if not details["state_match"]:
                    diffs.append("States")
                if compare_images:
                    if not details["agentview_match"]:
                        diffs.append("Agentview images")
                    if not details["wrist_match"]:
                        diffs.append("Wrist images")

                print(f"   What's different: {', '.join(diffs)}")

                # Show action details
                if not details["action_match"]:
                    print(f"\n   ❌ Actions:")
                    print(f"      {details['action_msg']}")

                # Show state details
                if not details["state_match"]:
                    print(f"\n   ❌ States:")
                    state_diffs = details["state_diffs"]
                    for diff in state_diffs[:10]:  # Show first 10 differences
                        print(f"      {diff}")
                    if len(state_diffs) > 10:
                        print(
                            f"      ... and {len(state_diffs) - 10} more state differences"
                        )

                # Show image details (only if comparing)
                if compare_images:
                    if not details["agentview_match"]:
                        print(f"\n   ❌ Agentview images:")
                        print(f"      {details['agentview_msg']}")

                    if not details["wrist_match"]:
                        print(f"\n   ❌ Wrist images:")
                        print(f"      {details['wrist_msg']}")
            else:
                print(f"   Reason: {result.get('reason', 'Unknown')}")

    if results["errors"]:
        print()
        print("=" * 80)
        print("ERRORS")
        print("=" * 80)
        for item in results["errors"]:
            print(f"  {item['key']}: {item['error']}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare LIBERO trajectory logs from two directories"
    )
    parser.add_argument(
        "dir1",
        type=str,
        help="Path to first trajectory base directory (e.g., trajectory_logs/aaronson)",
    )
    parser.add_argument(
        "dir2",
        type=str,
        help="Path to second trajectory base directory (e.g., trajectory_logs/pulisic)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Numerical tolerance for float comparisons (default: 1e-6)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed comparison for each trajectory",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Skip image comparison (only compare actions and states)",
    )

    args = parser.parse_args()

    compare_all_trajectories(
        args.dir1,
        args.dir2,
        args.tolerance,
        args.verbose,
        compare_images=not args.no_images,
    )


if __name__ == "__main__":
    main()
