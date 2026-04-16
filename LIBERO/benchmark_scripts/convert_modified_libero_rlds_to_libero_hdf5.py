#!/usr/bin/env python3
"""
Convert OpenVLA modified LIBERO RLDS TFRecords to LIBERO-style HDF5 demos.

Input (modified RLDS):
  <input_dir>/*.tfrecord-*

Output (LIBERO-style):
  <output_dir>/<task_name>_demo.hdf5
    data/
      demo_0/
        actions                [T, 7] float32
        obs/
          agentview_rgb        [T, H, W, 3] uint8
      demo_1/
      ...

This schema matches what the current OPD/SFT loader expects in
`rlinf/custom/libero_trajectory_dataset.py`.
"""

import argparse
import os
import subprocess
import hashlib
from collections import Counter
from collections import defaultdict

import h5py
import numpy as np
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Path containing RLDS TFRecord shards (*.tfrecord-*)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Target LIBERO dataset dir (e.g. LIBERO/libero/datasets/libero_spatial)",
    )
    parser.add_argument(
        "--demos-per-task",
        type=int,
        default=0,
        help="If >0, keep only first N demos per task (0 means all).",
    )
    parser.add_argument(
        "--revert-libero-actions",
        action="store_true",
        help=(
            "Apply the same action conversion used in rlds_logits_precompute_worker: "
            "gripper = -gripper, then map last dim from [-1,1] to [0,1]."
        ),
    )
    parser.add_argument(
        "--expected-height",
        type=int,
        default=256,
        help="Expected observation height for agentview_rgb.",
    )
    parser.add_argument(
        "--expected-width",
        type=int,
        default=256,
        help="Expected observation width for agentview_rgb.",
    )
    parser.add_argument(
        "--validate-written-files",
        action="store_true",
        help=(
            "Re-open each written HDF5 and assert per-task/per-demo parity: "
            "episode count, action chunks, num_samples, and global totals."
        ),
    )
    parser.add_argument(
        "--download-first",
        action="store_true",
        help="Download the modified RLDS dataset from Hugging Face before conversion.",
    )
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        default="openvla/modified_libero_rlds",
        help="Hugging Face dataset repo id.",
    )
    parser.add_argument(
        "--task-suite-name",
        type=str,
        default="libero_spatial",
        help="Task suite name used to resolve RLDS path (<suite>_no_noops/1.0.0).",
    )
    parser.add_argument(
        "--compare-with-dir",
        type=str,
        default=None,
        help=(
            "Optional existing LIBERO HDF5 directory to compare against. "
            "Checks task files, demos, num_samples, totals, and action chunk parity."
        ),
    )
    parser.add_argument(
        "--compare-mode",
        type=str,
        default="strict",
        choices=["strict", "allow-missing-episodes", "length-matched-actions"],
        help=(
            "Comparison mode for --compare-with-dir. "
            "'strict' requires exact parity. "
            "'allow-missing-episodes' allows fewer demos in converted dataset "
            "but enforces exact action parity on overlapping demo names. "
            "'length-matched-actions' greedily matches episodes by trajectory length "
            "and checks first-action agreement on those pairs."
        ),
    )
    parser.add_argument(
        "--length-match-max-abs-diff",
        type=float,
        default=1e-7,
        help=(
            "In compare-mode=length-matched-actions, threshold for max absolute "
            "difference on first action dims [0:6] to count as a match."
        ),
    )
    parser.add_argument(
        "--length-match-min-ratio",
        type=float,
        default=0.0,
        help=(
            "In compare-mode=length-matched-actions, fail if global matched ratio "
            "(first 6 dims within threshold) is below this value in [0,1]."
        ),
    )
    parser.add_argument(
        "--rotate-images-180",
        action="store_true",
        help="Rotate agentview/wrist images by 180 degrees before writing HDF5 (default: enabled).",
    )
    parser.add_argument(
        "--no-rotate-images-180",
        action="store_false",
        dest="rotate_images_180",
        help="Disable 180-degree image rotation.",
    )
    parser.add_argument(
        "--flip-last-action-dim",
        action="store_true",
        help="Multiply last action dimension by -1 before writing HDF5 (default: enabled).",
    )
    parser.add_argument(
        "--no-flip-last-action-dim",
        action="store_false",
        dest="flip_last_action_dim",
        help="Disable last action dimension sign flip.",
    )
    parser.set_defaults(rotate_images_180=True, flip_last_action_dim=True)
    return parser.parse_args()


def _parse_episode(raw_record):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    features = example.features.feature

    language_bytes = features["steps/language_instruction"].bytes_list.value
    task_name = language_bytes[0].decode("utf-8")

    num_steps = len(features["steps/is_first"].int64_list.value)
    actions_flat = list(features["steps/action"].float_list.value)
    actions = np.array(actions_flat, dtype=np.float32).reshape(num_steps, 7)

    image_bytes = list(features["steps/observation/image"].bytes_list.value)
    if len(image_bytes) != num_steps:
        raise ValueError(
            f"Malformed episode for task={task_name}: "
            f"len(image_bytes)={len(image_bytes)} != num_steps={num_steps}"
        )

    wrist_image_bytes = None
    if "steps/observation/wrist_image" in features:
        wrist_image_bytes = list(features["steps/observation/wrist_image"].bytes_list.value)
        if len(wrist_image_bytes) != num_steps:
            raise ValueError(
                f"Malformed episode for task={task_name}: "
                f"len(wrist_image_bytes)={len(wrist_image_bytes)} != num_steps={num_steps}"
            )

    return {
        "task_name": task_name,
        "actions": actions,
        "image_bytes": image_bytes,
        "wrist_image_bytes": wrist_image_bytes,
    }


def _decode_image(image_bytes):
    # decode_image returns uint8 [H, W, 3] for encoded RGB images
    image = tf.image.decode_image(image_bytes, channels=3)
    return image.numpy().astype(np.uint8)


def _revert_actions_from_libero(dataset_actions):
    # Same logic used in rlinf/custom/rlds_logits_precompute_worker.py
    out = dataset_actions.copy()
    out[..., -1] = -out[..., -1]
    out[..., -1] = (out[..., -1] + 1.0) / 2.0
    return out


def _safe_task_filename(task_name):
    return task_name.replace(" ", "_").replace("/", "_")


def _download_modified_rlds(repo_id, local_datasets_root):
    os.makedirs(local_datasets_root, exist_ok=True)
    cmd = [
        "hf",
        "download",
        repo_id,
        "--repo-type",
        "dataset",
        "--local-dir",
        local_datasets_root,
    ]
    print(f"[download] Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Could not find `hf` CLI in PATH. Install huggingface_hub CLI first."
        ) from exc


def _resolve_rlds_input_dir(datasets_root, task_suite_name):
    candidates = [
        os.path.join(datasets_root, f"{task_suite_name}_no_noops", "1.0.0"),
        os.path.join(
            datasets_root,
            "modified_libero_rlds",
            f"{task_suite_name}_no_noops",
            "1.0.0",
        ),
    ]

    for cand in candidates:
        if tf.io.gfile.glob(os.path.join(cand, "*.tfrecord-*")):
            return cand

    checked = "\n".join(f"  - {c}" for c in candidates)
    raise ValueError(
        "Could not locate RLDS TFRecords for requested suite. Checked:\n" + checked
    )


def _validate_written_hdf5(
    out_path,
    episodes,
    expected_h,
    expected_w,
    revert_libero_actions,
    flip_last_action_dim,
):
    with h5py.File(out_path, "r") as f_in:
        if "data" not in f_in:
            raise ValueError(f"[validate] Missing 'data' group in {out_path}")
        data = f_in["data"]

        demo_names = sorted(list(data.keys()))
        expected_num_demos = len(episodes)
        if len(demo_names) != expected_num_demos:
            raise ValueError(
                f"[validate] Demo count mismatch for {out_path}: "
                f"expected {expected_num_demos}, found {len(demo_names)}"
            )

        if int(data.attrs.get("num_demos", -1)) != expected_num_demos:
            raise ValueError(
                f"[validate] data.attrs['num_demos'] mismatch for {out_path}: "
                f"expected {expected_num_demos}, found {data.attrs.get('num_demos')}"
            )

        expected_total = 0
        for demo_idx, ep in enumerate(episodes):
            demo_name = f"demo_{demo_idx}"
            if demo_name not in data:
                raise ValueError(f"[validate] Missing {demo_name} in {out_path}")

            demo = data[demo_name]
            obs = demo["obs"]

            actions_expected = ep["actions"].astype(np.float32)
            if revert_libero_actions:
                actions_expected = _revert_actions_from_libero(actions_expected).astype(
                    np.float32
                )
            if flip_last_action_dim:
                actions_expected[..., -1] = -actions_expected[..., -1]

            actions_written = demo["actions"][:]
            if actions_written.shape != actions_expected.shape:
                raise ValueError(
                    f"[validate] Actions shape mismatch for {out_path}/{demo_name}: "
                    f"expected {actions_expected.shape}, found {actions_written.shape}"
                )
            if not np.array_equal(actions_written, actions_expected):
                raise ValueError(
                    f"[validate] Actions content mismatch for {out_path}/{demo_name}"
                )

            images = obs["agentview_rgb"][:]
            if images.ndim != 4 or images.shape[-1] != 3:
                raise ValueError(
                    f"[validate] Invalid image shape for {out_path}/{demo_name}: "
                    f"{images.shape} (expected [T,H,W,3])"
                )
            if images.shape[1] != expected_h or images.shape[2] != expected_w:
                raise ValueError(
                    f"[validate] Image size mismatch for {out_path}/{demo_name}: "
                    f"expected ({expected_h},{expected_w},3), "
                    f"found ({images.shape[1]},{images.shape[2]},{images.shape[3]})"
                )

            traj_len = int(actions_expected.shape[0])
            if int(images.shape[0]) != traj_len:
                raise ValueError(
                    f"[validate] T mismatch for {out_path}/{demo_name}: "
                    f"actions={traj_len}, images={images.shape[0]}"
                )

            num_samples_attr = int(demo.attrs.get("num_samples", -1))
            if num_samples_attr != traj_len:
                raise ValueError(
                    f"[validate] num_samples mismatch for {out_path}/{demo_name}: "
                    f"expected {traj_len}, found {num_samples_attr}"
                )

            dones = demo["dones"][:]
            rewards = demo["rewards"][:]
            if dones.shape[0] != traj_len or rewards.shape[0] != traj_len:
                raise ValueError(
                    f"[validate] dones/rewards length mismatch for {out_path}/{demo_name}"
                )
            if traj_len > 0 and (dones[-1] != 1 or rewards[-1] != 1):
                raise ValueError(
                    f"[validate] Final dones/rewards marker mismatch for {out_path}/{demo_name}"
                )

            expected_total += traj_len

        total_attr = int(data.attrs.get("total", -1))
        if total_attr != expected_total:
            raise ValueError(
                f"[validate] data.attrs['total'] mismatch for {out_path}: "
                f"expected {expected_total}, found {total_attr}"
            )


def _compare_dirs(
    reference_dir,
    converted_dir,
    compare_mode,
    length_match_max_abs_diff=1e-7,
    length_match_min_ratio=0.0,
):
    ref_files = sorted(
        [f for f in os.listdir(reference_dir) if f.endswith(".hdf5")]
    )
    out_files = sorted(
        [f for f in os.listdir(converted_dir) if f.endswith(".hdf5")]
    )

    errors = []

    ref_set = set(ref_files)
    out_set = set(out_files)
    missing_in_out = sorted(ref_set - out_set)
    extra_in_out = sorted(out_set - ref_set)
    if missing_in_out:
        errors.append(f"Missing task files in converted dir: {missing_in_out}")
    if extra_in_out:
        errors.append(f"Extra task files in converted dir: {extra_in_out}")

    common = sorted(ref_set & out_set)
    for fn in common:
        ref_path = os.path.join(reference_dir, fn)
        out_path = os.path.join(converted_dir, fn)
        with h5py.File(ref_path, "r") as ref_h5, h5py.File(out_path, "r") as out_h5:
            if "data" not in ref_h5 or "data" not in out_h5:
                errors.append(f"[{fn}] missing 'data' group in one file")
                continue

            ref_data = ref_h5["data"]
            out_data = out_h5["data"]

            ref_demos = sorted(list(ref_data.keys()))
            out_demos = sorted(list(out_data.keys()))
            ref_num_demos = int(ref_data.attrs.get("num_demos", -1))
            out_num_demos = int(out_data.attrs.get("num_demos", -1))

            def _print_first_action(prefix, data_group, demo_name):
                if demo_name is None or demo_name not in data_group:
                    print(f"[compare] {prefix}: demo missing")
                    return
                demo = data_group[demo_name]
                if "actions" not in demo:
                    print(f"[compare] {prefix}: actions missing")
                    return
                actions = demo["actions"][:]
                if actions.shape[0] == 0:
                    print(f"[compare] {prefix}: empty actions")
                    return
                vec = np.array2string(actions[0], precision=6, separator=", ")
                print(f"[compare] {prefix}: {demo_name} action[0]={vec}")

            if compare_mode == "strict":
                if ref_demos != out_demos:
                    errors.append(
                        f"[{fn}] demo list mismatch: ref={len(ref_demos)} out={len(out_demos)}"
                    )
                    continue
                if ref_num_demos != out_num_demos:
                    errors.append(
                        f"[{fn}] num_demos attr mismatch: ref={ref_num_demos} out={out_num_demos}"
                    )
                ref_total = int(ref_data.attrs.get("total", -1))
                out_total = int(out_data.attrs.get("total", -1))
                if ref_total != out_total:
                    errors.append(
                        f"[{fn}] total attr mismatch: ref={ref_total} out={out_total}"
                    )
                demos_to_compare = ref_demos
            elif compare_mode == "allow-missing-episodes":
                # In allow-missing mode, demo indices may not align (different ordering).
                # Compare episode content by action-sequence hashes instead.
                ref_hashes = Counter()
                out_hashes = Counter()
                for d in ref_demos:
                    ref_actions = ref_data[d]["actions"][:]
                    ref_hashes[hashlib.sha1(ref_actions.tobytes()).hexdigest()] += 1
                for d in out_demos:
                    out_actions = out_data[d]["actions"][:]
                    out_hashes[hashlib.sha1(out_actions.tobytes()).hexdigest()] += 1

                overlap_count = sum((ref_hashes & out_hashes).values())
                out_count = sum(out_hashes.values())
                ref_count = sum(ref_hashes.values())

                print(
                    f"[compare] info: {fn} reference={ref_count} converted={out_count} "
                    f"content-overlap={overlap_count}"
                )

                if overlap_count == 0:
                    # Helpful debug print for quick terminal-side inspection.
                    ref_demo0 = ref_demos[0] if ref_demos else None
                    out_demo0 = out_demos[0] if out_demos else None
                    _print_first_action("reference first", ref_data, ref_demo0)
                    _print_first_action("converted first", out_data, out_demo0)
                    errors.append(
                        f"[{fn}] no overlapping episodes by action content; "
                        "this is not a simple subset-with-missing-episodes case"
                    )
                    continue
                if overlap_count != out_count:
                    ref_demo0 = ref_demos[0] if ref_demos else None
                    out_demo0 = out_demos[0] if out_demos else None
                    _print_first_action("reference first", ref_data, ref_demo0)
                    _print_first_action("converted first", out_data, out_demo0)
                    errors.append(
                        f"[{fn}] converted episodes are not a pure subset of reference by action content: "
                        f"overlap={overlap_count}, converted={out_count}"
                    )
                    continue

                # If converted is a subset by content, we can skip demo-index checks.
                # No per-demo loop needed because matching is content-based.
                continue
            else:
                # compare_mode == "length-matched-actions"
                # Greedy one-to-one assignment by absolute trajectory-length difference.
                ref_meta = []
                out_meta = []
                for d in ref_demos:
                    ref_meta.append((int(d.split("_")[1]), int(ref_data[d]["actions"].shape[0]), d))
                for d in out_demos:
                    out_meta.append((int(d.split("_")[1]), int(out_data[d]["actions"].shape[0]), d))
                ref_meta.sort(key=lambda x: x[0])
                out_meta.sort(key=lambda x: x[0])

                candidates = []
                for oi, ol, od in ref_meta:
                    for ni, nl, nd in out_meta:
                        candidates.append((abs(ol - nl), oi, ni, od, nd))
                candidates.sort(key=lambda x: (x[0], x[1], x[2]))

                used_o = set()
                used_n = set()
                exact_len_pairs = []
                for len_diff, oi, ni, od, nd in candidates:
                    if oi in used_o or ni in used_n:
                        continue
                    used_o.add(oi)
                    used_n.add(ni)
                    if len_diff == 0:
                        exact_len_pairs.append((od, nd))
                    if len(used_n) == min(len(ref_meta), len(out_meta)):
                        break

                if not exact_len_pairs:
                    errors.append(
                        f"[{fn}] length-matched-actions found no exact-length episode pairs"
                    )
                    continue

                task_ok = 0
                for od, nd in exact_len_pairs:
                    ref_actions = ref_data[od]["actions"][:]
                    out_actions = out_data[nd]["actions"][:]
                    ref_first = ref_actions[0][:6].astype(np.float64)
                    out_first = out_actions[0][:6].astype(np.float64)
                    max_abs = float(np.max(np.abs(ref_first - out_first)))
                    task_ok += int(max_abs <= length_match_max_abs_diff)

                task_ratio = task_ok / len(exact_len_pairs)
                print(
                    f"[compare] length-match: {fn} "
                    f"pairs={len(exact_len_pairs)} "
                    f"matched={task_ok} ratio={task_ratio:.3f} "
                    f"(threshold={length_match_max_abs_diff:g})"
                )
                if task_ratio < length_match_min_ratio:
                    errors.append(
                        f"[{fn}] length-matched-actions ratio below minimum: "
                        f"{task_ratio:.3f} < {length_match_min_ratio:.3f}"
                    )
                continue

            for demo in demos_to_compare:
                ref_demo = ref_data[demo]
                out_demo = out_data[demo]

                if "actions" not in ref_demo or "actions" not in out_demo:
                    errors.append(f"[{fn}/{demo}] missing actions dataset")
                    continue

                ref_actions = ref_demo["actions"][:]
                out_actions = out_demo["actions"][:]
                if ref_actions.shape != out_actions.shape:
                    errors.append(
                        f"[{fn}/{demo}] action shape mismatch: ref={ref_actions.shape} out={out_actions.shape}"
                    )
                elif not np.array_equal(ref_actions, out_actions):
                    errors.append(f"[{fn}/{demo}] action chunk values mismatch")

                ref_num_samples = int(ref_demo.attrs.get("num_samples", -1))
                out_num_samples = int(out_demo.attrs.get("num_samples", -1))
                if ref_num_samples != out_num_samples:
                    errors.append(
                        f"[{fn}/{demo}] num_samples attr mismatch: ref={ref_num_samples} out={out_num_samples}"
                    )

                if "obs" not in ref_demo or "obs" not in out_demo:
                    errors.append(f"[{fn}/{demo}] missing obs group")
                    continue

                ref_obs = ref_demo["obs"]
                out_obs = out_demo["obs"]
                if "agentview_rgb" not in ref_obs or "agentview_rgb" not in out_obs:
                    errors.append(f"[{fn}/{demo}] missing obs/agentview_rgb")
                else:
                    ref_img = ref_obs["agentview_rgb"][:]
                    out_img = out_obs["agentview_rgb"][:]
                    if ref_img.ndim != out_img.ndim:
                        errors.append(
                            f"[{fn}/{demo}] image ndim mismatch: ref={ref_img.ndim} out={out_img.ndim}"
                        )
                    if ref_img.shape[0] != out_img.shape[0]:
                        errors.append(
                            f"[{fn}/{demo}] image length mismatch: ref={ref_img.shape[0]} out={out_img.shape[0]}"
                        )
                    if ref_img.shape[-1] != out_img.shape[-1]:
                        errors.append(
                            f"[{fn}/{demo}] channel mismatch: ref={ref_img.shape[-1]} out={out_img.shape[-1]}"
                        )

                if "dones" in ref_demo and "dones" in out_demo:
                    if ref_demo["dones"].shape != out_demo["dones"].shape:
                        errors.append(f"[{fn}/{demo}] dones shape mismatch")
                if "rewards" in ref_demo and "rewards" in out_demo:
                    if ref_demo["rewards"].shape != out_demo["rewards"].shape:
                        errors.append(f"[{fn}/{demo}] rewards shape mismatch")

    if errors:
        preview = "\n".join(f"- {e}" for e in errors[:50])
        extra_count = len(errors) - min(50, len(errors))
        if extra_count > 0:
            preview += f"\n- ... and {extra_count} more mismatch(es)"
        raise RuntimeError(
            "[compare] Converted dataset does NOT match reference dataset:\n"
            + preview
        )

    if compare_mode == "strict":
        print(f"[compare] OK (strict): '{converted_dir}' matches '{reference_dir}'")
    elif compare_mode == "allow-missing-episodes":
        print(
            f"[compare] OK (allow-missing-episodes): overlapping demos in "
            f"'{converted_dir}' match '{reference_dir}'"
        )
    else:
        print(
            f"[compare] OK (length-matched-actions): per-task length-matched "
            f"first-action checks passed for '{converted_dir}' vs '{reference_dir}'"
        )


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    datasets_root = os.path.abspath(os.path.join(args.output_dir, os.pardir))
    input_dir = args.input_dir

    if args.download_first:
        _download_modified_rlds(args.hf_repo_id, datasets_root)
        input_dir = _resolve_rlds_input_dir(
            datasets_root=datasets_root, task_suite_name=args.task_suite_name
        )
        print(f"[download] Resolved input-dir to: {input_dir}")

    if input_dir is None:
        raise ValueError("Provide --input-dir, or use --download-first.")

    files = tf.io.gfile.glob(os.path.join(input_dir, "*.tfrecord-*"))
    if not files:
        raise ValueError(f"No TFRecord files found under: {input_dir}")

    print(f"[convert] Loading TFRecords from: {input_dir}")
    raw_dataset = tf.data.TFRecordDataset(files)

    task_episodes = defaultdict(list)
    n_total = 0
    for raw_record in raw_dataset:
        ep = _parse_episode(raw_record)
        task_episodes[ep["task_name"]].append(ep)
        n_total += 1

    print(
        f"[convert] Parsed {n_total} episode(s) across {len(task_episodes)} task(s)."
    )

    for task_name in sorted(task_episodes.keys()):
        episodes = task_episodes[task_name]
        if args.demos_per_task > 0:
            episodes = episodes[: args.demos_per_task]

        task_file = f"{_safe_task_filename(task_name)}_demo.hdf5"
        out_path = os.path.join(args.output_dir, task_file)
        print(
            f"[convert] Writing {len(episodes)} demo(s) for task='{task_name}' "
            f"-> {out_path}"
        )

        with h5py.File(out_path, "w") as f_out:
            data_group = f_out.create_group("data")
            data_group.attrs["problem_info"] = task_name

            for demo_idx, ep in enumerate(episodes):
                demo_group = data_group.create_group(f"demo_{demo_idx}")
                obs_group = demo_group.create_group("obs")

                actions = ep["actions"].astype(np.float32)
                if args.revert_libero_actions:
                    actions = _revert_actions_from_libero(actions).astype(np.float32)
                if args.flip_last_action_dim:
                    actions[..., -1] = -actions[..., -1]

                images = np.stack(
                    [_decode_image(b) for b in ep["image_bytes"]], axis=0
                ).astype(np.uint8)
                if args.rotate_images_180:
                    # Rotate each frame by 180 deg (flip H and W axes).
                    images = images[:, ::-1, ::-1, :]
                if images.ndim != 4 or images.shape[-1] != 3:
                    raise ValueError(
                        f"Invalid decoded image tensor for task='{task_name}', "
                        f"demo={demo_idx}: {images.shape}"
                    )
                if (
                    images.shape[1] != args.expected_height
                    or images.shape[2] != args.expected_width
                ):
                    raise ValueError(
                        f"Unexpected agentview_rgb shape for task='{task_name}', demo={demo_idx}: "
                        f"got {images.shape}, expected [T,{args.expected_height},{args.expected_width},3]"
                    )
                wrist_images = None
                if ep["wrist_image_bytes"] is not None:
                    wrist_images = np.stack(
                        [_decode_image(b) for b in ep["wrist_image_bytes"]], axis=0
                    ).astype(np.uint8)
                    if args.rotate_images_180:
                        wrist_images = wrist_images[:, ::-1, ::-1, :]

                demo_group.create_dataset("actions", data=actions)
                obs_group.create_dataset("agentview_rgb", data=images)
                if wrist_images is not None:
                    obs_group.create_dataset("eye_in_hand_rgb", data=wrist_images)

                # Keep terminal markers compatible with LIBERO-style demos.
                dones = np.zeros(images.shape[0], dtype=np.uint8)
                rewards = np.zeros(images.shape[0], dtype=np.uint8)
                if images.shape[0] > 0:
                    dones[-1] = 1
                    rewards[-1] = 1
                demo_group.create_dataset("dones", data=dones)
                demo_group.create_dataset("rewards", data=rewards)
                demo_group.attrs["num_samples"] = int(images.shape[0])

            data_group.attrs["total"] = int(
                sum(ep["actions"].shape[0] for ep in episodes)
            )
            data_group.attrs["num_demos"] = int(len(episodes))

        if args.validate_written_files:
            _validate_written_hdf5(
                out_path=out_path,
                episodes=episodes,
                expected_h=args.expected_height,
                expected_w=args.expected_width,
                revert_libero_actions=args.revert_libero_actions,
                flip_last_action_dim=args.flip_last_action_dim,
            )
            print(f"[validate] OK: {out_path}")

    if args.compare_with_dir is not None:
        reference_dir = os.path.abspath(args.compare_with_dir)
        converted_dir = os.path.abspath(args.output_dir)
        if not os.path.isdir(reference_dir):
            raise ValueError(
                f"--compare-with-dir is not a directory: {reference_dir}"
            )
        _compare_dirs(
            reference_dir=reference_dir,
            converted_dir=converted_dir,
            compare_mode=args.compare_mode,
            length_match_max_abs_diff=args.length_match_max_abs_diff,
            length_match_min_ratio=args.length_match_min_ratio,
        )

    print("[convert] Done.")


if __name__ == "__main__":
    main()

