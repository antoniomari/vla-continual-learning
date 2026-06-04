#!/usr/bin/env python3
"""Export LIBERO HDF5 demonstration frames as videos for dataset inspection."""

from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


DEFAULT_DATASET_DIR = (
    "LIBERO/libero/datasets/libero_spatial_256_from_rlds_reverted"
)
TASK_MAP_PATH = "LIBERO/libero/libero/benchmark/libero_suite_task_map.py"


def load_task_map(repo_root: Path) -> dict[str, list[str]]:
    path = repo_root / TASK_MAP_PATH
    text = path.read_text()
    module = ast.parse(text, filename=str(path))
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "libero_task_map":
                    return ast.literal_eval(node.value)
    raise ValueError(f"Could not find libero_task_map in {path}")


def parse_ids(raw: str) -> list[int]:
    ids: list[int] = []
    for part in raw.replace(",", " ").split():
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            ids.extend(range(int(start), int(end) + 1))
        else:
            ids.append(int(part))
    return ids


def add_overlay(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    pad = 5
    line_h = 13
    box_w = min(image.width, max(draw.textlength(line) for line in lines) + 2 * pad + 2)
    box_h = line_h * len(lines) + 2 * pad
    draw.rectangle((0, 0, box_w, box_h), fill=(0, 0, 0))
    for i, line in enumerate(lines):
        draw.text((pad, pad + i * line_h), line, fill=(255, 255, 255))
    return np.asarray(image)


def write_video(
    out_path: Path,
    frames: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray | None,
    dones: np.ndarray | None,
    fps: int,
    max_frames: int | None,
    rotate180: bool,
    label: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = len(frames) if max_frames is None else min(len(frames), max_frames)
    try:
        writer_ctx = imageio.get_writer(
            str(out_path), format="FFMPEG", fps=fps, macro_block_size=1
        )
    except ImportError as exc:
        raise RuntimeError(
            "MP4 export requires the imageio ffmpeg plugin. Run this from the repo "
            "venv on the cluster, or install it with `pip install imageio[ffmpeg]`."
        ) from exc

    with writer_ctx as writer:
        for t in range(n_frames):
            frame = frames[t]
            if rotate180:
                frame = frame[::-1, ::-1]
            action = actions[t] if t < len(actions) else np.zeros(7)
            reward = float(rewards[t]) if rewards is not None and t < len(rewards) else 0.0
            done = int(dones[t]) if dones is not None and t < len(dones) else 0
            action_norm = float(np.linalg.norm(action[:6]))
            lines = [
                label,
                f"t={t:03d}/{len(frames)-1:03d} reward={reward:.1f} done={done}",
                f"|a[:6]|={action_norm:.3f} grip={action[-1]:.3f}",
            ]
            writer.append_data(add_overlay(frame, lines))


def export_demo(
    h5_path: Path,
    demo_name: str,
    output_dir: Path,
    camera: str,
    fps: int,
    max_frames: int | None,
    view: str,
) -> list[Path]:
    with h5py.File(h5_path, "r") as h5:
        demo = h5["data"][demo_name]
        frames = demo["obs"][camera][:]
        actions = demo["actions"][:]
        rewards = demo["rewards"][:] if "rewards" in demo else None
        dones = demo["dones"][:] if "dones" in demo else None

    written: list[Path] = []
    variants: list[tuple[str, bool]]
    if view == "both":
        variants = [("raw", False), ("rot180", True)]
    elif view == "rot180":
        variants = [("rot180", True)]
    else:
        variants = [("raw", False)]

    stem = h5_path.stem.replace("_demo", "")
    for suffix, rotate in variants:
        out = output_dir / stem / f"{demo_name}_{camera}_{suffix}.mp4"
        label = f"{stem} {demo_name} {camera} {suffix}"
        write_video(out, frames, actions, rewards, dones, fps, max_frames, rotate, label)
        written.append(out)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-ids", default="5", help="Task ids, e.g. '5' or '1 4 5 9'.")
    parser.add_argument("--max-demos", type=int, default=8)
    parser.add_argument("--demo-offset", type=int, default=0)
    parser.add_argument("--camera", default="agentview_rgb", choices=["agentview_rgb", "eye_in_hand_rgb"])
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=80, help="Use 0 for full episode.")
    parser.add_argument("--view", choices=["raw", "rot180", "both"], default="both")
    parser.add_argument(
        "--output-dir",
        default="visualization/results/dataset_videos/libero_spatial_256_from_rlds_reverted",
    )
    args = parser.parse_args()

    repo_root = Path.cwd()
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = repo_root / dataset_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    max_frames = None if args.max_frames <= 0 else args.max_frames

    task_map = load_task_map(repo_root)
    task_names = task_map[args.suite]
    written: list[Path] = []

    for task_id in parse_ids(args.task_ids):
        task_name = task_names[task_id]
        h5_path = dataset_dir / f"{task_name}_demo.hdf5"
        if not h5_path.exists():
            raise FileNotFoundError(f"Missing task {task_id} file: {h5_path}")
        with h5py.File(h5_path, "r") as h5:
            demos = sorted(h5["data"].keys(), key=lambda name: int(name.split("_")[-1]))
        selected = demos[args.demo_offset : args.demo_offset + args.max_demos]
        print(f"Task {task_id}: {task_name}")
        print(f"  file={h5_path}")
        print(f"  exporting demos={selected}")
        for demo_name in selected:
            written.extend(
                export_demo(
                    h5_path=h5_path,
                    demo_name=demo_name,
                    output_dir=output_dir / f"task_{task_id}",
                    camera=args.camera,
                    fps=args.fps,
                    max_frames=max_frames,
                    view=args.view,
                )
            )

    print("\nWrote videos:")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
