#!/usr/bin/env python3
"""Inspect LIBERO SFT HDF5 action conventions for OPD teacher debugging."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_libero_root() -> Path:
    return _repo_root() / "LIBERO"


def _task_file_for_id(suite: str, task_id: int) -> str:
    from libero.libero.benchmark import get_benchmark

    benchmark = get_benchmark(suite)()
    return f"{benchmark.get_task(task_id).name}_demo.hdf5"


def _summarize_actions(name: str, actions: np.ndarray) -> None:
    flat = actions.reshape(-1, actions.shape[-1])
    print(f"\n{name}")
    print(f"  shape={actions.shape} flat={flat.shape}")
    print(f"  min={np.array2string(flat.min(axis=0), precision=5)}")
    print(f"  max={np.array2string(flat.max(axis=0), precision=5)}")
    print(f"  mean={np.array2string(flat.mean(axis=0), precision=5)}")
    print(f"  std={np.array2string(flat.std(axis=0), precision=5)}")
    print(f"  q01={np.array2string(np.quantile(flat, 0.01, axis=0), precision=5)}")
    print(f"  q99={np.array2string(np.quantile(flat, 0.99, axis=0), precision=5)}")
    print(f"  outside[-1,1] per dim={np.array2string(((flat < -1) | (flat > 1)).mean(axis=0), precision=5)}")

    grip = flat[:, -1]
    print("  gripper:")
    print(f"    min={grip.min():.6f} max={grip.max():.6f} mean={grip.mean():.6f} std={grip.std():.6f}")
    print(f"    frac <0={np.mean(grip < 0):.4f}  ==0={np.mean(grip == 0):.4f}  ==1={np.mean(grip == 1):.4f}  >1={np.mean(grip > 1):.4f}")
    vals, counts = np.unique(np.round(grip, 4), return_counts=True)
    order = np.argsort(counts)[::-1][:12]
    top = ", ".join(f"{vals[i]:.4g}:{counts[i]}" for i in order)
    print(f"    top rounded values: {top}")

    libero_env_grip = np.sign(2 * grip - 1) * -1.0
    vals, counts = np.unique(libero_env_grip, return_counts=True)
    print(f"    after prepare_actions_for_libero gripper values: {dict(zip(vals.tolist(), counts.tolist()))}")


def _apply_sft_action_preprocessing(
    actions: np.ndarray,
    gripper_from_neg1_0_to_0_1: bool,
) -> np.ndarray:
    processed = actions.copy()
    if gripper_from_neg1_0_to_0_1:
        processed[..., -1] = np.clip(processed[..., -1] + 1.0, 0.0, 1.0)
    return processed


def _preflight_actions(
    task_id: int,
    actions: np.ndarray,
    gripper_from_neg1_0_to_0_1: bool,
) -> list[str]:
    errors = []
    processed = _apply_sft_action_preprocessing(
        actions,
        gripper_from_neg1_0_to_0_1=gripper_from_neg1_0_to_0_1,
    )
    flat = processed.reshape(-1, processed.shape[-1])
    grip = flat[:, -1]

    if processed.shape[-1] != 7:
        errors.append(f"task {task_id}: expected action_dim=7, got {processed.shape[-1]}")
    if np.any(~np.isfinite(flat)):
        errors.append(f"task {task_id}: actions contain non-finite values after preprocessing")
    outside = ((flat < -1.0001) | (flat > 1.0001)).mean(axis=0)
    if np.any(outside > 0):
        errors.append(
            f"task {task_id}: actions outside [-1,1] after preprocessing per dim "
            f"{np.array2string(outside, precision=5)}"
        )
    if grip.min() < -1e-6 or grip.max() > 1.0 + 1e-6:
        errors.append(
            f"task {task_id}: SFT gripper must be in [0,1] before LIBERO env conversion; "
            f"got min={grip.min():.6f}, max={grip.max():.6f}. "
            "Use --gripper-from-neg1-0-to-0-1 for datasets with {-1,0} labels."
        )

    has_close = np.any(grip < 0.5)
    has_open = np.any(grip > 0.5)
    if not (has_close and has_open):
        errors.append(
            f"task {task_id}: preprocessed gripper should contain both close(<0.5) "
            f"and open(>0.5); got min={grip.min():.6f}, max={grip.max():.6f}"
        )

    env_grip = np.sign(2 * grip - 1) * -1.0
    unique_env = np.unique(env_grip)
    if unique_env.size < 2:
        errors.append(
            f"task {task_id}: LIBERO env gripper conversion collapses to one command "
            f"{unique_env.tolist()}; raw/preprocessed gripper convention is likely wrong"
        )

    return errors


def _read_task_actions(path: Path, max_demos: int | None) -> tuple[np.ndarray, list[str]]:
    actions = []
    demo_names = []
    with h5py.File(path, "r") as f:
        names = sorted(f["data"].keys())
        if max_demos is not None:
            names = names[:max_demos]
        for name in names:
            arr = np.asarray(f["data"][name]["actions"], dtype=np.float32)
            actions.append(arr)
            demo_names.append(name)
    if not actions:
        raise RuntimeError(f"No actions found in {path}")
    return np.concatenate(actions, axis=0), demo_names


def _print_examples(path: Path, n: int) -> None:
    with h5py.File(path, "r") as f:
        first_demo = sorted(f["data"].keys())[0]
        actions = np.asarray(f["data"][first_demo]["actions"], dtype=np.float32)
    print(f"\nFirst demo examples from {path.name}/{first_demo}:")
    for i in range(min(n, len(actions))):
        env_action = actions[i].copy()
        env_action[-1] = np.sign(2 * env_action[-1] - 1) * -1.0
        print(
            f"  t={i:03d} raw={np.array2string(actions[i], precision=5)} "
            f"env_grip={env_action[-1]:.1f}"
        )


def _compare_action_stats_to_model(actions: np.ndarray, model_dir: str, unnorm_key: str) -> None:
    try:
        import torch
        from omegaconf import OmegaConf

        from rlinf.models import get_model
        from rlinf.models.embodiment.model_utils import compute_action_tokens_from_actions
    except Exception as exc:
        print(f"\n[roundtrip] skipped: could not import model stack: {exc}")
        return

    cfg = OmegaConf.create(
        {
            "model_name": "openvla_oft",
            "value_type": "step_level",
            "action_dim": 7,
            "num_action_chunks": 8,
            "use_proprio": False,
            "unnorm_key": unnorm_key,
            "center_crop": True,
            "precision": "bf16",
            "add_bias_linear": False,
            "add_qkv_bias": True,
            "vocab_size": 32000,
            "hidden_size": 4096,
            "policy_setup": "widowx_bridge",
            "vh_mode": "a0",
            "image_size": [224, 224],
            "is_lora": True,
            "lora_rank": 32,
            "partial_finetune": False,
            "layers_to_train": 50,
            "num_images_in_input": 1,
            "attn_implementation": "flash_attention_2",
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "gradient_checkpointing": False,
        }
    )

    print(f"\n[roundtrip] loading model stats from {model_dir} with unnorm_key={unnorm_key}")
    model = get_model(model_dir, cfg)
    chunks = []
    for start in range(0, min(len(actions) - 8 + 1, 64), 8):
        chunks.append(actions[start : start + 8])
    if not chunks:
        print("[roundtrip] skipped: not enough actions for one chunk")
        return
    batch = torch.tensor(np.stack(chunks, axis=0), dtype=torch.float32)
    token_ids = compute_action_tokens_from_actions(model, batch)
    token_ids_t = torch.tensor(token_ids)
    decoded, *_ = model._decode_action_tokens(token_ids_t)
    decoded = np.asarray(decoded, dtype=np.float32)
    original = batch.numpy()
    abs_err = np.abs(decoded - original)
    print(f"  token_ids range: {token_ids.min()}..{token_ids.max()}")
    print(f"  decoded-vs-original mean_abs per dim={np.array2string(abs_err.reshape(-1, 7).mean(axis=0), precision=5)}")
    print(f"  decoded-vs-original max_abs per dim={np.array2string(abs_err.reshape(-1, 7).max(axis=0), precision=5)}")
    print(f"  first original chunk[0]={np.array2string(original[0, 0], precision=5)}")
    print(f"  first decoded  chunk[0]={np.array2string(decoded[0, 0], precision=5)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero-root", default=str(_default_libero_root()))
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--dataset", default="libero_spatial_256_from_rlds_reverted")
    parser.add_argument("--tasks", nargs="+", type=int, default=[1, 4, 5, 9])
    parser.add_argument("--max-demos", type=int, default=None)
    parser.add_argument("--examples", type=int, default=12)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--unnorm-key", default="libero_spatial_no_noops")
    parser.add_argument("--roundtrip", action="store_true")
    parser.add_argument(
        "--gripper-from-neg1-0-to-0-1",
        action="store_true",
        help="Apply the OPD SFT BC gripper fix: {-1,0} -> {0,1} before checks.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Exit non-zero if action preprocessing invariants needed before training fail.",
    )
    args = parser.parse_args()

    repo = _repo_root()
    sys.path.insert(0, str(repo / "LIBERO"))
    sys.path.insert(0, str(repo / "openvla-oft"))

    dataset_dir = Path(args.libero_root) / "libero" / "datasets" / args.dataset
    print(f"dataset_dir={dataset_dir}")
    if not dataset_dir.is_dir():
        raise FileNotFoundError(dataset_dir)

    all_task_actions = {}
    for task_id in args.tasks:
        filename = _task_file_for_id(args.suite, task_id)
        path = dataset_dir / filename
        print(f"\nTASK {task_id}: {filename}")
        if not path.is_file():
            print(f"  MISSING: {path}")
            continue
        actions, demos = _read_task_actions(path, args.max_demos)
        print(f"  demos={len(demos)} first_demos={demos[:3]}")
        _summarize_actions(f"Task {task_id} actions", actions)
        if args.gripper_from_neg1_0_to_0_1:
            processed = _apply_sft_action_preprocessing(
                actions,
                gripper_from_neg1_0_to_0_1=True,
            )
            _summarize_actions(
                f"Task {task_id} actions after SFT gripper -1/0 -> 0/1",
                processed,
            )
        _print_examples(path, args.examples)
        if args.preflight:
            errors = _preflight_actions(
                task_id,
                actions,
                gripper_from_neg1_0_to_0_1=args.gripper_from_neg1_0_to_0_1,
            )
            if errors:
                print("\n[PREFLIGHT FAILED]")
                for err in errors:
                    print(f"  - {err}")
                return 2
            print(f"\n[PREFLIGHT OK] task {task_id} action preprocessing checks passed")
        all_task_actions[task_id] = actions

    if len(all_task_actions) > 1:
        print("\nCross-task gripper summary:")
        for task_id, actions in all_task_actions.items():
            grip = actions.reshape(-1, actions.shape[-1])[:, -1]
            print(
                f"  task {task_id}: mean={grip.mean():.5f} min={grip.min():.5f} "
                f"max={grip.max():.5f} frac<0={np.mean(grip < 0):.4f} "
                f"frac==0={np.mean(grip == 0):.4f} frac==1={np.mean(grip == 1):.4f}"
            )

    if args.roundtrip:
        if not args.model_dir:
            raise ValueError("--roundtrip requires --model-dir")
        first_actions = next(iter(all_task_actions.values()))
        _compare_action_stats_to_model(first_actions, args.model_dir, args.unnorm_key)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
