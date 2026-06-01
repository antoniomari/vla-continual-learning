#!/usr/bin/env python3
"""Upload/download a LIBERO dataset folder through a Hugging Face dataset repo."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def _require_hf():
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: huggingface_hub\n"
            "Install it in the active environment with:\n"
            "  python -m pip install -U huggingface_hub hf_transfer\n"
        ) from exc
    return HfApi, snapshot_download


def _count_files(path: Path) -> tuple[int, int]:
    n_files = 0
    n_bytes = 0
    for item in path.rglob("*"):
        if item.is_file():
            n_files += 1
            n_bytes += item.stat().st_size
    return n_files, n_bytes


def _format_bytes(n_bytes: int) -> str:
    value = float(n_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TB"


def _copy_tree_contents(src: Path, dst: Path, overwrite: bool) -> None:
    if not src.is_dir():
        raise FileNotFoundError(f"Downloaded source folder does not exist: {src}")
    if dst.exists():
        if not overwrite:
            raise FileExistsError(
                f"Target dataset folder already exists: {dst}\n"
                "Pass --overwrite to replace it."
            )
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def upload(args: argparse.Namespace) -> int:
    HfApi, _ = _require_hf()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder does not exist: {dataset_dir}")

    n_files, n_bytes = _count_files(dataset_dir)
    print("Uploading LIBERO dataset folder to Hugging Face")
    print(f"  local folder: {dataset_dir}")
    print(f"  files:        {n_files}")
    print(f"  size:         {_format_bytes(n_bytes)}")
    print(f"  repo_id:      {args.repo_id}")
    print(f"  repo_type:    {args.repo_type}")
    print(f"  path_in_repo: {args.path_in_repo}")
    if args.dry_run:
        print("DRY_RUN=1: not uploading.")
        return 0

    api = HfApi(token=args.token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        private=args.private,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=str(dataset_dir),
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        path_in_repo=args.path_in_repo,
        commit_message=args.commit_message,
        token=args.token,
    )
    print("Upload completed.")
    return 0


def download(args: argparse.Namespace) -> int:
    _, snapshot_download = _require_hf()
    staging_dir = Path(args.staging_dir).expanduser().resolve()
    target_dir = Path(args.dataset_dir).expanduser().resolve()
    path_in_repo = args.path_in_repo.strip("/")
    allow_patterns = [f"{path_in_repo}/**"] if path_in_repo else None

    print("Downloading LIBERO dataset folder from Hugging Face")
    print(f"  repo_id:      {args.repo_id}")
    print(f"  repo_type:    {args.repo_type}")
    print(f"  revision:     {args.revision}")
    print(f"  path_in_repo: {path_in_repo or '(repo root)'}")
    print(f"  staging:      {staging_dir}")
    print(f"  target:       {target_dir}")
    if args.dry_run:
        print("DRY_RUN=1: not downloading.")
        return 0

    staging_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = Path(
        snapshot_download(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
            local_dir=str(staging_dir),
            allow_patterns=allow_patterns,
            token=args.token,
        )
    )

    downloaded_folder = snapshot_path / path_in_repo if path_in_repo else snapshot_path
    _copy_tree_contents(downloaded_folder, target_dir, overwrite=args.overwrite)

    n_files, n_bytes = _count_files(target_dir)
    print("Download completed.")
    print(f"  installed folder: {target_dir}")
    print(f"  files:            {n_files}")
    print(f"  size:             {_format_bytes(n_bytes)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--repo-id", required=True)
    common.add_argument("--repo-type", default="dataset")
    common.add_argument("--path-in-repo", required=True)
    common.add_argument("--dataset-dir", required=True)
    common.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    common.add_argument("--dry-run", action="store_true")

    up = sub.add_parser("upload", parents=[common])
    up.add_argument("--private", action="store_true")
    up.add_argument(
        "--commit-message",
        default="Upload LIBERO dataset folder",
    )
    up.set_defaults(func=upload)

    down = sub.add_parser("download", parents=[common])
    down.add_argument("--staging-dir", required=True)
    down.add_argument("--revision", default="main")
    down.add_argument("--overwrite", action="store_true")
    down.set_defaults(func=download)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
