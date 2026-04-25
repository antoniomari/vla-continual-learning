# Copyright 2025 The RLinf Authors.
#
# Append final embodied eval metrics to results/eval_results.csv (one row per run).

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Union

import torch

from omegaconf import DictConfig, OmegaConf


def _to_scalar(v: Any) -> Union[float, int, str]:
    if v is None:
        return ""
    if isinstance(v, (float, int, str, bool)):
        if isinstance(v, bool):
            return int(v)
        return v
    if isinstance(v, torch.Tensor):
        return float(v.detach().float().cpu().item())
    return str(v)


def metric_key_to_csv_column(key: str) -> str:
    """Map keys like env_info/task_0_success to task_0_success (no eval/ or env_info/)."""
    s = key
    for prefix in ("eval/env_info/", "eval/"):
        if s.startswith(prefix):
            s = s[len(prefix) :]
            break
    if s.startswith("env_info/"):
        s = s[len("env_info/") :]
    return s


def _results_csv_path() -> Path:
    root = os.environ.get("REPO_PATH") or os.getcwd()
    return Path(root) / "results" / "eval_results.csv"


def _build_metadata_row(cfg: DictConfig) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "experiment_name": cfg.runner.logger.get("experiment_name", "") or "",
        "seed": _to_scalar(cfg.actor.get("seed", "")),
        "group_size": _to_scalar(cfg.algorithm.get("group_size", "")),
        "num_group_envs": _to_scalar(cfg.algorithm.get("num_group_envs", "")),
        "rollout_epoch": _to_scalar(cfg.algorithm.get("rollout_epoch", "")),
        "eval_rollout_epoch": _to_scalar(cfg.algorithm.get("eval_rollout_epoch", "")),
        "global_batch_size": _to_scalar(cfg.actor.get("global_batch_size", "")),
    }
    sp = cfg.algorithm.get("sampling_params")
    if sp is not None:
        t_eval = None
        if OmegaConf.is_config(sp):
            t_eval = sp.get("temperature_eval", None)
        elif isinstance(sp, dict):
            t_eval = sp.get("temperature_eval")
        if t_eval is not None:
            row["temperature_eval"] = _to_scalar(t_eval)
    # Shell exports (crl_experiment/eval_embodiment.sh, examples/embodiment/eval_embodiment.sh)
    row["hydra_config_name"] = os.environ.get("RLINF_EVAL_HYDRA_CONFIG_NAME", "")
    row["checkpoint_location"] = os.environ.get("RLINF_EVAL_CHECKPOINT_REL", "")
    if not row["checkpoint_location"]:
        row["checkpoint_location"] = os.environ.get("RLINF_EVAL_CHECKPOINT_LOCATION", "")
    step = os.environ.get("EVAL_STEP_NUMBER", "") or os.environ.get("RLINF_EVAL_GLOBAL_STEP", "")
    row["global_step"] = step
    lora = None
    try:
        lora = cfg.actor.model.get("lora_path") or cfg.actor.model.get("lora_paths")
    except Exception:
        pass
    if lora is not None and str(lora) not in ("None",):
        row["lora_path"] = str(lora)
    fids = cfg.env.get("fixed_task_ids", None)
    if fids is not None:
        row["fixed_task_ids"] = str(
            OmegaConf.to_container(fids) if OmegaConf.is_config(fids) else fids
        )
    return row


def append_eval_results_row(
    final_metrics: Mapping[str, Any],
    cfg: DictConfig,
    csv_path: Path | None = None,
) -> Path:
    """
    Append one row: metadata + one column per eval metric, keys without eval/env_info prefixes.
    Merges with existing file if new columns appear (rewrites with padded empty cells).
    """
    path = csv_path or _results_csv_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    metadata = _build_metadata_row(cfg)
    row: Dict[str, Any] = {**metadata}
    for k, v in final_metrics.items():
        col = metric_key_to_csv_column(k)
        if not col:
            col = k
        if col in row:
            col = f"metric_{col}"
        row[col] = _to_scalar(v)

    if not path.exists():
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sorted(row.keys()))
            w.writeheader()
            w.writerow({k: row[k] for k in w.fieldnames})
        return path

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        old_fieldnames = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]

    new_fieldnames = sorted(set(old_fieldnames) | set(row.keys()))
    for r in rows:
        for k in new_fieldnames:
            if k not in r:
                r[k] = ""
    out_row = {k: row.get(k, "") for k in new_fieldnames}
    rows.append(out_row)

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=new_fieldnames)
        w.writeheader()
        w.writerows(rows)

    return path
