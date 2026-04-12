# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Logging helpers for counting / tracing full VLA (or policy) weight loads."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)

MODEL_LOAD_TAG = "[MODEL_LOAD]"


def log_get_model(
    *,
    role: str,
    model_path: str,
    model_name: str,
    worker_rank: int | None = None,
    worker_world_size: int | None = None,
    extra: str = "",
) -> None:
    """Emit one line when a worker instantiates weights via `get_model`.

    To limit log volume, only rank 0 of the worker group prints; the line states
    ``world_size`` so it is clear how many parallel loads occur.
    """
    if worker_rank not in (None, 0):
        return
    ws = ""
    if worker_world_size is not None:
        ws = f" group_world_size={worker_world_size}"
    suffix = f" | {extra}" if extra else ""
    msg = (
        f"{MODEL_LOAD_TAG} role={role}{ws} path={model_path} "
        f"model_name={model_name}{suffix}"
    )
    print(msg, flush=True)


def log_embodied_driver_inventory(cfg: "DictConfig") -> None:
    """Print how many logical full-model loads the embodied layout implies (driver process)."""
    from rlinf.scheduler import Cluster
    from rlinf.utils.placement import HybridComponentPlacement

    cluster = Cluster(num_nodes=cfg.cluster.num_nodes)
    placement = HybridComponentPlacement(cfg, cluster)
    n_actor = placement.get_world_size("actor")
    n_rollout = placement.get_world_size("rollout")
    n_env = placement.get_world_size("env")
    adv = cfg.algorithm.get("adv_type", "")
    opd = adv == "embodied_opd"

    lines = [
        f"{MODEL_LOAD_TAG} Driver inventory (from cluster.component_placement GPU lists):",
        f"  Rollout workers: {n_rollout} → typically **{n_rollout} full inference weight load(s)** "
        f"(`rollout.model_dir`, one HF module per rollout rank).",
        f"  Actor workers: {n_actor} → **one FSDP-sharded trainable policy** across these ranks "
        f"(each rank participates in `get_model` + shard; not {n_actor} independent full copies in GPU RAM "
        f"when using full_shard).",
        f"  Env workers: {n_env} → **no** policy checkpoint load.",
    ]
    if opd:
        lines.append(
            f"  OPD (`adv_type=embodied_opd`): **+ up to {n_actor} full teacher load(s)** "
            f"(one unfrozen HF+PEFT module per actor rank, lazy on first advantage compute)."
        )
    else:
        lines.append(
            "  OPD teacher: not used (adv_type is not embodied_opd)."
        )
    lines.append(
        f"{MODEL_LOAD_TAG} Rough peak **full-weight** copies if everything resident: "
        f"~{n_rollout} (rollout)"
        + (f" + {n_actor} (OPD teacher)" if opd else "")
        + "; actor is sharded, not counted as N full copies."
    )
    print("\n".join(lines), flush=True)
