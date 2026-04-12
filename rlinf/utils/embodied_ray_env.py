# Copyright 2025 The RLinf Authors.
#
# Shared Ray driver environment for train_embodied_agent and eval_embodied_agent.
# Both use setup_ray_driver_resources + apply_ray_object_store_from_config (same as training).
# Optional helpers at the bottom are for manual troubleshooting (env exports), not used by default.

from __future__ import annotations

import os
from typing import Any


def setup_ray_driver_resources(*, cap_gpus_to_one: bool = False) -> None:
    """Set RLINF_RAY_NUM_CPUS / optionally RLINF_RAY_NUM_GPUS before Cluster() / ray.init.

    Ray otherwise auto-detects all host CPUs/GPUs, which can overload shared nodes; see
    cluster._ray_init_resource_kwargs.

    train_embodied_agent and eval_embodied_agent pass cap_gpus_to_one=False. Set
    RLINF_RAY_NUM_GPUS if you need a specific Ray GPU reservation.
    """
    if "RLINF_RAY_NUM_CPUS" not in os.environ:
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        if slurm_cpus and str(slurm_cpus).isdigit():
            os.environ["RLINF_RAY_NUM_CPUS"] = str(slurm_cpus)
        else:
            os.environ["RLINF_RAY_NUM_CPUS"] = str(min(32, max(4, os.cpu_count() or 8)))
    if cap_gpus_to_one and "RLINF_RAY_NUM_GPUS" not in os.environ:
        os.environ["RLINF_RAY_NUM_GPUS"] = "1"


def apply_ray_object_store_from_config(cfg: Any) -> int:
    """Mirror train_embodied_agent: set RAY_OBJECT_STORE_MEMORY before Cluster()."""
    ray_memory = int(cfg.cluster.get("ray_object_store_memory", 34359738368))
    os.environ["RAY_OBJECT_STORE_MEMORY"] = str(ray_memory)
    return ray_memory


def apply_eval_ray_object_store(cfg: Any) -> int:
    """Eval-only: cap plasma size so mmap of /dev/shm does not block ray.init() for minutes.

    Default cap 512 MiB. Disable with RLINF_EVAL_RAY_OBJECT_STORE_CAP=0 (use yaml value as-is).
    Raise cap with e.g. RLINF_EVAL_RAY_OBJECT_STORE_CAP=2147483648 (2 GiB).
    """
    raw = int(cfg.cluster.get("ray_object_store_memory", 2147483648))
    cap_s = os.environ.get("RLINF_EVAL_RAY_OBJECT_STORE_CAP", "536870912").strip()
    if cap_s in ("0", "", "none", "None"):
        chosen = raw
    else:
        chosen = min(raw, int(cap_s))
    os.environ["RAY_OBJECT_STORE_MEMORY"] = str(chosen)
    return chosen


def force_eval_ray_hpc_env() -> None:
    """Overwrite env so Hydra/shell cannot leave dashboard on or full runtime_env on."""
    os.environ["RLINF_RAY_INCLUDE_DASHBOARD"] = "0"
    os.environ["RLINF_RAY_SKIP_RUNTIME_ENV"] = "1"
    os.environ["RLINF_RAY_MINIMAL_INIT"] = "1"


def enable_eval_local_ray_only() -> None:
    """Eval-only: start local Ray without ray.init(address='auto') unless user opts out.

    Set RLINF_RAY_LOCAL_ONLY=0 to match training (try connect to existing cluster first).
    """
    v = os.environ.get("RLINF_RAY_LOCAL_ONLY", "1").strip().lower()
    if v in ("0", "false", "no"):
        os.environ.pop("RLINF_RAY_LOCAL_ONLY", None)
    else:
        os.environ["RLINF_RAY_LOCAL_ONLY"] = "1"


def setup_eval_hpc_defaults() -> None:
    """Eval-only: defaults before force_eval_ray_hpc_env() overwrites for a known-good eval stack."""
    os.environ.setdefault("RLINF_RAY_INCLUDE_DASHBOARD", "0")
    os.environ.setdefault("RLINF_RAY_SKIP_RUNTIME_ENV", "1")
    os.environ.setdefault("RLINF_RAY_MINIMAL_INIT", "1")
