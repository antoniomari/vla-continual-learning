# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import signal
import sys
import time
import warnings
from dataclasses import dataclass
from importlib.metadata import version
from typing import TYPE_CHECKING, Dict, List, Optional, Type

import ray
import ray.util.scheduling_strategies
from packaging import version as vs
from ray._private import ray_logging
from ray.actor import ActorHandle
from ray.util.state import list_actors

from .accelerator import Accelerator, AcceleratorType

ray_version = version("ray")
assert vs.parse(ray_version) >= vs.parse("2.47.0"), (
    "Ray version 2.47.0 or higher is required. Run pip install ray[default]==2.47.0"
)


def _rlinf_phase_print(msg: str) -> None:
    """Log to stderr so lines stay visible next to Ray (Hydra/tee can delay stdout)."""
    print(msg, file=sys.stderr, flush=True)


def _ray_init_resource_kwargs() -> Dict[str, int]:
    """Optional caps for ray.init on shared HPC nodes.

    If unset, Ray auto-detects all host CPUs/GPUs, which can spawn a huge default
    worker pool and stress memory (worker registration EOF / OOM). Set:

    - RLINF_RAY_NUM_CPUS to your Slurm allocation (e.g. $SLURM_CPUS_PER_TASK)
    - RLINF_RAY_NUM_GPUS to GPUs you own (often 1)
    """
    kwargs: Dict[str, int] = {}
    if os.environ.get("RLINF_RAY_NUM_CPUS"):
        kwargs["num_cpus"] = int(os.environ["RLINF_RAY_NUM_CPUS"])
    if os.environ.get("RLINF_RAY_NUM_GPUS"):
        kwargs["num_gpus"] = int(os.environ["RLINF_RAY_NUM_GPUS"])
    return kwargs


def _ray_init_extras() -> Dict:
    """Optional ray.init kwargs (dashboard, minimal init for eval, etc.)."""
    include_dash = os.environ.get("RLINF_RAY_INCLUDE_DASHBOARD", "1") == "1"
    extra: Dict = {
        # RLINF_RAY_INCLUDE_DASHBOARD=0 avoids binding port 8265 (often problematic on HPC).
        "include_dashboard": include_dash,
    }
    # Eval entrypoint sets RLINF_RAY_MINIMAL_INIT=1 to reduce work inside ray.init (logging setup, etc.).
    if os.environ.get("RLINF_RAY_MINIMAL_INIT", "0") == "1":
        extra["include_dashboard"] = False
        extra["configure_logging"] = False
    return extra


def _ray_init_object_store_kw() -> Dict:
    """Pass object_store_memory explicitly (plasma mmap can block if env is ignored)."""
    v = os.environ.get("RAY_OBJECT_STORE_MEMORY")
    if not v:
        return {}
    try:
        return {"object_store_memory": int(v)}
    except ValueError:
        return {}


def _ray_runtime_env_payload() -> Optional[Dict]:
    """Worker env injection for ray.init.

    Passing runtime_env={\"env_vars\": dict(os.environ)} can take a very long time or appear
    hung inside ray.init() on HPC (huge Slurm/Modules environments). Eval sets
    RLINF_RAY_SKIP_RUNTIME_ENV=1 to omit it; workers still inherit many vars from the parent.
    Set RLINF_RAY_SKIP_RUNTIME_ENV=0 to restore previous behavior.
    """
    if os.environ.get("RLINF_RAY_SKIP_RUNTIME_ENV", "0") == "1":
        return None
    return {"env_vars": dict(os.environ)}

if TYPE_CHECKING:
    from .worker import Worker


@dataclass
class NodeInfo:
    """Information about a node in the cluster."""

    node_rank: str
    """Rank of the node in the cluster."""

    ray_id: str
    """Ray's unique identifier for the node."""

    node_ip: str
    """IP address of the node."""

    accelerator_type: AcceleratorType
    """Type of accelerator available on the node."""

    num_accelerators: int
    """Number of accelerators available on the node."""

    num_cpus: int
    """Number of CPUs available on the node."""


class Cluster:
    """A singleton class that manages the cluster resources for Ray workers."""

    SYS_NAME = "RLinf"
    NAMESPACE = SYS_NAME
    LOGGING_LEVEL = "INFO"
    TIMEOUT_WARN_TIME = 60000

    @classmethod
    def find_free_port(cls):
        """Find a free port on the node."""
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    @classmethod
    def has_initialized(cls):
        """Check if the cluster has been initialized."""
        return hasattr(cls, "_instance") and cls._instance is not None

    def __new__(cls, *args, **kwargs):  # noqa D417
        """Create a singleton class that manages the cluster resources for Ray workers."""
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
            cls._instance._has_initialized = False
        return cls._instance

    def __init__(self, num_nodes: Optional[int] = None):
        """Initialize the cluster.

        Args:
            num_nodes (int): The number of nodes in the cluster. When you wish to acquire the cluster instance in a processes other than the main driver process, do not pass this argument. Instead, use the `Cluster()` constructor without arguments.
        """
        if self._has_initialized:
            return
        if num_nodes is not None:
            self._ray_instance_count = 0
            self._init_and_launch_managers(num_nodes)
        else:
            self._init_from_existing_managers()
        self._has_initialized = True

    def _init_and_launch_managers(self, num_nodes: int):
        assert num_nodes > 0, "num_nodes must be greater than 0."

        # Add logger
        self._logger = logging.getLogger(Cluster.SYS_NAME)
        self._logger.setLevel(Cluster.LOGGING_LEVEL)
        self._logger.propagate = False
        for handler in self._logger.handlers:
            self._logger.removeHandler(handler)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="[%(levelname)s %(asctime)s %(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)

        self._num_nodes = num_nodes
        self._set_default_env_vars()

        if ray.is_initialized():
            if self._ray_instance_count > 0:
                # For reinit Ray to switch namespace
                ray.shutdown()
            else:
                # Initializing Ray before us interferes with the namespace and logging level settings.
                raise RuntimeError(
                    "You have initialized Ray before creating the Cluster instance. This may be due to calling ray.init or creating certain Ray objects like Ray Queue before instantiating the Cluster class. Please ensure that the Cluster class is instantiated before Ray is initialized because it will interfere with our Ray namespace and logging settings."
                )

        # NOTE: Add os.environ variables to the worker environment.
        # When ray cluster has been started via `ray start` before running the Python script, ray will only capture the environment variables exported before `ray start` and ignore all subsequently exported environment variables.
        # To handle this, we need to manually pass the environment variables to Ray when initializing the cluster.
        # Any env vars conflicting with Worker env vars will be overwritten by Worker.
        if "RAY_DEDUP_LOGS" not in os.environ:
            # Default disabling deduplication of logs to ensure all logs are printed.
            ray_logging.RAY_DEDUP_LOGS = 0
        # Ensure Ray uses a user-writable temp directory to avoid /tmp permission issues
        self._ensure_ray_tmpdir()

        _ray_kw = _ray_init_resource_kwargs()
        _extra = _ray_init_extras()
        _os_mem = _ray_init_object_store_kw()
        _rt = _ray_runtime_env_payload()
        # Single-node eval / local jobs: set RLINF_RAY_LOCAL_ONLY=1 so we never call
        # ray.init(address="auto"), which can hang on HPC when RAY_ADDRESS points at a
        # stale or unreachable cluster head (num_envs does not affect this).
        _local_only = os.environ.get("RLINF_RAY_LOCAL_ONLY", "") == "1"
        _rlinf_phase_print("[RLinf] ray.init() starting ...")
        if _rt is None:
            _rlinf_phase_print(
                "[RLinf] ray.init: skipping runtime_env (RLINF_RAY_SKIP_RUNTIME_ENV=1)"
            )
        if _local_only:
            os.environ.pop("RAY_ADDRESS", None)
            _init = {
                "logging_level": Cluster.LOGGING_LEVEL,
                "namespace": Cluster.NAMESPACE,
                **_ray_kw,
                **_extra,
                **_os_mem,
            }
            if _rt is not None:
                _init["runtime_env"] = _rt
            _rlinf_phase_print(
                f"[RLinf] ray.init local kwargs keys: {list(_init.keys())} "
                f"include_dashboard={_init.get('include_dashboard', 'n/a')} "
                f"object_store_memory={_init.get('object_store_memory', 'default')}"
            )
            ray.init(**_init)
        else:
            try:
                # First try to connect to an existing Ray cluster
                _init = {
                    "address": "auto",
                    "logging_level": Cluster.LOGGING_LEVEL,
                    "namespace": Cluster.NAMESPACE,
                    **_ray_kw,
                    **_extra,
                    **_os_mem,
                }
                if _rt is not None:
                    _init["runtime_env"] = _rt
                ray.init(**_init)
            except (ConnectionError, PermissionError, OSError, ValueError):
                _init = {
                    "logging_level": Cluster.LOGGING_LEVEL,
                    "namespace": Cluster.NAMESPACE,
                    **_ray_kw,
                    **_extra,
                    **_os_mem,
                }
                if _rt is not None:
                    _init["runtime_env"] = _rt
                ray.init(**_init)
        # Ray may log "Started a local Ray instance" before ray.init() returns; our code runs
        # only after ray.init() completes. The next line is the first RLinf feedback after that.
        _rlinf_phase_print("[RLinf] ray.init() returned.")
        try:
            _n_after_init = len(ray.nodes())
        except Exception as e_exc:
            _rlinf_phase_print(f"[RLinf] ray.nodes() failed right after init: {e_exc}")
            raise
        _rlinf_phase_print(
            f"[RLinf] ray.nodes() count={_n_after_init}; need {self._num_nodes} "
            "(waiting if below threshold) ..."
        )

        # Wait for the cluster to be ready (can spin forever on misconfigured clusters).
        _wait_timeout = float(os.environ.get("RLINF_RAY_NODE_WAIT_TIMEOUT", "300"))
        _poll_sleep = float(os.environ.get("RLINF_RAY_NODE_POLL_SLEEP_SEC", "0.1"))
        _wait_start = time.time()
        _last_print = 0.0
        while len(ray.nodes()) < self._num_nodes:
            n = len(ray.nodes())
            elapsed = time.time() - _wait_start
            if elapsed > _wait_timeout:
                raise RuntimeError(
                    f"[RLinf] Timed out after {int(_wait_timeout)}s waiting for Ray nodes "
                    f"(have {n}, need {self._num_nodes}). "
                    "Check $RAY_TMPDIR/session_latest/logs or /tmp/ray/session_latest/logs. "
                    "Try: RLINF_RAY_INCLUDE_DASHBOARD=0, lower cluster.ray_object_store_memory, "
                    "or verify the node is not out of RAM/shm (df -h /dev/shm)."
                )
            now = time.time()
            if now - _last_print >= 5.0:
                remaining = max(0.0, _wait_timeout - elapsed)
                _rlinf_phase_print(
                    f"[RLinf] Waiting for Ray nodes: {n}/{self._num_nodes} "
                    f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s until timeout)"
                )
                self._logger.warning(
                    f"Waiting for {self._num_nodes} nodes to be ready, currently {n} nodes available."
                )
                _last_print = now
            time.sleep(_poll_sleep)

        self._nodes: List[NodeInfo] = []
        for node in ray.nodes():
            accelerator_type, num_accelerators = (
                Accelerator.get_node_accelerator_type_and_num(node)
            )
            self._nodes.append(
                NodeInfo(
                    node_rank=0,
                    ray_id=node["NodeID"],
                    node_ip=node["NodeManagerAddress"],
                    accelerator_type=accelerator_type,
                    num_accelerators=num_accelerators,
                    num_cpus=int(node["Resources"].get("CPU", 0)),
                )
            )

        # Sort nodes first by accelerator type, then by IP
        nodes_group_by_accel_type: Dict[AcceleratorType, List[NodeInfo]] = {
            accel_type: [] for accel_type in AcceleratorType
        }
        for node in self._nodes:
            nodes_group_by_accel_type[node.accelerator_type].append(node)
        for accel_type in nodes_group_by_accel_type.keys():
            nodes_group_by_accel_type[accel_type].sort(key=lambda x: x.node_ip)
        self._nodes = [
            node for nodes in nodes_group_by_accel_type.values() for node in nodes
        ]

        # Handle num_nodes configuration mismatch with actual node number
        if len(self._nodes) > self._num_nodes:
            warnings.warn(
                f"The cluster is initialized with {self._num_nodes} nodes, but detected {len(self._nodes)} nodes have joined the ray cluster. So only the first {self._num_nodes} nodes are used."
            )
            self._nodes = self._nodes[: self._num_nodes]

        _cluster_ready_msg = (
            f"{Cluster.SYS_NAME} is running on a cluster with {len(self._nodes)} node{'s' if len(self._nodes) > 1 else ''} "
            f"and {self.num_accelerators_in_cluster} accelerator{'s' if self.num_accelerators_in_cluster > 1 else ''}. "
            f"The nodes' details are: {self._nodes}"
        )
        self._logger.info(_cluster_ready_msg)
        sys.stderr.flush()

        # Launch managers
        from .manager import (
            CollectiveManager,
            NodeManager,
            WorkerManager,
        )

        try:
            self._worker_manager = (
                ray.remote(WorkerManager)
                .options(name=WorkerManager.MANAGER_NAME)
                .remote()
            )
            self._coll_manager = (
                ray.remote(CollectiveManager)
                .options(name=CollectiveManager.MANAGER_NAME)
                .remote()
            )
            self._node_manager = (
                ray.remote(NodeManager)
                .options(name=NodeManager.MANAGER_NAME)
                .remote(self._nodes)
            )
        except ValueError:
            # If the WorkerManager is already running, we need to switch the namespace
            self._ray_instance_count += 1
            Cluster.NAMESPACE = f"RLinf_{self._ray_instance_count}"
            return self._init_and_launch_managers(num_nodes)

        def signal_handler(sig, frame):
            # Exit the main process if SIGUSR1 is received, which is sent by the worker group when an exception occurs.
            sys.stdout.flush()
            sys.stderr.flush()

            # Try to clean up actors, but gracefully handle cases with multiple Ray instances
            try:
                alive_actors = list_actors(
                    filters=[
                        ("STATE", "=", "ALIVE"),
                        ("RAY_NAMESPACE", "=", Cluster.NAMESPACE),
                    ]
                )
                for actor_state in alive_actors:
                    try:
                        actor = ray.get_actor(actor_state.name, namespace=Cluster.NAMESPACE)
                        ray.kill(actor, no_restart=True)
                    except Exception:
                        # Ignore errors killing individual actors
                        pass
            except Exception:
                # If we can't list actors (e.g., multiple Ray instances), skip cleanup
                # Actors will be cleaned up when Ray shuts down
                pass

            if ray.is_initialized():
                # Mimic ray's sleep before shutdown to ensure log messages are flushed
                time.sleep(0.5)
                ray.shutdown(_exiting_interpreter=True)
            print("Exiting main process due to a failure upon worker execution.")
            exit(-1)

        signal.signal(signal.SIGUSR1, signal_handler)

    def _init_from_existing_managers(self):
        # Ensure Ray uses a user-writable temp directory to avoid /tmp permission issues
        self._ensure_ray_tmpdir()
        if not ray.is_initialized():
            _ray_kw = _ray_init_resource_kwargs()
            _extra = _ray_init_extras()
            _os_mem = _ray_init_object_store_kw()
            _rt = _ray_runtime_env_payload()
            try:
                _init = {
                    "address": "auto",
                    "namespace": Cluster.NAMESPACE,
                    "logging_level": Cluster.LOGGING_LEVEL,
                    **_ray_kw,
                    **_extra,
                    **_os_mem,
                }
                if _rt is not None:
                    _init["runtime_env"] = _rt
                ray.init(**_init)
            except (ConnectionError, PermissionError, OSError, ValueError):
                _init = {
                    "namespace": Cluster.NAMESPACE,
                    "logging_level": Cluster.LOGGING_LEVEL,
                    **_ray_kw,
                    **_extra,
                    **_os_mem,
                }
                if _rt is not None:
                    _init["runtime_env"] = _rt
                ray.init(**_init)

        from .manager.node_manager import NodeManager

        self._node_manager = NodeManager.get_proxy()
        self._nodes = self._node_manager.get_nodes()
        self._num_nodes = len(self._nodes)

    def _set_default_env_vars(self):
        """Set default environment variables for the system."""
        env_var_list = ["CATCH_FAILURE", "LOG_LEVEL", "TIMEOUT"]
        system_name = Cluster.SYS_NAME.upper()
        for env_var in env_var_list:
            env_var = f"{system_name}_{env_var}"
            if env_var not in os.environ:
                if env_var == f"{system_name}_CATCH_FAILURE":
                    os.environ[env_var] = "0"
                elif env_var == f"{system_name}_LOG_LEVEL":
                    os.environ[env_var] = "INFO"
                elif env_var == f"{system_name}_TIMEOUT":
                    os.environ[env_var] = "180"

    def _ensure_ray_tmpdir(self):
        """Ensure RAY_TMPDIR points to a user-writable directory and exists.

        Ray by default uses /tmp/ray. On shared clusters this can lead to
        PermissionError if that path is owned by another user. We redirect
        Ray's temp directory to a per-user path under the project scratch
        space if not already set.
        """
        if os.environ.get("RAY_TMPDIR"):
            # Respect an explicitly provided path; just make sure it exists.
            tmp_dir = os.environ["RAY_TMPDIR"]
        else:
            user = os.environ.get("USER") or os.environ.get("LOGNAME") or "unknown"
            # Use a short path: AF_UNIX socket paths must be <= 107 bytes. Putting sessions under
            # a long repo path (e.g. /cluster/home/.../vla-continual-learning/.ray_tmp/...) exceeds
            # that limit and ray.init fails with validate_socket_filename.
            tmp_dir = os.path.join(os.sep, "tmp", f"ray_{user}")
            os.environ["RAY_TMPDIR"] = tmp_dir

        try:
            os.makedirs(tmp_dir, mode=0o700, exist_ok=True)
        except Exception:
            # As a last resort, do not crash here; leave to Ray to error with context
            pass

        # Align Python/tmp-based temp resolution with Ray temp to avoid /tmp usage
        for k in ("TMPDIR", "TMP", "TEMP"):
            if not os.environ.get(k):
                os.environ[k] = tmp_dir

    @staticmethod
    def get_sys_env_var(env_var: str, default: Optional[str] = None) -> Optional[str]:
        """Get the system environment variable for the cluster."""
        system_name = Cluster.SYS_NAME.upper()
        env_var = f"{system_name}_{env_var}"
        return os.environ.get(env_var, default)

    @property
    def num_nodes(self):
        """Get the number of nodes in the cluster."""
        return self._num_nodes

    @property
    def num_accelerators_in_cluster(self):
        """Get the number of accelerators in the cluster."""
        return sum(node.num_accelerators for node in self._nodes)

    @property
    def node_accelerator_ids(self) -> List[List[int]]:
        """Get the global accelerator IDs for each node in the cluster."""
        node_start_accel_id = 0
        node_accel_ids = []
        for node in self._nodes:
            node_accel_ids.append(
                list(
                    range(
                        node_start_accel_id, node_start_accel_id + node.num_accelerators
                    )
                )
            )
            node_start_accel_id += node.num_accelerators
        return node_accel_ids

    def get_node_id_from_accel_id(self, accel_id: int) -> int:
        """Get the node ID from the global accelerator ID.

        Args:
            accel_id (int): The global accelerator ID.

        Returns:
            int: The node ID.
        """
        for i, ids in enumerate(self.node_accelerator_ids):
            if accel_id in ids:
                return i
        raise ValueError(f"Accelerator ID {accel_id} not found in any node.")

    def get_node_num_accelerators(self, node_id: int) -> int:
        """Get the number of accelerators in a specific node.

        Args:
            node_id (int): The ID of the node.

        Returns:
            int: The number of accelerators in the node.
        """
        if node_id < 0 or node_id >= self._num_nodes:
            raise ValueError(
                f"Invalid node_id: {node_id}. Must be between 0 and {self._num_nodes - 1}."
            )
        return self._nodes[node_id].num_accelerators

    def global_accel_id_to_local_accel_id(self, accel_id: int):
        """Get the local accelerator ID from the global accelerator ID.

        Args:
            accel_id (int): The global accelerator ID.

        Returns:
            int: The local accelerator ID.
        """
        node_id = self.get_node_id_from_accel_id(accel_id)
        node_accel_ids = self.node_accelerator_ids[node_id]
        assert accel_id in node_accel_ids, (
            f"Accelerator ID {accel_id} not found in node {node_id}."
        )
        return node_accel_ids.index(accel_id)

    def get_node_info(self, node_id: int):
        """Get the NodeInfo of a specific node rank."""
        return self._nodes[node_id]

    def get_node_ip(self, node_id: int) -> str:
        """Get the IP address of a specific node by its ID. Note that this is not the ray NodeID but the index of node in the cluster."""
        return self._nodes[node_id].node_ip

    def allocate(
        self,
        cls: Type["Worker"],
        worker_name: str,
        node_id: int,
        env_vars: dict,
        cls_args: List = [],
        cls_kwargs: dict = {},
    ) -> ActorHandle:
        """Allocate a ray remote class instance on a specific node and local rank.

        Args:
            cls (Type[Worker]): The class to allocate.
            worker_name (str): The name of the worker.
            node_id (int): The ID of the node to allocate on.
            env_vars (dict): Environment variables to set for the worker.
            cls_args (List): Positional arguments to pass to the class constructor.
            cls_kwargs (dict): Keyword arguments to pass to the class constructor.

        Returns:
            ray.ObjectRef: A reference to the allocated remote class instance.

        """
        if node_id < 0 or node_id >= self._num_nodes:
            raise ValueError(
                f"Invalid node_id: {node_id}. Must be between 0 and {self._num_nodes - 1}."
            )

        node = self._nodes[node_id]
        remote_cls = ray.remote(cls)

        options = {
            "runtime_env": {"env_vars": env_vars},
            "name": worker_name,
            "scheduling_strategy": ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=node.ray_id,
                soft=False,
            ),
        }

        return remote_cls.options(**options).remote(*cls_args, **cls_kwargs)
