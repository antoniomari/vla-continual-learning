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

import hydra
import torch.multiprocessing as mp

from rlinf.config import validate_cfg
from rlinf.custom.bc_only_fsdp_actor_worker import BCOnlyFSDPActor
from rlinf.custom.bc_runner import BCOnlyRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement

mp.set_start_method("spawn", force=True)


@hydra.main(
    version_base="1.1", config_path="config", config_name="maniskill_ppo_openvlaoft"
)
def main(cfg) -> None:
    cfg = validate_cfg(cfg)

    # Export Ray object store memory from config to environment variable
    # This allows start_ray.sh to use the configured value
    import os

    # ray_memory = cfg.cluster.get("ray_object_store_memory", 461708984320)
    # os.environ["RAY_OBJECT_STORE_MEMORY"] = str(ray_memory)

    cluster = Cluster(num_nodes=cfg.cluster.num_nodes)
    component_placement = HybridComponentPlacement(cfg, cluster)

    # Create actor worker group
    actor_placement = component_placement.get_strategy("actor")
    actor_group = BCOnlyFSDPActor.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
    )

    runner = BCOnlyRunner(cfg=cfg, actor=actor_group)

    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
