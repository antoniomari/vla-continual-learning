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
from omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.custom.logits_precompute_worker import LogitsPrecomputeWorker
from rlinf.custom.rlds_logits_precompute_worker import RLDSLogitsPrecomputeWorker
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement

mp.set_start_method("spawn", force=True)


@hydra.main(
    version_base="1.1", config_path="config", config_name="maniskill_ppo_openvlaoft"
)
def main(cfg) -> None:
    cfg = validate_cfg(cfg)

    if "logits_precompute" not in cfg:
        # Get paths from environment or set defaults

        import os

        OmegaConf.set_struct(cfg, False)
        cfg.logits_precompute = {
            "root_dir": os.getenv("LIBERO_REPO_PATH"),
            "output_dir": None,
            "demos_per_task": 1,
        }

    cfg.runner.only_eval = True
    print(cfg)

    # Export Ray object store memory from config to environment variable
    # This allows start_ray.sh to use the configured value
    import os

    ray_memory = cfg.cluster.get("ray_object_store_memory", 461708984320)
    os.environ["RAY_OBJECT_STORE_MEMORY"] = str(ray_memory)

    cluster = Cluster(num_nodes=cfg.cluster.num_nodes)
    component_placement = HybridComponentPlacement(cfg, cluster)

    # Create rollout worker group
    rollout_placement = component_placement.get_strategy("rollout")
    rollout_group = RLDSLogitsPrecomputeWorker.create_group(cfg).launch(
        cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
    )
    rollout_group.init_worker().wait()
    rollout_group.process_all_tasks().wait()

    # rollout_group = LogitsPrecomputeWorker.create_group(cfg).launch(
    #     cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
    # )
    # rollout_group.init_worker().wait()
    # rollout_group.process_all_files().wait()


if __name__ == "__main__":
    main()
