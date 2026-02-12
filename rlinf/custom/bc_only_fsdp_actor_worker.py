import gc
import os
from itertools import cycle

import numpy as np
import torch
from omegaconf import DictConfig
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.data import DataLoader
from tqdm import tqdm

from rlinf.custom.libero_trajectory_dataset import LiberoSFTDataset
from rlinf.custom.loss import behavior_cloning_ce_loss
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
from rlinf.models import get_model
from rlinf.models.embodiment.model_utils import compute_action_tokens_from_actions
from rlinf.scheduler import Cluster, Worker
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import append_to_dict
from rlinf.utils.placement import HybridComponentPlacement


def bc_only_forward(
    model,
    bc_batch,
    action_token_len=None,
):
    """
    Forward pass for BC only - no RL batch, no temperature/top-k.

    Args:
        model: The model to use
        bc_batch: BC batch with input_ids, attention_mask, pixel_values, actions
        action_token_len: Number of action tokens (e.g., 56)

    Returns:
        output_dict: Dictionary with intermediate_logits for BC loss
    """
    outputs = model(
        input_ids=bc_batch["input_ids"],
        attention_mask=bc_batch["attention_mask"],
        pixel_values=bc_batch["pixel_values"],
        output_hidden_states=False,
    )

    # Extract action token logits - NO temperature, NO top-k
    logits_tensor = outputs.logits[:, -action_token_len - 1 : -1]  # [B, 56, vocab_size]

    output_dict = {
        "intermediate_logits": logits_tensor,  # Raw logits for BC loss
        "raw_logits": logits_tensor.clone(),
    }

    return output_dict


class BCOnlyFSDPActor(FSDPModelManager, Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        super().__init__(cfg.actor)

        self.cfg = cfg
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        self.device = torch.cuda.current_device()
        world_size = self._world_size
        self.device_mesh = init_device_mesh(
            "cuda", mesh_shape=(world_size,), mesh_dim_names=["fsdp"]
        )

        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        # Initialize BC dataset
        self._init_bc_dataset()

    def _init_bc_dataset(self):
        dataset_path = os.environ.get("LIBERO_REPO_PATH")
        if self._rank == 0:
            print(f"Initializing BC dataset on rank {self._rank}")

        self.bc_dataset = LiberoSFTDataset(
            cfg=self.cfg,
            root_dir=dataset_path,
            demos_per_task=1,
            rank=self._rank,
            world_size=self._world_size,
            use_cached_logits=False,
            logits_type="",
            use_preprocessed=True,
        )

        self.bc_dataloader = cycle(
            DataLoader(
                self.bc_dataset,
                batch_size=self.cfg.actor.micro_batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
                drop_last=True,
            )
        )

        self.bc_iterator = iter(self.bc_dataloader)
        if self._rank == 0:
            print(f"BC dataset initialized: {len(self.bc_dataset)} samples")

    def init_worker(self):
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        else:
            if torch.distributed.get_backend() != "nccl":
                torch.distributed.destroy_process_group()
                torch.distributed.init_process_group(backend="nccl")

        self.setup_model_and_optimizer()

    def model_provider_func(self):
        model = get_model(self.cfg.actor.checkpoint_load_path, self.cfg.actor.model)
        if model is not None:
            return model
        return super().model_provider_func()

    def run_training(self):
        """Run one epoch of BC training."""
        self.model.train()
        self.optimizer.zero_grad()
        rollout_size = 64 * 64

        # Calculate gradient accumulation steps
        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )

        gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )

        bc_coeff = 1.0
        action_token_len = self.model.action_dim * self.model.num_action_chunks

        metrics = {}
        step_count = 0
        for batch_idx in tqdm(
            range(
                rollout_size
                * self.cfg.algorithm.rollout_epoch
                // self.cfg.actor.global_batch_size
            ),
            desc="get loss and metrics",
        ):
            self.optimizer.zero_grad()

            # Accumulate gradients over micro-batches
            for micro_batch_idx in range(gradient_accumulation):
                bc_batch = next(self.bc_iterator)

                # Move to GPU
                for k, v in bc_batch.items():
                    bc_batch[k] = v.to(f"cuda:{int(os.environ['LOCAL_RANK'])}")

                # Forward pass - NO temperature, NO top-k
                output_dict = bc_only_forward(
                    self.model,
                    bc_batch=bc_batch,
                    action_token_len=action_token_len,
                )

                # Compute expert action tokens
                expert_actions_tokens = torch.tensor(
                    compute_action_tokens_from_actions(self.model, bc_batch["actions"]),
                    device=f"cuda:{int(os.environ['LOCAL_RANK'])}",
                )

                # Compute BC loss
                kwargs = {
                    "intermediate_logits": output_dict["intermediate_logits"],
                    "expert_actions_tokens": expert_actions_tokens,
                    "bc_coeff": bc_coeff,
                    "vocab_size": self.model.vocab_size,
                    "n_action_bins": self.model.config.n_action_bins,
                }
                bc_loss, bc_metrics_data = behavior_cloning_ce_loss(**kwargs)

                # Scale loss by gradient accumulation
                loss = bc_loss / gradient_accumulation
                loss.backward()

                # Collect metrics
                bc_metrics_data["bc/loss_unscaled"] = bc_loss.detach().item()
                bc_metrics_data["total/loss"] = loss.detach().item()
                append_to_dict(metrics, bc_metrics_data)

            torch.cuda.empty_cache()

            # Clip gradients and step
            grad_norm = self.model.clip_grad_norm_(
                max_norm=self.cfg.actor.optim.clip_grad
            )
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Log learning rate and grad norm
            data = {
                "actor/grad_norm": grad_norm.detach().item(),
                "actor/lr": self.optimizer.param_groups[0]["lr"],
            }
            append_to_dict(metrics, data)

            step_count += 1

        # Aggregate metrics across all batches
        mean_metric_dict = {key: np.mean(value) for key, value in metrics.items()}
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )

        self.optimizer.zero_grad()
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()

        return mean_metric_dict

    def save_checkpoint(self, save_base_path, step):
        """Save model checkpoint."""
        torch.distributed.barrier()
        is_lora = self.cfg.actor.model.get("is_lora", False)
        model = self.model

        optim_state = self.get_optimizer_state_dict()

        if is_lora:
            from peft import get_peft_model_state_dict
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(
                model, StateDictType.FULL_STATE_DICT, save_policy
            ):
                cpu_state = get_peft_model_state_dict(model, model.state_dict())

                if self._rank == 0:
                    os.makedirs(save_base_path, exist_ok=True)
                    print(f"Saving model checkpoint to {save_base_path}")
                    torch.save(optim_state, os.path.join(save_base_path, "optim.pt"))
                    torch.save(
                        cpu_state, os.path.join(save_base_path, "adapter_model.bin")
                    )
                    model.peft_config["default"].save_pretrained(save_base_path)
        else:
            model_state = self.get_model_state_dict()
            if self._rank == 0:
                os.makedirs(save_base_path, exist_ok=True)
                torch.save(model_state, os.path.join(save_base_path, "model.pt"))
                torch.save(optim_state, os.path.join(save_base_path, "optim.pt"))

        torch.distributed.barrier()
