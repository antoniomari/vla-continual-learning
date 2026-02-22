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

import os

import torch
import torch.optim as optim
from omegaconf import DictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from transformers import AutoModelForCausalLM

from rlinf.config import torch_dtype_from_precision
from rlinf.hybrid_engines.fsdp.utils import (
    get_fsdp_wrap_policy,
    init_fn,
)
from rlinf.utils.utils import clear_memory


class FSDPModelManager:
    """
    FSDP Model Manager for RL training.
    Uses FSDP for all models (including simple_cnn with NO_SHARD strategy).
    """

    def __init__(self, cfg: DictConfig):
        self._cfg = cfg
        self.torch_dtype = torch_dtype_from_precision(self._cfg.model.precision)

        assert (
            self.torch_dtype == torch.float16 or self.torch_dtype == torch.bfloat16
        ), (
            f"Precision {self._cfg.model.precision} is not supported, only support bf16 and fp16."
        )

    def model_provider_func(self) -> torch.nn.Module:
        raise Exception("Should not reach here.")
        if self._cfg.model.get("gptq_model", False):
            from auto_gptq import AutoGPTQForCausalLM

            model_wrapper = AutoGPTQForCausalLM.from_quantized(
                self._cfg.model.model_path, device="cuda:0", use_triton=True
            )
            model = model_wrapper.model
        elif self._cfg.model.get("load_in_8bit", False):
            model = AutoModelForCausalLM.from_pretrained(
                self._cfg.model.model_path,
                device_map=self._cfg.model.get("device_map", "auto"),
                load_in_8bit=True,
            )
        else:
            # default load in float16
            model = AutoModelForCausalLM.from_pretrained(
                self._cfg.model.model_path,
                torch_dtype=self.torch_dtype,
                device_map=self._cfg.model.get("device_map", "auto"),
                trust_remote_code=True,
                use_safetensors=self._cfg.model.get("use_safetensors", False),
            )
            if torch.cuda.is_available():
                model = model.cuda()
            if self.torch_dtype == torch.float16:
                model = model.half()

        return model

    def setup_model_and_optimizer(self):
        """Setup model and optimizer."""
        module = self.model_provider_func()

        # Enable gradient checkpointing if the model supports it (transformer models)
        # Simple models like SimpleCNNPolicy don't have this method
        if hasattr(module, "gradient_checkpointing_enable"):
            module.gradient_checkpointing_enable()

        # Use FSDP for all models (including simple_cnn with NO_SHARD).
        # FSDP is preferred over DDP because it forwards attribute access via __getattr__,
        # which is needed by the training loop (e.g., self.model.action_dim).
        # Note: For CNN models, BatchNorm is frozen (eval mode) during training,
        # so buffers don't accumulate during backward pass. Keeping buffer_dtype
        # matching param_dtype avoids FSDP dtype mismatch errors.
        mixed_precision = MixedPrecision(
            param_dtype=self.torch_dtype,
            reduce_dtype=self.torch_dtype,
            buffer_dtype=self.torch_dtype,
        )

        if self._cfg.model.sharding_strategy == "full_shard":
            sharding_strategy = ShardingStrategy.FULL_SHARD
        elif self._cfg.model.sharding_strategy == "shard_grad_op":
            sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
        else:
            sharding_strategy = ShardingStrategy.NO_SHARD
        # For CNN models, skip importing utils.py entirely (avoids PrismaticProjector
        # import chain that triggers init_process_group at module-load time).
        # For VLA models, import get_fsdp_wrap_policy lazily — by this point
        # init_process_group has already been called, so the prismatic import
        # chain's PartialState() is a harmless no-op.
        is_cnn = self._cfg.model.get("model_name") == "simple_cnn"
        if is_cnn:
            auto_wrap_policy = None
        else:
            auto_wrap_policy = get_fsdp_wrap_policy(
                module=module, config=None, is_lora=self._cfg.model.is_lora
            )

        betas = (self._cfg.optim.adam_beta1, self._cfg.optim.adam_beta2)

        self.model = FSDP(
            module,
            param_init_fn=init_fn,
            use_orig_params=True,  ### CUSTOM: False
            auto_wrap_policy=auto_wrap_policy,
            device_id=int(os.environ["LOCAL_RANK"]),
            sharding_strategy=sharding_strategy,  # zero3
            mixed_precision=mixed_precision,
            sync_module_states=True,
        )

        param_groups = self._build_param_groups(betas)
        self.optimizer = optim.AdamW(param_groups)

    def _build_param_groups(self, betas):
        use_component_lrs = self._cfg.optim.get("use_component_lrs", False)
        use_layer_decay = self._cfg.optim.get("use_layer_decay", False)

        # build original param groups
        if not (use_component_lrs or use_layer_decay):
            return self._build_default_param_groups(betas)

        param_dict = self._categorize_parameters()

        param_groups = []
        if use_component_lrs:
            param_groups.extend(self._build_component_groups(param_dict, betas))
        elif use_layer_decay:
            raise NotImplementedError()

        return param_groups

    def _build_component_groups(self, param_dict, betas):
        """Build parameter groups based on model components."""
        groups = []

        # Component learning rates from config
        component_lrs = {
            "vision_lora": self._cfg.optim.get("vision_lora_lr", self._cfg.optim.lr),
            "llm_lora": self._cfg.optim.get("llm_lora_lr", self._cfg.optim.lr),
            "lm_head_lora": self._cfg.optim.get("lm_head_lora_lr", self._cfg.optim.lr),
        }

        for component, params in param_dict.items():
            lr = component_lrs[component]
            groups.append(
                {"params": params, "lr": lr, "betas": betas, "name": component}
            )

            if self._rank == 0:
                print(f"Component {component}: {len(params)} params, LR={lr}")

        return groups

    def _categorize_parameters(self):
        param_dict = {
            # "value_head": [],
            "vision_lora": [],
            "llm_lora": [],
            "lm_head_lora": [],
        }

        # Store layer info for layer-wise decay
        # param_layer_info = {}  # param_id -> (param, name, layer_num)

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # param_id = id(param)
            # layer_num = self._extract_layer_number(name)
            # param_layer_info[param_id] = (param, name, layer_num)

            # Categorize by component
            if "lora_" in name:
                if self._is_vision_param(name):
                    param_dict["vision_lora"].append(param)
                elif self._is_llm_param(name):
                    param_dict["llm_lora"].append(param)
                elif self._is_lm_head(name):
                    param_dict["lm_head_lora"].append(param)
                else:
                    raise NotImplementedError()

        # Store for layer decay if needed
        # self._param_layer_info = param_layer_info
        return param_dict

    # def _extract_layer_number(self, name):
    #     """Extract layer number from parameter name."""
    #     if "layers." in name:
    #         try:
    #             return int(name.split("layers.")[1].split(".")[0])
    #         except:
    #             pass
    #     elif "layer." in name:
    #         try:
    #             return int(name.split("layer.")[1].split(".")[0])
    #         except:
    #             pass
    #     return None

    def _is_vision_param(self, name):
        """Check if parameter belongs to vision backbone."""
        return "vision_backbone" in name or ".projector." in name

    def _is_llm_param(self, name):
        """Check if parameter belongs to LLM."""
        return "language_model" in name and "lm_head" not in name

    def _is_lm_head(self, name):
        """Check if parameter belongs to LM head."""
        return "lm_head" in name

    def _build_default_param_groups(self, betas):
        # NOTE: Currently we assume that only the value head contains "value_head" in its name.
        # The value head only serves for value prediction in RL algorithms like PPO.
        param_groups = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if "value_head" not in n and p.requires_grad
                ],
                "lr": self._cfg.optim.lr,
                "betas": betas,
            },
        ]

        # Always include value_head params in the optimizer when they exist.
        # FSDP requires all managed parameters to be in the optimizer for
        # state dict operations (e.g. FSDP.optim_state_dict).
        # When vh_mode is active, use value_lr; otherwise use the regular lr.
        # (When vh_mode is "none", no gradients flow to value_head so
        #  the weights won't actually be updated.)
        value_head_params = [
            p
            for n, p in self.model.named_parameters()
            if "value_head" in n and p.requires_grad
        ]
        if value_head_params:
            vh_lr = (
                self._cfg.optim.value_lr
                if self._cfg.model.vh_mode in ["a", "a0", "a6"]
                else self._cfg.optim.lr
            )
            param_groups.append(
                {
                    "params": value_head_params,
                    "lr": vh_lr,
                    "betas": betas,
                }
            )

        return param_groups

    def get_model_state_dict(self):
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT):
            state_dict = self.model.state_dict()
        return state_dict

    def get_optimizer_state_dict(self):
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT):
            state_dict = FSDP.optim_state_dict(self.model, self.optimizer)
        return state_dict

    def offload_fsdp_grad(self):
        for _, param in self.model.named_parameters():
            if param.grad is not None:
                param.grad = param.grad.to("cpu", non_blocking=True)
        clear_memory()

    def load_fsdp_grad(self, device_id):
        for _, param in self.model.named_parameters():
            if param.grad is not None:
                param.grad = param.grad.to(device_id, non_blocking=True)
        clear_memory()

    def offload_fsdp_param_and_grad(self, offload_grad=False):
        for _, param in self.model.named_parameters():
            if hasattr(param, "_handle") and param._handle is not None:
                flat_param = param._handle.flat_param
                if (
                    hasattr(flat_param, "_local_shard")
                    and flat_param._local_shard is not None
                ):
                    flat_param._local_shard = flat_param._local_shard.to(
                        "cpu", non_blocking=True
                    )
                if flat_param.data is not None:
                    flat_param.data = flat_param.data.to("cpu", non_blocking=True)
                    flat_param._local_shard = flat_param.data
            elif hasattr(param, "_local_shard") and param._local_shard is not None:
                param._local_shard = param._local_shard.to("cpu", non_blocking=True)

            if param.data is not None:
                param.data = param.data.to("cpu", non_blocking=True)

            if offload_grad and param.grad is not None:
                param.grad = param.grad.to("cpu", non_blocking=True)
        clear_memory()

    def load_fsdp_param_and_grad(self, device_id, load_grad=False):
        for _, param in self.model.named_parameters():
            if hasattr(param, "_handle") and param._handle is not None:
                flat_param = param._handle.flat_param
                if (
                    hasattr(flat_param, "_local_shard")
                    and flat_param._local_shard is not None
                ):
                    flat_param._local_shard = flat_param._local_shard.to(
                        device_id, non_blocking=True
                    )
                if flat_param.data is not None:
                    flat_param.data = flat_param.data.to(device_id, non_blocking=True)
                    flat_param._local_shard = flat_param.data
            elif hasattr(param, "_local_shard") and param._local_shard is not None:
                param._local_shard = param._local_shard.to(device_id, non_blocking=True)

            if param.data is not None:
                param.data = param.data.to(device_id, non_blocking=True)

            if load_grad and param.grad is not None:
                param.grad = param.grad.to(device_id, non_blocking=True)
        clear_memory()

    def offload_fsdp_optimizer(self):
        if not self.optimizer.state:
            return
        for param_group in self.optimizer.param_groups:
            for param in param_group["params"]:
                state = self.optimizer.state[param]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to("cpu", non_blocking=True)
        clear_memory()

    def load_fsdp_optimizer(self, device_id):
        if not self.optimizer.state:
            return
        for param_group in self.optimizer.param_groups:
            for param in param_group["params"]:
                state = self.optimizer.state[param]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(device_id, non_blocking=True)
        clear_memory()
