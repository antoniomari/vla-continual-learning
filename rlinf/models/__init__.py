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

import json
import os
from typing import Optional

import torch
from omegaconf import DictConfig
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoProcessor,
    AutoTokenizer,
)

from rlinf.config import torch_dtype_from_precision
from rlinf.utils.model_load_info import log_get_model


def _apply_lora_scale_if_present(model, lora_scale: float):
    """
    Apply a global multiplier to LoRA contribution.

    This scales the per-adapter scaling factors inside PEFT LoRA layers so that:
      - lora_scale=0.0 behaves like the base model (adapter disabled)
      - lora_scale=1.0 is the default LoRA behavior
      - values in between interpolate smoothly
    
    Raises:
        ValueError: If lora_scale is None or cannot be converted to float
        RuntimeError: If no LoRA modules with scaling are found in the model
    """
    if lora_scale is None:
        raise ValueError("lora_scale cannot be None. It must be a numeric value (default: 1.0)")
    
    try:
        lora_scale = float(lora_scale)
    except (TypeError, ValueError) as e:
        raise ValueError(f"lora_scale must be a numeric value, got {type(lora_scale).__name__}: {lora_scale}") from e
    
    if lora_scale == 1.0:
        return model

    # PEFT LoRA modules commonly expose a `scaling` dict keyed by adapter name.
    # We update all adapters we find to keep behavior consistent.
    scaled_count = 0
    for module in model.modules():
        scaling = getattr(module, "scaling", None)
        if isinstance(scaling, dict) and len(scaling) > 0:
            for adapter_name in list(scaling.keys()):
                try:
                    old_value = scaling[adapter_name]
                    scaling[adapter_name] = float(old_value) * lora_scale
                    scaled_count += 1
                except (TypeError, ValueError) as e:
                    raise ValueError(
                        f"Failed to scale LoRA adapter '{adapter_name}' in module {type(module).__name__}: "
                        f"scaling value {old_value} is not numeric"
                    ) from e

    if scaled_count == 0:
        raise RuntimeError(
            "No LoRA modules with scaling found in the model. "
            "This may indicate that the model is not a PEFT model or LoRA adapters are not properly configured."
        )

    return model


def get_model_config_and_processor(cfg: DictConfig):
    if cfg.model.model_name == "openvla":
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig

        from .embodiment.prismatic.processing_prismatic import (
            PrismaticImageProcessor,
            PrismaticProcessor,
        )

        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)

        model_config = AutoConfig.from_pretrained(cfg.tokenizer.tokenizer_model)

        dataset_statistics_path = os.path.join(
            cfg.tokenizer.tokenizer_model, "dataset_statistics.json"
        )
        if os.path.isfile(dataset_statistics_path):
            with open(dataset_statistics_path, "r") as f:
                new_norm_stats = json.load(f)
                norm_stats = getattr(model_config, "norm_stats", {})
                norm_stats.update(new_norm_stats)
                setattr(model_config, "norm_stats", norm_stats)
        image_processor = PrismaticImageProcessor.from_pretrained(
            cfg.tokenizer.tokenizer_model, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.tokenizer.tokenizer_model, trust_remote_code=True, padding_side="left"
        )
        input_processor = PrismaticProcessor.from_pretrained(
            cfg.tokenizer.tokenizer_model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            trust_remote_code=True,
        )
    elif cfg.model.model_name == "openvla_oft":
        from prismatic.extern.hf.configuration_prismatic import (
            OpenVLAConfig as OpenVLAOFTConfig,
        )

        from .embodiment.prismatic.processing_prismatic import (
            MultiInputPrismaticProcessor as PrismaticProcessorOFT,
        )
        from .embodiment.prismatic.processing_prismatic import PrismaticImageProcessor

        AutoConfig.register("openvla", OpenVLAOFTConfig)
        AutoImageProcessor.register(OpenVLAOFTConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAOFTConfig, PrismaticProcessorOFT)

        model_config = OpenVLAOFTConfig.from_pretrained(
            cfg.tokenizer.tokenizer_model, center_crop=cfg.model.center_crop
        )
        image_processor = PrismaticImageProcessor.from_pretrained(
            cfg.tokenizer.tokenizer_model, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.tokenizer.tokenizer_model, trust_remote_code=True, padding_side="left"
        )
        input_processor = PrismaticProcessorOFT.from_pretrained(
            cfg.tokenizer.tokenizer_model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            trust_remote_code=True,
        )

    return model_config, input_processor


def get_model(
    model_path,
    cfg: DictConfig,
    override_config_kwargs=None,
    *,
    load_role: Optional[str] = None,
    worker_rank: Optional[int] = None,
    worker_world_size: Optional[int] = None,
):
    if load_role is not None:
        log_get_model(
            role=load_role,
            model_path=model_path,
            model_name=str(cfg.model_name),
            worker_rank=worker_rank,
            worker_world_size=worker_world_size,
        )

    torch_dtype = torch_dtype_from_precision(cfg.precision)
    if cfg.model_name == "openvla":
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig

        actor_model_config = OpenVLAConfig.from_pretrained(
            model_path, trust_remote_code=cfg.trust_remote_code
        )

        dataset_statistics_path = os.path.join(model_path, "dataset_statistics.json")
        if os.path.isfile(dataset_statistics_path):
            with open(dataset_statistics_path, "r") as f:
                new_norm_stats = json.load(f)
                norm_stats = getattr(actor_model_config, "norm_stats", {})
                norm_stats.update(new_norm_stats)
                setattr(actor_model_config, "norm_stats", norm_stats)

        from .embodiment.openvla_action_model import OpenVLAForRLActionPrediction

        model = OpenVLAForRLActionPrediction.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            hidden_size=cfg.hidden_size,
            unnorm_key=cfg.unnorm_key,
            config=actor_model_config,
            vh_mode=cfg.vh_mode,
            action_dim=cfg.action_dim,
            num_action_chunks=cfg.num_action_chunks,
            attn_implementation=cfg.attn_implementation,
            low_cpu_mem_usage=cfg.low_cpu_mem_usage,
            trust_remote_code=cfg.trust_remote_code,
        )
    elif cfg.model_name == "openvla_oft":
        from prismatic.extern.hf.configuration_prismatic import (
            OpenVLAConfig as OpenVLAOFTConfig,
        )

        from .embodiment.openvla_oft_action_model import OpenVLAOFTForRLActionPrediction

        AutoConfig.register("openvla", OpenVLAOFTConfig)
        actor_model_config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=cfg.trust_remote_code
        )

        dataset_statistics_path = os.path.join(model_path, "dataset_statistics.json")
        if os.path.isfile(dataset_statistics_path):
            with open(dataset_statistics_path, "r") as f:
                new_norm_stats = json.load(f)
                norm_stats = getattr(actor_model_config, "norm_stats", {})
                norm_stats.update(new_norm_stats)
                setattr(actor_model_config, "norm_stats", norm_stats)

        override_config_kwargs = cfg
        if override_config_kwargs is not None:
            for key, val in override_config_kwargs.items():
                setattr(actor_model_config, key, val)

        model = OpenVLAOFTForRLActionPrediction.from_pretrained(
            pretrained_model_name_or_path=model_path,
            torch_dtype=torch_dtype,
            # attn_implementation="flash_attention_2",
            config=actor_model_config,
            action_dim=cfg.action_dim,
            num_action_chunks=cfg.num_action_chunks,
            trust_remote_code=True,
        )

        # oft add
        model.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)
    elif cfg.model_name == "simple_cnn":
        from .simple_cnn_policy import SimpleCNNPolicy
        from rlinf.custom.simple_cnn_utils import compute_action_statistics
        
        # Load checkpoint if provided
        checkpoint_path = model_path
        if not os.path.exists(checkpoint_path):
            # If model_path doesn't exist, try to load from checkpoint_load_path
            checkpoint_path = cfg.get("checkpoint_load_path", None)
            if checkpoint_path is None or not os.path.exists(checkpoint_path):
                raise ValueError(
                    f"simple_cnn model requires a checkpoint path. "
                    f"Provided model_path: {model_path}, "
                    f"checkpoint_load_path: {cfg.get('checkpoint_load_path', None)}"
                )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Extract model parameters from checkpoint or config
        task_id_map = checkpoint.get("task_id_map", {})
        num_tasks = checkpoint.get("num_tasks", len(task_id_map) if task_id_map else cfg.get("num_tasks", 10))
        norm_stats = checkpoint.get("norm_stats", None)
        unnorm_key = checkpoint.get("unnorm_key", cfg.get("unnorm_key", "libero_spatial_no_noops"))
        vocab_size = checkpoint.get("vocab_size", cfg.get("vocab_size", 32000))
        n_action_bins = checkpoint.get("n_action_bins", cfg.get("n_action_bins", 256))
        action_dim = checkpoint.get("action_dim", cfg.get("action_dim", 7))
        num_action_chunks = checkpoint.get("num_action_chunks", cfg.get("num_action_chunks", 8))
        image_size = cfg.get("image_size", 224)
        if isinstance(image_size, (list, tuple)):
            image_size = int(image_size[0])
        
        # If norm_stats not in checkpoint, try to compute from dataset
        if norm_stats is None or len(norm_stats) == 0:
            # Try to compute from dataset
            dataset_path = os.environ.get("LIBERO_REPO_PATH", "")
            if dataset_path:
                dataset_path = os.path.join(
                    dataset_path, "libero", "datasets_with_logits", "libero_spatial_simplevla_trajall"
                )
                if os.path.exists(dataset_path):
                    print(f"Computing action statistics from {dataset_path}...")
                    norm_stats = compute_action_statistics(dataset_path, unnorm_key=unnorm_key)
                else:
                    raise ValueError(
                        f"norm_stats not found in checkpoint and cannot compute from dataset. "
                        f"Dataset path {dataset_path} does not exist."
                    )
            else:
                raise ValueError(
                    "norm_stats not found in checkpoint and LIBERO_REPO_PATH not set. "
                    "Cannot compute action statistics."
                )
        
        # Create model
        model = SimpleCNNPolicy(
            action_dim=action_dim,
            num_action_chunks=num_action_chunks,
            image_size=image_size,
            num_tasks=num_tasks,
            use_task_embedding=True,
            vocab_size=vocab_size,
            n_action_bins=n_action_bins,
            norm_stats=norm_stats,
            unnorm_key=unnorm_key,
        )
        
        # Load weights if available
        # Handle two checkpoint formats:
        # 1. From supervised training: checkpoint has "model_state_dict" key
        # 2. From RL training (FSDP): checkpoint IS the state dict directly
        if "model_state_dict" in checkpoint:
            # Supervised training format
            try:
                model.load_state_dict(checkpoint["model_state_dict"])
                print(f"Loaded simple_cnn model weights from {checkpoint_path} (supervised training format)")
            except Exception as e:
                print(f"Warning: Failed to load model_state_dict from checkpoint: {e}")
                print(f"  Attempting to load checkpoint directly as state dict...")
                # Fallback: try loading checkpoint directly
                try:
                    model.load_state_dict(checkpoint)
                    print(f"Loaded simple_cnn model weights from {checkpoint_path} (direct state dict)")
                except Exception as e2:
                    print(f"Warning: Failed to load checkpoint as state dict: {e2}")
                    print(f"  Using random initialization")
        elif isinstance(checkpoint, dict):
            # Try loading checkpoint directly as state dict (RL training format)
            try:
                model.load_state_dict(checkpoint)
                print(f"Loaded simple_cnn model weights from {checkpoint_path} (RL training format)")
            except Exception as e:
                print(f"Warning: Failed to load checkpoint as state dict: {e}")
                print(f"  Checkpoint keys: {list(checkpoint.keys())[:10]}...")  # Show first 10 keys
                print(f"  Using random initialization")
        else:
            print(f"Warning: Checkpoint is not a dict, cannot load weights. Using random initialization.")
        
        # Simple CNN doesn't use LoRA, so skip LoRA loading and return early
        if torch.cuda.is_available():
            model = model.cuda()
        return model
    else:
        return None
    
    if torch.cuda.is_available():
        model = model.cuda()

    if cfg.is_lora:
        # Default is 1.0 (no scaling change) for backward compatibility.
        lora_scale = getattr(cfg, "lora_scale", 1.0)
        
        # Support for multiple LoRA adapters (lora_paths) or single adapter (lora_path)
        # For multiple adapters, we merge them sequentially into the base model
        lora_paths = getattr(cfg, "lora_paths", None)
        lora_path = getattr(cfg, "lora_path", None)
        
        if lora_paths is not None and len(lora_paths) > 0:
            # Multiple LoRA adapters: merge them sequentially into base model
            # This makes all previous adapters "active" (part of the base model)
            # Then we'll add a new trainable adapter on top
            print(f"Loading and merging {len(lora_paths)} previous LoRA adapters into base model:")
            print(f"  Adapters: {lora_paths}")
            
            # Get merge coefficient for previous adapters (default 1.0)
            merge_coefficient = getattr(cfg, "previous_lora_merge_coefficient", 1.0)
            merge_coefficient = float(merge_coefficient)
            if merge_coefficient < 0.0 or merge_coefficient > 1.0:
                raise ValueError(f"previous_lora_merge_coefficient must be between 0.0 and 1.0, got {merge_coefficient}")
            
            if merge_coefficient != 1.0:
                print(f"  Merge coefficient: {merge_coefficient} (will scale previous adapter weights)")
            
            # Merge all previous adapters sequentially into base model
            merged_count = 0
            for idx, adapter_path in enumerate(lora_paths):
                if os.path.exists(adapter_path):
                    print(f"  [{idx+1}/{len(lora_paths)}] Loading adapter from {adapter_path}")
                    peft_model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
                    
                    # Apply merge coefficient by scaling the adapter weights before merging
                    if merge_coefficient != 1.0:
                        print(f"    Applying merge coefficient {merge_coefficient} to adapter weights...")
                        # Scale LoRA adapter contribution by modifying the scaling factors
                        # This is cleaner than directly modifying weights
                        for name, module in peft_model.named_modules():
                            # Scale the adapter scaling factors
                            scaling = getattr(module, "scaling", None)
                            if isinstance(scaling, dict) and len(scaling) > 0:
                                for adapter_name in list(scaling.keys()):
                                    old_scaling = scaling[adapter_name]
                                    scaling[adapter_name] = float(old_scaling) * merge_coefficient
                                    print(f"      Scaled {name}.scaling[{adapter_name}]: {old_scaling} -> {scaling[adapter_name]}")
                            
                            # Also scale lora_B weights directly if scaling dict doesn't exist
                            # (some PEFT implementations don't use scaling dict)
                            if hasattr(module, "lora_B") and not isinstance(getattr(module, "scaling", None), dict):
                                for adapter_name in module.lora_B.keys():
                                    if module.lora_B[adapter_name] is not None:
                                        module.lora_B[adapter_name].data *= merge_coefficient
                    
                    # Merge this adapter into the base model (makes it part of the base)
                    print(f"    Merging adapter into base model...")
                    model = peft_model.merge_and_unload()
                    merged_count += 1
                else:
                    raise ValueError(f"Adapter path {adapter_path} does not exist")
            
            print(f"  ✓ Successfully merged {merged_count} adapter(s) into base model with coefficient {merge_coefficient}")
            print(f"  → Base model now contains knowledge from {merged_count} previous task(s)")
            
            # After merging all previous adapters, either load an existing adapter or create a new one
            if lora_path is not None:
                # Load existing adapter (for evaluation or resuming training)
                print(f"  Loading current LoRA adapter from {cfg.lora_path}")
                model = PeftModel.from_pretrained(model, cfg.lora_path, is_trainable=True)
                print(f"  ✓ Current LoRA adapter loaded on top of merged model")
            else:
                # Create a NEW trainable LoRA adapter on top (for new task training)
                # This new adapter will be the only trainable one
                print(f"  Creating new trainable LoRA adapter on top of merged model...")
                lora_config = LoraConfig(
                    r=cfg.lora_rank,
                    lora_alpha=cfg.lora_rank,
                    lora_dropout=0.0,
                    target_modules=[
                        "proj", "qkv", "fc1", "fc2", "q", "kv", "fc3",
                        "q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj", "lm_head",
                    ],
                    init_lora_weights="gaussian",
                )
                model = get_peft_model(model, lora_config)
                print(f"  ✓ New trainable LoRA adapter created (only this adapter will be updated during training)")
        elif lora_path is not None:
            # Single LoRA adapter (backward compatibility)
            print(f"Loading LoRA adapter from {cfg.lora_path}")
            model = PeftModel.from_pretrained(model, cfg.lora_path, is_trainable=True)
        else:
            # Create new LoRA adapter (for new task training)
            lora_config = LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=cfg.lora_rank,
                lora_dropout=0.0,
                target_modules=[
                    "proj",
                    "qkv",
                    "fc1",
                    "fc2",  # vision
                    "q",
                    "kv",
                    "fc3",  # project
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                    "lm_head",  # llm
                ],
                init_lora_weights="gaussian",
            )
            model = get_peft_model(model, lora_config)

        model = _apply_lora_scale_if_present(model, lora_scale)

        if hasattr(model, "value_head"):
            for param in model.value_head.parameters():
                param.requires_grad = True

    if cfg.partial_finetune:
        params = []
        for _, p in model.named_parameters():
            p.requires_grad = False
            params.append(p)

        layers_to_train = getattr(cfg, "layers_to_train", 50)
        for p in params[-layers_to_train:]:
            p.requires_grad = True

        print(
            f"Partial finetune enabled. Training last {layers_to_train} layers. Total params: {len(params)}")

    if not cfg.is_lora and not cfg.partial_finetune:
        for param in model.parameters():
            param.requires_grad = True
        # OPD teacher base uses is_lora=False then PeftModel.from_pretrained on the adapter;
        # this is not the RL student (student uses cfg.is_lora=True from actor.model).
        if load_role is not None and "opd_teacher" in load_role:
            print(
                "[OPD teacher] Base HF weights loaded without LoRA in get_model; "
                "adapter is attached in load_opd_teacher_model (RL student is unchanged).",
                flush=True,
            )
        else:
            print("Full base model set to trainable (no LoRA, no partial finetune).")

    if hasattr(cfg, "ckpt_path") and cfg.ckpt_path is not None:
        print(f"LOADING MODEL CHECKPOINT from: {cfg.ckpt_path}")
        print(f"Checkpoint exists: {os.path.exists(cfg.ckpt_path)}")
        model_dict = torch.load(cfg.ckpt_path)
        filtered_state_dict = {
            k: v for k, v in model_dict.items() if "value_head" not in k
        }
        missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
        print(f"Missing keys (likely value head): {missing}")
        print(f"Unexpected keys: {unexpected}")
        print(f"CHECKPOINT LOADED SUCCESSFULLY (critic reinitialized)")
    return model
