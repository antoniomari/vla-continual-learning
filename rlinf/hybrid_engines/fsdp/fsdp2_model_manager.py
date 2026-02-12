from torch.distributed.fsdp import fully_shard, FSDPModule, CPUOffloadPolicy, MixedPrecisionPolicy
from torch.distributed.device_mesh import init_device_mesh
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig

class FSDP2ModelManager:
    """
    FSDP2 Model Manager for RL training with LoRA support
    """

    def __init__(self, cfg: DictConfig):
        self._cfg = cfg
        self.torch_dtype = torch_dtype_from_precision(self._cfg.model.precision)
        
        # Get distributed info
        import torch.distributed as dist
        if dist.is_initialized():
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()
        else:
            self._rank = 0
            self._world_size = 1

        assert (
            self.torch_dtype == torch.float16 or self.torch_dtype == torch.bfloat16
        ), (
            f"Precision {self._cfg.model.precision} is not supported, only support bf16 and fp16."
        )

    def model_provider_func(self) -> torch.nn.Module:
        """Load the base model (same as before)"""
        # Your existing model loading logic
        model = AutoModelForCausalLM.from_pretrained(
            self._cfg.model.model_path,
            torch_dtype=self.torch_dtype,
            device_map="cpu",  # Load to CPU first, FSDP2 will handle device placement
            trust_remote_code=True,
            use_safetensors=self._cfg.model.get("use_safetensors", False),
        )
        return model

    def setup_model_and_optimizer(self):
        """Setup model and optimizer with FSDP2."""
        
        # Step 1: Load the base model
        module = self.model_provider_func()
        
        # Step 2: Enable gradient checkpointing
        module.gradient_checkpointing_enable()
        
        # Step 3: Create device mesh for FSDP2
        mesh = init_device_mesh(
            device_type="cuda",
            mesh_shape=(self._world_size,),
            mesh_dim_names=("data_parallel",)
        )
        
        # Step 4: Configure mixed precision policy
        mp_policy = MixedPrecisionPolicy(
            param_dtype=self.torch_dtype,  # Parameters stored in bf16/fp16
            reduce_dtype=torch.float32,     # Gradients reduced in fp32 for stability
            cast_forward_inputs=True,
        )
        
        # Step 5: Configure CPU offload policy (optional)
        if self._cfg.model.get("cpu_offload", False):
            offload_policy = CPUOffloadPolicy(pin_memory=True)
        else:
            offload_policy = OffloadPolicy()  # No offloading
        
        # Step 6: Apply FSDP2 wrapping in bottom-up manner
        # First wrap individual transformer layers/blocks
        # This depends on your model architecture - adjust layer names accordingly
        for name, layer in module.named_modules():
            # Example: for LLaMA models, wrap each transformer block
            # Adjust this based on your actual model architecture
            if self._should_wrap_layer(name, layer):
                fully_shard(
                    layer,
                    mesh=mesh,
                    reshard_after_forward=self._cfg.model.get("reshard_after_forward", True),
                    mp_policy=mp_policy,
                    offload_policy=offload_policy,
                )
        
        # Step 7: Wrap the root model
        self.model = fully_shard(
            module,
            mesh=mesh,
            reshard_after_forward=self._cfg.model.get("reshard_after_forward", True),
            mp_policy=mp_policy,
            offload_policy=offload_policy,
        )
        
        # Verify the model is now an FSDPModule
        assert isinstance(self.model, FSDPModule), "Model should be an FSDPModule"
        
        # Step 8: Setup optimizer AFTER FSDP wrapping
        # FSDP2 converts parameters to DTensor, optimizer must use these
        betas = (self._cfg.optim.adam_beta1, self._cfg.optim.adam_beta2)
        
        # Separate parameter groups for base model and value head
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
        
        if self._cfg.model.vh_mode in ["a", "a0", "a6"]:
            param_groups.append(
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if "value_head" in n and p.requires_grad
                    ],
                    "lr": self._cfg.optim.value_lr,
                    "betas": betas,
                }
            )
        
        self.optimizer = optim.AdamW(param_groups)
        
        if self._rank == 0:
            print(f"[FSDP2] Model setup complete")
            print(f"[FSDP2] Total parameters: {sum(p.numel() for p in self.model.parameters()) / 1e9:.2f}B")
            print(f"[FSDP2] Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e9:.2f}B")

    def _should_wrap_layer(self, name: str, layer: nn.Module) -> bool:
        """
        Determine if a layer should be wrapped with FSDP2.
        Adjust this based on your model architecture.
        """
        # Example wrapping logic - customize for your model
        # For LLaMA: wrap each TransformerBlock
        # For GPT: wrap each GPTBlock
        # For T5: wrap each T5Block
        
        # Example patterns (adjust to your model):
        wrap_patterns = [
            "TransformerBlock",
            "GPTBlock", 
            "T5Block",
            "LlamaDecoderLayer",
            # Add your model's layer types here
        ]
        
        return any(pattern in type(layer).__name__ for pattern in wrap_patterns)

    # =================================================================
    # MANUAL OFFLOADING METHODS - MUCH SIMPLER WITH FSDP2!
    # =================================================================
    
    # def offload_fsdp_param_and_grad(self, offload_grad=False):
    #     """
    #     Offload FSDP2 model parameters and gradients to CPU.
    #     FSDP2 makes this MUCH simpler than FSDP1!
    #     """
    #     import gc
        
    #     # With FSDP2, simply move the model to CPU
    #     # FSDP2 handles DTensor sharding automatically
    #     self.model.to(device="cpu")
        
    #     # Offload gradients if requested
    #     if offload_grad:
    #         for param in self.model.parameters():
    #             if param.grad is not None:
    #                 param.grad = param.grad.cpu()
        
    #     # Aggressive memory cleanup
    #     if torch.cuda.is_available():
    #         torch.cuda.synchronize()
        
    #     for _ in range(5):
    #         gc.collect()
    #         if torch.cuda.is_available():
    #             torch.cuda.empty_cache()
    #             torch.cuda.synchronize()
        
    #     if self._rank == 0:
    #         if torch.cuda.is_available():
    #             allocated = torch.cuda.memory_allocated() / 1024**3
    #             reserved = torch.cuda.memory_reserved() / 1024**3
    #             print(f"[Offload] GPU memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

    # def load_fsdp_param_and_grad(self, device_id, load_grad=False):
    #     """
    #     Load FSDP2 model parameters and gradients back to GPU.
    #     FSDP2 makes this MUCH simpler than FSDP1!
    #     """
    #     import gc
        
    #     # With FSDP2, simply move the model to GPU
    #     # FSDP2 handles DTensor sharding automatically
    #     self.model.to(device=device_id)
        
    #     # Load gradients if requested
    #     if load_grad:
    #         for param in self.model.parameters():
    #             if param.grad is not None:
    #                 param.grad = param.grad.to(device_id)
        
    #     # Cleanup
    #     if torch.cuda.is_available():
    #         torch.cuda.synchronize()
        
    #     for _ in range(3):
    #         gc.collect()
    #         if torch.cuda.is_available():
    #             torch.cuda.empty_cache()
    #             torch.cuda.synchronize()
        
    #     if self._rank == 0:
    #         if torch.cuda.is_available():
    #             print(f"[Load] GPU memory after load: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

    def offload_fsdp_param_and_grad(self, offload_grad=False):
        """
        Offload FSDP2 model parameters to CPU and release GPU memory.
        Includes libc malloc_trim to fix the 'stuck' memory on Rank 0/GPU 1.
        """
        import gc
        import ctypes
        import torch.distributed as dist

        # 1. Synchronization
        if dist.is_initialized():
            dist.barrier()

        # 2. Move Model to CPU
        self.model.to("cpu")

        # 3. Clear Gradients
        if offload_grad:
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad = param.grad.to("cpu", non_blocking=False)
        else:
            self.model.zero_grad(set_to_none=True)

        # 4. Aggressive Cleanup with malloc_trim
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            for _ in range(3):
                gc.collect()
                torch.cuda.empty_cache()
            
            # --- THE CRITICAL FIX FOR STUCK MEMORY ---
            # This forces the C library to release freed memory back to the OS
            try:
                libc = ctypes.CDLL("libc.so.6")
                libc.malloc_trim(0)
            except Exception:
                pass  # Ignore if not on Linux
            # -----------------------------------------

        # 5. Final Barrier
        if dist.is_initialized():
            dist.barrier()

        if self._rank == 0:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                print(f"[Offload] Complete. GPU Memory: {allocated:.2f}GB")
    
    def load_fsdp_param_and_grad(self, device_id, load_grad=False):
        """
        Load FSDP2 model parameters back to GPU.
        Blocks until fully loaded to ensure consistent state.
        """
        import gc
        import torch.distributed as dist

        # 1. SYNCHRONIZE: Wait for all ranks to be ready before allocating massive memory.
        if dist.is_initialized():
            dist.barrier()

        # 2. Move Model to GPU
        # FSDP2 handles re-sharding automatically.
        self.model.to(device_id)

        # 3. Handle Gradients (if they were preserved)
        if load_grad:
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad = param.grad.to(device_id, non_blocking=False)

        # 4. Cleanup CPU-side artifacts
        # Moving to GPU might leave temporary CPU copies; clear them now.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # 5. FINAL BARRIER: Ensure all ranks are loaded and ready to compute.
        if dist.is_initialized():
            dist.barrier()

        if self._rank == 0:
            if torch.cuda.is_available():
                print(f"[Load] Model loaded. GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

    def offload_fsdp_optimizer(self):
        """Offload optimizer states to CPU."""
        import gc
        
        for state in self.optimizer.state.values():
            if not isinstance(state, dict):
                continue
            for key, value in list(state.items()):
                if torch.is_tensor(value):
                    if value.device.type != "cpu":
                        state[key] = value.detach().to("cpu", non_blocking=True)
                        del value
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        for _ in range(3):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def load_fsdp_optimizer(self, device_id):
        """Load optimizer states to GPU."""
        import gc
        
        for state in self.optimizer.state.values():
            if not isinstance(state, dict):
                continue
            for key, value in list(state.items()):
                if torch.is_tensor(value):
                    if value.device != device_id:
                        state[key] = value.detach().to(device_id, non_blocking=True)
                        del value
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        for _ in range(3):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # =================================================================
    # STATE DICT METHODS - UPDATED FOR FSDP2
    # =================================================================
    
    def get_model_state_dict(self):
        """
        Get model state dict using FSDP2's DTensor-based approach.
        """
        # Optimization: If model is on CPU, gather on CPU directly to avoid GPU move
        # Check first parameter's device
        first_param = next(self.model.parameters(), None)
        if first_param is not None and first_param.device.type == "cpu":
            return self._get_cpu_model_state_dict()

        from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
        
        # Use FSDP2's built-in state dict API
        state_dict = get_model_state_dict(
            model=self.model,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,  # Offload to CPU to save GPU memory
            )
        )
        return state_dict

    def _get_cpu_model_state_dict(self):
        """Gather state dict on CPU using a CPU device mesh."""
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import DTensor
        import torch
        
        # Initialize CPU mesh if not exists (using Gloo backend implicitly)
        if not hasattr(self, "_cpu_mesh"):
             # Ensure world_size matches
             self._cpu_mesh = init_device_mesh("cpu", (self._world_size,))
             
        state_dict = {}
        with torch.no_grad():
            full_name_to_param = dict(self.model.named_parameters())
            full_name_to_buffer = dict(self.model.named_buffers())
            
            # Handle parameters
            for name, param in full_name_to_param.items():
                if isinstance(param, DTensor):
                    # Create a CPU DTensor from local shard
                    local_shard = param.to_local()
                    # Reconstruct DTensor on CPU mesh: use same placements (Shard/Replicate)
                    # NOTE: We assume the same placements as the original DTensor, just on CPU mesh
                    cpu_dt = DTensor.from_local(local_shard, self._cpu_mesh, param.placements)
                    state_dict[name] = cpu_dt.full_tensor()
                else:
                    state_dict[name] = param
            
            # Handle buffers
            for name, buf in full_name_to_buffer.items():
                if isinstance(buf, DTensor):
                    local_shard = buf.to_local()
                    cpu_dt = DTensor.from_local(local_shard, self._cpu_mesh, buf.placements)
                    state_dict[name] = cpu_dt.full_tensor()
                else:
                    state_dict[name] = buf
                    
        return state_dict

    def get_optimizer_state_dict(self):
        """
        Get optimizer state dict using FSDP2's DTensor-based approach.
        """
        from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict, StateDictOptions
        
        state_dict = get_optimizer_state_dict(
            model=self.model,
            optimizer=self.optimizer,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
            )
        )
        return state_dict
