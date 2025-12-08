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

import pytest
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import socket
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType

from rlinf.utils.utils import fsdp_cpu_weight_swap


class SimpleModel(nn.Module):
    """Simple model for testing FSDP weight swap."""
    
    def __init__(self, dim=10):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x


@pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.cuda.is_available(),
    reason="Requires distributed and CUDA support"
)
class TestFSDPWeightSwap:
    """Test suite for FSDP weight swap utility."""
    
    @pytest.fixture(autouse=True)
    def setup_distributed(self):
        """Setup distributed environment if not already initialized."""
        if not torch.distributed.is_initialized():
            # Initialize a simple single-process group for testing
            # In real usage, this would be done by the training script
            try:
                torch.distributed.init_process_group(
                    backend="nccl" if torch.cuda.is_available() else "gloo",
                    init_method="tcp://localhost:12355",
                    rank=0,
                    world_size=1,
                )
            except RuntimeError:
                # Already initialized or can't initialize
                pass
    
    def test_fsdp_weight_swap_basic(self):
        """Test basic FSDP weight swap functionality."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Create a simple model
        model = SimpleModel(dim=10).cuda()
        
        # Wrap with FSDP
        fsdp_model = FSDP(
            model,
            device_id=torch.cuda.current_device(),
            use_orig_params=True,
        )
        
        # Get initial state dict using SHARDED_STATE_DICT (matches implementation)
        with FSDP.state_dict_type(fsdp_model, StateDictType.SHARDED_STATE_DICT):
            initial_state = fsdp_model.state_dict()
        
        # Create reference weights (different from initial)
        # Note: For testing, we'll create a full state dict that will be sharded on load
        with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT):
            full_initial_state = fsdp_model.state_dict()
        
        reference_state = {}
        for k, v in full_initial_state.items():
            if torch.is_tensor(v):
                reference_state[k] = v.cpu().clone() + 1.0  # Add 1.0 to all weights
            else:
                reference_state[k] = v
        
        # Test forward pass with initial weights
        x = torch.randn(2, 10).cuda()
        output_before = fsdp_model(x)
        
        # Swap to reference weights (using full state dict, will be sharded automatically)
        with fsdp_cpu_weight_swap(fsdp_model, reference_state):
            # Forward pass with reference weights
            output_during = fsdp_model(x)
            
            # Verify weights changed (check using SHARDED_STATE_DICT)
            with FSDP.state_dict_type(fsdp_model, StateDictType.SHARDED_STATE_DICT):
                current_state = fsdp_model.state_dict()
                # Check that weights are different (approximately)
                # Note: We can only check keys that exist in the shard
                for k in current_state:
                    if k in initial_state and torch.is_tensor(initial_state[k]) and torch.is_tensor(current_state[k]):
                        diff = (current_state[k].cpu() - initial_state[k].cpu()).abs().max()
                        assert diff > 0.5  # Should be approximately 1.0
        
        # After context, weights should be restored
        output_after = fsdp_model(x)
        
        # Outputs before and after should be similar (within numerical precision)
        diff_before_after = (output_before - output_after).abs().max()
        assert diff_before_after < 1e-5, "Weights should be restored after context"
    
    def test_fsdp_weight_swap_exception_handling(self):
        """Test that weights are restored even if exception occurs."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        model = SimpleModel(dim=5).cuda()
        fsdp_model = FSDP(
            model,
            device_id=torch.cuda.current_device(),
            use_orig_params=True,
        )
        
        # Get initial state using SHARDED_STATE_DICT
        with FSDP.state_dict_type(fsdp_model, StateDictType.SHARDED_STATE_DICT):
            initial_state = fsdp_model.state_dict()
        
        # Create reference weights as full state dict (will be sharded on load)
        with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT):
            full_initial_state = fsdp_model.state_dict()
        
        reference_state = {
            k: v.cpu().clone() + 0.5 if torch.is_tensor(v) else v
            for k, v in full_initial_state.items()
        }
        
        # Test that exception doesn't prevent restoration
        try:
            with fsdp_cpu_weight_swap(fsdp_model, reference_state):
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Verify weights are restored (check using SHARDED_STATE_DICT)
        with FSDP.state_dict_type(fsdp_model, StateDictType.SHARDED_STATE_DICT):
            restored_state = fsdp_model.state_dict()
            for k in restored_state:
                if k in initial_state and torch.is_tensor(initial_state[k]) and torch.is_tensor(restored_state[k]):
                    diff = (restored_state[k].cpu() - initial_state[k].cpu()).abs().max()
                    assert diff < 1e-5, "Weights should be restored after exception"
    
    def test_fsdp_weight_swap_nested(self):
        """Test nested weight swaps."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        model = SimpleModel(dim=8).cuda()
        fsdp_model = FSDP(
            model,
            device_id=torch.cuda.current_device(),
            use_orig_params=True,
        )
        
        # Get initial state as full state dict for creating reference states
        with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT):
            state1_full = fsdp_model.state_dict()
        
        # Create reference states as full state dicts (will be sharded on load)
        state2 = {k: v.cpu().clone() + 1.0 if torch.is_tensor(v) else v for k, v in state1_full.items()}
        state3 = {k: v.cpu().clone() + 2.0 if torch.is_tensor(v) else v for k, v in state1_full.items()}
        
        x = torch.randn(1, 8).cuda()
        out1 = fsdp_model(x)
        
        with fsdp_cpu_weight_swap(fsdp_model, state2):
            out2 = fsdp_model(x)
            
            with fsdp_cpu_weight_swap(fsdp_model, state3):
                out3 = fsdp_model(x)
            
            # After inner context, should be back to state2
            out2_after = fsdp_model(x)
            diff = (out2 - out2_after).abs().max()
            assert diff < 1e-4
        
        # After outer context, should be back to state1
        out1_after = fsdp_model(x)
        diff = (out1 - out1_after).abs().max()
        assert diff < 1e-4


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _fsdp_swap_worker(rank, world_size, backend, port):
    torch.distributed.init_process_group(
        backend=backend,
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)

        model = SimpleModel(dim=6).to(device)
        # use_orig_params=False forces FlatParameter/real sharding
        fsdp_model = FSDP(model, device_id=device, use_orig_params=False)

        x = torch.randn(2, 6, device=device)

        # Save local state (LOCAL_STATE_DICT) for restore path
        with FSDP.state_dict_type(fsdp_model, StateDictType.LOCAL_STATE_DICT):
            local_state = fsdp_model.state_dict()
            local_state = {
                k: (v.detach().cpu().clone() if torch.is_tensor(v) else v)
                for k, v in local_state.items()
            }

        # Build full reference state (FULL_STATE_DICT) to load during swap
        with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT):
            full_state = fsdp_model.state_dict()
        reference_state = {
            k: (v.cpu().clone() + 1.0 if torch.is_tensor(v) else v)
            for k, v in full_state.items()
        }

        out_before = fsdp_model(x)

        with fsdp_cpu_weight_swap(fsdp_model, reference_state):
            out_during = fsdp_model(x)
            # Should differ because weights were shifted by +1
            assert (out_during - out_before).abs().max() > 1e-3

        out_after = fsdp_model(x)
        # After context, should match original output
        assert (out_after - out_before).abs().max() < 1e-5
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.cuda.is_available(),
    reason="Requires distributed and CUDA support",
)
def test_fsdp_weight_swap_multi_rank_local_full():
    """Multi-rank regression: LOCAL save/restore + FULL load should work without errors."""
    world_size = 2
    backend = "nccl"
    port = _find_free_port()
    mp.spawn(
        _fsdp_swap_worker,
        args=(world_size, backend, port),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
