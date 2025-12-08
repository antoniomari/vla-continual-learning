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

from rlinf.custom.loss import (
    behavior_cloning_loss,
    behavior_cloning_loss_with_reference_logits,
)


class TestBehaviorCloningLoss:
    """Test suite for behavior cloning loss functions."""

    def test_behavior_cloning_loss_basic(self):
        """Test basic behavior cloning loss computation."""
        batch_size = 4
        action_dim = 7
        
        predicted_actions = torch.randn(batch_size, action_dim, requires_grad=True)
        expert_actions = torch.randn(batch_size, action_dim)
        bc_coeff = 0.5
        
        kwargs = {
            "action_tokens": predicted_actions,
            "expert_action_tokens": expert_actions,
            "bc_coeff": bc_coeff,
        }
        
        weighted_loss, metrics = behavior_cloning_loss(**kwargs)
        
        # Check that loss is a scalar tensor
        assert weighted_loss.dim() == 0
        assert weighted_loss.requires_grad
        
        # Check metrics structure
        assert "bc/loss" in metrics
        assert "bc/weighted_loss" in metrics
        assert "bc/coeff" in metrics
        assert "bc/mean_abs_error" in metrics
        
        # Check that weighted loss equals coeff * loss
        expected_weighted = bc_coeff * metrics["bc/loss"]
        assert abs(weighted_loss.item() - expected_weighted) < 1e-6
        
        # Check that loss is positive
        assert metrics["bc/loss"] >= 0
        assert metrics["bc/mean_abs_error"] >= 0

    def test_behavior_cloning_loss_zero_coeff(self):
        """Test that zero coefficient results in zero loss."""
        predicted_actions = torch.randn(2, 5)
        expert_actions = torch.randn(2, 5)
        
        kwargs = {
            "action_tokens": predicted_actions,
            "expert_action_tokens": expert_actions,
            "bc_coeff": 0.0,
        }
        
        weighted_loss, metrics = behavior_cloning_loss(**kwargs)
        
        assert weighted_loss.item() == 0.0
        assert metrics["bc/weighted_loss"] == 0.0

    def test_behavior_cloning_loss_identical_actions(self):
        """Test that identical actions result in zero loss."""
        actions = torch.randn(3, 6)
        bc_coeff = 1.0
        
        kwargs = {
            "action_tokens": actions,
            "expert_action_tokens": actions.clone(),
            "bc_coeff": bc_coeff,
        }
        
        weighted_loss, metrics = behavior_cloning_loss(**kwargs)
        
        assert abs(weighted_loss.item()) < 1e-6
        assert abs(metrics["bc/loss"]) < 1e-6
        assert abs(metrics["bc/mean_abs_error"]) < 1e-6


class TestBehaviorCloningLossWithReferenceLogits:
    """Test suite for reference logits-based BC loss."""

    def test_reference_logits_loss_basic(self):
        """Test basic reference logits loss computation."""
        batch_size = 4
        action_dim = 7
        vocab_range = 64  # n_action_bins
        
        current_logits = torch.randn(batch_size, action_dim, vocab_range, requires_grad=True)
        reference_logits = torch.randn(batch_size, action_dim, vocab_range)
        bc_coeff = 0.5
        
        kwargs = {
            "current_logits": current_logits,
            "reference_logits": reference_logits,
            "bc_coeff": bc_coeff,
        }
        
        weighted_loss, metrics = behavior_cloning_loss_with_reference_logits(**kwargs)
        
        # Check that loss is a scalar tensor
        assert weighted_loss.dim() == 0
        assert weighted_loss.requires_grad
        
        # Check that reference logits are detached (no gradients)
        assert not reference_logits.requires_grad
        
        # Check metrics structure
        assert "bc/loss" in metrics
        assert "bc/weighted_loss" in metrics
        assert "bc/coeff" in metrics
        assert "bc/mean_abs_error" in metrics
        assert "bc/max_abs_error" in metrics
        assert "bc/logits_mse" in metrics
        
        # Check that weighted loss equals coeff * loss
        expected_weighted = bc_coeff * metrics["bc/loss"]
        assert abs(weighted_loss.item() - expected_weighted) < 1e-5
        
        # Check that loss is positive
        assert metrics["bc/loss"] >= 0
        assert metrics["bc/mean_abs_error"] >= 0
        assert metrics["bc/max_abs_error"] >= 0

    def test_reference_logits_loss_zero_coeff(self):
        """Test that zero coefficient results in zero loss."""
        current_logits = torch.randn(2, 5, 32, requires_grad=True)
        reference_logits = torch.randn(2, 5, 32)
        
        kwargs = {
            "current_logits": current_logits,
            "reference_logits": reference_logits,
            "bc_coeff": 0.0,
        }
        
        weighted_loss, metrics = behavior_cloning_loss_with_reference_logits(**kwargs)
        
        assert weighted_loss.item() == 0.0
        assert metrics["bc/weighted_loss"] == 0.0

    def test_reference_logits_loss_identical_logits(self):
        """Test that identical logits result in zero loss."""
        logits = torch.randn(3, 6, 64, requires_grad=True)
        bc_coeff = 1.0
        
        kwargs = {
            "current_logits": logits,
            "reference_logits": logits.detach().clone(),
            "bc_coeff": bc_coeff,
        }
        
        weighted_loss, metrics = behavior_cloning_loss_with_reference_logits(**kwargs)
        
        assert abs(weighted_loss.item()) < 1e-5
        assert abs(metrics["bc/loss"]) < 1e-5
        assert abs(metrics["bc/mean_abs_error"]) < 1e-5

    def test_reference_logits_loss_gradient_flow(self):
        """Test that gradients flow through current logits but not reference."""
        batch_size = 2
        action_dim = 3
        vocab_range = 16
        
        current_logits = torch.randn(batch_size, action_dim, vocab_range, requires_grad=True)
        reference_logits = torch.randn(batch_size, action_dim, vocab_range)
        bc_coeff = 1.0
        
        kwargs = {
            "current_logits": current_logits,
            "reference_logits": reference_logits,
            "bc_coeff": bc_coeff,
        }
        
        weighted_loss, _ = behavior_cloning_loss_with_reference_logits(**kwargs)
        weighted_loss.backward()
        
        # Check that gradients exist for current logits
        assert current_logits.grad is not None
        assert current_logits.grad.abs().sum() > 0
        
        # Check that reference logits remain detached
        assert not reference_logits.requires_grad

    def test_reference_logits_loss_different_shapes(self):
        """Test that loss works with different batch sizes and dimensions."""
        test_cases = [
            (1, 1, 8),   # Single sample, single action
            (10, 5, 32), # Multiple samples, multiple actions
            (100, 7, 64), # Large batch
        ]
        
        for batch_size, action_dim, vocab_range in test_cases:
            current_logits = torch.randn(batch_size, action_dim, vocab_range, requires_grad=True)
            reference_logits = torch.randn(batch_size, action_dim, vocab_range)
            
            kwargs = {
                "current_logits": current_logits,
                "reference_logits": reference_logits,
                "bc_coeff": 0.5,
            }
            
            weighted_loss, metrics = behavior_cloning_loss_with_reference_logits(**kwargs)
            
            assert weighted_loss.dim() == 0
            assert metrics["bc/loss"] >= 0

    def test_reference_logits_loss_consistency(self):
        """Test that loss is consistent with manual MSE computation."""
        current_logits = torch.randn(4, 6, 32, requires_grad=True)
        reference_logits = torch.randn(4, 6, 32)
        bc_coeff = 0.7
        
        kwargs = {
            "current_logits": current_logits,
            "reference_logits": reference_logits,
            "bc_coeff": bc_coeff,
        }
        
        weighted_loss, metrics = behavior_cloning_loss_with_reference_logits(**kwargs)
        
        # Manual computation
        manual_mse = torch.nn.functional.mse_loss(current_logits, reference_logits)
        manual_weighted = bc_coeff * manual_mse
        
        assert abs(weighted_loss.item() - manual_weighted.item()) < 1e-5
        assert abs(metrics["bc/loss"] - manual_mse.item()) < 1e-5


class TestLossComparison:
    """Test suite comparing different loss functions."""

    def test_loss_scales_differently(self):
        """Test that action token loss and logits loss scale differently."""
        # Create logits that would produce specific actions
        batch_size = 2
        action_dim = 3
        vocab_size = 100
        n_action_bins = 32
        
        # Create logits where argmax gives specific action tokens
        current_logits = torch.zeros(batch_size, action_dim, vocab_size)
        reference_logits = torch.zeros(batch_size, action_dim, vocab_size)
        
        # Set argmax to different values
        current_logits[:, :, vocab_size - 10] = 10.0  # Action token = vocab_size - 10
        reference_logits[:, :, vocab_size - 15] = 10.0  # Action token = vocab_size - 15
        
        # Get action tokens from logits
        current_actions = torch.argmax(current_logits, dim=-1)
        reference_actions = torch.argmax(reference_logits, dim=-1)
        
        # Compute both losses
        action_loss, _ = behavior_cloning_loss(
            action_tokens=current_actions.float(),
            expert_action_tokens=reference_actions.float(),
            bc_coeff=1.0,
        )
        
        logits_loss, _ = behavior_cloning_loss_with_reference_logits(
            current_logits=current_logits,
            reference_logits=reference_logits,
            bc_coeff=1.0,
        )
        
        # Both should be positive but different
        assert action_loss.item() > 0
        assert logits_loss.item() > 0
        # Logits loss should generally be larger since it considers full distribution
        assert logits_loss.item() != action_loss.item()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
