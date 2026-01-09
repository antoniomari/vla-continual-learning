from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from rlinf.models.embodiment.model_utils import compute_action_tokens_from_actions


class TestComputeActionTokensFromActions:
    @pytest.fixture
    def mock_model(self):
        """Create a mock model with necessary attributes"""
        model = Mock()
        model.action_dim = 7
        model.vocab_size = 32000
        model.config = Mock()
        model.config.n_action_bins = 256

        # Create bin centers from -1 to 1
        model.bin_centers = np.linspace(-1.0, 1.0, 256)

        # Mock normalization key
        model.unnorm_key = "default"

        # Mock the _unnormalize_actions method
        def _unnormalize_actions(normalized_actions, unnorm_key=None):
            # Simple identity function for testing
            if isinstance(normalized_actions, torch.Tensor):
                return normalized_actions.cpu().numpy()
            return normalized_actions

        model._unnormalize_actions = _unnormalize_actions

        return model

    def test_basic_conversion(self, mock_model):
        """Test basic conversion from actions to tokens"""
        B, T, D = 2, 3, 7
        actions = np.random.uniform(-1.0, 1.0, size=(B, T, D))

        # Patch _normalize_actions at the module level
        with patch(
            "rlinf.models.embodiment.model_utils._normalize_actions"
        ) as mock_norm:
            mock_norm.return_value = actions  # Return input unchanged

            token_ids = compute_action_tokens_from_actions(mock_model, actions)

        # Check output shape
        assert token_ids.shape == (B, T * D)

        # Check token range
        min_token = mock_model.vocab_size - mock_model.config.n_action_bins
        max_token = mock_model.vocab_size - 1
        assert np.all(token_ids >= min_token)
        assert np.all(token_ids <= max_token)

    def test_roundtrip_conversion(self, mock_model):
        """Test that converting actions->tokens->actions approximately recovers original"""
        B, T, D = 2, 3, 7
        original_actions = np.random.uniform(-1.0, 1.0, size=(B, T, D))

        with patch(
            "rlinf.models.embodiment.model_utils._normalize_actions"
        ) as mock_norm:
            mock_norm.return_value = original_actions

            # Convert to tokens
            token_ids = compute_action_tokens_from_actions(mock_model, original_actions)

        # Convert back to actions (simulate the forward process)
        token_ids_reshaped = token_ids.reshape(B * T, D)
        discretized_actions = mock_model.vocab_size - token_ids_reshaped
        discretized_actions = np.clip(
            discretized_actions - 1, a_min=0, a_max=mock_model.bin_centers.shape[0] - 1
        )
        normalized_actions = np.asarray(
            [mock_model.bin_centers[da] for da in discretized_actions]
        )
        recovered_actions = normalized_actions.reshape(B, T, D)

        # Check that recovered actions are close to original
        max_error = np.max(np.abs(recovered_actions - original_actions))
        bin_width = (mock_model.bin_centers[-1] - mock_model.bin_centers[0]) / len(
            mock_model.bin_centers
        )

        # Error should be at most half a bin width
        assert max_error <= bin_width / 2 + 1e-5, (
            f"Max error {max_error} exceeds half bin width {bin_width / 2}"
        )

    def test_extreme_values(self, mock_model):
        """Test that extreme action values are clipped correctly"""
        B, T, D = 1, 1, 7

        # Test with values at boundaries
        actions_min = np.full((B, T, D), -1.0)
        actions_max = np.full((B, T, D), 1.0)

        with patch(
            "rlinf.models.embodiment.model_utils._normalize_actions"
        ) as mock_norm:
            mock_norm.side_effect = lambda m, a, norm_key=None: a

            tokens_min = compute_action_tokens_from_actions(mock_model, actions_min)
            tokens_max = compute_action_tokens_from_actions(mock_model, actions_max)

        min_token = mock_model.vocab_size - mock_model.config.n_action_bins
        max_token = mock_model.vocab_size - 1

        assert np.all(tokens_min >= min_token)
        assert np.all(tokens_min <= max_token)
        assert np.all(tokens_max >= min_token)
        assert np.all(tokens_max <= max_token)

    def test_out_of_range_clipping(self, mock_model):
        """Test that out-of-range actions are clipped to valid token range"""
        B, T, D = 1, 2, 7

        # Actions outside [-1, 1] range
        actions = np.array(
            [
                [
                    [-2.0, 2.0, 0.5, 1.5, -1.5, 0.0, 0.8],
                    [-3.0, 3.0, -0.5, 0.2, 1.2, -0.3, 0.1],
                ]
            ]
        )

        with patch(
            "rlinf.models.embodiment.model_utils._normalize_actions"
        ) as mock_norm:
            mock_norm.return_value = actions  # Don't clip

            token_ids = compute_action_tokens_from_actions(mock_model, actions)

        # All tokens should still be in valid range due to clipping
        min_token = mock_model.vocab_size - mock_model.config.n_action_bins
        max_token = mock_model.vocab_size - 1

        assert np.all(token_ids >= min_token)
        assert np.all(token_ids <= max_token)

    def test_deterministic(self, mock_model):
        """Test that same actions produce same tokens"""
        B, T, D = 2, 3, 7
        actions = np.random.uniform(-1.0, 1.0, size=(B, T, D))

        with patch(
            "rlinf.models.embodiment.model_utils._normalize_actions"
        ) as mock_norm:
            mock_norm.return_value = actions

            tokens1 = compute_action_tokens_from_actions(mock_model, actions)
            tokens2 = compute_action_tokens_from_actions(mock_model, actions)

        assert np.array_equal(tokens1, tokens2)

    def test_different_batch_sizes(self, mock_model):
        """Test with various batch and time dimensions"""
        test_cases = [
            (1, 1, 7),  # Single timestep, single batch
            (5, 10, 7),  # Multiple timesteps, multiple batches
            (32, 50, 7),  # Large batch
        ]

        for B, T, D in test_cases:
            actions = np.random.uniform(-1.0, 1.0, size=(B, T, D))

            with patch(
                "rlinf.models.embodiment.model_utils._normalize_actions"
            ) as mock_norm:
                mock_norm.return_value = actions

                token_ids = compute_action_tokens_from_actions(mock_model, actions)

            assert token_ids.shape == (B, T * D), f"Failed for shape ({B}, {T}, {D})"

            min_token = mock_model.vocab_size - mock_model.config.n_action_bins
            max_token = mock_model.vocab_size - 1
            assert np.all(token_ids >= min_token)
            assert np.all(token_ids <= max_token)

    def test_zero_actions(self, mock_model):
        """Test with all zero actions"""
        B, T, D = 2, 3, 7
        actions = np.zeros((B, T, D))

        with patch(
            "rlinf.models.embodiment.model_utils._normalize_actions"
        ) as mock_norm:
            mock_norm.return_value = actions

            token_ids = compute_action_tokens_from_actions(mock_model, actions)

        # Check all tokens are in valid range
        min_token = mock_model.vocab_size - mock_model.config.n_action_bins
        max_token = mock_model.vocab_size - 1
        assert np.all(token_ids >= min_token)
        assert np.all(token_ids <= max_token)

        # Check tokens are all the same (since all actions are same)
        assert np.all(token_ids == token_ids[0, 0])

    def test_dimension_mismatch_raises(self, mock_model):
        """Test that incorrect action dimensions raise an assertion error"""
        B, T = 2, 3
        wrong_D = 5  # Should be 7

        actions = np.random.uniform(-1.0, 1.0, size=(B, T, wrong_D))

        with patch(
            "rlinf.models.embodiment.model_utils._normalize_actions"
        ) as mock_norm:
            mock_norm.return_value = actions

            with pytest.raises(AssertionError):
                compute_action_tokens_from_actions(mock_model, actions)

    def test_nearest_bin_selection(self, mock_model):
        """Test that actions are mapped to nearest bin"""
        B, T, D = 1, 1, 7

        # Create actions that are exactly at bin centers
        bin_indices = [0, 64, 128, 192, 255, 100, 200]
        actions = np.array([[[mock_model.bin_centers[idx] for idx in bin_indices]]])

        with patch(
            "rlinf.models.embodiment.model_utils._normalize_actions"
        ) as mock_norm:
            mock_norm.return_value = actions

            token_ids = compute_action_tokens_from_actions(mock_model, actions)

        # Convert back to check
        expected_tokens = mock_model.vocab_size - 1 - np.array(bin_indices)

        assert np.allclose(token_ids.flatten(), expected_tokens), (
            f"Expected {expected_tokens}, got {token_ids.flatten()}"
        )
