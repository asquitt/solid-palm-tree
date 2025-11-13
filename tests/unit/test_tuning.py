"""Unit Tests for Hyperparameter Tuning"""

import pytest
from src.tuning.search_spaces import (
    get_search_space,
    estimate_tuning_cost,
)


class TestSearchSpaces:
    """Test search space definitions."""

    def test_default_search_space(self):
        """Test default search space."""
        space = get_search_space("default")

        assert "learning_rate" in space
        assert "batch_size" in space
        assert "num_epochs" in space

    def test_quick_search_space(self):
        """Test quick search space."""
        space = get_search_space("quick")
        assert space is not None

    def test_lora_search_space(self):
        """Test LoRA search space."""
        space = get_search_space("lora")

        assert "lora_r" in space
        assert "lora_alpha" in space

    def test_invalid_search_space(self):
        """Test invalid search space name."""
        with pytest.raises(ValueError):
            get_search_space("invalid_name")


class TestCostEstimation:
    """Test cost estimation functions."""

    def test_cost_estimation(self):
        """Test tuning cost estimation."""
        result = estimate_tuning_cost(
            num_samples=20,
            hours_per_trial=2,
            cost_per_hour=1.5,
            early_stopping_factor=0.7,
        )

        assert "total_trials" in result
        assert "cost_with_early_stopping" in result
        assert "estimated_savings" in result
        assert result["estimated_savings"] > 0
