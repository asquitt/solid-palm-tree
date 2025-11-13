"""
Hyperparameter Search Spaces

This module defines predefined search spaces for common LLM fine-tuning
scenarios. Search spaces define the range of hyperparameters to explore.

Search Space Types in Ray Tune:
- tune.choice([...]): Discrete categorical values
- tune.uniform(min, max): Continuous uniform distribution
- tune.loguniform(min, max): Log-scale uniform (good for learning rates)
- tune.randint(min, max): Random integer in range
- tune.quniform(min, max, q): Quantized uniform (step size q)

Tips for Defining Search Spaces:
1. Start small: Test with 5-10 trials first
2. Use log-scale for learning rates (orders of magnitude matter)
3. Focus on high-impact parameters: LR, batch size, warmup
4. Add more parameters once basics are tuned
5. Use domain knowledge to set reasonable ranges
"""

from ray import tune
from typing import Dict, Any


def get_search_space(preset: str = "default") -> Dict[str, Any]:
    """
    Get predefined search space by name.

    Available Presets:
    - "default": Balanced search for general fine-tuning
    - "quick": Small search space for fast experiments (5-10 trials)
    - "extensive": Comprehensive search for production models (50+ trials)
    - "lora": Optimized for LoRA parameter-efficient fine-tuning
    - "large_model": For models 7B+ parameters

    Args:
        preset: Name of predefined search space

    Returns:
        Dictionary defining the search space
    """
    search_spaces = {
        "default": get_default_search_space(),
        "quick": get_quick_search_space(),
        "extensive": get_extensive_search_space(),
        "lora": get_lora_search_space(),
        "large_model": get_large_model_search_space(),
    }

    if preset not in search_spaces:
        raise ValueError(
            f"Unknown search space preset: {preset}. "
            f"Available: {list(search_spaces.keys())}"
        )

    return search_spaces[preset]


def get_default_search_space() -> Dict[str, Any]:
    """
    Default search space for general LLM fine-tuning.

    Hyperparameters Included:
    - Learning rate: Most important! Can make 10x difference
    - Batch size: Affects convergence and memory
    - Warmup ratio: Helps with training stability
    - Weight decay: Regularization to prevent overfitting

    Expected trials: 10-20
    Expected time: 2-4 hours (depends on model size and data)
    """
    return {
        # Learning rate: Use log-scale as order of magnitude matters
        # Range: 1e-6 to 1e-3 covers most LLM fine-tuning scenarios
        "learning_rate": tune.loguniform(1e-6, 1e-3),

        # Batch size: Power of 2 for efficiency
        # Larger batch = more stable gradients but needs more memory
        "batch_size": tune.choice([4, 8, 16, 32]),

        # Gradient accumulation: Simulate larger batches
        # Effective batch = batch_size * gradient_accumulation_steps
        "gradient_accumulation_steps": tune.choice([1, 2, 4]),

        # Warmup ratio: Fraction of training for LR warmup
        # 0.1 = warm up for first 10% of training
        "warmup_ratio": tune.uniform(0.05, 0.15),

        # Weight decay: L2 regularization
        # Higher = more regularization (prevent overfitting)
        "weight_decay": tune.loguniform(1e-3, 1e-1),

        # Number of epochs: How many passes through data
        "num_epochs": tune.choice([3, 5, 10]),
    }


def get_quick_search_space() -> Dict[str, Any]:
    """
    Quick search space for rapid experimentation.

    Use this for:
    - Initial exploration
    - Debugging
    - Small datasets
    - Limited compute budget

    Expected trials: 5-10
    Expected time: 30-60 minutes
    """
    return {
        # Only tune the most important hyperparameter
        "learning_rate": tune.choice([1e-5, 3e-5, 5e-5, 1e-4]),

        # Fixed reasonable defaults for others
        "batch_size": 8,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "num_epochs": 3,
    }


def get_extensive_search_space() -> Dict[str, Any]:
    """
    Comprehensive search space for production models.

    Use this when:
    - You have compute budget for 50+ trials
    - Model will be deployed to production
    - You need to squeeze out every % of performance

    Expected trials: 50-100
    Expected time: 1-2 days
    """
    return {
        # Learning rate: Finer granularity
        "learning_rate": tune.loguniform(5e-7, 5e-4),

        # Batch size: More options
        "batch_size": tune.choice([4, 8, 16, 32, 64]),

        # Gradient accumulation
        "gradient_accumulation_steps": tune.choice([1, 2, 4, 8]),

        # Optimizer: Try different optimizers
        "optimizer": tune.choice(["adamw", "adafactor"]),

        # Learning rate scheduler
        "scheduler": tune.choice(["cosine", "linear", "polynomial"]),

        # Warmup
        "warmup_ratio": tune.uniform(0.0, 0.2),
        "warmup_steps": tune.choice([0, 100, 500, 1000]),

        # Regularization
        "weight_decay": tune.loguniform(1e-4, 1e-1),

        # Max gradient norm (gradient clipping)
        "max_grad_norm": tune.uniform(0.5, 2.0),

        # Training duration
        "num_epochs": tune.choice([3, 5, 7, 10]),

        # Precision
        "precision": tune.choice(["fp16", "bf16"]),
    }


def get_lora_search_space() -> Dict[str, Any]:
    """
    Search space optimized for LoRA fine-tuning.

    LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning
    method that trains small adapter layers instead of all parameters.

    LoRA-specific hyperparameters:
    - lora_r: Rank of low-rank matrices (higher = more capacity)
    - lora_alpha: Scaling factor (typically 2 * lora_r)
    - lora_dropout: Dropout for regularization

    Use this when:
    - Fine-tuning large models (7B+ parameters)
    - Limited GPU memory
    - Want faster training

    Expected trials: 15-30
    """
    return {
        # LoRA rank: Controls adapter capacity
        # Higher = more expressiveness but more memory
        "lora_r": tune.choice([4, 8, 16, 32]),

        # LoRA alpha: Scaling parameter
        # Typically set to 2 * lora_r, but can be tuned
        "lora_alpha": tune.choice([8, 16, 32, 64]),

        # LoRA dropout: Regularization
        "lora_dropout": tune.uniform(0.0, 0.1),

        # Learning rate: LoRA often needs higher LR than full fine-tuning
        "learning_rate": tune.loguniform(1e-4, 1e-3),

        # Batch size
        "batch_size": tune.choice([8, 16, 32]),

        # Epochs: LoRA typically needs fewer epochs
        "num_epochs": tune.choice([3, 5, 7]),

        # Warmup
        "warmup_ratio": tune.uniform(0.05, 0.15),
    }


def get_large_model_search_space() -> Dict[str, Any]:
    """
    Search space for large models (7B+ parameters).

    Large models have different optimization characteristics:
    - Need smaller learning rates for stability
    - Benefit from longer warmup
    - More sensitive to gradient clipping
    - Often use BF16 or INT8 for memory efficiency

    Use this for:
    - Llama-2-7b and larger
    - GPT-3 scale models
    - Any model that doesn't fit in single GPU

    Expected trials: 20-40
    """
    return {
        # Smaller learning rates for stability
        "learning_rate": tune.loguniform(1e-6, 5e-5),

        # Smaller batch sizes (memory constraints)
        "batch_size": tune.choice([1, 2, 4, 8]),

        # More aggressive gradient accumulation
        "gradient_accumulation_steps": tune.choice([4, 8, 16, 32]),

        # Longer warmup for stability
        "warmup_ratio": tune.uniform(0.1, 0.3),

        # Gradient clipping important for large models
        "max_grad_norm": tune.uniform(0.3, 1.0),

        # Precision: BF16 better for large models (wider range than FP16)
        "precision": tune.choice(["bf16", "int8"]),

        # Usually enable gradient checkpointing for memory
        "gradient_checkpointing": True,

        # Shorter training (large models overfit faster)
        "num_epochs": tune.choice([1, 2, 3]),
    }


def create_custom_search_space(
    learning_rate_range: tuple = (1e-6, 1e-3),
    batch_sizes: list = [8, 16, 32],
    num_epochs_options: list = [3, 5, 7],
    **kwargs
) -> Dict[str, Any]:
    """
    Create a custom search space with common parameters.

    This helper function makes it easy to define custom search spaces
    without writing the full Ray Tune syntax.

    Args:
        learning_rate_range: (min, max) for learning rate
        batch_sizes: List of batch sizes to try
        num_epochs_options: List of epoch counts to try
        **kwargs: Additional tune.* specifications

    Returns:
        Custom search space dictionary

    Example:
        >>> search_space = create_custom_search_space(
        ...     learning_rate_range=(1e-5, 1e-4),
        ...     batch_sizes=[4, 8],
        ...     num_epochs_options=[5],
        ...     weight_decay=tune.uniform(0.01, 0.1),
        ... )
    """
    search_space = {
        "learning_rate": tune.loguniform(*learning_rate_range),
        "batch_size": tune.choice(batch_sizes),
        "num_epochs": tune.choice(num_epochs_options),
    }

    # Add any additional parameters
    search_space.update(kwargs)

    return search_space


# Cost Analysis Functions
def estimate_tuning_cost(
    num_samples: int,
    hours_per_trial: float,
    cost_per_hour: float,
    early_stopping_factor: float = 0.3,
) -> Dict[str, float]:
    """
    Estimate total cost of hyperparameter tuning.

    Args:
        num_samples: Number of trials
        hours_per_trial: Expected hours per full trial
        cost_per_hour: Cost per GPU hour (e.g., $1.50 for g4dn.xlarge)
        early_stopping_factor: Fraction of compute saved by early stopping
                              (ASHA typically saves 60-90%)

    Returns:
        Dictionary with cost estimates

    Example:
        >>> # 20 trials, 2 hours each, $1.50/hour
        >>> estimate_tuning_cost(20, 2, 1.50)
        {'without_early_stopping': 60.0, 'with_early_stopping': 18.0, 'savings': 42.0}
    """
    total_hours_no_stopping = num_samples * hours_per_trial
    total_hours_with_stopping = total_hours_no_stopping * (1 - early_stopping_factor)

    cost_no_stopping = total_hours_no_stopping * cost_per_hour
    cost_with_stopping = total_hours_with_stopping * cost_per_hour

    return {
        "total_trials": num_samples,
        "hours_without_early_stopping": total_hours_no_stopping,
        "hours_with_early_stopping": total_hours_with_stopping,
        "cost_without_early_stopping": cost_no_stopping,
        "cost_with_early_stopping": cost_with_stopping,
        "estimated_savings": cost_no_stopping - cost_with_stopping,
        "savings_percentage": early_stopping_factor * 100,
    }
