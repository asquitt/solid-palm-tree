"""
Hyperparameter Tuning Module - Ray Tune Integration

This module provides distributed hyperparameter optimization using Ray Tune
with support for various search algorithms and schedulers.

Key Features:
- Multiple search algorithms (Grid, Random, Bayesian, ASHA, PBT)
- Early stopping to save compute
- Parallel trial execution
- Automatic checkpointing and resumption
"""

from src.tuning.tuner import HyperparameterTuner, TuningConfig
from src.tuning.search_spaces import get_search_space

__all__ = [
    "HyperparameterTuner",
    "TuningConfig",
    "get_search_space",
]
