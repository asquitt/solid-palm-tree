"""
Training Module - Ray Train Integration

This module provides distributed training capabilities using Ray Train,
supporting data-parallel fine-tuning of LLMs across multiple GPUs and nodes.

Key Components:
- Trainer: High-level training orchestrator
- DistributedTrainer: Multi-node distributed training
- TrainingConfig: Configuration dataclass for training parameters
- callbacks: Training callbacks for logging, checkpointing, etc.
"""

from src.training.trainer import Trainer, DistributedTrainer
from src.training.config import TrainingConfig
from src.training.callbacks import CheckpointCallback, LoggingCallback

__all__ = [
    "Trainer",
    "DistributedTrainer",
    "TrainingConfig",
    "CheckpointCallback",
    "LoggingCallback",
]
