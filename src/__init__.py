"""
Distributed LLM Fine-tuning & Evaluation Platform

A production-ready platform for distributed fine-tuning and evaluation of
Large Language Models using Ray, Kubernetes, and modern MLOps tools.

Main modules:
- training: Ray Train integration for distributed fine-tuning
- tuning: Ray Tune for hyperparameter optimization
- serving: Ray Serve for model deployment
- evaluation: Custom LLM evaluation metrics
- infrastructure: Kubernetes and cloud resource management
- data: Feast integration and data pipelines
- checkpointing: DVC and cloud storage integration
- optimization: Cost optimization and spot instance management
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__license__ = "MIT"

# Expose main components for convenient imports
from src.training import Trainer, DistributedTrainer
from src.evaluation import LLMEvaluator
from src.serving import ModelServer

__all__ = [
    "Trainer",
    "DistributedTrainer",
    "LLMEvaluator",
    "ModelServer",
]
