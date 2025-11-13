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
- security: Authentication, authorization, and secrets management
- utils: Retry logic, health checks, and monitoring
"""

__version__ = "0.3.0"
__author__ = "Your Name"
__license__ = "MIT"

# Lazy imports to avoid importing heavy dependencies (torch, ray, transformers)
# when they're not needed. Import only when actually used.

__all__ = [
    "Trainer",
    "DistributedTrainer",
    "LLMEvaluator",
    "ModelServer",
]


def __getattr__(name):
    """Lazy import main components only when accessed."""
    if name == "Trainer" or name == "DistributedTrainer":
        from src.training import Trainer, DistributedTrainer
        return Trainer if name == "Trainer" else DistributedTrainer
    elif name == "LLMEvaluator":
        from src.evaluation import LLMEvaluator
        return LLMEvaluator
    elif name == "ModelServer":
        from src.serving import ModelServer
        return ModelServer
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
