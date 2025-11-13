"""
Evaluation Module - Comprehensive LLM Metrics

This module provides evaluation metrics specifically designed for
Large Language Models, including both automated metrics and
human preference simulation.

Key Metrics:
- Perplexity: How well the model predicts text
- ROUGE: Recall-oriented text overlap
- BLEU: Precision-oriented translation quality
- BERTScore: Contextual similarity using embeddings
- Custom: Task-specific metrics
"""

from src.evaluation.metrics import LLMEvaluator, compute_perplexity
from src.evaluation.benchmark import ModelBenchmark

__all__ = [
    "LLMEvaluator",
    "compute_perplexity",
    "ModelBenchmark",
]
