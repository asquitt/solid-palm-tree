"""
Week 1 - Evaluator

Your task: Implement evaluation metrics for language models.

Difficulty: ⭐⭐ Medium
Estimated time: 1 hour
"""

import torch
import math
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM
from typing import Dict


def evaluate(model: AutoModelForCausalLM, dataloader: DataLoader, device: str = "cpu") -> Dict[str, float]:
    """
    Evaluate the model.

    Args:
        model: Model to evaluate
        dataloader: Evaluation data
        device: Device to use

    Returns:
        Dictionary with metrics:
        - "loss": Average loss
        - "perplexity": Perplexity (exp(loss))

    TODO: Implement evaluation
    HINT: 1. Set model to eval mode: model.eval()
    HINT: 2. Use torch.no_grad() for inference
    HINT: 3. Loop over batches, calculate loss
    HINT: 4. Return average loss and perplexity
    HINT: perplexity = math.exp(average_loss)
    """
    # YOUR CODE HERE
    raise NotImplementedError("Implement evaluate")


if __name__ == "__main__":
    print("Test your Evaluator implementation:")
    print("pytest tests/test_evaluator.py -v")
