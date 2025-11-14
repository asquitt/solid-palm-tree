"""
Week 1 - Trainer

Your task: Implement a training loop for fine-tuning LLMs.

Difficulty: ⭐⭐⭐ Hard
Estimated time: 2-3 hours
"""

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, get_linear_schedule_with_warmup
from typing import Dict, Optional
from tqdm import tqdm


class Trainer:
    """Simple trainer for fine-tuning language models."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        train_dataloader: DataLoader,
        learning_rate: float = 5e-5,
        device: str = "cpu"
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.device = device

        # TODO: Create optimizer
        # HINT: Use AdamW optimizer
        # HINT: Pass model.parameters() and lr=learning_rate
        # YOUR CODE HERE
        self.optimizer = None  # Replace with AdamW optimizer

    def train_epoch(self) -> float:
        """
        Train for one epoch.

        Returns:
            Average loss for the epoch

        TODO: Implement the training loop
        HINT: 1. Set model to training mode: model.train()
        HINT: 2. Loop over batches in train_dataloader
        HINT: 3. Move batch to device
        HINT: 4. Forward pass: model(**batch)
        HINT: 5. Get loss from outputs
        HINT: 6. Backward: loss.backward()
        HINT: 7. Update: optimizer.step()
        HINT: 8. Reset: optimizer.zero_grad()
        HINT: 9. Track total loss and return average
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement train_epoch")


    def save_checkpoint(self, path: str):
        """
        Save model checkpoint.

        Args:
            path: Where to save the checkpoint

        TODO: Implement checkpoint saving
        HINT: Save model.state_dict() and optimizer.state_dict()
        HINT: Use torch.save()
        """
        # YOUR CODE HERE
        raise NotImplementedError("Implement save_checkpoint")


if __name__ == "__main__":
    print("Test your Trainer implementation:")
    print("pytest tests/test_trainer.py -v")
