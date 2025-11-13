"""
Distributed Trainer Implementation

This module implements the core training logic using Ray Train for
distributed fine-tuning of LLMs across multiple GPUs and nodes.

Key Components:
- Trainer: High-level training orchestrator
- DistributedTrainer: Ray Train integration for multi-GPU/multi-node
- Training loop with checkpointing, logging, and evaluation

Architecture:
1. Initialize model, tokenizer, and datasets
2. Setup Ray Train for distributed execution
3. Run training loop with:
   - Forward pass (compute loss)
   - Backward pass (compute gradients)
   - Optimizer step (update weights)
   - Checkpointing and logging
4. Evaluate and save final model
"""

import os
import time
import logging
from typing import Optional, Dict, Any, Callable
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_scheduler,
    set_seed,
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
import ray
from ray import train
from ray.train import ScalingConfig, RunConfig, CheckpointConfig
from ray.train.torch import TorchTrainer

from src.training.config import TrainingConfig, OptimizerType, PrecisionType
from src.training.callbacks import CheckpointCallback, LoggingCallback
from src.evaluation.metrics import LLMEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Trainer:
    """
    High-level trainer for LLM fine-tuning.

    This class handles the entire training pipeline:
    1. Loading model and tokenizer
    2. Preparing datasets
    3. Setting up optimizer and scheduler
    4. Running the training loop
    5. Checkpointing and evaluation

    Usage:
        >>> config = TrainingConfig(model_name="gpt2", num_epochs=3)
        >>> trainer = Trainer(config)
        >>> trainer.train()

    Attributes:
        config: Training configuration
        model: The transformer model being fine-tuned
        tokenizer: Tokenizer for text processing
        train_dataloader: DataLoader for training data
        eval_dataloader: DataLoader for evaluation data
        optimizer: Optimizer for weight updates
        scheduler: Learning rate scheduler
    """

    def __init__(self, config: TrainingConfig):
        """
        Initialize the trainer with configuration.

        Args:
            config: Training configuration object
        """
        self.config = config
        self.device = self._get_device()

        # Set random seed for reproducibility
        set_seed(config.seed)

        # Initialize components (lazily loaded)
        self.model = None
        self.tokenizer = None
        self.train_dataloader = None
        self.eval_dataloader = None
        self.optimizer = None
        self.scheduler = None
        self.evaluator = None

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_loss = float('inf')

        logger.info(f"Trainer initialized with config: {config.model_name}")
        logger.info(f"Device: {self.device}")

    def _get_device(self) -> torch.device:
        """Determine the device to use (GPU/CPU)."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")  # Apple Silicon
        else:
            return torch.device("cpu")

    def load_model_and_tokenizer(self):
        """
        Load the pre-trained model and tokenizer from HuggingFace.

        This method handles:
        1. Loading the tokenizer with special tokens
        2. Loading the model with appropriate dtype
        3. Applying LoRA if configured
        4. Moving model to device

        Memory Optimization Tips:
        - Use FP16/BF16 to reduce memory by 50%
        - Enable gradient_checkpointing to trade compute for memory
        - Use LoRA to only train <1% of parameters
        """
        logger.info(f"Loading model: {self.config.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_name,
            cache_dir=self.config.cache_dir,
            use_fast=True,  # Use fast Rust-based tokenizer
        )

        # Add padding token if not present (needed for batching)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Determine model dtype based on precision
        torch_dtype_map = {
            PrecisionType.FP32: torch.float32,
            PrecisionType.FP16: torch.float16,
            PrecisionType.BF16: torch.bfloat16,
            PrecisionType.INT8: torch.float16,  # Load as FP16, quantize later
        }
        torch_dtype = torch_dtype_map[self.config.precision]

        # Load model with appropriate settings
        load_kwargs = {
            "pretrained_model_name_or_path": self.config.model_name,
            "cache_dir": self.config.cache_dir,
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,  # Load model piece by piece (saves RAM)
        }

        # For 8-bit quantization (QLoRA)
        if self.config.precision == PrecisionType.INT8:
            load_kwargs["load_in_8bit"] = True
            load_kwargs["device_map"] = "auto"  # Automatic device placement

        self.model = AutoModelForCausalLM.from_pretrained(**load_kwargs)

        # Enable gradient checkpointing if configured
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled (saves memory)")

        # Apply LoRA for parameter-efficient fine-tuning
        if self.config.use_lora:
            logger.info(f"Applying LoRA with rank={self.config.lora_r}")

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        # Move model to device (unless using 8-bit which handles this automatically)
        if self.config.precision != PrecisionType.INT8:
            self.model = self.model.to(self.device)

        logger.info(f"Model loaded successfully")
        logger.info(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

    def prepare_datasets(self):
        """
        Load and preprocess training and evaluation datasets.

        This method handles:
        1. Loading datasets from HuggingFace or local files
        2. Tokenizing text data
        3. Grouping tokens into sequences of max_seq_length
        4. Creating DataLoaders for batching

        Dataset Processing Pipeline:
        Raw Text -> Tokenize -> Group into sequences -> DataLoader -> Batches
        """
        logger.info(f"Loading dataset: {self.config.dataset_name}")

        # Load dataset from HuggingFace
        dataset = load_dataset(
            self.config.dataset_name,
            self.config.dataset_config,
            cache_dir=self.config.cache_dir,
        )

        # Get train and validation splits
        train_dataset = dataset[self.config.train_split]
        eval_dataset = dataset[self.config.validation_split]

        logger.info(f"Train examples: {len(train_dataset)}")
        logger.info(f"Eval examples: {len(eval_dataset)}")

        # Tokenization function
        def tokenize_function(examples):
            """Tokenize text examples."""
            # Concatenate all text with EOS token separator
            texts = examples.get("text", examples.get("content", []))
            return self.tokenizer(
                texts,
                truncation=True,
                max_length=self.config.max_seq_length,
                padding="max_length",
                return_tensors=None,  # Return lists, not tensors
            )

        # Tokenize datasets in parallel
        logger.info("Tokenizing datasets...")
        tokenized_train = train_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=self.config.preprocessing_num_workers,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing train dataset",
        )
        tokenized_eval = eval_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=self.config.preprocessing_num_workers,
            remove_columns=eval_dataset.column_names,
            desc="Tokenizing eval dataset",
        )

        # Set format for PyTorch
        tokenized_train.set_format(type="torch")
        tokenized_eval.set_format(type="torch")

        # Create DataLoaders
        self.train_dataloader = DataLoader(
            tokenized_train,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=self.config.dataloader_pin_memory,
        )
        self.eval_dataloader = DataLoader(
            tokenized_eval,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=self.config.dataloader_pin_memory,
        )

        logger.info(f"DataLoaders created: {len(self.train_dataloader)} train batches")

    def setup_optimizer_and_scheduler(self):
        """
        Configure optimizer and learning rate scheduler.

        Optimizer Options:
        - AdamW: Adam with weight decay (most popular for LLMs)
        - Adam: Standard Adam
        - SGD: Stochastic Gradient Descent
        - Adafactor: Memory-efficient optimizer

        Scheduler Options:
        - Cosine: Smooth decay following cosine curve (recommended)
        - Linear: Linear decay to 0
        - Constant: No decay
        """
        # Separate parameters with and without weight decay
        # Typically exclude bias and LayerNorm from weight decay
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if p.requires_grad and any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        # Create optimizer
        optimizer_class = {
            OptimizerType.ADAMW: torch.optim.AdamW,
            OptimizerType.ADAM: torch.optim.Adam,
            OptimizerType.SGD: torch.optim.SGD,
        }[self.config.optimizer]

        self.optimizer = optimizer_class(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
        )

        # Calculate total training steps
        num_update_steps_per_epoch = len(self.train_dataloader) // self.config.gradient_accumulation_steps
        max_train_steps = self.config.num_epochs * num_update_steps_per_epoch

        # Calculate warmup steps
        if self.config.warmup_steps > 0:
            num_warmup_steps = self.config.warmup_steps
        else:
            num_warmup_steps = int(max_train_steps * self.config.warmup_ratio)

        # Create learning rate scheduler
        self.scheduler = get_scheduler(
            name=self.config.scheduler.value,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_train_steps,
        )

        logger.info(f"Optimizer: {self.config.optimizer.value}")
        logger.info(f"Scheduler: {self.config.scheduler.value}")
        logger.info(f"Total training steps: {max_train_steps}")
        logger.info(f"Warmup steps: {num_warmup_steps}")

    def train(self):
        """
        Main training loop.

        Training Loop Steps:
        1. Forward pass: compute loss
        2. Backward pass: compute gradients
        3. Gradient accumulation: accumulate over multiple micro-batches
        4. Optimizer step: update weights
        5. Scheduler step: adjust learning rate
        6. Logging: track metrics
        7. Checkpointing: save model periodically
        8. Evaluation: validate on held-out data
        """
        # Initialize everything
        if self.model is None:
            self.load_model_and_tokenizer()
        if self.train_dataloader is None:
            self.prepare_datasets()
        if self.optimizer is None:
            self.setup_optimizer_and_scheduler()

        # Initialize evaluator
        self.evaluator = LLMEvaluator()

        # Create output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 80)
        logger.info("Starting training")
        logger.info("=" * 80)

        # Training metrics
        total_loss = 0.0
        total_steps = 0
        start_time = time.time()

        # Set model to training mode
        self.model.train()

        # Main training loop
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            logger.info(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")

            epoch_loss = 0.0
            epoch_steps = 0

            for step, batch in enumerate(self.train_dataloader):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["input_ids"],  # For causal LM, labels = input_ids
                )
                loss = outputs.loss

                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps

                # Backward pass
                loss.backward()

                # Accumulate loss for logging
                total_loss += loss.item()
                epoch_loss += loss.item()

                # Update weights after accumulating gradients
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )

                    # Optimizer step
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    self.global_step += 1
                    total_steps += 1
                    epoch_steps += 1

                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        avg_loss = total_loss / self.config.logging_steps
                        lr = self.scheduler.get_last_lr()[0]
                        elapsed = time.time() - start_time
                        steps_per_sec = total_steps / elapsed

                        logger.info(
                            f"Step {self.global_step} | "
                            f"Loss: {avg_loss:.4f} | "
                            f"LR: {lr:.2e} | "
                            f"Speed: {steps_per_sec:.2f} steps/s"
                        )
                        total_loss = 0.0

                    # Checkpointing
                    if self.global_step % self.config.checkpoint_frequency == 0:
                        self.save_checkpoint(step=self.global_step)

                    # Evaluation
                    if self.global_step % self.config.evaluation_steps == 0:
                        eval_metrics = self.evaluate()
                        logger.info(f"Evaluation metrics: {eval_metrics}")

                        # Save best model
                        if eval_metrics["loss"] < self.best_eval_loss:
                            self.best_eval_loss = eval_metrics["loss"]
                            self.save_checkpoint(step=self.global_step, is_best=True)
                            logger.info(f"New best model saved! Loss: {self.best_eval_loss:.4f}")

            # End of epoch
            avg_epoch_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0
            logger.info(f"Epoch {epoch + 1} completed | Average loss: {avg_epoch_loss:.4f}")

        # Training completed
        total_time = time.time() - start_time
        logger.info("=" * 80)
        logger.info(f"Training completed in {total_time / 3600:.2f} hours")
        logger.info("=" * 80)

        # Save final model
        self.save_checkpoint(step=self.global_step, is_final=True)

        return {
            "final_loss": avg_epoch_loss,
            "best_eval_loss": self.best_eval_loss,
            "total_steps": self.global_step,
            "training_time": total_time,
        }

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model on validation set.

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Running evaluation...")
        self.model.eval()

        total_loss = 0.0
        total_steps = 0

        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["input_ids"],
                )
                total_loss += outputs.loss.item()
                total_steps += 1

        avg_loss = total_loss / total_steps
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        self.model.train()

        return {
            "loss": avg_loss,
            "perplexity": perplexity,
        }

    def save_checkpoint(self, step: int, is_best: bool = False, is_final: bool = False):
        """
        Save model checkpoint.

        Args:
            step: Current training step
            is_best: Whether this is the best model so far
            is_final: Whether this is the final model
        """
        checkpoint_dir = Path(self.config.output_dir) / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model and tokenizer
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        # Save training state
        state = {
            "global_step": self.global_step,
            "epoch": self.current_epoch,
            "best_eval_loss": self.best_eval_loss,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "config": self.config.to_dict(),
        }
        torch.save(state, checkpoint_dir / "training_state.pt")

        logger.info(f"Checkpoint saved: {checkpoint_dir}")

        # Save symlinks for best and final models
        if is_best:
            best_dir = Path(self.config.output_dir) / "best_model"
            if best_dir.exists():
                best_dir.unlink()
            best_dir.symlink_to(checkpoint_dir.name)

        if is_final:
            final_dir = Path(self.config.output_dir) / "final_model"
            if final_dir.exists():
                final_dir.unlink()
            final_dir.symlink_to(checkpoint_dir.name)

class DistributedTrainer(Trainer):
    """
    Distributed trainer using Ray Train for multi-GPU/multi-node training.

    This class extends Trainer to support distributed training across
    multiple GPUs and nodes using Ray Train's data parallelism.

    Key Features:
    - Automatic data sharding across workers
    - Gradient synchronization using AllReduce
    - Fault tolerance with checkpointing
    - Seamless scaling from 1 GPU to 100s of GPUs

    Usage:
        >>> config = TrainingConfig(num_workers=4)
        >>> trainer = DistributedTrainer(config)
        >>> trainer.train_distributed()
    """

    def __init__(self, config: TrainingConfig):
        """Initialize distributed trainer."""
        super().__init__(config)

    def train_distributed(self):
        """
        Launch distributed training using Ray Train.

        This method creates a Ray Train TorchTrainer that handles:
        1. Launching workers on multiple GPUs/nodes
        2. Setting up distributed process groups
        3. Coordinating training across workers
        4. Checkpointing and fault recovery
        """
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        # Define training function for each worker
        def train_func(config_dict: Dict[str, Any]):
            """Training function executed on each distributed worker."""
            # Recreate config from dictionary
            config = TrainingConfig.from_dict(config_dict)

            # Create trainer for this worker
            trainer = Trainer(config)

            # Setup distributed training
            trainer.model = train.torch.prepare_model(trainer.model)
            trainer.train_dataloader = train.torch.prepare_data_loader(
                trainer.train_dataloader
            )

            # Run training loop
            return trainer.train()

        # Configure scaling
        scaling_config = ScalingConfig(
            num_workers=self.config.num_workers,
            use_gpu=True,
            resources_per_worker={
                "CPU": 4,
                "GPU": 1,
            },
        )

        # Configure checkpointing
        checkpoint_config = CheckpointConfig(
            num_to_keep=self.config.save_total_limit,
            checkpoint_score_attribute="loss",
            checkpoint_score_order="min",
        )

        # Configure run
        run_config = RunConfig(
            name=self.config.mlflow_run_name or f"run-{int(time.time())}",
            storage_path=self.config.output_dir,
            checkpoint_config=checkpoint_config,
        )

        # Create Ray TorchTrainer
        trainer = TorchTrainer(
            train_loop_per_worker=train_func,
            train_loop_config=self.config.to_dict(),
            scaling_config=scaling_config,
            run_config=run_config,
        )

        # Launch training
        logger.info(f"Launching distributed training with {self.config.num_workers} workers")
        result = trainer.fit()

        logger.info("Distributed training completed")
        return result
