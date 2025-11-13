"""
Training CLI Interface

Command-line interface for training LLMs.

Usage:
    llm-train --model gpt2 --dataset wikitext --epochs 3
    llm-train --config path/to/config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml

from src.training.config import TrainingConfig
from src.training.trainer import Trainer, DistributedTrainer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Distributed LLM Fine-tuning Platform - Training CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file
    parser.add_argument(
        "--config", type=str, help="Path to YAML configuration file"
    )

    # Model settings
    parser.add_argument(
        "--model", type=str, default="gpt2", help="Model name or path"
    )
    parser.add_argument("--dataset", type=str, default="wikitext", help="Dataset name")
    parser.add_argument(
        "--dataset-config", type=str, help="Dataset configuration/subset"
    )

    # Training settings
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=5e-5, help="Learning rate"
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=512, help="Maximum sequence length"
    )

    # Optimization
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "fp16", "bf16", "int8"],
        default="fp16",
        help="Training precision",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing",
    )
    parser.add_argument(
        "--use-lora", action="store_true", help="Use LoRA for efficient fine-tuning"
    )
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")

    # Distributed training
    parser.add_argument(
        "--num-workers", type=int, default=1, help="Number of distributed workers"
    )
    parser.add_argument(
        "--distributed", action="store_true", help="Use distributed training"
    )

    # Output
    parser.add_argument(
        "--output-dir", type=str, default="./outputs", help="Output directory"
    )

    # Logging
    parser.add_argument(
        "--use-mlflow", action="store_true", default=True, help="Use MLflow tracking"
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default="http://localhost:5000",
        help="MLflow tracking URI",
    )

    # Cost optimization
    parser.add_argument(
        "--use-spot-instances",
        action="store_true",
        help="Use spot instances for cost savings",
    )

    return parser.parse_args()


def load_config_from_yaml(path: str) -> TrainingConfig:
    """Load configuration from YAML file."""
    with open(path, "r") as f:
        config_dict = yaml.safe_load(f)
    return TrainingConfig.from_dict(config_dict)


def main():
    """Main entry point for training CLI."""
    args = parse_args()

    logger.info("=" * 80)
    logger.info("Distributed LLM Fine-tuning Platform - Training")
    logger.info("=" * 80)

    # Load or create config
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config = load_config_from_yaml(args.config)
    else:
        logger.info("Creating configuration from arguments")
        config = TrainingConfig(
            model_name=args.model,
            dataset_name=args.dataset,
            dataset_config=args.dataset_config,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_seq_length=args.max_seq_length,
            precision=args.precision,
            gradient_checkpointing=args.gradient_checkpointing,
            use_lora=args.use_lora,
            lora_r=args.lora_r,
            num_workers=args.num_workers,
            output_dir=args.output_dir,
            use_mlflow=args.use_mlflow,
            mlflow_tracking_uri=args.mlflow_uri,
            use_spot_instances=args.use_spot_instances,
        )

    # Display configuration
    logger.info("\nConfiguration:")
    logger.info(f"  Model: {config.model_name}")
    logger.info(f"  Dataset: {config.dataset_name}")
    logger.info(f"  Epochs: {config.num_epochs}")
    logger.info(f"  Batch Size: {config.batch_size}")
    logger.info(f"  Learning Rate: {config.learning_rate}")
    logger.info(f"  Precision: {config.precision.value}")
    logger.info(f"  Workers: {config.num_workers}")
    logger.info(f"  Output: {config.output_dir}")

    # Create trainer
    if args.distributed or config.num_workers > 1:
        logger.info("\nUsing distributed training")
        trainer = DistributedTrainer(config)
        results = trainer.train_distributed()
    else:
        logger.info("\nUsing single-node training")
        trainer = Trainer(config)
        results = trainer.train()

    # Display results
    logger.info("\n" + "=" * 80)
    logger.info("Training completed!")
    logger.info("=" * 80)
    logger.info(f"Results: {results}")
    logger.info(f"Model saved to: {config.output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
