#!/usr/bin/env python3
"""
Quick Start Example - Fine-tune GPT-2 on WikiText

This script demonstrates the simplest way to fine-tune a model using
the platform. Perfect for learning and testing.

Usage:
    python scripts/examples/quick_start.py

Expected runtime: 10-15 minutes on a single GPU
Expected cost: $0.25-0.50 (on g4dn.xlarge)
"""

import argparse
import logging

from src.training.config import TrainingConfig
from src.training.trainer import Trainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run quick start fine-tuning."""

    # Parse arguments
    parser = argparse.ArgumentParser(description="Quick start fine-tuning example")
    parser.add_argument("--model", default="gpt2", help="Model name")
    parser.add_argument("--dataset", default="wikitext", help="Dataset name")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--output-dir", default="./outputs/quick_start", help="Output directory")
    parser.add_argument("--local", action="store_true", help="Run locally (no Ray)")
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Quick Start: Fine-tuning LLM")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")

    # Create training configuration
    config = TrainingConfig(
        # Model settings
        model_name=args.model,

        # Dataset settings
        dataset_name=args.dataset,
        dataset_config="wikitext-2-raw-v1" if args.dataset == "wikitext" else None,
        max_seq_length=512,

        # Training hyperparameters
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=5e-5,
        warmup_ratio=0.1,

        # Optimization
        precision="fp16",  # Use FP16 for 2x speedup
        gradient_checkpointing=False,  # Disable for faster training in quick start

        # Output
        output_dir=args.output_dir,
        logging_steps=10,
        checkpoint_frequency=500,
        evaluation_steps=500,

        # MLflow tracking
        use_mlflow=True,
        mlflow_tracking_uri="http://localhost:5000",
        mlflow_experiment_name="quick-start",
    )

    # Display estimated memory usage
    # GPT-2 has 124M parameters
    if args.model == "gpt2":
        memory_est = config.estimate_memory_usage(124_000_000)
        logger.info("\nEstimated GPU memory usage:")
        for key, value in memory_est.items():
            logger.info(f"  {key}: {value} GB")
        logger.info(f"\nRecommended GPU: NVIDIA T4 (16GB) or better")

    # Create trainer
    logger.info("\nInitializing trainer...")
    trainer = Trainer(config)

    # Run training
    logger.info("\nStarting training...")
    logger.info("You can monitor progress at http://localhost:5000 (MLflow)")

    results = trainer.train()

    # Display results
    logger.info("\n" + "=" * 80)
    logger.info("Training completed!")
    logger.info("=" * 80)
    logger.info(f"Final loss: {results['final_loss']:.4f}")
    logger.info(f"Best eval loss: {results['best_eval_loss']:.4f}")
    logger.info(f"Total steps: {results['total_steps']}")
    logger.info(f"Training time: {results['training_time'] / 3600:.2f} hours")
    logger.info(f"\nModel saved to: {args.output_dir}/best_model")
    logger.info("\nNext steps:")
    logger.info("  1. View training metrics: mlflow ui --host 0.0.0.0 --port 5000")
    logger.info("  2. Run evaluation: python scripts/examples/evaluate_model.py")
    logger.info("  3. Deploy model: python scripts/examples/serve_model.py")


if __name__ == "__main__":
    main()
