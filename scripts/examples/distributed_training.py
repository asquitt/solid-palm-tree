#!/usr/bin/env python3
"""
Distributed Training Example - Multi-GPU Fine-tuning

Demonstrates distributed training across multiple GPUs using Ray Train.

Usage:
    python scripts/examples/distributed_training.py --num-workers 4
"""

import argparse
import logging
from src.training.config import TrainingConfig
from src.training.trainer import DistributedTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2", help="Model name")
    parser.add_argument("--dataset", default="wikitext", help="Dataset name")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of GPUs")
    parser.add_argument("--use-spot-instances", action="store_true")
    args = parser.parse_args()

    logger.info(f"Launching distributed training with {args.num_workers} workers")

    config = TrainingConfig(
        model_name=args.model,
        dataset_name=args.dataset,
        dataset_config="wikitext-2-raw-v1" if args.dataset == "wikitext" else None,
        num_epochs=3,
        batch_size=8,
        num_workers=args.num_workers,
        use_spot_instances=args.use_spot_instances,
        output_dir="./outputs/distributed",
    )

    trainer = DistributedTrainer(config)
    results = trainer.train_distributed()

    logger.info(f"Training completed: {results}")

if __name__ == "__main__":
    main()
