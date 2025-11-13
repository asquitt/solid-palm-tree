"""
Hyperparameter Tuning CLI Interface

Command-line interface for hyperparameter optimization.

Usage:
    llm-tune --model gpt2 --dataset wikitext --num-trials 20
"""

import argparse
import logging
import sys

from src.training.config import TrainingConfig
from src.tuning.tuner import HyperparameterTuner, TuningConfig
from src.tuning.search_spaces import get_search_space

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Distributed LLM Platform - Hyperparameter Tuning CLI"
    )

    parser.add_argument("--model", type=str, default="gpt2", help="Model name")
    parser.add_argument("--dataset", type=str, default="wikitext", help="Dataset name")
    parser.add_argument(
        "--num-trials", type=int, default=10, help="Number of trials"
    )
    parser.add_argument(
        "--search-space",
        type=str,
        choices=["quick", "default", "extensive", "lora", "large_model"],
        default="default",
        help="Search space preset",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["asha", "pbt", "fifo"],
        default="asha",
        help="Trial scheduler",
    )
    parser.add_argument(
        "--max-concurrent-trials",
        type=int,
        default=2,
        help="Max concurrent trials",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./outputs/tuning", help="Output directory"
    )

    return parser.parse_args()


def main():
    """Main entry point for tuning CLI."""
    args = parse_args()

    logger.info("Starting hyperparameter optimization")

    # Create base config
    base_config = TrainingConfig(
        model_name=args.model,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
    )

    # Get search space
    search_space = get_search_space(args.search_space)

    # Create tuning config
    tuning_config = TuningConfig(
        num_samples=args.num_trials,
        scheduler=args.scheduler,
        max_concurrent_trials=args.max_concurrent_trials,
        storage_path=args.output_dir,
    )

    # Run tuning
    tuner = HyperparameterTuner(base_config, search_space, tuning_config)
    results = tuner.tune()

    # Get best config
    best_config = tuner.get_best_config(results)

    logger.info("=" * 80)
    logger.info("Hyperparameter optimization completed!")
    logger.info(f"Best configuration: {best_config.to_dict()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
