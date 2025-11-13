#!/usr/bin/env python3
"""
Hyperparameter Search Example

Demonstrates distributed hyperparameter optimization using Ray Tune.

Usage:
    python scripts/examples/hyperparameter_search.py --num-trials 20
"""

import argparse
import logging
from src.training.config import TrainingConfig
from src.tuning.tuner import HyperparameterTuner, TuningConfig
from src.tuning.search_spaces import get_search_space

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--scheduler", default="asha", choices=["asha", "pbt", "fifo"])
    args = parser.parse_args()

    logger.info("Starting hyperparameter optimization")

    base_config = TrainingConfig(
        model_name=args.model,
        dataset_name=args.dataset,
        dataset_config="wikitext-2-raw-v1",
        output_dir="./outputs/tuning",
    )

    search_space = get_search_space("quick")

    tuning_config = TuningConfig(
        num_samples=args.num_trials,
        scheduler=args.scheduler,
        max_concurrent_trials=2,
    )

    tuner = HyperparameterTuner(base_config, search_space, tuning_config)
    results = tuner.tune()

    best_config = tuner.get_best_config(results)
    logger.info(f"Best config: {best_config}")

if __name__ == "__main__":
    main()
