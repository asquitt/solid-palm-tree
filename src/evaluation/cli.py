"""
Evaluation CLI Interface

Command-line interface for evaluating models.

Usage:
    llm-eval --model ./outputs/my-model --dataset wikitext
"""

import argparse
import logging
import sys

from src.evaluation.benchmark import ModelBenchmark, quick_benchmark

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Distributed LLM Platform - Evaluation CLI"
    )

    parser.add_argument(
        "--model", type=str, required=True, help="Model path or HuggingFace ID"
    )
    parser.add_argument(
        "--baseline", type=str, default="gpt2", help="Baseline model for comparison"
    )
    parser.add_argument(
        "--dataset", type=str, default="wikitext", help="Test dataset"
    )
    parser.add_argument(
        "--num-samples", type=int, default=100, help="Number of test samples"
    )
    parser.add_argument(
        "--output", type=str, help="Output path for report"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run quick benchmark"
    )

    return parser.parse_args()


def main():
    """Main entry point for evaluation CLI."""
    args = parse_args()

    logger.info("Starting model evaluation")

    if args.quick:
        report = quick_benchmark(args.model, args.baseline)
    else:
        benchmark = ModelBenchmark(
            test_dataset=args.dataset,
            num_samples=args.num_samples,
        )

        results = benchmark.compare_models(
            [args.baseline, args.model],
            ["Baseline", "Fine-tuned"],
        )

        report = benchmark.generate_report(results, args.output)

    print("\n" + report)

    if args.output:
        logger.info(f"Report saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
