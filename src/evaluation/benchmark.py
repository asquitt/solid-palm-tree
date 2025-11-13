"""
Model Benchmark Suite

This module provides comprehensive benchmarking tools for comparing
fine-tuned models against baselines and each other.

Features:
- Automatic baseline comparison
- Performance metrics tracking
- Cost-per-token analysis
- Inference speed benchmarking
- Memory usage profiling
"""

import time
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from src.evaluation.metrics import LLMEvaluator, compute_perplexity

logger = logging.getLogger(__name__)


class ModelBenchmark:
    """
    Comprehensive model benchmarking suite.

    This class helps you:
    1. Compare your fine-tuned model against baselines
    2. Measure inference speed and memory usage
    3. Calculate cost-per-token for deployment planning
    4. Generate comparison reports

    Usage:
        >>> benchmark = ModelBenchmark()
        >>> results = benchmark.compare_models([
        ...     "gpt2",  # baseline
        ...     "./outputs/my-finetuned-model"  # your model
        ... ])
        >>> benchmark.generate_report(results)
    """

    def __init__(
        self,
        device: Optional[str] = None,
        test_dataset: str = "wikitext",
        test_config: str = "wikitext-2-raw-v1",
        num_samples: int = 100,
    ):
        """
        Initialize benchmark suite.

        Args:
            device: Device to run benchmarks on (cuda/cpu)
            test_dataset: Dataset for evaluation
            test_config: Dataset configuration
            num_samples: Number of samples to test on
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.test_dataset = test_dataset
        self.test_config = test_config
        self.num_samples = num_samples
        self.evaluator = LLMEvaluator(device=self.device)

        logger.info(f"ModelBenchmark initialized on {self.device}")

    def benchmark_model(
        self,
        model_path: str,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Benchmark a single model comprehensively.

        Metrics computed:
        - Perplexity on test set
        - Inference speed (tokens/sec)
        - Memory usage (GB)
        - Generation quality (ROUGE, BLEU, etc.)
        - Cost per 1K tokens (estimated)

        Args:
            model_path: Path to model or HuggingFace model ID
            model_name: Display name for model (optional)

        Returns:
            Dictionary with benchmark results
        """
        model_name = model_name or Path(model_path).name

        logger.info(f"Benchmarking model: {model_name}")

        # Load model and tokenizer
        logger.info("Loading model...")
        start_time = time.time()
        model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        load_time = time.time() - start_time

        # Get model size
        num_params = sum(p.numel() for p in model.parameters())
        model_size_mb = num_params * 4 / (1024**2)  # Assuming FP32

        # Load test data
        logger.info("Loading test dataset...")
        dataset = load_dataset(
            self.test_dataset, self.test_config, split="test"
        )

        # Sample data
        if len(dataset) > self.num_samples:
            dataset = dataset.shuffle(seed=42).select(range(self.num_samples))

        # Prepare data
        def tokenize(examples):
            return tokenizer(
                examples.get("text", examples.get("content", [])),
                truncation=True,
                padding="max_length",
                max_length=512,
            )

        tokenized = dataset.map(tokenize, batched=True)
        tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

        from torch.utils.data import DataLoader

        dataloader = DataLoader(tokenized, batch_size=8)

        # Benchmark perplexity
        logger.info("Computing perplexity...")
        perplexity = compute_perplexity(model, dataloader, self.device)

        # Benchmark inference speed
        logger.info("Benchmarking inference speed...")
        inference_times = []
        total_tokens = 0

        model.eval()
        with torch.no_grad():
            for batch in list(dataloader)[:10]:  # Test on 10 batches
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                start = time.time()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                elapsed = time.time() - start

                batch_tokens = attention_mask.sum().item()
                total_tokens += batch_tokens
                inference_times.append(elapsed)

        avg_inference_time = sum(inference_times) / len(inference_times)
        tokens_per_second = total_tokens / sum(inference_times)

        # Benchmark generation speed
        logger.info("Benchmarking generation speed...")
        prompt = "The quick brown fox"
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

        generation_times = []
        for _ in range(10):
            start = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_length=50, do_sample=False, pad_token_id=tokenizer.eos_token_id
                )
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            generation_times.append(time.time() - start)

        avg_generation_time = sum(generation_times) / len(generation_times)
        tokens_generated = outputs.shape[1] - inputs["input_ids"].shape[1]
        generation_tokens_per_sec = tokens_generated / avg_generation_time

        # Memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.max_memory_allocated(self.device) / (1024**3)
            memory_reserved = torch.cuda.max_memory_reserved(self.device) / (1024**3)
        else:
            memory_allocated = 0
            memory_reserved = 0

        # Estimate cost (based on typical cloud GPU pricing)
        # Assuming $1.50/hour for a T4 GPU
        cost_per_hour = 1.50
        cost_per_1k_tokens = (cost_per_hour / 3600) * (1000 / tokens_per_second)

        results = {
            "model_name": model_name,
            "model_path": model_path,
            "num_parameters": num_params,
            "model_size_mb": round(model_size_mb, 2),
            "load_time_seconds": round(load_time, 2),
            "perplexity": round(perplexity, 4),
            "inference_tokens_per_second": round(tokens_per_second, 2),
            "generation_tokens_per_second": round(generation_tokens_per_sec, 2),
            "avg_inference_time_ms": round(avg_inference_time * 1000, 2),
            "avg_generation_time_ms": round(avg_generation_time * 1000, 2),
            "memory_allocated_gb": round(memory_allocated, 2),
            "memory_reserved_gb": round(memory_reserved, 2),
            "estimated_cost_per_1k_tokens": round(cost_per_1k_tokens, 6),
            "device": str(self.device),
        }

        logger.info(f"Benchmark complete for {model_name}")

        # Clean up
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return results

    def compare_models(
        self,
        model_paths: List[str],
        model_names: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Compare multiple models.

        Args:
            model_paths: List of model paths or HuggingFace IDs
            model_names: Optional display names

        Returns:
            List of benchmark results for each model
        """
        if model_names is None:
            model_names = [None] * len(model_paths)

        results = []
        for path, name in zip(model_paths, model_names):
            try:
                result = self.benchmark_model(path, name)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to benchmark {path}: {e}")
                results.append({"model_name": name or path, "error": str(e)})

        return results

    def generate_report(
        self,
        results: List[Dict[str, Any]],
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate a comparison report.

        Args:
            results: List of benchmark results
            output_path: Optional path to save report

        Returns:
            Report as markdown string
        """
        report_lines = [
            "# Model Benchmark Report",
            "",
            f"**Test Dataset**: {self.test_dataset} ({self.test_config})",
            f"**Number of Samples**: {self.num_samples}",
            f"**Device**: {self.device}",
            "",
            "## Summary Comparison",
            "",
        ]

        # Create comparison table
        headers = [
            "Model",
            "Perplexity ↓",
            "Speed (tok/s) ↑",
            "Memory (GB)",
            "Cost/1K tok",
        ]

        report_lines.append("| " + " | ".join(headers) + " |")
        report_lines.append("|" + "|".join(["---"] * len(headers)) + "|")

        for result in results:
            if "error" in result:
                continue

            row = [
                result["model_name"],
                f"{result['perplexity']:.2f}",
                f"{result['inference_tokens_per_second']:.1f}",
                f"{result['memory_allocated_gb']:.2f}",
                f"${result['estimated_cost_per_1k_tokens']:.6f}",
            ]
            report_lines.append("| " + " | ".join(row) + " |")

        report_lines.extend(
            [
                "",
                "## Detailed Metrics",
                "",
            ]
        )

        # Detailed breakdown for each model
        for result in results:
            if "error" in result:
                report_lines.extend(
                    [
                        f"### {result['model_name']}",
                        "",
                        f"**Error**: {result['error']}",
                        "",
                    ]
                )
                continue

            report_lines.extend(
                [
                    f"### {result['model_name']}",
                    "",
                    "**Model Info**:",
                    f"- Parameters: {result['num_parameters']:,}",
                    f"- Size: {result['model_size_mb']:.2f} MB",
                    f"- Load Time: {result['load_time_seconds']:.2f}s",
                    "",
                    "**Quality Metrics**:",
                    f"- Perplexity: {result['perplexity']:.4f} (lower is better)",
                    "",
                    "**Performance Metrics**:",
                    f"- Inference Speed: {result['inference_tokens_per_second']:.2f} tokens/sec",
                    f"- Generation Speed: {result['generation_tokens_per_second']:.2f} tokens/sec",
                    f"- Avg Inference Time: {result['avg_inference_time_ms']:.2f} ms",
                    f"- Avg Generation Time: {result['avg_generation_time_ms']:.2f} ms",
                    "",
                    "**Resource Usage**:",
                    f"- Memory Allocated: {result['memory_allocated_gb']:.2f} GB",
                    f"- Memory Reserved: {result['memory_reserved_gb']:.2f} GB",
                    "",
                    "**Cost Estimate**:",
                    f"- Cost per 1K tokens: ${result['estimated_cost_per_1k_tokens']:.6f}",
                    f"- Estimated monthly cost (1M tokens/day): ${result['estimated_cost_per_1k_tokens'] * 1000 * 30:.2f}",
                    "",
                ]
            )

        # Add recommendations
        if len([r for r in results if "error" not in r]) > 1:
            report_lines.extend(
                [
                    "## Recommendations",
                    "",
                ]
            )

            valid_results = [r for r in results if "error" not in r]

            # Best perplexity
            best_quality = min(valid_results, key=lambda x: x["perplexity"])
            report_lines.append(
                f"- **Best Quality**: {best_quality['model_name']} "
                f"(perplexity: {best_quality['perplexity']:.2f})"
            )

            # Fastest inference
            fastest = max(
                valid_results, key=lambda x: x["inference_tokens_per_second"]
            )
            report_lines.append(
                f"- **Fastest Inference**: {fastest['model_name']} "
                f"({fastest['inference_tokens_per_second']:.1f} tokens/sec)"
            )

            # Most cost-effective
            cheapest = min(
                valid_results, key=lambda x: x["estimated_cost_per_1k_tokens"]
            )
            report_lines.append(
                f"- **Most Cost-Effective**: {cheapest['model_name']} "
                f"(${cheapest['estimated_cost_per_1k_tokens']:.6f} per 1K tokens)"
            )

            # Memory efficient
            efficient = min(valid_results, key=lambda x: x["memory_allocated_gb"])
            report_lines.append(
                f"- **Most Memory Efficient**: {efficient['model_name']} "
                f"({efficient['memory_allocated_gb']:.2f} GB)"
            )

        report = "\n".join(report_lines)

        # Save if path provided
        if output_path:
            with open(output_path, "w") as f:
                f.write(report)
            logger.info(f"Report saved to {output_path}")

            # Also save JSON
            json_path = Path(output_path).with_suffix(".json")
            with open(json_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"JSON results saved to {json_path}")

        return report


def quick_benchmark(model_path: str, baseline: str = "gpt2") -> str:
    """
    Quick benchmark comparing a model against baseline.

    Args:
        model_path: Path to fine-tuned model
        baseline: Baseline model to compare against

    Returns:
        Markdown report

    Example:
        >>> report = quick_benchmark("./outputs/my-model")
        >>> print(report)
    """
    benchmark = ModelBenchmark(num_samples=50)

    logger.info("Running quick benchmark...")
    results = benchmark.compare_models(
        [baseline, model_path], ["Baseline (GPT-2)", "Fine-tuned Model"]
    )

    report = benchmark.generate_report(results)
    return report
