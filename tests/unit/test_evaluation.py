"""Unit Tests for Evaluation Metrics"""

import pytest
from src.evaluation.metrics import (
    compute_rouge,
    compute_bleu,
    compute_diversity_metrics,
    LLMEvaluator,
)


class TestROUGE:
    """Test ROUGE metrics."""

    def test_perfect_match(self):
        """Test ROUGE with perfect match."""
        predictions = ["The cat sat on the mat"]
        references = ["The cat sat on the mat"]

        scores = compute_rouge(predictions, references)

        # Perfect match should have score near 1.0
        assert scores["rouge1"] > 0.99
        assert scores["rouge2"] > 0.99
        assert scores["rougeL"] > 0.99

    def test_no_overlap(self):
        """Test ROUGE with no overlap."""
        predictions = ["The quick brown fox"]
        references = ["A lazy sleeping dog"]

        scores = compute_rouge(predictions, references)

        # No overlap should have score near 0
        assert scores["rouge1"] < 0.1


class TestBLEU:
    """Test BLEU metrics."""

    def test_perfect_match(self):
        """Test BLEU with perfect match."""
        predictions = ["The cat sat on the mat"]
        references = ["The cat sat on the mat"]

        scores = compute_bleu(predictions, references)

        # Perfect match should have high score
        assert scores["bleu_corpus"] > 90


class TestDiversity:
    """Test diversity metrics."""

    def test_diversity_calculation(self):
        """Test diversity metric calculation."""
        texts = [
            "The cat sat on the mat",
            "A dog ran in the park",
            "Birds fly in the sky",
        ]

        scores = compute_diversity_metrics(texts)

        assert "distinct_1" in scores
        assert "distinct_2" in scores
        assert "entropy" in scores
        assert 0 <= scores["distinct_1"] <= 1
        assert 0 <= scores["distinct_2"] <= 1
        assert scores["entropy"] > 0


class TestLLMEvaluator:
    """Test LLMEvaluator class."""

    def test_initialization(self):
        """Test evaluator initialization."""
        evaluator = LLMEvaluator()
        assert evaluator is not None

    def test_evaluate_with_all_metrics(self):
        """Test evaluation with all metrics."""
        evaluator = LLMEvaluator()

        predictions = ["The cat sat on the mat"]
        references = ["A cat was sitting on a mat"]

        scores = evaluator.evaluate(
            predictions,
            references,
            metrics=["rouge", "bleu", "diversity"],
        )

        assert "rouge1" in scores
        assert "bleu_corpus" in scores
        assert "distinct_1" in scores
