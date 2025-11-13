"""
LLM Evaluation Metrics

This module implements comprehensive evaluation metrics for assessing
the quality of fine-tuned language models.

Metric Categories:
1. Perplexity: Measures how well model predicts text (lower is better)
2. ROUGE: Measures text overlap for summarization tasks
3. BLEU: Measures translation quality
4. BERTScore: Measures semantic similarity using contextual embeddings
5. Task-specific: Custom metrics for specific use cases

When to use each metric:
- Perplexity: General language modeling quality
- ROUGE: Summarization, abstractive generation
- BLEU: Translation, exact match tasks
- BERTScore: Semantic similarity, paraphrase detection
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import math

import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu
import bert_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

# Download NLTK data for BLEU computation
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

logger = logging.getLogger(__name__)


def compute_perplexity(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> float:
    """
    Compute perplexity on a dataset.

    Perplexity measures how well the model predicts the next token.
    Lower perplexity = better model.

    Formula: perplexity = exp(average_cross_entropy_loss)

    Interpretation:
    - Perplexity of N means the model is as confused as if it had to
      randomly choose between N equally likely tokens
    - Good LLMs have perplexity < 20 on test data
    - Random baseline would have perplexity = vocabulary_size

    Args:
        model: The language model to evaluate
        dataloader: DataLoader with evaluation data
        device: Device to run evaluation on

    Returns:
        Perplexity score (float)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,  # For causal LM, labels = inputs
            )

            # Accumulate loss weighted by number of tokens
            loss = outputs.loss
            num_tokens = attention_mask.sum().item()

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    # Calculate average loss and perplexity
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return perplexity


def compute_rouge(
    predictions: List[str],
    references: List[str],
    rouge_types: List[str] = ["rouge1", "rouge2", "rougeL"]
) -> Dict[str, float]:
    """
    Compute ROUGE scores for text generation tasks.

    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures
    text overlap between generated and reference text.

    ROUGE Types:
    - ROUGE-1: Unigram (single word) overlap
    - ROUGE-2: Bigram (two word) overlap
    - ROUGE-L: Longest Common Subsequence

    When to use:
    - Summarization: ROUGE-L captures long-form coherence
    - Question answering: ROUGE-1/2 for factual overlap
    - Abstractive generation: All types for comprehensive evaluation

    Args:
        predictions: List of generated texts
        references: List of reference (gold) texts
        rouge_types: Which ROUGE scores to compute

    Returns:
        Dictionary of ROUGE scores
    """
    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)

    # Compute scores for each pair
    scores = {rouge_type: [] for rouge_type in rouge_types}

    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)

        for rouge_type in rouge_types:
            # Use F1 score (harmonic mean of precision and recall)
            scores[rouge_type].append(score[rouge_type].fmeasure)

    # Average scores
    avg_scores = {
        rouge_type: np.mean(score_list)
        for rouge_type, score_list in scores.items()
    }

    return avg_scores


def compute_bleu(
    predictions: List[str],
    references: List[str],
    max_order: int = 4
) -> Dict[str, float]:
    """
    Compute BLEU score for translation/generation tasks.

    BLEU (Bilingual Evaluation Understudy) measures precision of
    n-gram overlap between generated and reference text.

    How it works:
    1. Compute precision for n-grams of order 1, 2, 3, 4
    2. Apply brevity penalty for short generations
    3. Geometric mean of precisions

    BLEU ranges from 0 to 100:
    - 0-10: Almost useless
    - 10-20: Hard to get the gist
    - 20-30: Understandable
    - 30-40: Good translations
    - 40-50: Very good translations
    - 50-60: High quality
    - >60: Often better than human

    When to use:
    - Machine translation (original use case)
    - Code generation (exact match matters)
    - Tasks where precision is more important than recall

    Args:
        predictions: List of generated texts
        references: List of reference texts
        max_order: Maximum n-gram order (default: 4)

    Returns:
        Dictionary with BLEU scores
    """
    # Tokenize texts
    predictions_tokenized = [pred.split() for pred in predictions]
    references_tokenized = [[ref.split()] for ref in references]

    # Compute corpus-level BLEU using sacrebleu
    bleu = corpus_bleu(
        predictions,
        [references],
        lowercase=True,
        tokenize="13a",  # Standard tokenization
    )

    # Compute sentence-level BLEU and average
    smoothing = SmoothingFunction().method1
    sentence_bleus = []

    for pred_tokens, ref_tokens in zip(predictions_tokenized, references_tokenized):
        score = sentence_bleu(
            ref_tokens,
            pred_tokens,
            smoothing_function=smoothing,
            weights=[1.0/max_order] * max_order,
        )
        sentence_bleus.append(score * 100)  # Scale to 0-100

    return {
        "bleu_corpus": bleu.score,
        "bleu_sentence_avg": np.mean(sentence_bleus),
        "bleu_precision": bleu.precisions[0],  # Unigram precision
    }


def compute_bertscore(
    predictions: List[str],
    references: List[str],
    model_type: str = "microsoft/deberta-base-mnli",
    device: Optional[str] = None
) -> Dict[str, float]:
    """
    Compute BERTScore for semantic similarity evaluation.

    BERTScore computes similarity between generated and reference text
    using contextual embeddings from BERT-like models.

    Advantages over ROUGE/BLEU:
    - Captures semantic similarity (not just lexical overlap)
    - Robust to paraphrasing
    - Better correlation with human judgment

    How it works:
    1. Encode texts with BERT to get contextual embeddings
    2. Compute cosine similarity between token embeddings
    3. Match tokens with maximum similarity
    4. Aggregate to precision, recall, and F1

    When to use:
    - Semantic similarity is more important than exact match
    - Paraphrasing tasks
    - When references may be phrased differently

    Args:
        predictions: Generated texts
        references: Reference texts
        model_type: BERT model to use for embeddings
        device: Device to run on (cuda/cpu)

    Returns:
        Dictionary with precision, recall, F1 scores
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Compute BERTScore
    P, R, F1 = bert_score.score(
        predictions,
        references,
        model_type=model_type,
        device=device,
        verbose=False,
    )

    return {
        "bertscore_precision": P.mean().item(),
        "bertscore_recall": R.mean().item(),
        "bertscore_f1": F1.mean().item(),
    }


def compute_diversity_metrics(texts: List[str]) -> Dict[str, float]:
    """
    Compute diversity metrics for generated text.

    Diversity metrics measure how varied and non-repetitive
    the generated text is.

    Metrics:
    - Distinct-1: Ratio of unique unigrams to total unigrams
    - Distinct-2: Ratio of unique bigrams to total bigrams
    - Entropy: Information-theoretic measure of diversity

    Why it matters:
    - Low diversity = repetitive, boring text
    - High diversity = varied, interesting text
    - Balance needed: too high = incoherent

    Args:
        texts: List of generated texts

    Returns:
        Dictionary of diversity metrics
    """
    all_tokens = []
    all_bigrams = []

    for text in texts:
        tokens = text.split()
        all_tokens.extend(tokens)

        # Generate bigrams
        bigrams = list(zip(tokens[:-1], tokens[1:]))
        all_bigrams.extend(bigrams)

    # Distinct-1: unique unigrams / total unigrams
    distinct_1 = len(set(all_tokens)) / len(all_tokens) if all_tokens else 0

    # Distinct-2: unique bigrams / total bigrams
    distinct_2 = len(set(all_bigrams)) / len(all_bigrams) if all_bigrams else 0

    # Entropy: -sum(p * log(p))
    token_counts = {}
    for token in all_tokens:
        token_counts[token] = token_counts.get(token, 0) + 1

    total_count = len(all_tokens)
    entropy = 0.0
    if total_count > 0:
        for count in token_counts.values():
            prob = count / total_count
            entropy -= prob * math.log(prob)

    return {
        "distinct_1": distinct_1,
        "distinct_2": distinct_2,
        "entropy": entropy,
    }


class LLMEvaluator:
    """
    Comprehensive LLM evaluator with multiple metrics.

    This class provides a unified interface for evaluating LLMs
    using various metrics appropriate for different tasks.

    Usage:
        >>> evaluator = LLMEvaluator()
        >>> predictions = ["The cat sat on the mat"]
        >>> references = ["A cat was sitting on a mat"]
        >>> scores = evaluator.evaluate(predictions, references)

    Attributes:
        device: Device for computation (cuda/cpu)
        rouge_types: ROUGE variants to compute
        bertscore_model: Model for BERTScore computation
    """

    def __init__(
        self,
        device: Optional[str] = None,
        rouge_types: List[str] = ["rouge1", "rouge2", "rougeL"],
        bertscore_model: str = "microsoft/deberta-base-mnli",
    ):
        """
        Initialize evaluator.

        Args:
            device: Device for computation
            rouge_types: ROUGE scores to compute
            bertscore_model: Model for BERTScore
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.rouge_types = rouge_types
        self.bertscore_model = bertscore_model

        logger.info(f"LLMEvaluator initialized on device: {self.device}")

    def evaluate(
        self,
        predictions: List[str],
        references: List[str],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate predictions against references using multiple metrics.

        Args:
            predictions: Generated texts
            references: Reference (gold) texts
            metrics: Which metrics to compute (None = all)
                    Options: ["rouge", "bleu", "bertscore", "diversity"]

        Returns:
            Dictionary of metric scores
        """
        if metrics is None:
            metrics = ["rouge", "bleu", "bertscore", "diversity"]

        results = {}

        # Validate inputs
        if len(predictions) != len(references):
            raise ValueError(
                f"Number of predictions ({len(predictions)}) must equal "
                f"number of references ({len(references)})"
            )

        logger.info(f"Evaluating {len(predictions)} samples with metrics: {metrics}")

        # Compute each metric
        if "rouge" in metrics:
            logger.info("Computing ROUGE scores...")
            rouge_scores = compute_rouge(predictions, references, self.rouge_types)
            results.update(rouge_scores)

        if "bleu" in metrics:
            logger.info("Computing BLEU scores...")
            bleu_scores = compute_bleu(predictions, references)
            results.update(bleu_scores)

        if "bertscore" in metrics:
            logger.info("Computing BERTScore...")
            bertscore_scores = compute_bertscore(
                predictions,
                references,
                model_type=self.bertscore_model,
                device=self.device,
            )
            results.update(bertscore_scores)

        if "diversity" in metrics:
            logger.info("Computing diversity metrics...")
            diversity_scores = compute_diversity_metrics(predictions)
            results.update(diversity_scores)

        logger.info("Evaluation completed")
        return results

    def evaluate_model(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
    ) -> Dict[str, float]:
        """
        Evaluate a model's perplexity on a dataset.

        Args:
            model: The model to evaluate
            dataloader: DataLoader with evaluation data

        Returns:
            Dictionary with perplexity and loss
        """
        device = next(model.parameters()).device

        # Compute perplexity
        perplexity = compute_perplexity(model, dataloader, device)

        return {
            "perplexity": perplexity,
            "loss": math.log(perplexity),
        }

    def generate_and_evaluate(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        prompts: List[str],
        references: List[str],
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate text and evaluate against references.

        This is useful for end-to-end evaluation where you want to:
        1. Generate text from prompts
        2. Compare to reference texts
        3. Compute quality metrics

        Args:
            model: Language model for generation
            tokenizer: Tokenizer for encoding/decoding
            prompts: Input prompts for generation
            references: Reference texts for comparison
            generation_kwargs: Generation parameters (temperature, top_p, etc.)

        Returns:
            Dictionary with generated texts and metric scores
        """
        if generation_kwargs is None:
            generation_kwargs = {
                "max_length": 100,
                "num_beams": 4,
                "early_stopping": True,
                "no_repeat_ngram_size": 3,
            }

        logger.info(f"Generating text for {len(prompts)} prompts...")

        # Generate text
        model.eval()
        predictions = []

        with torch.no_grad():
            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

                outputs = model.generate(
                    **inputs,
                    **generation_kwargs,
                )

                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Remove prompt from generated text
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()

                predictions.append(generated_text)

        # Evaluate predictions
        scores = self.evaluate(predictions, references)

        return {
            "predictions": predictions,
            "scores": scores,
        }
