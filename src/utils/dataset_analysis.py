"""
Dataset Analysis Utilities

Tools for analyzing and understanding datasets before training.
"""

import logging
from typing import Dict, List, Optional, Any
from collections import Counter
import json

import numpy as np
from datasets import load_dataset

logger = logging.getLogger(__name__)


class DatasetAnalyzer:
    """
    Analyze datasets for LLM training.
    
    Provides insights on:
    - Dataset statistics (size, splits, features)
    - Text length distributions
    - Vocabulary statistics
    - Data quality issues
    - Class balance (for classification)
    
    Usage:
        >>> analyzer = DatasetAnalyzer("wikitext", "wikitext-2-raw-v1")
        >>> stats = analyzer.analyze()
        >>> analyzer.print_summary(stats)
    """
    
    def __init__(
        self,
        dataset_name: str,
        config_name: Optional[str] = None,
        split: str = "train",
    ):
        """Initialize dataset analyzer."""
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.split = split
        
        logger.info(f"Loading dataset: {dataset_name}")
        self.dataset = load_dataset(dataset_name, config_name, split=split)
        
    def analyze(self, sample_size: int = 1000) -> Dict[str, Any]:
        """
        Perform comprehensive dataset analysis.
        
        Args:
            sample_size: Number of samples to analyze (for speed)
            
        Returns:
            Dictionary with analysis results
        """
        logger.info("Analyzing dataset...")
        
        # Sample dataset if too large
        if len(self.dataset) > sample_size:
            sample = self.dataset.shuffle(seed=42).select(range(sample_size))
        else:
            sample = self.dataset
            
        stats = {
            "dataset_name": self.dataset_name,
            "config": self.config_name,
            "split": self.split,
            "total_samples": len(self.dataset),
            "analyzed_samples": len(sample),
            "features": list(self.dataset.features.keys()),
        }
        
        # Analyze text fields
        text_field = self._find_text_field()
        if text_field:
            stats["text_statistics"] = self._analyze_text(sample, text_field)
            
        return stats
        
    def _find_text_field(self) -> Optional[str]:
        """Find the main text field in dataset."""
        text_fields = ["text", "content", "document", "article", "sentence"]
        for field in text_fields:
            if field in self.dataset.features:
                return field
        return None
        
    def _analyze_text(self, sample, field: str) -> Dict[str, Any]:
        """Analyze text statistics."""
        texts = [item[field] for item in sample if item[field]]
        
        # Length statistics
        lengths = [len(text.split()) for text in texts]
        char_lengths = [len(text) for text in texts]
        
        # Vocabulary
        all_words = " ".join(texts).lower().split()
        vocab = set(all_words)
        word_freq = Counter(all_words)
        
        return {
            "num_samples": len(texts),
            "avg_word_length": np.mean(lengths),
            "std_word_length": np.std(lengths),
            "min_word_length": min(lengths),
            "max_word_length": max(lengths),
            "median_word_length": np.median(lengths),
            "avg_char_length": np.mean(char_lengths),
            "vocabulary_size": len(vocab),
            "most_common_words": word_freq.most_common(20),
            "total_words": len(all_words),
        }
        
    def print_summary(self, stats: Dict[str, Any]):
        """Print analysis summary."""
        print("\n" + "=" * 80)
        print("DATASET ANALYSIS SUMMARY")
        print("=" * 80)
        print(f"\nDataset: {stats['dataset_name']}")
        if stats['config']:
            print(f"Config: {stats['config']}")
        print(f"Split: {stats['split']}")
        print(f"Total Samples: {stats['total_samples']:,}")
        print(f"Analyzed Samples: {stats['analyzed_samples']:,}")
        print(f"Features: {', '.join(stats['features'])}")
        
        if "text_statistics" in stats:
            text_stats = stats["text_statistics"]
            print("\nText Statistics:")
            print(f"  Average words per sample: {text_stats['avg_word_length']:.1f}")
            print(f"  Std deviation: {text_stats['std_word_length']:.1f}")
            print(f"  Min length: {text_stats['min_word_length']} words")
            print(f"  Max length: {text_stats['max_word_length']} words")
            print(f"  Median length: {text_stats['median_word_length']:.1f} words")
            print(f"  Vocabulary size: {text_stats['vocabulary_size']:,} unique words")
            print(f"  Total words: {text_stats['total_words']:,}")
            
            print("\nMost Common Words:")
            for word, count in text_stats['most_common_words'][:10]:
                print(f"  {word}: {count:,}")
                
        print("\n" + "=" * 80)
        
    def save_report(self, stats: Dict[str, Any], output_path: str):
        """Save analysis report to JSON."""
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Report saved to {output_path}")


def quick_analysis(dataset_name: str, config: Optional[str] = None):
    """
    Quick dataset analysis and summary.
    
    Args:
        dataset_name: Dataset name from HuggingFace
        config: Optional dataset configuration
    """
    analyzer = DatasetAnalyzer(dataset_name, config)
    stats = analyzer.analyze()
    analyzer.print_summary(stats)
    return stats
