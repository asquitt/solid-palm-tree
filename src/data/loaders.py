"""Dataset loaders for common LLM fine-tuning datasets."""

from datasets import load_dataset
from typing import Optional

class DatasetLoader:
    """Loader for HuggingFace datasets."""
    
    @staticmethod
    def load(name: str, config: Optional[str] = None, split: str = "train"):
        """Load dataset from HuggingFace."""
        return load_dataset(name, config, split=split)
