"""
Week 1 - Data Preparation

Your task: Implement functions to prepare and tokenize datasets for training.

Learning goals:
- Load datasets from Hugging Face
- Tokenize text data
- Create DataLoaders for batching
- Handle padding and truncation

Difficulty: ⭐⭐ Medium
Estimated time: 1 hour
"""

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


def load_text_dataset(
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "train"
):
    """
    Load a text dataset from Hugging Face.

    Args:
        dataset_name: Name of the dataset
        dataset_config: Configuration/subset of the dataset
        split: Which split to load ("train", "validation", "test")

    Returns:
        Loaded dataset

    Example:
        >>> dataset = load_text_dataset("wikitext", "wikitext-2-raw-v1", "train")
        >>> print(dataset[0])

    TODO: Implement this function
    HINT: Use datasets.load_dataset(dataset_name, dataset_config, split=split)
    """
    # YOUR CODE HERE
    raise NotImplementedError("Remove this line and implement load_text_dataset")


def tokenize_function(
    examples: Dict[str, List],
    tokenizer: AutoTokenizer,
    text_column: str = "text",
    max_length: int = 512
) -> Dict[str, List]:
    """
    Tokenize a batch of examples.

    Args:
        examples: Dictionary with text examples
        tokenizer: Tokenizer to use
        text_column: Name of the column containing text
        max_length: Maximum sequence length

    Returns:
        Dictionary with tokenized data

    Example:
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> examples = {"text": ["Hello world", "How are you?"]}
        >>> tokenized = tokenize_function(examples, tokenizer)
        >>> print(tokenized.keys())

    TODO: Implement this function
    HINT: Use tokenizer(examples[text_column], ...)
    HINT: Set truncation=True and max_length=max_length
    HINT: Return the tokenized result directly
    """
    # YOUR CODE HERE
    raise NotImplementedError("Remove this line and implement tokenize_function")


def prepare_dataset(
    dataset,
    tokenizer: AutoTokenizer,
    text_column: str = "text",
    max_length: int = 512
):
    """
    Tokenize an entire dataset.

    Args:
        dataset: Dataset to tokenize
        tokenizer: Tokenizer to use
        text_column: Name of the column containing text
        max_length: Maximum sequence length

    Returns:
        Tokenized dataset

    Example:
        >>> dataset = load_text_dataset()
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> tokenized = prepare_dataset(dataset, tokenizer)

    TODO: Implement this function
    HINT: Use dataset.map() to apply tokenization
    HINT: Pass a lambda function: lambda x: tokenize_function(x, tokenizer, text_column, max_length)
    HINT: Set batched=True for faster processing
    HINT: Remove original text column: remove_columns=[text_column]
    """
    # YOUR CODE HERE
    raise NotImplementedError("Remove this line and implement prepare_dataset")


@dataclass
class DataCollatorForLanguageModeling:
    """
    Data collator for causal language modeling.

    This collator:
    - Pads sequences to the same length in a batch
    - Creates labels (shifted input_ids for causal LM)
    """

    tokenizer: AutoTokenizer
    max_length: int = 512

    def __call__(self, examples: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        """
        Collate examples into a batch.

        Args:
            examples: List of examples, each with "input_ids" and optionally "attention_mask"

        Returns:
            Dictionary with batched tensors:
            - input_ids: shape [batch_size, seq_len]
            - attention_mask: shape [batch_size, seq_len]
            - labels: shape [batch_size, seq_len] (same as input_ids for causal LM)

        TODO: Implement this function
        HINT: 1. Extract input_ids from examples
        HINT: 2. Use tokenizer.pad() to pad sequences
        HINT: 3. Set padding=True and return_tensors="pt"
        HINT: 4. Create labels (for causal LM, labels = input_ids)
        HINT: 5. Set padding tokens in labels to -100 (ignored in loss)
        """
        # YOUR CODE HERE
        raise NotImplementedError("Remove this line and implement DataCollatorForLanguageModeling.__call__")


def create_dataloader(
    dataset,
    tokenizer: AutoTokenizer,
    batch_size: int = 8,
    shuffle: bool = True,
    max_length: int = 512
) -> DataLoader:
    """
    Create a DataLoader from a tokenized dataset.

    Args:
        dataset: Tokenized dataset
        tokenizer: Tokenizer (for data collator)
        batch_size: Batch size
        shuffle: Whether to shuffle data
        max_length: Maximum sequence length

    Returns:
        DataLoader

    Example:
        >>> dataset = prepare_dataset(load_text_dataset(), tokenizer)
        >>> dataloader = create_dataloader(dataset, tokenizer, batch_size=4)
        >>> batch = next(iter(dataloader))
        >>> print(batch.keys())

    TODO: Implement this function
    HINT: Create a DataCollatorForLanguageModeling
    HINT: Use DataLoader with:
          - dataset=dataset
          - batch_size=batch_size
          - shuffle=shuffle
          - collate_fn=data_collator
    """
    # YOUR CODE HERE
    raise NotImplementedError("Remove this line and implement create_dataloader")


# ============================================================================
# BONUS CHALLENGES (Optional)
# ============================================================================

def filter_short_examples(dataset, min_length: int = 10):
    """
    BONUS: Filter out examples that are too short.

    Args:
        dataset: Dataset to filter
        min_length: Minimum number of tokens

    Returns:
        Filtered dataset

    HINT: Use dataset.filter()
    HINT: Check len(example["input_ids"]) >= min_length
    """
    # YOUR CODE HERE (BONUS)
    pass


def create_validation_split(dataset, validation_size: float = 0.1):
    """
    BONUS: Split dataset into train and validation.

    Args:
        dataset: Dataset to split
        validation_size: Fraction for validation (0.0 to 1.0)

    Returns:
        Dictionary with "train" and "validation" datasets

    HINT: Use dataset.train_test_split(test_size=validation_size)
    """
    # YOUR CODE HERE (BONUS)
    pass


# ============================================================================
# TESTING SECTION
# ============================================================================

if __name__ == "__main__":
    """
    Run this file to test your implementations!

    Usage:
        python data_prep.py

    Note: This will download data (~18MB for wikitext-2)
    """
    print("=" * 60)
    print("Testing Data Preparation Implementation")
    print("=" * 60)

    # Test 1: Load dataset
    print("\n[Test 1] Loading dataset...")
    try:
        dataset = load_text_dataset("wikitext", "wikitext-2-raw-v1", "train")
        print(f"✓ Dataset loaded: {len(dataset)} examples")
        print(f"  Example: {dataset[0]['text'][:100]}...")
    except NotImplementedError:
        print("✗ load_text_dataset not implemented yet")

    # Test 2: Tokenization
    print("\n[Test 2] Testing tokenization...")
    try:
        from model_loader import load_tokenizer
        tokenizer = load_tokenizer("gpt2")
        examples = {"text": ["Hello world!", "This is a test."]}
        tokenized = tokenize_function(examples, tokenizer)
        print(f"✓ Tokenization successful")
        print(f"  Keys: {tokenized.keys()}")
        print(f"  First example length: {len(tokenized['input_ids'][0])}")
    except NotImplementedError:
        print("✗ tokenize_function not implemented yet")

    # Test 3: Prepare dataset
    print("\n[Test 3] Preparing full dataset...")
    try:
        dataset = load_text_dataset("wikitext", "wikitext-2-raw-v1", "train")
        from model_loader import load_tokenizer
        tokenizer = load_tokenizer("gpt2")
        tokenized_dataset = prepare_dataset(dataset, tokenizer, max_length=128)
        print(f"✓ Dataset tokenized: {len(tokenized_dataset)} examples")
        print(f"  Example keys: {tokenized_dataset[0].keys()}")
    except NotImplementedError:
        print("✗ prepare_dataset not implemented yet")

    # Test 4: Data collator
    print("\n[Test 4] Testing data collator...")
    try:
        from model_loader import load_tokenizer
        tokenizer = load_tokenizer("gpt2")
        examples = [
            {"input_ids": [1, 2, 3, 4, 5]},
            {"input_ids": [1, 2, 3]},  # Shorter
        ]
        collator = DataCollatorForLanguageModeling(tokenizer)
        batch = collator(examples)
        print(f"✓ Data collator works")
        print(f"  Batch keys: {batch.keys()}")
        print(f"  input_ids shape: {batch['input_ids'].shape}")
        print(f"  labels shape: {batch['labels'].shape}")
    except NotImplementedError:
        print("✗ DataCollatorForLanguageModeling not implemented yet")

    # Test 5: DataLoader
    print("\n[Test 5] Creating DataLoader...")
    try:
        dataset = load_text_dataset("wikitext", "wikitext-2-raw-v1", "train")
        from model_loader import load_tokenizer
        tokenizer = load_tokenizer("gpt2")
        tokenized_dataset = prepare_dataset(dataset, tokenizer, max_length=128)
        dataloader = create_dataloader(tokenized_dataset, tokenizer, batch_size=4)
        batch = next(iter(dataloader))
        print(f"✓ DataLoader created")
        print(f"  Batch size: {batch['input_ids'].shape[0]}")
        print(f"  Sequence length: {batch['input_ids'].shape[1]}")
    except NotImplementedError:
        print("✗ create_dataloader not implemented yet")

    print("\n" + "=" * 60)
    print("Basic testing complete!")
    print("Run full tests: pytest tests/test_data_prep.py -v")
    print("=" * 60)
