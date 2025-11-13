"""
Mock Datasets and DataLoaders for Testing

These mocks provide fast dataset access without downloading real data.
"""

import random
from typing import List, Dict, Any, Optional, Iterator


class MockDataset:
    """
    Lightweight mock dataset for testing.

    Simulates HuggingFace datasets interface with realistic text samples.
    """

    def __init__(
        self,
        size: int = 1000,
        split: str = "train",
        text_length_range: tuple = (50, 500)
    ):
        self.size = size
        self.split = split
        self.text_length_range = text_length_range

        # Generate realistic-looking samples
        self.data = self._generate_samples()

    def _generate_samples(self) -> List[Dict[str, Any]]:
        """Generate mock text samples."""
        samples = []

        topics = [
            "machine learning", "artificial intelligence", "neural networks",
            "deep learning", "natural language processing", "computer vision",
            "robotics", "data science", "algorithm", "optimization"
        ]

        for i in range(self.size):
            # Generate text with random length
            num_words = random.randint(
                self.text_length_range[0] // 5,
                self.text_length_range[1] // 5
            )

            # Create somewhat realistic text
            topic = random.choice(topics)
            words = [topic] + [f"word{j}" for j in range(num_words - 1)]
            random.shuffle(words)
            text = " ".join(words)

            samples.append({
                "text": text,
                "id": i,
                "split": self.split,
                "label": random.randint(0, 1) if random.random() > 0.5 else None
            })

        return samples

    def __len__(self) -> int:
        """Return dataset size."""
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item by index."""
        if idx >= self.size:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.size}")
        return self.data[idx]

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over dataset."""
        return iter(self.data)

    def shuffle(self, seed: Optional[int] = None):
        """Shuffle dataset."""
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.data)
        return self

    def select(self, indices: List[int]) -> "MockDataset":
        """Select subset of dataset."""
        new_dataset = MockDataset(size=len(indices), split=self.split)
        new_dataset.data = [self.data[i] for i in indices]
        new_dataset.size = len(indices)
        return new_dataset

    def train_test_split(self, test_size: float = 0.2, seed: Optional[int] = None) -> Dict[str, "MockDataset"]:
        """Split into train and test sets."""
        if seed is not None:
            random.seed(seed)

        indices = list(range(self.size))
        random.shuffle(indices)

        split_idx = int(self.size * (1 - test_size))
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        return {
            "train": self.select(train_indices),
            "test": self.select(test_indices)
        }

    def map(self, function, batched: bool = False, **kwargs) -> "MockDataset":
        """Apply function to dataset (simplified mock)."""
        if batched:
            # Process in batches
            batch_size = kwargs.get("batch_size", 1000)
            for i in range(0, len(self.data), batch_size):
                batch = self.data[i:i + batch_size]
                # In real implementation, would apply function
                pass
        else:
            # Process one by one
            self.data = [function(item) for item in self.data]

        return self

    def filter(self, function) -> "MockDataset":
        """Filter dataset by function."""
        filtered_data = [item for item in self.data if function(item)]
        new_dataset = MockDataset(size=len(filtered_data), split=self.split)
        new_dataset.data = filtered_data
        new_dataset.size = len(filtered_data)
        return new_dataset


class MockDataLoader:
    """
    Mock PyTorch DataLoader for testing.

    Provides batched iteration over datasets without PyTorch dependency.
    """

    def __init__(
        self,
        dataset: MockDataset,
        batch_size: int = 8,
        shuffle: bool = False,
        num_workers: int = 0,
        **kwargs
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        self._prepare_batches()

    def _prepare_batches(self):
        """Prepare batches from dataset."""
        indices = list(range(len(self.dataset)))

        if self.shuffle:
            random.shuffle(indices)

        self.batches = []
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch = [self.dataset[idx] for idx in batch_indices]
            self.batches.append(batch)

    def __len__(self) -> int:
        """Return number of batches."""
        return len(self.batches)

    def __iter__(self) -> Iterator[List[Dict[str, Any]]]:
        """Iterate over batches."""
        if self.shuffle:
            self._prepare_batches()  # Re-shuffle for new epoch
        return iter(self.batches)


class MockHFDataset:
    """
    Mock for datasets.load_dataset() return value.

    Simulates DatasetDict from HuggingFace datasets library.
    """

    def __init__(self, splits: Optional[List[str]] = None):
        if splits is None:
            splits = ["train", "validation", "test"]

        self._splits = {}
        for split in splits:
            size = 1000 if split == "train" else 200
            self._splits[split] = MockDataset(size=size, split=split)

    def __getitem__(self, split: str) -> MockDataset:
        """Get dataset split."""
        if split not in self._splits:
            raise KeyError(f"Split '{split}' not found. Available: {list(self._splits.keys())}")
        return self._splits[split]

    def __contains__(self, split: str) -> bool:
        """Check if split exists."""
        return split in self._splits

    def keys(self) -> List[str]:
        """Get available splits."""
        return list(self._splits.keys())


def load_mock_dataset(dataset_name: str, split: Optional[str] = None, **kwargs) -> Any:
    """
    Mock version of datasets.load_dataset().

    Args:
        dataset_name: Name of dataset (ignored in mock)
        split: Optional split name
        **kwargs: Additional arguments (ignored in mock)

    Returns:
        MockDataset or MockHFDataset
    """
    if split is not None:
        # Return single split
        size = kwargs.get("size", 1000)
        return MockDataset(size=size, split=split)
    else:
        # Return DatasetDict with multiple splits
        return MockHFDataset()


class MockTextDataset:
    """
    Mock for simple text datasets.

    Useful for quick testing of text processing pipelines.
    """

    SAMPLE_TEXTS = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks are inspired by biological neural networks.",
        "Deep learning uses multiple layers to learn representations.",
        "Natural language processing enables computers to understand text.",
        "Computer vision allows machines to interpret visual information.",
        "Reinforcement learning learns through interaction with environment.",
        "Transfer learning applies knowledge from one task to another.",
        "Attention mechanisms help models focus on relevant information.",
        "Transformers have revolutionized natural language processing.",
    ]

    def __init__(self, num_samples: int = 100):
        self.num_samples = num_samples
        self.texts = []

        # Generate samples by repeating and slightly modifying base texts
        for i in range(num_samples):
            base_text = random.choice(self.SAMPLE_TEXTS)
            # Add some variation
            text = f"{base_text} Sample {i}."
            self.texts.append(text)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]

    def __iter__(self) -> Iterator[str]:
        return iter(self.texts)


class MockCollator:
    """
    Mock data collator for batching.

    Simulates transformers.DataCollatorForLanguageModeling.
    """

    def __init__(self, tokenizer, mlm: bool = False, mlm_probability: float = 0.15):
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.mlm_probability = mlm_probability

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate examples into batch."""
        # Extract texts
        texts = [ex.get("text", "") for ex in examples]

        # Tokenize
        batch = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors=None
        )

        # For language modeling, labels are same as input_ids
        batch["labels"] = batch["input_ids"].copy() if "input_ids" in batch else None

        return batch
