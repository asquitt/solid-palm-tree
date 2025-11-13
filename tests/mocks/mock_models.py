"""
Mock Models and Tokenizers for Testing

These mocks allow testing training/inference logic without loading real models.
They simulate the interface of HuggingFace transformers models.
"""

import random
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class MockModelOutput:
    """Mock output from model forward pass."""
    loss: Optional[float] = None
    logits: Optional[List[List[float]]] = None
    hidden_states: Optional[List[List[float]]] = None

    def __getitem__(self, key):
        """Support dict-like access."""
        return getattr(self, key)


class MockModel:
    """
    Lightweight mock model for testing without real transformers models.

    Simulates GPT-2/LLaMA-like interface with realistic loss values.
    Supports training mode, gradient tracking, and device placement.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {"vocab_size": 50257, "hidden_size": 768, "num_layers": 12}
        self._training = False
        self._device = "cpu"
        self.training_step = 0
        self.num_parameters = 124_000_000  # Simulate GPT-2 size

        # Simulate parameters
        self.parameters_list = [MockParameter() for _ in range(12)]

    def forward(
        self,
        input_ids: Any = None,
        attention_mask: Any = None,
        labels: Any = None,
        **kwargs
    ) -> MockModelOutput:
        """Mock forward pass with realistic loss."""
        batch_size = 8 if input_ids is None else len(input_ids)
        seq_length = 512 if input_ids is None else (
            len(input_ids[0]) if isinstance(input_ids[0], list) else 512
        )
        vocab_size = self.config["vocab_size"]

        # Generate realistic loss (starts high, decreases with training)
        if labels is not None:
            # Simulate learning: loss decreases over training steps
            base_loss = 5.0  # Untrained model
            improvement = min(self.training_step * 0.01, 3.0)
            noise = random.uniform(-0.1, 0.1)
            loss = max(base_loss - improvement + noise, 1.0)
        else:
            loss = None

        # Generate random logits
        logits = [
            [random.gauss(0, 1) for _ in range(vocab_size)]
            for _ in range(batch_size)
        ]

        if self._training:
            self.training_step += 1

        return MockModelOutput(loss=loss, logits=logits)

    def __call__(self, *args, **kwargs):
        """Allow calling model like a function."""
        return self.forward(*args, **kwargs)

    def train(self, mode: bool = True):
        """Set model to training mode."""
        self._training = mode
        return self

    def eval(self):
        """Set model to evaluation mode."""
        self._training = False
        return self

    def to(self, device: str):
        """Move model to device."""
        self._device = device
        return self

    def parameters(self):
        """Return model parameters (for optimizer)."""
        return iter(self.parameters_list)

    def named_parameters(self):
        """Return named parameters."""
        return [(f"layer.{i}.weight", param) for i, param in enumerate(self.parameters_list)]

    def state_dict(self) -> Dict:
        """Return state dict for checkpointing."""
        return {
            "training_step": self.training_step,
            "config": self.config,
            "device": self._device,
        }

    def load_state_dict(self, state_dict: Dict):
        """Load state dict from checkpoint."""
        self.training_step = state_dict.get("training_step", 0)
        self.config = state_dict.get("config", self.config)
        self._device = state_dict.get("device", "cpu")

    def save_pretrained(self, path: str):
        """Mock save to directory."""
        pass

    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs):
        """Mock loading pretrained model."""
        return cls()

    def generate(
        self,
        input_ids: Any = None,
        max_length: int = 50,
        temperature: float = 1.0,
        top_p: float = 0.9,
        **kwargs
    ) -> List[List[int]]:
        """Mock text generation."""
        batch_size = 1 if input_ids is None else len(input_ids)

        # Generate random token IDs
        generated = [
            [random.randint(0, self.config["vocab_size"] - 1) for _ in range(max_length)]
            for _ in range(batch_size)
        ]
        return generated

    def gradient_checkpointing_enable(self):
        """Mock gradient checkpointing."""
        pass


class MockParameter:
    """Mock PyTorch parameter."""

    def __init__(self):
        self.data = [random.gauss(0, 0.02) for _ in range(100)]
        self.grad = None
        self.requires_grad = True

    def zero_grad(self):
        """Reset gradients."""
        self.grad = None


class MockTokenizer:
    """
    Lightweight mock tokenizer for testing.

    Simulates HuggingFace tokenizer interface.
    """

    def __init__(self, vocab_size: int = 50257):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 50256
        self.bos_token_id = 50256
        self.model_max_length = 1024

    def __call__(
        self,
        text: Any,
        padding: str = "max_length",
        truncation: bool = True,
        max_length: int = 512,
        return_tensors: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Tokenize text (returns mock token IDs)."""
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        # Generate mock token IDs (realistic length based on text)
        encodings = []
        for t in texts:
            # Approximate: 1 token per 4 characters
            num_tokens = min(len(t) // 4 + 1, max_length)
            token_ids = [random.randint(1, self.vocab_size - 1) for _ in range(num_tokens)]

            # Pad if needed
            if padding == "max_length":
                token_ids += [self.pad_token_id] * (max_length - num_tokens)

            encodings.append(token_ids[:max_length])

        # Create attention masks
        attention_masks = [
            [1 if token != self.pad_token_id else 0 for token in enc]
            for enc in encodings
        ]

        return {
            "input_ids": encodings,
            "attention_mask": attention_masks,
        }

    def encode(self, text: str, **kwargs) -> List[int]:
        """Encode text to token IDs."""
        result = self(text, **kwargs)
        return result["input_ids"][0]

    def decode(self, token_ids: List[int], **kwargs) -> str:
        """Decode token IDs to text (mock)."""
        # Generate plausible text
        num_words = len(token_ids) // 3
        words = [f"word{i}" for i in range(num_words)]
        return " ".join(words)

    def batch_decode(self, token_ids_batch: List[List[int]], **kwargs) -> List[str]:
        """Batch decode token IDs."""
        return [self.decode(ids, **kwargs) for ids in token_ids_batch]

    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs):
        """Mock loading pretrained tokenizer."""
        return cls()

    def save_pretrained(self, path: str):
        """Mock save to directory."""
        pass


class MockLoRAModel(MockModel):
    """Mock model with LoRA adapters."""

    def __init__(self, base_model: MockModel, r: int = 8, alpha: int = 16):
        super().__init__(base_model.config)
        self.base_model = base_model
        self.r = r
        self.alpha = alpha

        # LoRA has much fewer trainable parameters
        self.num_parameters = 124_000_000  # Base model
        self.trainable_parameters = int(self.num_parameters * 0.01)  # ~1% trainable with LoRA

    def print_trainable_parameters(self):
        """Print trainable parameter statistics."""
        trainable = self.trainable_parameters
        total = self.num_parameters
        percentage = 100 * trainable / total
        print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {percentage:.2f}")
