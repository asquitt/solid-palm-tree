"""
Week 1 - Model Loader (SOLUTION)

This is the reference implementation. Compare with your solution!
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple, Dict, Any


def load_tokenizer(model_name: str = "gpt2") -> AutoTokenizer:
    """Load a tokenizer from Hugging Face Hub."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Some models don't have a pad token - use eos_token as padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_model(
    model_name: str = "gpt2",
    device: str = None
) -> AutoModelForCausalLM:
    """Load a pre-trained model from Hugging Face Hub."""
    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Move to device
    model = model.to(device)

    return model


def get_device() -> str:
    """Get the best available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_model_info(model: AutoModelForCausalLM) -> Dict[str, Any]:
    """Get information about a model."""
    config = model.config

    return {
        "num_parameters": model.num_parameters(),
        "num_layers": config.n_layer,
        "hidden_size": config.n_embd if hasattr(config, 'n_embd') else config.hidden_size,
        "vocab_size": config.vocab_size,
        "device": str(model.device)
    }


def test_model_forward(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str = "Hello, world!"
) -> torch.Tensor:
    """Test the model with a forward pass."""
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt")

    # Move to same device as model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Forward pass (no gradients needed for inference)
    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.logits


def count_trainable_parameters(model: AutoModelForCausalLM) -> int:
    """Count only trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_model_layers(model: AutoModelForCausalLM, num_layers_to_freeze: int = 6):
    """Freeze the first N layers of the model."""
    # Freeze embedding layer
    for param in model.transformer.wte.parameters():
        param.requires_grad = False
    for param in model.transformer.wpe.parameters():
        param.requires_grad = False

    # Freeze transformer layers
    for i in range(min(num_layers_to_freeze, len(model.transformer.h))):
        for param in model.transformer.h[i].parameters():
            param.requires_grad = False


if __name__ == "__main__":
    """Test the implementation."""
    print("=" * 60)
    print("Model Loader Solution - Testing")
    print("=" * 60)

    print("\n[1] Loading tokenizer...")
    tokenizer = load_tokenizer("gpt2")
    print(f"✓ Tokenizer loaded: vocab_size = {len(tokenizer)}")

    print("\n[2] Loading model...")
    model = load_model("gpt2")
    print(f"✓ Model loaded on {model.device}")

    print("\n[3] Model info...")
    info = get_model_info(model)
    for key, value in info.items():
        print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")

    print("\n[4] Forward pass...")
    logits = test_model_forward(model, tokenizer, "Hello, world!")
    print(f"✓ Output shape: {logits.shape}")

    print("\n[5] Trainable parameters...")
    trainable = count_trainable_parameters(model)
    total = model.num_parameters()
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    print("\n[6] Freezing layers...")
    freeze_model_layers(model, num_layers_to_freeze=6)
    trainable_after = count_trainable_parameters(model)
    print(f"  Trainable after freezing: {trainable_after:,} ({100*trainable_after/total:.1f}%)")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
