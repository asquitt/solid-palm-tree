"""
Week 1 - Model Loader

Your task: Implement functions to load pre-trained models and tokenizers from Hugging Face.

Learning goals:
- Understand how to load models from Hugging Face Hub
- Learn about tokenizers and their configuration
- Handle device placement (CPU vs GPU)

Difficulty: ⭐ Easy
Estimated time: 30 minutes
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple, Dict, Any


def load_tokenizer(model_name: str = "gpt2") -> AutoTokenizer:
    """
    Load a tokenizer from Hugging Face Hub.

    Args:
        model_name: Name of the model (e.g., "gpt2", "gpt2-medium")

    Returns:
        Loaded tokenizer

    Example:
        >>> tokenizer = load_tokenizer("gpt2")
        >>> tokens = tokenizer("Hello world")
        >>> print(tokens)

    TODO: Implement this function
    HINT: Use AutoTokenizer.from_pretrained()
    HINT: Some models need a padding token - set tokenizer.pad_token = tokenizer.eos_token if pad_token is None
    """
    # YOUR CODE HERE
    raise NotImplementedError("Remove this line and implement load_tokenizer")


def load_model(
    model_name: str = "gpt2",
    device: str = None
) -> AutoModelForCausalLM:
    """
    Load a pre-trained model from Hugging Face Hub.

    Args:
        model_name: Name of the model (e.g., "gpt2", "gpt2-medium")
        device: Device to load model on ("cuda" or "cpu"). If None, auto-detect.

    Returns:
        Loaded model on specified device

    Example:
        >>> model = load_model("gpt2", device="cpu")
        >>> print(f"Model loaded on: {model.device}")

    TODO: Implement this function
    HINT: Use AutoModelForCausalLM.from_pretrained()
    HINT: If device is None, check torch.cuda.is_available()
    HINT: Use model.to(device) to move model to device
    """
    # YOUR CODE HERE
    raise NotImplementedError("Remove this line and implement load_model")


def get_device() -> str:
    """
    Get the best available device (CUDA if available, else CPU).

    Returns:
        Device string ("cuda" or "cpu")

    Example:
        >>> device = get_device()
        >>> print(f"Using device: {device}")

    TODO: Implement this function
    HINT: Check torch.cuda.is_available()
    HINT: Return "cuda" if GPU is available, else "cpu"
    """
    # YOUR CODE HERE
    raise NotImplementedError("Remove this line and implement get_device")


def get_model_info(model: AutoModelForCausalLM) -> Dict[str, Any]:
    """
    Get information about a model.

    Args:
        model: The model to analyze

    Returns:
        Dictionary with model information:
        - "num_parameters": Total number of parameters
        - "num_layers": Number of transformer layers
        - "hidden_size": Size of hidden layers
        - "vocab_size": Size of vocabulary
        - "device": Device the model is on

    Example:
        >>> model = load_model("gpt2")
        >>> info = get_model_info(model)
        >>> print(f"Parameters: {info['num_parameters']:,}")

    TODO: Implement this function
    HINT: Use model.num_parameters() for total parameters
    HINT: Access model.config for architecture details
    HINT: model.config.n_layer gives number of layers
    HINT: model.config.hidden_size or n_embd gives hidden size
    HINT: model.config.vocab_size gives vocabulary size
    HINT: Use str(model.device) for device
    """
    # YOUR CODE HERE
    raise NotImplementedError("Remove this line and implement get_model_info")


def test_model_forward(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str = "Hello, world!"
) -> torch.Tensor:
    """
    Test the model with a forward pass.

    Args:
        model: The model to test
        tokenizer: The tokenizer to use
        text: Input text to process

    Returns:
        Model output logits

    Example:
        >>> model = load_model("gpt2")
        >>> tokenizer = load_tokenizer("gpt2")
        >>> logits = test_model_forward(model, tokenizer, "Hello")
        >>> print(f"Output shape: {logits.shape}")

    TODO: Implement this function
    HINT: 1. Tokenize the text: tokenizer(text, return_tensors="pt")
    HINT: 2. Move inputs to same device as model: inputs.to(model.device)
    HINT: 3. Run forward pass: model(**inputs)
    HINT: 4. Return the logits from outputs
    HINT: Use torch.no_grad() for inference (no gradients needed)
    """
    # YOUR CODE HERE
    raise NotImplementedError("Remove this line and implement test_model_forward")


# ============================================================================
# BONUS CHALLENGES (Optional)
# ============================================================================

def count_trainable_parameters(model: AutoModelForCausalLM) -> int:
    """
    BONUS: Count only trainable parameters.

    Args:
        model: The model to analyze

    Returns:
        Number of trainable parameters

    HINT: Use param.requires_grad to check if parameter is trainable
    HINT: Use param.numel() to count elements in a parameter
    """
    # YOUR CODE HERE (BONUS)
    pass


def freeze_model_layers(model: AutoModelForCausalLM, num_layers_to_freeze: int = 6):
    """
    BONUS: Freeze the first N layers of the model.

    This is useful for faster fine-tuning - freeze early layers, train only later layers.

    Args:
        model: The model to modify
        num_layers_to_freeze: Number of initial layers to freeze

    HINT: Access layers via model.transformer.h (for GPT-2)
    HINT: Set param.requires_grad = False to freeze
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
        python model_loader.py

    This will test basic functionality. Run the full test suite with:
        pytest tests/test_model_loader.py -v
    """
    print("=" * 60)
    print("Testing Model Loader Implementation")
    print("=" * 60)

    # Test 1: Load tokenizer
    print("\n[Test 1] Loading tokenizer...")
    try:
        tokenizer = load_tokenizer("gpt2")
        print(f"✓ Tokenizer loaded: {type(tokenizer).__name__}")
        print(f"  Vocab size: {len(tokenizer)}")
    except NotImplementedError:
        print("✗ load_tokenizer not implemented yet")

    # Test 2: Load model
    print("\n[Test 2] Loading model...")
    try:
        model = load_model("gpt2")
        print(f"✓ Model loaded: {type(model).__name__}")
    except NotImplementedError:
        print("✗ load_model not implemented yet")

    # Test 3: Get device
    print("\n[Test 3] Getting device...")
    try:
        device = get_device()
        print(f"✓ Device detected: {device}")
    except NotImplementedError:
        print("✗ get_device not implemented yet")

    # Test 4: Model info
    print("\n[Test 4] Getting model info...")
    try:
        model = load_model("gpt2")
        info = get_model_info(model)
        print(f"✓ Model info:")
        print(f"  Parameters: {info['num_parameters']:,}")
        print(f"  Layers: {info['num_layers']}")
        print(f"  Hidden size: {info['hidden_size']}")
        print(f"  Vocab size: {info['vocab_size']}")
        print(f"  Device: {info['device']}")
    except NotImplementedError:
        print("✗ get_model_info not implemented yet")

    # Test 5: Forward pass
    print("\n[Test 5] Testing forward pass...")
    try:
        model = load_model("gpt2")
        tokenizer = load_tokenizer("gpt2")
        logits = test_model_forward(model, tokenizer, "Hello, world!")
        print(f"✓ Forward pass successful")
        print(f"  Output shape: {logits.shape}")
    except NotImplementedError:
        print("✗ test_model_forward not implemented yet")

    print("\n" + "=" * 60)
    print("Basic testing complete!")
    print("Run full tests: pytest tests/test_model_loader.py -v")
    print("=" * 60)
