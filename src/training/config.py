"""
Training Configuration

This module defines the configuration dataclass for distributed LLM training.
It includes all hyperparameters, infrastructure settings, and optimization flags.

Key Concepts:
- Dataclasses: Python's built-in way to define configuration objects
- Type hints: Ensures type safety and better IDE support
- Default values: Sensible defaults based on best practices
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class OptimizerType(Enum):
    """Supported optimizer types for training."""
    ADAMW = "adamw"  # AdamW: Adam with weight decay (most common for LLMs)
    ADAM = "adam"    # Standard Adam optimizer
    SGD = "sgd"      # Stochastic Gradient Descent (rarely used for LLMs)
    ADAFACTOR = "adafactor"  # Memory-efficient optimizer for large models


class SchedulerType(Enum):
    """Learning rate scheduler types."""
    LINEAR = "linear"           # Linear decay to 0
    COSINE = "cosine"           # Cosine annealing (recommended)
    CONSTANT = "constant"       # No decay
    POLYNOMIAL = "polynomial"   # Polynomial decay


class PrecisionType(Enum):
    """Training precision for memory and speed optimization."""
    FP32 = "fp32"      # Full precision (32-bit) - most accurate, most memory
    FP16 = "fp16"      # Half precision (16-bit) - 2x faster, 50% memory
    BF16 = "bf16"      # Brain float 16 - better range than fp16, requires A100+
    INT8 = "int8"      # 8-bit quantization - 4x memory savings (QLoRA)


@dataclass
class TrainingConfig:
    """
    Comprehensive training configuration for distributed LLM fine-tuning.

    This configuration class contains all parameters needed for training,
    from model selection to optimization settings. Each parameter is
    documented with its purpose and recommended values.

    Usage:
        >>> config = TrainingConfig(
        ...     model_name="gpt2",
        ...     num_epochs=3,
        ...     batch_size=8
        ... )
    """

    # ============================================================================
    # Model Configuration
    # ============================================================================

    model_name: str = "gpt2"
    """
    HuggingFace model identifier or path to local model.
    Examples: "gpt2", "meta-llama/Llama-2-7b-hf", "EleutherAI/gpt-neo-2.7B"
    """

    model_revision: Optional[str] = None
    """Specific model revision/version to use (for reproducibility)."""

    tokenizer_name: Optional[str] = None
    """Tokenizer name if different from model_name."""

    cache_dir: Optional[str] = None
    """Directory to cache downloaded models and tokenizers."""

    # ============================================================================
    # Dataset Configuration
    # ============================================================================

    dataset_name: str = "wikitext"
    """
    Dataset name from HuggingFace datasets or path to local dataset.
    Examples: "wikitext", "alpaca", "squad", "your-username/custom-dataset"
    """

    dataset_config: Optional[str] = "wikitext-2-raw-v1"
    """Specific dataset configuration/subset."""

    train_split: str = "train"
    """Name of the training split."""

    validation_split: str = "validation"
    """Name of the validation split."""

    test_split: Optional[str] = "test"
    """Name of the test split (optional)."""

    max_seq_length: int = 512
    """
    Maximum sequence length for tokenization.
    Trade-off: Longer = more context but more memory.
    Common values: 512 (GPT-2), 2048 (GPT-3), 4096 (Llama-2)
    """

    preprocessing_num_workers: int = 4
    """Number of parallel processes for dataset preprocessing."""

    # ============================================================================
    # Training Hyperparameters
    # ============================================================================

    num_epochs: int = 3
    """
    Number of complete passes through the training dataset.
    Rule of thumb: 3-5 epochs for fine-tuning, 1-2 for large datasets.
    """

    batch_size: int = 8
    """
    Training batch size per GPU device.
    Must balance: GPU memory, training speed, gradient quality.
    Start with 8 and adjust based on GPU memory (OOM errors).
    """

    gradient_accumulation_steps: int = 1
    """
    Number of steps to accumulate gradients before updating weights.
    Effective batch size = batch_size * num_gpus * gradient_accumulation_steps
    Use this to simulate larger batches when GPU memory is limited.
    """

    learning_rate: float = 5e-5
    """
    Initial learning rate for the optimizer.
    Typical range for LLM fine-tuning: 1e-5 to 5e-5
    Larger models usually need smaller learning rates.
    """

    weight_decay: float = 0.01
    """
    L2 regularization coefficient to prevent overfitting.
    Typical range: 0.01 to 0.1
    """

    warmup_steps: int = 500
    """
    Number of steps for learning rate warmup (gradual increase from 0).
    Helps stabilize training at the start.
    Rule of thumb: 5-10% of total training steps.
    """

    warmup_ratio: float = 0.1
    """
    Alternative to warmup_steps: fraction of total steps for warmup.
    If both are set, warmup_steps takes precedence.
    """

    max_grad_norm: float = 1.0
    """
    Maximum gradient norm for gradient clipping.
    Prevents exploding gradients (common in RNNs/Transformers).
    Typical value: 1.0
    """

    optimizer: OptimizerType = OptimizerType.ADAMW
    """Optimizer algorithm for training."""

    scheduler: SchedulerType = SchedulerType.COSINE
    """Learning rate scheduler type."""

    # ============================================================================
    # Precision & Memory Optimization
    # ============================================================================

    precision: PrecisionType = PrecisionType.FP16
    """
    Training precision for speed and memory optimization.
    - FP32: Baseline, most accurate
    - FP16: 2x speedup, requires gradient scaling
    - BF16: Better than FP16 for large models (A100 GPUs)
    - INT8: 4x memory savings with QLoRA (slight accuracy loss)
    """

    gradient_checkpointing: bool = False
    """
    Enable gradient checkpointing to trade compute for memory.
    Saves memory by not storing all activations (recomputes during backward).
    Cost: ~20% slower training, Benefit: 30-50% memory savings.
    Enable when: Getting OOM errors with large models.
    """

    use_lora: bool = False
    """
    Enable LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.
    LoRA trains small adapter layers instead of all parameters.
    Benefits: 90% less memory, faster training, smaller checkpoints.
    Best for: Large models (7B+) with limited compute.
    """

    lora_r: int = 8
    """
    LoRA rank (dimensionality of low-rank matrices).
    Higher = more expressiveness but more memory.
    Typical values: 4, 8, 16, 32
    """

    lora_alpha: int = 16
    """
    LoRA scaling parameter.
    Typically set to 2 * lora_r.
    """

    lora_dropout: float = 0.1
    """Dropout probability for LoRA layers."""

    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    """
    Which modules to apply LoRA to.
    Common choices:
    - ["q_proj", "v_proj"]: Query and value projections (most common)
    - ["q_proj", "k_proj", "v_proj", "o_proj"]: All attention layers
    """

    # ============================================================================
    # Distributed Training Configuration
    # ============================================================================

    num_workers: int = 1
    """
    Number of distributed training workers (typically = number of GPUs).
    For single GPU: 1
    For multi-GPU: torch.cuda.device_count()
    For multi-node: num_nodes * gpus_per_node
    """

    use_horovod: bool = False
    """
    Use Horovod for multi-node distributed training.
    Horovod provides efficient ring-allreduce for gradient synchronization.
    Enable when: Training across multiple nodes (>1 machine).
    """

    backend: str = "nccl"
    """
    Distributed backend for PyTorch.
    - "nccl": Best for GPU (NVIDIA only)
    - "gloo": CPU or mixed GPU/CPU
    - "mpi": For Horovod
    """

    # ============================================================================
    # Checkpointing & Logging
    # ============================================================================

    output_dir: str = "./outputs"
    """Directory to save model checkpoints and logs."""

    checkpoint_frequency: int = 500
    """Save checkpoint every N training steps."""

    checkpoint_to_cloud: bool = True
    """Upload checkpoints to cloud storage (S3/GCS/Azure)."""

    cloud_storage_path: Optional[str] = None
    """
    Cloud storage path for checkpoints.
    Examples:
    - s3://my-bucket/checkpoints/
    - gs://my-bucket/checkpoints/
    - azure://my-container/checkpoints/
    """

    use_dvc: bool = True
    """Enable DVC for checkpoint versioning and reproducibility."""

    logging_steps: int = 10
    """Log training metrics every N steps."""

    evaluation_steps: int = 500
    """Run evaluation on validation set every N steps."""

    save_total_limit: int = 3
    """Maximum number of checkpoints to keep (oldest deleted first)."""

    # ============================================================================
    # MLflow Experiment Tracking
    # ============================================================================

    use_mlflow: bool = True
    """Enable MLflow for experiment tracking."""

    mlflow_tracking_uri: str = "http://localhost:5000"
    """MLflow tracking server URI."""

    mlflow_experiment_name: str = "llm-finetuning"
    """MLflow experiment name."""

    mlflow_run_name: Optional[str] = None
    """MLflow run name (auto-generated if None)."""

    # ============================================================================
    # Cost Optimization
    # ============================================================================

    use_spot_instances: bool = False
    """Use spot/preemptible instances for cost savings (60-80% cheaper)."""

    spot_fallback_enabled: bool = True
    """Automatically fallback to on-demand if spot instances are unavailable."""

    max_cost_per_hour: Optional[float] = None
    """Maximum cost per hour budget (training stops if exceeded)."""

    auto_scale_enabled: bool = True
    """Enable auto-scaling of compute resources based on workload."""

    # ============================================================================
    # Advanced Options
    # ============================================================================

    seed: int = 42
    """Random seed for reproducibility."""

    dataloader_num_workers: int = 4
    """Number of subprocesses for data loading."""

    dataloader_pin_memory: bool = True
    """Pin memory for faster GPU transfer (set False if low RAM)."""

    resume_from_checkpoint: Optional[str] = None
    """Path to checkpoint to resume training from."""

    ignore_data_skip: bool = False
    """
    When resuming, skip data already processed.
    Set False to reprocess all data (useful for debugging).
    """

    debug_mode: bool = False
    """Enable debug mode with verbose logging and assertions."""

    extra_args: Dict[str, Any] = field(default_factory=dict)
    """Additional arguments for custom configurations."""

    def __post_init__(self):
        """Validation and derived configurations."""
        # Validate learning rate
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")

        # Validate batch size
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        # Set tokenizer_name to model_name if not specified
        if self.tokenizer_name is None:
            self.tokenizer_name = self.model_name

        # Validate cloud storage configuration
        if self.checkpoint_to_cloud and self.cloud_storage_path is None:
            raise ValueError(
                "cloud_storage_path must be set when checkpoint_to_cloud=True"
            )

        # Validate spot instance configuration
        if self.use_spot_instances and not self.use_dvc:
            print(
                "WARNING: Spot instances without DVC may lose checkpoints on interruption. "
                "Consider enabling use_dvc=True for checkpoint recovery."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for logging/serialization."""
        return {
            k: v.value if isinstance(v, Enum) else v
            for k, v in self.__dict__.items()
            if not k.startswith('_')
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig":
        """Create config from dictionary."""
        # Convert string enums back to Enum objects
        if "optimizer" in config_dict and isinstance(config_dict["optimizer"], str):
            config_dict["optimizer"] = OptimizerType(config_dict["optimizer"])
        if "scheduler" in config_dict and isinstance(config_dict["scheduler"], str):
            config_dict["scheduler"] = SchedulerType(config_dict["scheduler"])
        if "precision" in config_dict and isinstance(config_dict["precision"], str):
            config_dict["precision"] = PrecisionType(config_dict["precision"])

        return cls(**config_dict)

    def get_effective_batch_size(self) -> int:
        """Calculate effective batch size across all devices."""
        return self.batch_size * self.num_workers * self.gradient_accumulation_steps

    def estimate_memory_usage(self, model_params: int) -> Dict[str, float]:
        """
        Estimate GPU memory usage in GB.

        Args:
            model_params: Number of model parameters

        Returns:
            Dictionary with memory breakdown
        """
        # Bytes per parameter based on precision
        bytes_per_param = {
            PrecisionType.FP32: 4,
            PrecisionType.FP16: 2,
            PrecisionType.BF16: 2,
            PrecisionType.INT8: 1,
        }[self.precision]

        # Model weights
        model_memory = model_params * bytes_per_param / 1e9

        # Optimizer states (AdamW stores 2 states per parameter)
        optimizer_memory = model_params * 8 / 1e9  # Always FP32 for stability

        # Gradients
        gradient_memory = model_params * bytes_per_param / 1e9

        # Activations (rough estimate based on batch size and sequence length)
        activation_memory = (
            self.batch_size * self.max_seq_length * model_params * bytes_per_param / 1e9 / 1000
        )

        # Gradient checkpointing reduces activation memory by ~50%
        if self.gradient_checkpointing:
            activation_memory *= 0.5

        # LoRA only trains adapter weights (~0.5-1% of parameters)
        if self.use_lora:
            trainable_params = model_params * (self.lora_r * 2 / 768)  # Rough estimate
            optimizer_memory = trainable_params * 8 / 1e9
            gradient_memory = trainable_params * bytes_per_param / 1e9

        total = model_memory + optimizer_memory + gradient_memory + activation_memory

        return {
            "model_gb": round(model_memory, 2),
            "optimizer_gb": round(optimizer_memory, 2),
            "gradients_gb": round(gradient_memory, 2),
            "activations_gb": round(activation_memory, 2),
            "total_gb": round(total, 2),
        }
