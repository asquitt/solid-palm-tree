"""
Unit Tests for Training Configuration

Tests the TrainingConfig dataclass and its validation logic.
"""

import pytest
from src.training.config import (
    TrainingConfig,
    OptimizerType,
    SchedulerType,
    PrecisionType,
)


class TestTrainingConfig:
    """Test suite for TrainingConfig."""

    def test_default_config(self):
        """Test creating config with default values."""
        config = TrainingConfig()

        assert config.model_name == "gpt2"
        assert config.num_epochs == 3
        assert config.batch_size == 8
        assert config.learning_rate == 5e-5

    def test_custom_config(self):
        """Test creating config with custom values."""
        config = TrainingConfig(
            model_name="meta-llama/Llama-2-7b-hf",
            num_epochs=5,
            batch_size=16,
            learning_rate=1e-4,
        )

        assert config.model_name == "meta-llama/Llama-2-7b-hf"
        assert config.num_epochs == 5
        assert config.batch_size == 16
        assert config.learning_rate == 1e-4

    def test_invalid_learning_rate(self):
        """Test that negative learning rate raises error."""
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            TrainingConfig(learning_rate=-1e-5)

    def test_invalid_batch_size(self):
        """Test that zero/negative batch size raises error."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            TrainingConfig(batch_size=0)

        with pytest.raises(ValueError, match="batch_size must be positive"):
            TrainingConfig(batch_size=-1)

    def test_tokenizer_name_defaults_to_model_name(self):
        """Test that tokenizer_name defaults to model_name."""
        config = TrainingConfig(model_name="gpt2")
        assert config.tokenizer_name == "gpt2"

        config = TrainingConfig(model_name="gpt2", tokenizer_name="custom-tokenizer")
        assert config.tokenizer_name == "custom-tokenizer"

    def test_cloud_storage_validation(self):
        """Test cloud storage path validation."""
        # Should raise error if checkpoint_to_cloud=True but no path
        with pytest.raises(ValueError, match="cloud_storage_path must be set"):
            TrainingConfig(
                checkpoint_to_cloud=True,
                cloud_storage_path=None,
            )

        # Should work if both are set
        config = TrainingConfig(
            checkpoint_to_cloud=True,
            cloud_storage_path="s3://my-bucket/checkpoints",
        )
        assert config.cloud_storage_path == "s3://my-bucket/checkpoints"

    def test_to_dict_conversion(self):
        """Test converting config to dictionary."""
        config = TrainingConfig(
            model_name="gpt2",
            learning_rate=1e-4,
            optimizer=OptimizerType.ADAMW,
        )

        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["model_name"] == "gpt2"
        assert config_dict["learning_rate"] == 1e-4
        assert config_dict["optimizer"] == "adamw"  # Enum converted to string

    def test_from_dict_conversion(self):
        """Test creating config from dictionary."""
        config_dict = {
            "model_name": "gpt2",
            "learning_rate": 1e-4,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "precision": "fp16",
        }

        config = TrainingConfig.from_dict(config_dict)

        assert config.model_name == "gpt2"
        assert config.learning_rate == 1e-4
        assert config.optimizer == OptimizerType.ADAMW
        assert config.scheduler == SchedulerType.COSINE
        assert config.precision == PrecisionType.FP16

    def test_effective_batch_size_calculation(self):
        """Test calculating effective batch size."""
        config = TrainingConfig(
            batch_size=8,
            num_workers=4,
            gradient_accumulation_steps=2,
        )

        effective_batch_size = config.get_effective_batch_size()

        # 8 * 4 * 2 = 64
        assert effective_batch_size == 64

    def test_memory_usage_estimation(self):
        """Test GPU memory usage estimation."""
        config = TrainingConfig(
            batch_size=8,
            max_seq_length=512,
            precision=PrecisionType.FP16,
        )

        # GPT-2 has 124M parameters
        memory_est = config.estimate_memory_usage(model_params=124_000_000)

        assert isinstance(memory_est, dict)
        assert "model_gb" in memory_est
        assert "optimizer_gb" in memory_est
        assert "gradients_gb" in memory_est
        assert "activations_gb" in memory_est
        assert "total_gb" in memory_est

        # Check that values are reasonable
        assert memory_est["total_gb"] > 0
        assert memory_est["total_gb"] < 100  # Should fit in modern GPUs

    def test_memory_estimation_with_lora(self):
        """Test memory estimation with LoRA enabled."""
        config_full = TrainingConfig(use_lora=False)
        config_lora = TrainingConfig(use_lora=True, lora_r=8)

        memory_full = config_full.estimate_memory_usage(124_000_000)
        memory_lora = config_lora.estimate_memory_usage(124_000_000)

        # LoRA should use significantly less memory
        assert memory_lora["optimizer_gb"] < memory_full["optimizer_gb"]
        assert memory_lora["gradients_gb"] < memory_full["gradients_gb"]

    def test_memory_estimation_with_gradient_checkpointing(self):
        """Test memory estimation with gradient checkpointing."""
        config_no_gc = TrainingConfig(gradient_checkpointing=False)
        config_gc = TrainingConfig(gradient_checkpointing=True)

        memory_no_gc = config_no_gc.estimate_memory_usage(124_000_000)
        memory_gc = config_gc.estimate_memory_usage(124_000_000)

        # Gradient checkpointing should reduce activation memory
        assert memory_gc["activations_gb"] < memory_no_gc["activations_gb"]

    def test_enum_types(self):
        """Test enum type conversions."""
        config = TrainingConfig(
            optimizer=OptimizerType.ADAMW,
            scheduler=SchedulerType.COSINE,
            precision=PrecisionType.FP16,
        )

        assert isinstance(config.optimizer, OptimizerType)
        assert isinstance(config.scheduler, SchedulerType)
        assert isinstance(config.precision, PrecisionType)

        assert config.optimizer.value == "adamw"
        assert config.scheduler.value == "cosine"
        assert config.precision.value == "fp16"


class TestOptimizerTypes:
    """Test optimizer enum."""

    def test_optimizer_values(self):
        """Test optimizer enum values."""
        assert OptimizerType.ADAMW.value == "adamw"
        assert OptimizerType.ADAM.value == "adam"
        assert OptimizerType.SGD.value == "sgd"
        assert OptimizerType.ADAFACTOR.value == "adafactor"


class TestSchedulerTypes:
    """Test scheduler enum."""

    def test_scheduler_values(self):
        """Test scheduler enum values."""
        assert SchedulerType.LINEAR.value == "linear"
        assert SchedulerType.COSINE.value == "cosine"
        assert SchedulerType.CONSTANT.value == "constant"
        assert SchedulerType.POLYNOMIAL.value == "polynomial"


class TestPrecisionTypes:
    """Test precision enum."""

    def test_precision_values(self):
        """Test precision enum values."""
        assert PrecisionType.FP32.value == "fp32"
        assert PrecisionType.FP16.value == "fp16"
        assert PrecisionType.BF16.value == "bf16"
        assert PrecisionType.INT8.value == "int8"
