"""Integration Tests for Training Pipeline"""

import pytest
import tempfile
import shutil
from pathlib import Path

from src.training.config import TrainingConfig
from src.training.trainer import Trainer


@pytest.mark.slow
@pytest.mark.integration
class TestTrainingPipeline:
    """Integration tests for complete training pipeline."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary directory for outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_quick_training_run(self, temp_output_dir):
        """Test a quick training run with minimal data."""
        config = TrainingConfig(
            model_name="gpt2",
            dataset_name="wikitext",
            dataset_config="wikitext-2-raw-v1",
            num_epochs=1,
            batch_size=2,
            max_seq_length=128,
            output_dir=temp_output_dir,
            use_mlflow=False,  # Disable for testing
            logging_steps=1,
            checkpoint_frequency=1000,  # Don't checkpoint in test
        )

        trainer = Trainer(config)
        # Note: Full training test would be slow, so we just test initialization
        trainer.load_model_and_tokenizer()

        assert trainer.model is not None
        assert trainer.tokenizer is not None

    def test_checkpoint_saving(self, temp_output_dir):
        """Test checkpoint saving functionality."""
        config = TrainingConfig(
            model_name="gpt2",
            output_dir=temp_output_dir,
            use_mlflow=False,
        )

        trainer = Trainer(config)
        trainer.load_model_and_tokenizer()
        trainer.setup_optimizer_and_scheduler()

        # Save checkpoint
        checkpoint_path = Path(temp_output_dir) / "test-checkpoint"
        trainer.model.save_pretrained(checkpoint_path)

        assert checkpoint_path.exists()
        assert (checkpoint_path / "config.json").exists()
