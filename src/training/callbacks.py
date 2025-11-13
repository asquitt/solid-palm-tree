"""
Training Callbacks

This module provides callback classes for training events like
checkpointing, logging, early stopping, and custom hooks.

Callbacks Pattern:
Callbacks allow you to inject custom behavior at specific points
in the training loop without modifying the core training code.

Common callback events:
- on_train_begin/end
- on_epoch_begin/end
- on_step_begin/end
- on_checkpoint_save
- on_evaluation
"""

import os
import time
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import json

import mlflow
from ray import train as ray_train

logger = logging.getLogger(__name__)


class BaseCallback:
    """
    Base class for training callbacks.

    All custom callbacks should inherit from this class and override
    the methods they need to customize.
    """

    def on_train_begin(self, trainer, **kwargs):
        """Called at the beginning of training."""
        pass

    def on_train_end(self, trainer, **kwargs):
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, trainer, epoch: int, **kwargs):
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float], **kwargs):
        """Called at the end of each epoch."""
        pass

    def on_step_begin(self, trainer, step: int, **kwargs):
        """Called at the beginning of each training step."""
        pass

    def on_step_end(self, trainer, step: int, loss: float, **kwargs):
        """Called at the end of each training step."""
        pass

    def on_checkpoint_save(self, trainer, checkpoint_dir: str, **kwargs):
        """Called when a checkpoint is saved."""
        pass

    def on_evaluation(self, trainer, metrics: Dict[str, float], **kwargs):
        """Called after evaluation."""
        pass


class CheckpointCallback(BaseCallback):
    """
    Callback for saving model checkpoints.

    This callback handles:
    1. Periodic checkpoint saving during training
    2. Best model tracking based on evaluation metrics
    3. Cloud storage upload (S3/GCS/Azure)
    4. DVC versioning integration

    Args:
        checkpoint_dir: Directory to save checkpoints
        save_frequency: Save checkpoint every N steps
        keep_n_checkpoints: Maximum number of checkpoints to keep
        upload_to_cloud: Whether to upload to cloud storage
        cloud_path: Cloud storage path (e.g., s3://bucket/path)
    """

    def __init__(
        self,
        checkpoint_dir: str,
        save_frequency: int = 500,
        keep_n_checkpoints: int = 3,
        upload_to_cloud: bool = False,
        cloud_path: Optional[str] = None,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_frequency = save_frequency
        self.keep_n_checkpoints = keep_n_checkpoints
        self.upload_to_cloud = upload_to_cloud
        self.cloud_path = cloud_path

        self.checkpoint_history = []  # Track saved checkpoints
        self.best_metric = float('inf')

    def on_train_begin(self, trainer, **kwargs):
        """Create checkpoint directory."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Checkpoints will be saved to: {self.checkpoint_dir}")

    def on_step_end(self, trainer, step: int, loss: float, **kwargs):
        """
        Check if checkpoint should be saved.

        Checkpoint Frequency:
        - Every save_frequency steps
        - Best model based on evaluation loss
        - Final checkpoint at end of training
        """
        if step % self.save_frequency == 0:
            checkpoint_path = self.checkpoint_dir / f"checkpoint-step-{step}"
            self._save_checkpoint(trainer, checkpoint_path, step)

    def on_evaluation(self, trainer, metrics: Dict[str, float], **kwargs):
        """Save checkpoint if evaluation metric improved."""
        eval_loss = metrics.get("loss", float('inf'))

        if eval_loss < self.best_metric:
            self.best_metric = eval_loss
            best_path = self.checkpoint_dir / "best_model"

            logger.info(f"New best model! Loss: {eval_loss:.4f}")
            self._save_checkpoint(trainer, best_path, trainer.global_step, is_best=True)

    def _save_checkpoint(
        self,
        trainer,
        checkpoint_path: Path,
        step: int,
        is_best: bool = False
    ):
        """
        Save model checkpoint with metadata.

        Checkpoint Contents:
        - Model weights
        - Tokenizer configuration
        - Optimizer state
        - Scheduler state
        - Training metadata (step, epoch, metrics)
        """
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save model and tokenizer
        trainer.model.save_pretrained(checkpoint_path)
        trainer.tokenizer.save_pretrained(checkpoint_path)

        # Save optimizer and scheduler state
        training_state = {
            "global_step": trainer.global_step,
            "epoch": trainer.current_epoch,
            "best_eval_loss": trainer.best_eval_loss,
            "optimizer": trainer.optimizer.state_dict(),
            "scheduler": trainer.scheduler.state_dict(),
        }
        import torch
        torch.save(training_state, checkpoint_path / "training_state.pt")

        # Save metadata
        metadata = {
            "step": step,
            "timestamp": time.time(),
            "is_best": is_best,
            "config": trainer.config.to_dict(),
        }
        with open(checkpoint_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Checkpoint saved: {checkpoint_path}")

        # Track checkpoint history
        self.checkpoint_history.append({
            "path": str(checkpoint_path),
            "step": step,
            "timestamp": time.time(),
        })

        # Remove old checkpoints if exceeding limit
        self._cleanup_old_checkpoints()

        # Upload to cloud if configured
        if self.upload_to_cloud and self.cloud_path:
            self._upload_to_cloud(checkpoint_path)

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to save disk space."""
        if len(self.checkpoint_history) <= self.keep_n_checkpoints:
            return

        # Sort by timestamp
        self.checkpoint_history.sort(key=lambda x: x["timestamp"])

        # Remove oldest checkpoints
        num_to_remove = len(self.checkpoint_history) - self.keep_n_checkpoints

        for i in range(num_to_remove):
            checkpoint = self.checkpoint_history[i]
            checkpoint_path = Path(checkpoint["path"])

            # Don't remove best model
            if checkpoint_path.name == "best_model":
                continue

            # Remove checkpoint directory
            if checkpoint_path.exists():
                import shutil
                shutil.rmtree(checkpoint_path)
                logger.info(f"Removed old checkpoint: {checkpoint_path}")

        # Update history
        self.checkpoint_history = self.checkpoint_history[num_to_remove:]

    def _upload_to_cloud(self, checkpoint_path: Path):
        """
        Upload checkpoint to cloud storage (S3/GCS/Azure).

        Cloud Storage Benefits:
        - Durability: Checkpoints survive spot instance interruptions
        - Sharing: Team members can access checkpoints
        - Archival: Long-term storage for reproducibility
        """
        try:
            if self.cloud_path.startswith("s3://"):
                self._upload_to_s3(checkpoint_path)
            elif self.cloud_path.startswith("gs://"):
                self._upload_to_gcs(checkpoint_path)
            elif self.cloud_path.startswith("azure://"):
                self._upload_to_azure(checkpoint_path)
            else:
                logger.warning(f"Unknown cloud path format: {self.cloud_path}")
        except Exception as e:
            logger.error(f"Failed to upload checkpoint to cloud: {e}")

    def _upload_to_s3(self, checkpoint_path: Path):
        """Upload to AWS S3."""
        import boto3
        from botocore.exceptions import ClientError

        # Parse S3 path: s3://bucket/prefix
        s3_path = self.cloud_path.replace("s3://", "")
        bucket, prefix = s3_path.split("/", 1)

        s3_client = boto3.client("s3")

        # Upload all files in checkpoint directory
        for file_path in checkpoint_path.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(checkpoint_path)
                s3_key = f"{prefix}/{checkpoint_path.name}/{relative_path}"

                try:
                    s3_client.upload_file(str(file_path), bucket, s3_key)
                    logger.debug(f"Uploaded: {s3_key}")
                except ClientError as e:
                    logger.error(f"S3 upload failed for {s3_key}: {e}")

        logger.info(f"Checkpoint uploaded to S3: {self.cloud_path}/{checkpoint_path.name}")

    def _upload_to_gcs(self, checkpoint_path: Path):
        """Upload to Google Cloud Storage."""
        from google.cloud import storage

        # Parse GCS path: gs://bucket/prefix
        gcs_path = self.cloud_path.replace("gs://", "")
        bucket_name, prefix = gcs_path.split("/", 1)

        client = storage.Client()
        bucket = client.bucket(bucket_name)

        # Upload all files
        for file_path in checkpoint_path.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(checkpoint_path)
                blob_name = f"{prefix}/{checkpoint_path.name}/{relative_path}"

                blob = bucket.blob(blob_name)
                blob.upload_from_filename(str(file_path))
                logger.debug(f"Uploaded: {blob_name}")

        logger.info(f"Checkpoint uploaded to GCS: {self.cloud_path}/{checkpoint_path.name}")

    def _upload_to_azure(self, checkpoint_path: Path):
        """Upload to Azure Blob Storage."""
        from azure.storage.blob import BlobServiceClient

        # Parse Azure path: azure://container/prefix
        azure_path = self.cloud_path.replace("azure://", "")
        container_name, prefix = azure_path.split("/", 1)

        # Get connection string from environment
        connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        if not connection_string:
            logger.error("AZURE_STORAGE_CONNECTION_STRING not set")
            return

        blob_service_client = BlobServiceClient.from_connection_string(connection_string)

        # Upload all files
        for file_path in checkpoint_path.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(checkpoint_path)
                blob_name = f"{prefix}/{checkpoint_path.name}/{relative_path}"

                blob_client = blob_service_client.get_blob_client(
                    container=container_name,
                    blob=blob_name
                )
                with open(file_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)
                logger.debug(f"Uploaded: {blob_name}")

        logger.info(f"Checkpoint uploaded to Azure: {self.cloud_path}/{checkpoint_path.name}")


class LoggingCallback(BaseCallback):
    """
    Callback for logging training metrics.

    This callback logs metrics to:
    1. Console (stdout)
    2. MLflow for experiment tracking
    3. Ray Train for distributed logging
    4. TensorBoard (optional)

    Args:
        log_frequency: Log metrics every N steps
        use_mlflow: Whether to log to MLflow
        use_tensorboard: Whether to log to TensorBoard
    """

    def __init__(
        self,
        log_frequency: int = 10,
        use_mlflow: bool = True,
        use_tensorboard: bool = False,
    ):
        self.log_frequency = log_frequency
        self.use_mlflow = use_mlflow
        self.use_tensorboard = use_tensorboard

        self.start_time = None
        self.step_times = []

    def on_train_begin(self, trainer, **kwargs):
        """Initialize logging."""
        self.start_time = time.time()

        # Initialize MLflow
        if self.use_mlflow and trainer.config.use_mlflow:
            mlflow.set_tracking_uri(trainer.config.mlflow_tracking_uri)
            mlflow.set_experiment(trainer.config.mlflow_experiment_name)

            # Start MLflow run
            run_name = trainer.config.mlflow_run_name or f"run-{int(time.time())}"
            mlflow.start_run(run_name=run_name)

            # Log configuration
            mlflow.log_params(trainer.config.to_dict())

            logger.info(f"MLflow tracking enabled: {trainer.config.mlflow_tracking_uri}")

    def on_train_end(self, trainer, **kwargs):
        """Finalize logging."""
        total_time = time.time() - self.start_time

        # Log final metrics
        final_metrics = {
            "total_training_time": total_time,
            "total_steps": trainer.global_step,
            "final_epoch": trainer.current_epoch,
        }

        if self.use_mlflow:
            mlflow.log_metrics(final_metrics)
            mlflow.end_run()

        logger.info(f"Training completed in {total_time / 3600:.2f} hours")

    def on_step_end(self, trainer, step: int, loss: float, **kwargs):
        """Log step metrics."""
        if step % self.log_frequency != 0:
            return

        # Calculate metrics
        elapsed = time.time() - self.start_time
        lr = trainer.scheduler.get_last_lr()[0]

        # Calculate throughput
        self.step_times.append(time.time())
        if len(self.step_times) > self.log_frequency:
            self.step_times = self.step_times[-self.log_frequency:]

        if len(self.step_times) >= 2:
            time_diff = self.step_times[-1] - self.step_times[0]
            steps_per_sec = (len(self.step_times) - 1) / time_diff if time_diff > 0 else 0
        else:
            steps_per_sec = 0

        metrics = {
            "train/loss": loss,
            "train/learning_rate": lr,
            "train/global_step": step,
            "train/steps_per_second": steps_per_sec,
            "train/elapsed_time": elapsed,
        }

        # Log to console
        logger.info(
            f"Step {step} | "
            f"Loss: {loss:.4f} | "
            f"LR: {lr:.2e} | "
            f"Speed: {steps_per_sec:.2f} steps/s"
        )

        # Log to MLflow
        if self.use_mlflow:
            mlflow.log_metrics(metrics, step=step)

        # Log to Ray Train (for distributed training)
        try:
            ray_train.report(metrics)
        except Exception:
            pass  # Not in Ray Train context

    def on_evaluation(self, trainer, metrics: Dict[str, float], **kwargs):
        """Log evaluation metrics."""
        # Prefix metrics with "eval/"
        eval_metrics = {f"eval/{k}": v for k, v in metrics.items()}

        # Log to console
        logger.info(f"Evaluation | {metrics}")

        # Log to MLflow
        if self.use_mlflow:
            mlflow.log_metrics(eval_metrics, step=trainer.global_step)

        # Log to Ray Train
        try:
            ray_train.report(eval_metrics)
        except Exception:
            pass


class EarlyStoppingCallback(BaseCallback):
    """
    Callback for early stopping based on evaluation metrics.

    Early stopping prevents overfitting by stopping training when
    the validation loss stops improving.

    Args:
        patience: Number of evaluations to wait for improvement
        min_delta: Minimum change to qualify as improvement
        metric_name: Metric to monitor (default: "loss")
    """

    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.001,
        metric_name: str = "loss",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.metric_name = metric_name

        self.best_metric = float('inf')
        self.patience_counter = 0
        self.should_stop = False

    def on_evaluation(self, trainer, metrics: Dict[str, float], **kwargs):
        """Check if training should stop."""
        current_metric = metrics.get(self.metric_name, float('inf'))

        # Check if metric improved
        if current_metric < self.best_metric - self.min_delta:
            self.best_metric = current_metric
            self.patience_counter = 0
            logger.info(f"Metric improved: {current_metric:.4f}")
        else:
            self.patience_counter += 1
            logger.info(
                f"No improvement for {self.patience_counter}/{self.patience} evaluations"
            )

            if self.patience_counter >= self.patience:
                self.should_stop = True
                logger.info(
                    f"Early stopping triggered! Best {self.metric_name}: {self.best_metric:.4f}"
                )
