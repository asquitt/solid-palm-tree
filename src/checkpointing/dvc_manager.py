"""DVC (Data Version Control) integration for checkpoint versioning."""

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

class DVCManager:
    """Manager for DVC versioning of checkpoints."""
    
    def __init__(self, remote_storage: str):
        """Initialize DVC manager."""
        self.remote_storage = remote_storage
        logger.info(f"DVCManager initialized with remote: {remote_storage}")
    
    def add_checkpoint(self, checkpoint_path: str):
        """Add checkpoint to DVC tracking."""
        try:
            subprocess.run(["dvc", "add", checkpoint_path], check=True)
            subprocess.run(["git", "add", f"{checkpoint_path}.dvc"], check=True)
            logger.info(f"Checkpoint added to DVC: {checkpoint_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to add checkpoint to DVC: {e}")
    
    def push_checkpoint(self):
        """Push checkpoint to remote storage."""
        try:
            subprocess.run(["dvc", "push"], check=True)
            logger.info("Checkpoint pushed to remote storage")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to push checkpoint: {e}")
