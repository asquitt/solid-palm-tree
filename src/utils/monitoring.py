"""
Progress Monitoring Utilities

Real-time progress tracking with tqdm and custom monitors.
"""

import logging
import time
from typing import Optional, Dict, Any
from contextlib import contextmanager

from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class ProgressMonitor:
    """
    Enhanced progress monitoring with tqdm integration.
    
    Features:
    - Progress bars for training loops
    - ETA estimation
    - Speed metrics (samples/sec, tokens/sec)
    - Memory monitoring
    - Nested progress bars for multi-stage tasks
    
    Usage:
        >>> monitor = ProgressMonitor(total=1000, desc="Training")
        >>> for batch in dataloader:
        ...     # training code
        ...     monitor.update(1, loss=batch_loss)
        >>> monitor.close()
    """
    
    def __init__(
        self,
        total: Optional[int] = None,
        desc: str = "Progress",
        unit: str = "it",
        leave: bool = True,
    ):
        """
        Initialize progress monitor.
        
        Args:
            total: Total number of iterations
            desc: Description text
            unit: Unit name (it, batch, sample, etc.)
            leave: Keep progress bar after completion
        """
        self.pbar = tqdm(
            total=total,
            desc=desc,
            unit=unit,
            leave=leave,
            dynamic_ncols=True,
        )
        self.start_time = time.time()
        
    def update(self, n: int = 1, **metrics):
        """
        Update progress and display metrics.
        
        Args:
            n: Number of iterations to advance
            **metrics: Additional metrics to display (loss, accuracy, etc.)
        """
        # Format metrics for display
        postfix = {}
        for key, value in metrics.items():
            if isinstance(value, float):
                postfix[key] = f"{value:.4f}"
            else:
                postfix[key] = value
                
        self.pbar.set_postfix(postfix)
        self.pbar.update(n)
        
    def close(self):
        """Close progress bar."""
        elapsed = time.time() - self.start_time
        self.pbar.set_postfix_str(f"Total time: {elapsed:.2f}s")
        self.pbar.close()
        
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


@contextmanager
def progress_context(total: int, desc: str = "Processing", **kwargs):
    """
    Context manager for progress monitoring.
    
    Usage:
        >>> with progress_context(total=100, desc="Training") as pbar:
        ...     for i in range(100):
        ...         # do work
        ...         pbar.update(1, loss=0.5)
    """
    monitor = ProgressMonitor(total=total, desc=desc, **kwargs)
    try:
        yield monitor
    finally:
        monitor.close()


class TrainingMonitor:
    """
    Specialized monitor for training loops.
    
    Tracks:
    - Epoch progress
    - Batch progress
    - Loss trends
    - Speed metrics
    - Time remaining
    """
    
    def __init__(self, num_epochs: int, steps_per_epoch: int):
        """Initialize training monitor."""
        self.num_epochs = num_epochs
        self.steps_per_epoch = steps_per_epoch
        self.epoch_pbar = None
        self.step_pbar = None
        
    def start_epoch(self, epoch: int):
        """Start tracking a new epoch."""
        if self.epoch_pbar is None:
            self.epoch_pbar = tqdm(
                total=self.num_epochs,
                desc="Epochs",
                position=0,
                leave=True
            )
        
        if self.step_pbar is not None:
            self.step_pbar.close()
            
        self.step_pbar = tqdm(
            total=self.steps_per_epoch,
            desc=f"Epoch {epoch + 1}/{self.num_epochs}",
            position=1,
            leave=False
        )
        
    def update_step(self, **metrics):
        """Update step progress."""
        if self.step_pbar:
            postfix = {k: f"{v:.4f}" if isinstance(v, float) else v 
                      for k, v in metrics.items()}
            self.step_pbar.set_postfix(postfix)
            self.step_pbar.update(1)
            
    def end_epoch(self, **metrics):
        """End current epoch."""
        if self.step_pbar:
            self.step_pbar.close()
        if self.epoch_pbar:
            postfix = {k: f"{v:.4f}" if isinstance(v, float) else v 
                      for k, v in metrics.items()}
            self.epoch_pbar.set_postfix(postfix)
            self.epoch_pbar.update(1)
            
    def close(self):
        """Close all progress bars."""
        if self.step_pbar:
            self.step_pbar.close()
        if self.epoch_pbar:
            self.epoch_pbar.close()
