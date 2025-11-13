"""
Utilities Module

Helper utilities for visualization, monitoring, and analysis.
"""

__all__ = [
    "TrainingVisualizer",
    "plot_training_history",
    "ProgressMonitor",
    "DatasetAnalyzer",
    "setup_logger",
    "get_device",
    "set_seed",
]


def __getattr__(name):
    """Lazy import utilities to avoid importing heavy dependencies."""
    if name in ["TrainingVisualizer", "plot_training_history"]:
        from src.utils.visualization import TrainingVisualizer, plot_training_history
        return TrainingVisualizer if name == "TrainingVisualizer" else plot_training_history
    elif name == "ProgressMonitor":
        from src.utils.monitoring import ProgressMonitor
        return ProgressMonitor
    elif name == "DatasetAnalyzer":
        from src.utils.dataset_analysis import DatasetAnalyzer
        return DatasetAnalyzer
    elif name in ["setup_logger", "get_device", "set_seed"]:
        from src.utils.logging import setup_logger
        from src.utils.helpers import get_device, set_seed
        if name == "setup_logger":
            return setup_logger
        elif name == "get_device":
            return get_device
        elif name == "set_seed":
            return set_seed
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
