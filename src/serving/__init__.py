"""
Model Serving Module - Ray Serve Integration

This module provides production model serving with Ray Serve,
including A/B testing, autoscaling, and deployment management.
"""

__all__ = ["ModelServer", "DeploymentConfig", "create_default_health_manager"]


def __getattr__(name):
    """Lazy import to avoid importing heavy dependencies."""
    if name == "ModelServer" or name == "DeploymentConfig":
        from src.serving.server import ModelServer, DeploymentConfig
        return ModelServer if name == "ModelServer" else DeploymentConfig
    elif name == "create_default_health_manager":
        from src.serving.health import create_default_health_manager
        return create_default_health_manager
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
