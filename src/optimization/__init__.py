"""
Cost Optimization Module

Spot instance management, cost tracking, and resource optimization.
"""

from src.optimization.spot_manager import SpotInstanceManager
from src.optimization.cost_tracker import CostTracker

__all__ = ["SpotInstanceManager", "CostTracker"]
