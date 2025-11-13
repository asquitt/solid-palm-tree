"""
Spot Instance Manager

Manages spot/preemptible instances for cost-optimized training.

Spot instances can save 60-80% on compute costs but can be interrupted.
This manager handles:
- Automatic fallback to on-demand instances
- Checkpoint-based recovery from interruptions
- Spot price monitoring and bidding strategies
"""

import logging
from typing import Dict, Any, Optional
import time

logger = logging.getLogger(__name__)


class SpotInstanceManager:
    """
    Manager for spot/preemptible instance lifecycle.

    Spot Instance Concepts:
    - Spot instances are spare cloud capacity offered at discounts
    - Can be interrupted with 2-minute warning (AWS) or 30s (GCP)
    - Save 60-80% vs on-demand pricing
    - Best for fault-tolerant workloads with checkpointing

    Strategies:
    1. Checkpointing: Save frequently to recover from interruptions
    2. Fallback: Switch to on-demand if spots unavailable
    3. Diversification: Use multiple instance types
    4. Monitoring: Track spot prices and interruption rates
    """

    def __init__(
        self,
        provider: str = "aws",
        fallback_to_ondemand: bool = True,
        max_interruptions: int = 3,
    ):
        """
        Initialize spot instance manager.

        Args:
            provider: Cloud provider ("aws", "gcp", "azure")
            fallback_to_ondemand: Fallback to on-demand if spots unavailable
            max_interruptions: Max interruptions before switching to on-demand
        """
        self.provider = provider.lower()
        self.fallback_to_ondemand = fallback_to_ondemand
        self.max_interruptions = max_interruptions
        self.interruption_count = 0

        logger.info(f"SpotInstanceManager initialized for {provider}")

    def monitor_spot_interruption(self) -> bool:
        """
        Check if spot instance interruption is imminent.

        Cloud providers give warnings before termination:
        - AWS: 2-minute warning via metadata service
        - GCP: 30-second warning via metadata
        - Azure: 30-second warning via scheduled events API

        Returns:
            True if interruption detected
        """
        if self.provider == "aws":
            return self._check_aws_interruption()
        elif self.provider == "gcp":
            return self._check_gcp_preemption()
        elif self.provider == "azure":
            return self._check_azure_eviction()
        return False

    def _check_aws_interruption(self) -> bool:
        """Check AWS spot interruption notice."""
        try:
            import requests
            # AWS metadata service for spot interruption
            url = "http://169.254.169.254/latest/meta-data/spot/instance-action"
            response = requests.get(url, timeout=1)
            if response.status_code == 200:
                logger.warning("AWS spot interruption detected!")
                return True
        except Exception:
            pass
        return False

    def _check_gcp_preemption(self) -> bool:
        """Check GCP preemptible VM termination notice."""
        try:
            import requests
            url = "http://metadata.google.internal/computeMetadata/v1/instance/preempted"
            headers = {"Metadata-Flavor": "Google"}
            response = requests.get(url, headers=headers, timeout=1)
            if response.text == "TRUE":
                logger.warning("GCP preemption detected!")
                return True
        except Exception:
            pass
        return False

    def _check_azure_eviction(self) -> bool:
        """Check Azure spot VM eviction notice."""
        try:
            import requests
            url = "http://169.254.169.254/metadata/scheduledevents?api-version=2019-08-01"
            headers = {"Metadata": "true"}
            response = requests.get(url, headers=headers, timeout=1)
            data = response.json()
            if data.get("Events"):
                logger.warning("Azure spot eviction detected!")
                return True
        except Exception:
            pass
        return False

    def handle_interruption(self, checkpoint_callback) -> bool:
        """
        Handle spot instance interruption.

        Steps:
        1. Save checkpoint immediately
        2. Log interruption event
        3. Increment interruption counter
        4. Decide whether to request new spot or switch to on-demand

        Args:
            checkpoint_callback: Function to save checkpoint

        Returns:
            True if should continue with spots, False if should switch
        """
        logger.warning(f"Handling spot interruption #{self.interruption_count + 1}")

        # Save checkpoint immediately
        try:
            checkpoint_callback()
            logger.info("Checkpoint saved before interruption")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

        self.interruption_count += 1

        # Decide whether to continue with spots
        if self.interruption_count >= self.max_interruptions:
            if self.fallback_to_ondemand:
                logger.info("Max interruptions reached, switching to on-demand")
                return False
            else:
                logger.warning("Max interruptions reached but fallback disabled")

        return True

    def estimate_cost_savings(
        self,
        ondemand_price: float,
        spot_price: float,
        training_hours: float,
    ) -> Dict[str, float]:
        """
        Calculate cost savings from using spot instances.

        Args:
            ondemand_price: On-demand price per hour
            spot_price: Spot price per hour
            training_hours: Expected training duration

        Returns:
            Dictionary with cost analysis
        """
        ondemand_cost = ondemand_price * training_hours
        spot_cost = spot_price * training_hours

        # Add overhead for interruption recovery (typically 5-10%)
        interruption_overhead = 1.05
        spot_cost_with_overhead = spot_cost * interruption_overhead

        savings = ondemand_cost - spot_cost_with_overhead
        savings_percent = (savings / ondemand_cost) * 100

        return {
            "ondemand_cost": ondemand_cost,
            "spot_cost": spot_cost_with_overhead,
            "savings": savings,
            "savings_percent": savings_percent,
            "discount": ((ondemand_price - spot_price) / ondemand_price) * 100,
        }


class CostTracker:
    """Track and monitor training costs in real-time."""

    def __init__(self, budget_per_hour: Optional[float] = None):
        """
        Initialize cost tracker.

        Args:
            budget_per_hour: Maximum cost per hour (training stops if exceeded)
        """
        self.budget_per_hour = budget_per_hour
        self.start_time = time.time()
        self.total_cost = 0.0

        logger.info(f"CostTracker initialized with budget: ${budget_per_hour}/hour")

    def update_cost(self, cost_per_hour: float) -> bool:
        """
        Update total cost and check budget.

        Args:
            cost_per_hour: Current cost rate

        Returns:
            True if under budget, False if budget exceeded
        """
        elapsed_hours = (time.time() - self.start_time) / 3600
        self.total_cost = cost_per_hour * elapsed_hours

        if self.budget_per_hour and cost_per_hour > self.budget_per_hour:
            logger.error(
                f"Cost per hour (${cost_per_hour:.2f}) exceeds budget "
                f"(${self.budget_per_hour:.2f})"
            )
            return False

        return True

    def get_stats(self) -> Dict[str, float]:
        """Get cost statistics."""
        elapsed_hours = (time.time() - self.start_time) / 3600

        return {
            "total_cost": self.total_cost,
            "elapsed_hours": elapsed_hours,
            "budget_remaining": (self.budget_per_hour * elapsed_hours - self.total_cost)
            if self.budget_per_hour
            else None,
        }
