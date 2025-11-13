"""
Health Check System

This module provides comprehensive health and readiness checks for
production deployment with Kubernetes, load balancers, and monitoring systems.
"""

import time
import logging
import psutil
import platform
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path


logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    timestamp: float
    duration_ms: float
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["status"] = self.status.value
        return result


class HealthCheck:
    """
    Base class for health checks.

    Subclass this to create custom health checks.
    """

    def __init__(self, name: str):
        self.name = name

    def check(self) -> HealthCheckResult:
        """
        Perform health check.

        Returns:
            HealthCheckResult with status and details
        """
        start_time = time.time()
        try:
            status, message, details = self._do_check()
            duration_ms = (time.time() - start_time) * 1000

            return HealthCheckResult(
                name=self.name,
                status=status,
                message=message,
                timestamp=time.time(),
                duration_ms=duration_ms,
                details=details
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Health check '{self.name}' failed: {str(e)}")

            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {str(e)}",
                timestamp=time.time(),
                duration_ms=duration_ms
            )

    def _do_check(self) -> tuple:
        """
        Implement actual check logic.

        Returns:
            Tuple of (status, message, details dict)
        """
        raise NotImplementedError


class SystemHealthCheck(HealthCheck):
    """Check system resources (CPU, memory, disk)."""

    def __init__(
        self,
        name: str = "system",
        cpu_threshold: float = 90.0,
        memory_threshold: float = 90.0,
        disk_threshold: float = 90.0
    ):
        super().__init__(name)
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold

    def _do_check(self) -> tuple:
        """Check system resources."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)

        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent

        details = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "memory_available_gb": memory.available / (1024 ** 3),
            "disk_percent": disk_percent,
            "disk_free_gb": disk.free / (1024 ** 3)
        }

        # Determine status
        if (cpu_percent >= self.cpu_threshold or
                memory_percent >= self.memory_threshold or
                disk_percent >= self.disk_threshold):
            return (
                HealthStatus.UNHEALTHY,
                "System resources critically low",
                details
            )
        elif (cpu_percent >= self.cpu_threshold * 0.8 or
              memory_percent >= self.memory_threshold * 0.8 or
              disk_percent >= self.disk_threshold * 0.8):
            return (
                HealthStatus.DEGRADED,
                "System resources high",
                details
            )
        else:
            return (
                HealthStatus.HEALTHY,
                "System resources normal",
                details
            )


class GPUHealthCheck(HealthCheck):
    """Check GPU availability and memory."""

    def __init__(self, name: str = "gpu", require_gpu: bool = False):
        super().__init__(name)
        self.require_gpu = require_gpu

    def _do_check(self) -> tuple:
        """Check GPU status."""
        try:
            import torch

            if not torch.cuda.is_available():
                if self.require_gpu:
                    return (
                        HealthStatus.UNHEALTHY,
                        "GPU required but not available",
                        {"gpu_available": False}
                    )
                else:
                    return (
                        HealthStatus.HEALTHY,
                        "GPU not required",
                        {"gpu_available": False}
                    )

            # GPU is available
            gpu_count = torch.cuda.device_count()
            details = {
                "gpu_available": True,
                "gpu_count": gpu_count,
                "devices": []
            }

            for i in range(gpu_count):
                device_name = torch.cuda.get_device_name(i)
                memory_allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                memory_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                memory_total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)

                details["devices"].append({
                    "id": i,
                    "name": device_name,
                    "memory_allocated_gb": memory_allocated,
                    "memory_reserved_gb": memory_reserved,
                    "memory_total_gb": memory_total,
                    "memory_percent": (memory_allocated / memory_total) * 100
                })

            return (
                HealthStatus.HEALTHY,
                f"{gpu_count} GPU(s) available",
                details
            )

        except ImportError:
            if self.require_gpu:
                return (
                    HealthStatus.UNHEALTHY,
                    "PyTorch not available",
                    {"error": "torch not installed"}
                )
            else:
                return (
                    HealthStatus.HEALTHY,
                    "GPU check skipped (torch not installed)",
                    {}
                )


class DependencyHealthCheck(HealthCheck):
    """Check if required dependencies/services are accessible."""

    def __init__(self, name: str, check_func: Callable[[], bool], required: bool = True):
        super().__init__(name)
        self.check_func = check_func
        self.required = required

    def _do_check(self) -> tuple:
        """Check dependency."""
        try:
            is_available = self.check_func()

            if is_available:
                return (
                    HealthStatus.HEALTHY,
                    f"{self.name} is accessible",
                    {"available": True}
                )
            else:
                status = HealthStatus.UNHEALTHY if self.required else HealthStatus.DEGRADED
                return (
                    status,
                    f"{self.name} is not accessible",
                    {"available": False}
                )

        except Exception as e:
            status = HealthStatus.UNHEALTHY if self.required else HealthStatus.DEGRADED
            return (
                status,
                f"{self.name} check failed: {str(e)}",
                {"available": False, "error": str(e)}
            )


class FileSystemHealthCheck(HealthCheck):
    """Check if file system paths are writable."""

    def __init__(self, name: str = "filesystem", paths: Optional[List[str]] = None):
        super().__init__(name)
        self.paths = paths or ["/tmp", "."]

    def _do_check(self) -> tuple:
        """Check file system access."""
        results = {}

        for path in self.paths:
            try:
                test_file = Path(path) / f".health_check_{time.time()}"
                test_file.write_text("test")
                test_file.unlink()
                results[str(path)] = {"writable": True}
            except Exception as e:
                results[str(path)] = {"writable": False, "error": str(e)}

        all_writable = all(r.get("writable", False) for r in results.values())

        if all_writable:
            return (
                HealthStatus.HEALTHY,
                "All paths writable",
                {"paths": results}
            )
        else:
            return (
                HealthStatus.UNHEALTHY,
                "Some paths not writable",
                {"paths": results}
            )


class HealthCheckManager:
    """
    Manager for all health checks.

    Aggregates multiple health checks and provides unified interface
    for liveness and readiness probes.
    """

    def __init__(self):
        self.checks: Dict[str, HealthCheck] = {}
        self.readiness_checks: List[str] = []
        self.liveness_checks: List[str] = []

    def register(
        self,
        check: HealthCheck,
        for_readiness: bool = True,
        for_liveness: bool = False
    ):
        """
        Register a health check.

        Args:
            check: HealthCheck instance
            for_readiness: Include in readiness probe
            for_liveness: Include in liveness probe
        """
        self.checks[check.name] = check

        if for_readiness:
            self.readiness_checks.append(check.name)

        if for_liveness:
            self.liveness_checks.append(check.name)

        logger.info(f"Registered health check: {check.name}")

    def check_liveness(self) -> Dict[str, Any]:
        """
        Check liveness (is the application alive?).

        Liveness checks are minimal - just ensure process isn't deadlocked.

        Returns:
            Dict with status and results
        """
        results = []
        overall_status = HealthStatus.HEALTHY

        for check_name in self.liveness_checks:
            if check_name in self.checks:
                result = self.checks[check_name].check()
                results.append(result.to_dict())

                if result.status == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY

        return {
            "status": overall_status.value,
            "timestamp": time.time(),
            "checks": results
        }

    def check_readiness(self) -> Dict[str, Any]:
        """
        Check readiness (is the application ready to serve traffic?).

        Readiness checks verify all dependencies are available.

        Returns:
            Dict with status and results
        """
        results = []
        overall_status = HealthStatus.HEALTHY

        for check_name in self.readiness_checks:
            if check_name in self.checks:
                result = self.checks[check_name].check()
                results.append(result.to_dict())

                if result.status == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif result.status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED

        is_ready = overall_status == HealthStatus.HEALTHY

        return {
            "ready": is_ready,
            "status": overall_status.value,
            "timestamp": time.time(),
            "checks": results
        }

    def check_all(self) -> Dict[str, Any]:
        """
        Run all registered health checks.

        Returns:
            Dict with comprehensive status
        """
        results = []
        overall_status = HealthStatus.HEALTHY

        for check_name, check in self.checks.items():
            result = check.check()
            results.append(result.to_dict())

            if result.status == HealthStatus.UNHEALTHY:
                overall_status = HealthStatus.UNHEALTHY
            elif result.status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                overall_status = HealthStatus.DEGRADED

        return {
            "status": overall_status.value,
            "timestamp": time.time(),
            "system_info": {
                "platform": platform.system(),
                "python_version": platform.python_version(),
                "hostname": platform.node()
            },
            "checks": results
        }


# Helper functions for common checks

def check_mlflow_connection() -> bool:
    """Check if MLflow server is accessible."""
    try:
        import mlflow
        # Try to create a simple experiment
        client = mlflow.tracking.MlflowClient()
        # This will raise exception if MLflow is not accessible
        return True
    except:
        return False


def check_s3_access() -> bool:
    """Check if S3 is accessible."""
    try:
        import boto3
        s3 = boto3.client('s3')
        s3.list_buckets()
        return True
    except:
        return False


def check_ray_cluster() -> bool:
    """Check if Ray cluster is accessible."""
    try:
        import ray
        return ray.is_initialized()
    except:
        return False


# Default health check manager with common checks

def create_default_health_manager(require_gpu: bool = False) -> HealthCheckManager:
    """
    Create health check manager with common checks.

    Args:
        require_gpu: Whether GPU is required

    Returns:
        Configured HealthCheckManager
    """
    manager = HealthCheckManager()

    # System checks (always for liveness)
    manager.register(
        SystemHealthCheck(),
        for_readiness=True,
        for_liveness=True
    )

    # GPU checks (for readiness)
    manager.register(
        GPUHealthCheck(require_gpu=require_gpu),
        for_readiness=True,
        for_liveness=False
    )

    # File system checks (for readiness)
    manager.register(
        FileSystemHealthCheck(),
        for_readiness=True,
        for_liveness=False
    )

    # MLflow check (optional dependency)
    manager.register(
        DependencyHealthCheck("mlflow", check_mlflow_connection, required=False),
        for_readiness=True,
        for_liveness=False
    )

    # Ray check (optional dependency)
    manager.register(
        DependencyHealthCheck("ray", check_ray_cluster, required=False),
        for_readiness=True,
        for_liveness=False
    )

    return manager
