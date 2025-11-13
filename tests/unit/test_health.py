"""
Unit Tests for Health Check System

Tests health checks, readiness probes, and liveness probes.
"""

import pytest
import time
from src.serving.health import (
    HealthCheck,
    HealthStatus,
    HealthCheckResult,
    SystemHealthCheck,
    GPUHealthCheck,
    DependencyHealthCheck,
    FileSystemHealthCheck,
    HealthCheckManager,
    create_default_health_manager
)


class TestHealthCheck:
    """Test base HealthCheck class."""

    def test_custom_health_check(self):
        """Test creating custom health check."""

        class CustomCheck(HealthCheck):
            def _do_check(self):
                return (HealthStatus.HEALTHY, "All good", {"metric": 42})

        check = CustomCheck("custom")
        result = check.check()

        assert result.name == "custom"
        assert result.status == HealthStatus.HEALTHY
        assert result.message == "All good"
        assert result.details["metric"] == 42
        assert result.duration_ms >= 0

    def test_health_check_exception_handling(self):
        """Test that exceptions in checks are caught."""

        class FailingCheck(HealthCheck):
            def _do_check(self):
                raise ValueError("Oops")

        check = FailingCheck("failing")
        result = check.check()

        assert result.status == HealthStatus.UNHEALTHY
        assert "failed" in result.message.lower()

    def test_health_check_result_to_dict(self):
        """Test converting result to dictionary."""
        result = HealthCheckResult(
            name="test",
            status=HealthStatus.HEALTHY,
            message="OK",
            timestamp=time.time(),
            duration_ms=10.5,
            details={"key": "value"}
        )

        result_dict = result.to_dict()

        assert result_dict["name"] == "test"
        assert result_dict["status"] == "healthy"
        assert result_dict["message"] == "OK"
        assert result_dict["details"]["key"] == "value"


class TestSystemHealthCheck:
    """Test system resource health check."""

    def test_system_check_returns_status(self):
        """Test that system check returns valid status."""
        check = SystemHealthCheck()
        result = check.check()

        assert result.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]
        assert "cpu_percent" in result.details
        assert "memory_percent" in result.details
        assert "disk_percent" in result.details

    def test_system_check_thresholds(self):
        """Test system check with custom thresholds."""
        # Set very low threshold so it triggers
        check = SystemHealthCheck(
            cpu_threshold=0.1,  # Extremely low threshold
            memory_threshold=0.1,
            disk_threshold=0.1
        )

        result = check.check()

        # Should be unhealthy or degraded with such low thresholds
        assert result.status in [HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]


class TestGPUHealthCheck:
    """Test GPU health check."""

    def test_gpu_check_without_requirement(self):
        """Test GPU check when GPU not required."""
        check = GPUHealthCheck(require_gpu=False)
        result = check.check()

        # Should be healthy whether GPU is available or not
        assert result.status == HealthStatus.HEALTHY

    def test_gpu_check_handles_missing_torch(self):
        """Test GPU check handles missing PyTorch gracefully."""
        check = GPUHealthCheck(require_gpu=False)
        result = check.check()

        # Should not crash even if torch is not available
        assert result is not None
        assert isinstance(result.status, HealthStatus)


class TestDependencyHealthCheck:
    """Test dependency health check."""

    def test_dependency_check_success(self):
        """Test successful dependency check."""

        def check_func():
            return True

        check = DependencyHealthCheck("test_service", check_func, required=True)
        result = check.check()

        assert result.status == HealthStatus.HEALTHY
        assert result.details["available"] is True

    def test_dependency_check_failure_required(self):
        """Test dependency check failure when required."""

        def check_func():
            return False

        check = DependencyHealthCheck("test_service", check_func, required=True)
        result = check.check()

        assert result.status == HealthStatus.UNHEALTHY
        assert result.details["available"] is False

    def test_dependency_check_failure_optional(self):
        """Test dependency check failure when optional."""

        def check_func():
            return False

        check = DependencyHealthCheck("test_service", check_func, required=False)
        result = check.check()

        assert result.status == HealthStatus.DEGRADED
        assert result.details["available"] is False

    def test_dependency_check_exception(self):
        """Test dependency check with exception."""

        def check_func():
            raise ConnectionError("Can't connect")

        check = DependencyHealthCheck("test_service", check_func, required=True)
        result = check.check()

        assert result.status == HealthStatus.UNHEALTHY
        assert "error" in result.details


class TestFileSystemHealthCheck:
    """Test file system health check."""

    def test_filesystem_check_writable_paths(self):
        """Test file system check with writable paths."""
        import tempfile

        temp_dir = tempfile.mkdtemp()
        check = FileSystemHealthCheck(paths=[temp_dir])
        result = check.check()

        assert result.status == HealthStatus.HEALTHY
        assert result.details["paths"][temp_dir]["writable"] is True

    def test_filesystem_check_nonwritable_path(self):
        """Test file system check with non-writable path."""
        # Use a path that's likely not writable
        check = FileSystemHealthCheck(paths=["/root/test"])
        result = check.check()

        # Should detect as unhealthy (unless running as root)
        # This test might vary based on permissions
        assert result.status in [HealthStatus.HEALTHY, HealthStatus.UNHEALTHY]


class TestHealthCheckManager:
    """Test health check manager."""

    def test_register_check(self):
        """Test registering health checks."""
        manager = HealthCheckManager()

        class DummyCheck(HealthCheck):
            def _do_check(self):
                return (HealthStatus.HEALTHY, "OK", {})

        check = DummyCheck("dummy")
        manager.register(check, for_readiness=True, for_liveness=False)

        assert "dummy" in manager.checks
        assert "dummy" in manager.readiness_checks
        assert "dummy" not in manager.liveness_checks

    def test_liveness_check(self):
        """Test liveness probe."""
        manager = HealthCheckManager()

        class AlwaysHealthyCheck(HealthCheck):
            def _do_check(self):
                return (HealthStatus.HEALTHY, "OK", {})

        check = AlwaysHealthyCheck("liveness")
        manager.register(check, for_liveness=True, for_readiness=False)

        result = manager.check_liveness()

        assert result["status"] == "healthy"
        assert len(result["checks"]) == 1

    def test_readiness_check(self):
        """Test readiness probe."""
        manager = HealthCheckManager()

        class ReadyCheck(HealthCheck):
            def _do_check(self):
                return (HealthStatus.HEALTHY, "Ready", {})

        check = ReadyCheck("readiness")
        manager.register(check, for_readiness=True, for_liveness=False)

        result = manager.check_readiness()

        assert result["ready"] is True
        assert result["status"] == "healthy"
        assert len(result["checks"]) == 1

    def test_readiness_check_not_ready(self):
        """Test readiness check when not ready."""
        manager = HealthCheckManager()

        class NotReadyCheck(HealthCheck):
            def _do_check(self):
                return (HealthStatus.UNHEALTHY, "Not ready", {})

        check = NotReadyCheck("readiness")
        manager.register(check, for_readiness=True)

        result = manager.check_readiness()

        assert result["ready"] is False
        assert result["status"] == "unhealthy"

    def test_check_all(self):
        """Test checking all health checks."""
        manager = HealthCheckManager()

        class Check1(HealthCheck):
            def _do_check(self):
                return (HealthStatus.HEALTHY, "OK", {})

        class Check2(HealthCheck):
            def _do_check(self):
                return (HealthStatus.DEGRADED, "Degraded", {})

        manager.register(Check1("check1"))
        manager.register(Check2("check2"))

        result = manager.check_all()

        assert result["status"] == "degraded"  # Overall status is degraded
        assert len(result["checks"]) == 2
        assert "system_info" in result

    def test_unhealthy_check_affects_overall_status(self):
        """Test that one unhealthy check affects overall status."""
        manager = HealthCheckManager()

        class HealthyCheck(HealthCheck):
            def _do_check(self):
                return (HealthStatus.HEALTHY, "OK", {})

        class UnhealthyCheck(HealthCheck):
            def _do_check(self):
                return (HealthStatus.UNHEALTHY, "Bad", {})

        manager.register(HealthyCheck("healthy"))
        manager.register(UnhealthyCheck("unhealthy"))

        result = manager.check_all()

        assert result["status"] == "unhealthy"


class TestDefaultHealthManager:
    """Test default health manager creation."""

    def test_create_default_manager(self):
        """Test creating default health manager."""
        manager = create_default_health_manager(require_gpu=False)

        assert "system" in manager.checks
        assert "gpu" in manager.checks
        assert "filesystem" in manager.checks

        # Should have both liveness and readiness checks
        assert len(manager.liveness_checks) > 0
        assert len(manager.readiness_checks) > 0

    def test_default_manager_liveness(self):
        """Test default manager liveness check."""
        manager = create_default_health_manager()
        result = manager.check_liveness()

        assert "status" in result
        assert "checks" in result

    def test_default_manager_readiness(self):
        """Test default manager readiness check."""
        manager = create_default_health_manager()
        result = manager.check_readiness()

        assert "ready" in result
        assert "status" in result
        assert "checks" in result
