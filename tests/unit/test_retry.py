"""
Unit Tests for Retry Logic

Tests retry decorators, circuit breakers, and backoff strategies.
"""

import pytest
import time
from src.utils.retry import (
    retry,
    RetryStrategy,
    RetryContext,
    CircuitBreaker,
    upload_with_retry,
    download_with_retry
)


class TestRetryDecorator:
    """Test retry decorator functionality."""

    def test_successful_function_no_retry(self):
        """Test that successful function doesn't retry."""
        call_count = {"count": 0}

        @retry(max_attempts=3)
        def success_func():
            call_count["count"] += 1
            return "success"

        result = success_func()

        assert result == "success"
        assert call_count["count"] == 1  # Called only once

    def test_retry_on_exception(self):
        """Test retrying on exception."""
        call_count = {"count": 0}

        @retry(max_attempts=3, delay=0.01)  # Small delay for fast tests
        def failing_func():
            call_count["count"] += 1
            if call_count["count"] < 3:
                raise ValueError("Not yet")
            return "success"

        result = failing_func()

        assert result == "success"
        assert call_count["count"] == 3  # Retried twice, succeeded on 3rd

    def test_max_attempts_reached(self):
        """Test that exception is raised after max attempts."""
        call_count = {"count": 0}

        @retry(max_attempts=3, delay=0.01)
        def always_fails():
            call_count["count"] += 1
            raise ValueError("Always fails")

        with pytest.raises(ValueError, match="Always fails"):
            always_fails()

        assert call_count["count"] == 3  # Tried 3 times

    def test_exponential_backoff(self):
        """Test exponential backoff strategy."""
        times = []

        @retry(
            max_attempts=4,
            delay=0.1,
            backoff_multiplier=2.0,
            strategy=RetryStrategy.EXPONENTIAL
        )
        def failing_func():
            times.append(time.time())
            if len(times) < 4:
                raise ValueError("Retry")
            return "success"

        failing_func()

        # Check delays are increasing exponentially
        # Delays should be approximately: 0.1, 0.2, 0.4
        if len(times) >= 3:
            delay1 = times[1] - times[0]
            delay2 = times[2] - times[1]
            assert delay2 > delay1  # Later delay is longer

    def test_specific_exceptions_only(self):
        """Test retrying only on specific exceptions."""
        call_count = {"count": 0}

        @retry(
            max_attempts=3,
            delay=0.01,
            exceptions=(ValueError,)  # Only retry on ValueError
        )
        def func():
            call_count["count"] += 1
            if call_count["count"] == 1:
                raise ValueError("Retry this")
            else:
                raise TypeError("Don't retry this")

        with pytest.raises(TypeError):
            func()

        assert call_count["count"] == 2  # ValueError retried once, TypeError raised

    def test_on_retry_callback(self):
        """Test on_retry callback is called."""
        retry_info = []

        def on_retry_callback(exception, attempt):
            retry_info.append({"exception": exception, "attempt": attempt})

        @retry(max_attempts=3, delay=0.01, on_retry=on_retry_callback)
        def failing_func():
            if len(retry_info) < 2:
                raise ValueError("Retry")
            return "success"

        failing_func()

        assert len(retry_info) == 2  # Called on each retry
        assert retry_info[0]["attempt"] == 1
        assert retry_info[1]["attempt"] == 2

    def test_on_failure_callback(self):
        """Test on_failure callback is called."""
        failure_info = []

        def on_failure_callback(exception):
            failure_info.append(exception)

        @retry(max_attempts=2, delay=0.01, on_failure=on_failure_callback)
        def always_fails():
            raise ValueError("Always fails")

        with pytest.raises(ValueError):
            always_fails()

        assert len(failure_info) == 1  # Called once on final failure


class TestRetryContext:
    """Test RetryContext context manager."""

    def test_retry_context_success(self):
        """Test successful operation in retry context."""
        with RetryContext(max_attempts=3, delay=0.01) as ctx:
            attempts = 0
            while ctx.should_retry():
                attempts += 1
                try:
                    result = "success"
                    break
                except Exception as e:
                    ctx.record_failure(e)

        assert attempts == 1
        assert result == "success"

    def test_retry_context_with_failures(self):
        """Test retry context with failures."""
        with RetryContext(max_attempts=3, delay=0.01) as ctx:
            attempts = 0
            while ctx.should_retry():
                attempts += 1
                try:
                    if attempts < 3:
                        raise ValueError("Not yet")
                    result = "success"
                    break
                except Exception as e:
                    ctx.record_failure(e)

        assert attempts == 3
        assert result == "success"

    def test_retry_context_max_attempts(self):
        """Test retry context raises after max attempts."""
        with pytest.raises(ValueError):
            with RetryContext(max_attempts=3, delay=0.01) as ctx:
                while ctx.should_retry():
                    ctx.record_failure(ValueError("Always fails"))
                    raise ValueError("Always fails")


class TestCircuitBreaker:
    """Test circuit breaker pattern."""

    def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed (normal) state."""
        breaker = CircuitBreaker(failure_threshold=3, timeout=1.0)

        # Successful calls should work
        result = breaker.call(lambda: "success")
        assert result == "success"

    def test_circuit_breaker_opens_after_failures(self):
        """Test circuit breaker opens after threshold failures."""
        breaker = CircuitBreaker(failure_threshold=3, timeout=1.0)

        # Cause 3 failures
        for i in range(3):
            try:
                breaker.call(lambda: 1 / 0)  # Raises ZeroDivisionError
            except ZeroDivisionError:
                pass

        # Circuit should be open now
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            breaker.call(lambda: "should fail")

    def test_circuit_breaker_half_open_state(self):
        """Test circuit breaker transitions to half-open after timeout."""
        breaker = CircuitBreaker(failure_threshold=2, timeout=0.1)

        # Cause failures to open circuit
        for i in range(2):
            try:
                breaker.call(lambda: 1 / 0)
            except ZeroDivisionError:
                pass

        # Wait for timeout
        time.sleep(0.15)

        # Next call should be allowed (half-open state)
        # Successful call should close circuit
        result = breaker.call(lambda: "success")
        assert result == "success"

        # Circuit should be closed now
        result = breaker.call(lambda: "success again")
        assert result == "success again"

    def test_circuit_breaker_decorator(self):
        """Test circuit breaker as decorator."""
        breaker = CircuitBreaker(failure_threshold=2, timeout=0.1)

        call_count = {"count": 0}

        @breaker.protected
        def risky_operation():
            call_count["count"] += 1
            if call_count["count"] <= 2:
                raise ValueError("Failure")
            return "success"

        # First two calls fail
        for i in range(2):
            try:
                risky_operation()
            except ValueError:
                pass

        # Circuit is open, next call should fail immediately
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            risky_operation()

        # Call count should still be 2 (third call blocked by circuit)
        assert call_count["count"] == 2


class TestConvenienceFunctions:
    """Test convenience retry functions."""

    def test_upload_with_retry(self):
        """Test upload with retry function."""
        call_count = {"count": 0}

        def mock_upload(src, bucket, key):
            call_count["count"] += 1
            if call_count["count"] < 2:
                raise ConnectionError("Network error")
            return {"success": True}

        result = upload_with_retry(mock_upload, "file.txt", "bucket", "key")

        assert result["success"] is True
        assert call_count["count"] == 2  # Failed once, succeeded on retry

    def test_download_with_retry(self):
        """Test download with retry function."""
        call_count = {"count": 0}

        def mock_download(bucket, key, dest):
            call_count["count"] += 1
            if call_count["count"] < 2:
                raise TimeoutError("Timeout")
            return {"success": True}

        result = download_with_retry(mock_download, "bucket", "key", "file.txt")

        assert result["success"] is True
        assert call_count["count"] == 2
