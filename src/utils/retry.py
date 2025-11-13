"""
Retry Logic Utilities

This module provides robust retry mechanisms with exponential backoff,
circuit breaker pattern, and configurable retry strategies for handling
transient failures in distributed systems and cloud operations.
"""

import time
import logging
import functools
from typing import Callable, Optional, Tuple, Type, Union, Any
from enum import Enum
import random


logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    """Retry strategies."""
    FIXED = "fixed"  # Fixed delay between retries
    EXPONENTIAL = "exponential"  # Exponential backoff
    LINEAR = "linear"  # Linear backoff
    RANDOM = "random"  # Random jitter


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_multiplier: float = 2.0,
    max_delay: float = 60.0,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable] = None,
    on_failure: Optional[Callable] = None,
):
    """
    Decorator for retrying functions with configurable backoff strategies.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff_multiplier: Multiplier for exponential backoff
        max_delay: Maximum delay between retries (seconds)
        strategy: Retry strategy to use
        exceptions: Tuple of exceptions to catch and retry
        on_retry: Callback function called on each retry (receives exception, attempt number)
        on_failure: Callback function called when all retries fail (receives exception)

    Returns:
        Decorated function with retry logic

    Example:
        >>> @retry(max_attempts=3, delay=1.0, strategy=RetryStrategy.EXPONENTIAL)
        ... def upload_to_cloud(file_path):
        ...     # Your upload logic here
        ...     pass
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)

                except exceptions as e:
                    last_exception = e

                    if attempt == max_attempts:
                        # Final attempt failed
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {str(e)}"
                        )
                        if on_failure:
                            on_failure(e)
                        raise

                    # Calculate delay based on strategy
                    retry_delay = _calculate_delay(
                        strategy=strategy,
                        attempt=attempt,
                        base_delay=delay,
                        multiplier=backoff_multiplier,
                        max_delay=max_delay
                    )

                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt}/{max_attempts}). "
                        f"Retrying in {retry_delay:.2f}s... Error: {str(e)}"
                    )

                    if on_retry:
                        on_retry(e, attempt)

                    time.sleep(retry_delay)

            # Should never reach here, but just in case
            if last_exception:
                raise last_exception

        return wrapper

    return decorator


def _calculate_delay(
    strategy: RetryStrategy,
    attempt: int,
    base_delay: float,
    multiplier: float,
    max_delay: float
) -> float:
    """Calculate delay based on retry strategy."""

    if strategy == RetryStrategy.FIXED:
        delay = base_delay

    elif strategy == RetryStrategy.EXPONENTIAL:
        # Exponential: delay * (multiplier ^ attempt)
        delay = base_delay * (multiplier ** (attempt - 1))

    elif strategy == RetryStrategy.LINEAR:
        # Linear: delay * attempt
        delay = base_delay * attempt

    elif strategy == RetryStrategy.RANDOM:
        # Random between 0 and base_delay
        delay = random.uniform(0, base_delay)

    else:
        delay = base_delay

    # Cap at max_delay
    return min(delay, max_delay)


class RetryContext:
    """
    Context manager for retry logic.

    Provides more control over retry behavior within a code block.

    Example:
        >>> with RetryContext(max_attempts=3, delay=1.0) as retry_ctx:
        ...     while retry_ctx.should_retry():
        ...         try:
        ...             result = api_call()
        ...             break
        ...         except Exception as e:
        ...             retry_ctx.record_failure(e)
    """

    def __init__(
        self,
        max_attempts: int = 3,
        delay: float = 1.0,
        backoff_multiplier: float = 2.0,
        max_delay: float = 60.0,
        strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
        exceptions: Tuple[Type[Exception], ...] = (Exception,),
    ):
        self.max_attempts = max_attempts
        self.delay = delay
        self.backoff_multiplier = backoff_multiplier
        self.max_delay = max_delay
        self.strategy = strategy
        self.exceptions = exceptions

        self.attempt = 0
        self.last_exception = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type and exc_val:
            if isinstance(exc_val, self.exceptions):
                self.record_failure(exc_val)
        return False  # Don't suppress exceptions

    def should_retry(self) -> bool:
        """Check if should attempt another retry."""
        if self.attempt > 0 and self.last_exception:
            if self.attempt >= self.max_attempts:
                logger.error(f"Max retry attempts ({self.max_attempts}) reached")
                raise self.last_exception

            # Wait before retry
            retry_delay = _calculate_delay(
                strategy=self.strategy,
                attempt=self.attempt,
                base_delay=self.delay,
                multiplier=self.backoff_multiplier,
                max_delay=self.max_delay
            )
            logger.info(f"Retrying in {retry_delay:.2f}s (attempt {self.attempt + 1}/{self.max_attempts})")
            time.sleep(retry_delay)

        self.attempt += 1
        return self.attempt <= self.max_attempts

    def record_failure(self, exception: Exception):
        """Record a failed attempt."""
        self.last_exception = exception
        logger.warning(f"Attempt {self.attempt} failed: {str(exception)}")


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.

    Prevents cascading failures by failing fast when a service is unavailable.
    Opens circuit after threshold failures, tries again after timeout.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Service failing, requests fail immediately
    - HALF_OPEN: Testing if service recovered

    Example:
        >>> breaker = CircuitBreaker(failure_threshold=5, timeout=60)
        >>>
        >>> @breaker.protected
        ... def call_external_api():
        ...     return api.fetch_data()
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        """
        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Seconds to wait before trying again (OPEN -> HALF_OPEN)
            expected_exception: Exception type that triggers circuit breaker
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Call function through circuit breaker.

        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            Exception if circuit is open or function fails
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                raise Exception(f"Circuit breaker is OPEN. Try again after {self.timeout}s")

        try:
            result = func(*args, **kwargs)

            # Success - reset failure count
            if self.state == CircuitState.HALF_OPEN:
                self._reset()
                logger.info("Circuit breaker reset to CLOSED state")

            self.failure_count = 0
            return result

        except self.expected_exception as e:
            self._record_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return (time.time() - self.last_failure_time) >= self.timeout

    def _record_failure(self):
        """Record a failure and potentially open circuit."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.error(
                f"Circuit breaker opened after {self.failure_count} failures. "
                f"Will retry after {self.timeout}s"
            )

    def _reset(self):
        """Reset circuit breaker to closed state."""
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def protected(self, func: Callable) -> Callable:
        """
        Decorator to protect function with circuit breaker.

        Example:
            >>> breaker = CircuitBreaker(failure_threshold=3)
            >>> @breaker.protected
            ... def risky_operation():
            ...     pass
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper


# Convenience functions for common cloud operations

@retry(
    max_attempts=4,
    delay=2.0,
    backoff_multiplier=2.0,
    strategy=RetryStrategy.EXPONENTIAL,
    exceptions=(ConnectionError, TimeoutError, OSError)
)
def upload_with_retry(upload_func: Callable, *args, **kwargs):
    """
    Upload file with automatic retry on network errors.

    Args:
        upload_func: Upload function to call
        *args: Positional arguments for upload function
        **kwargs: Keyword arguments for upload function

    Example:
        >>> upload_with_retry(s3_client.upload_file, "local.txt", "bucket", "remote.txt")
    """
    return upload_func(*args, **kwargs)


@retry(
    max_attempts=4,
    delay=2.0,
    backoff_multiplier=2.0,
    strategy=RetryStrategy.EXPONENTIAL,
    exceptions=(ConnectionError, TimeoutError, OSError)
)
def download_with_retry(download_func: Callable, *args, **kwargs):
    """
    Download file with automatic retry on network errors.

    Args:
        download_func: Download function to call
        *args: Positional arguments for download function
        **kwargs: Keyword arguments for download function

    Example:
        >>> download_with_retry(s3_client.download_file, "bucket", "remote.txt", "local.txt")
    """
    return download_func(*args, **kwargs)


def retry_on_rate_limit(
    max_attempts: int = 5,
    initial_delay: float = 1.0,
    max_delay: float = 120.0
):
    """
    Retry decorator specifically for API rate limiting.

    Uses exponential backoff with jitter to handle rate limits gracefully.

    Example:
        >>> @retry_on_rate_limit(max_attempts=5)
        ... def call_api():
        ...     return api.get("/endpoint")
    """
    return retry(
        max_attempts=max_attempts,
        delay=initial_delay,
        backoff_multiplier=2.0,
        max_delay=max_delay,
        strategy=RetryStrategy.EXPONENTIAL,
        exceptions=(Exception,),  # Catch all, but check for rate limit in function
    )
