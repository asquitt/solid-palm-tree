"""
Rate Limiting

This module provides rate limiting functionality to prevent API abuse
and ensure fair resource usage across users.
"""

import time
import logging
from typing import Dict, Optional, Callable
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum


logger = logging.getLogger(__name__)


class RateLimitStrategy(Enum):
    """Rate limiting strategies."""
    FIXED_WINDOW = "fixed_window"  # Simple fixed time window
    SLIDING_WINDOW = "sliding_window"  # Sliding time window (more accurate)
    TOKEN_BUCKET = "token_bucket"  # Token bucket algorithm
    LEAKY_BUCKET = "leaky_bucket"  # Leaky bucket algorithm


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    max_requests: int  # Maximum requests
    window_seconds: int  # Time window in seconds
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW


class RateLimiter:
    """
    Rate limiter with multiple strategies.

    Tracks request rates per client and enforces limits.
    """

    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: int = 60,
        strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW
    ):
        """
        Args:
            max_requests: Maximum requests allowed
            window_seconds: Time window in seconds
            strategy: Rate limiting strategy
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.strategy = strategy

        # Storage for tracking requests
        self._fixed_window_counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self._sliding_window_timestamps: Dict[str, deque] = defaultdict(deque)
        self._token_buckets: Dict[str, Dict] = {}
        self._leaky_buckets: Dict[str, Dict] = {}

    def check_rate_limit(self, client_id: str) -> tuple:
        """
        Check if client has exceeded rate limit.

        Args:
            client_id: Unique client identifier

        Returns:
            Tuple of (allowed: bool, retry_after: int)
        """
        if self.strategy == RateLimitStrategy.FIXED_WINDOW:
            return self._check_fixed_window(client_id)

        elif self.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return self._check_sliding_window(client_id)

        elif self.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return self._check_token_bucket(client_id)

        elif self.strategy == RateLimitStrategy.LEAKY_BUCKET:
            return self._check_leaky_bucket(client_id)

        return True, 0

    def _check_fixed_window(self, client_id: str) -> tuple:
        """Fixed window rate limiting."""
        now = time.time()
        window = int(now // self.window_seconds)

        # Get count for current window
        count = self._fixed_window_counts[client_id][window]

        if count >= self.max_requests:
            # Calculate retry after
            window_end = (window + 1) * self.window_seconds
            retry_after = int(window_end - now)
            return False, retry_after

        # Increment count
        self._fixed_window_counts[client_id][window] += 1

        # Clean old windows
        old_windows = [w for w in self._fixed_window_counts[client_id].keys() if w < window - 1]
        for w in old_windows:
            del self._fixed_window_counts[client_id][w]

        return True, 0

    def _check_sliding_window(self, client_id: str) -> tuple:
        """Sliding window rate limiting (more accurate)."""
        now = time.time()
        window_start = now - self.window_seconds

        # Remove expired timestamps
        timestamps = self._sliding_window_timestamps[client_id]
        while timestamps and timestamps[0] < window_start:
            timestamps.popleft()

        # Check limit
        if len(timestamps) >= self.max_requests:
            # Calculate retry after (when oldest request expires)
            retry_after = int(timestamps[0] + self.window_seconds - now) + 1
            return False, retry_after

        # Add current timestamp
        timestamps.append(now)

        return True, 0

    def _check_token_bucket(self, client_id: str) -> tuple:
        """Token bucket algorithm."""
        now = time.time()

        if client_id not in self._token_buckets:
            # Initialize bucket
            self._token_buckets[client_id] = {
                "tokens": self.max_requests,
                "last_update": now
            }

        bucket = self._token_buckets[client_id]

        # Calculate new tokens based on time passed
        time_passed = now - bucket["last_update"]
        refill_rate = self.max_requests / self.window_seconds  # tokens per second
        new_tokens = time_passed * refill_rate

        # Update tokens (capped at max)
        bucket["tokens"] = min(self.max_requests, bucket["tokens"] + new_tokens)
        bucket["last_update"] = now

        # Check if we have a token available
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            return True, 0
        else:
            # Calculate how long until next token
            tokens_needed = 1 - bucket["tokens"]
            retry_after = int(tokens_needed / refill_rate) + 1
            return False, retry_after

    def _check_leaky_bucket(self, client_id: str) -> tuple:
        """Leaky bucket algorithm."""
        now = time.time()

        if client_id not in self._leaky_buckets:
            # Initialize bucket
            self._leaky_buckets[client_id] = {
                "level": 0,
                "last_leak": now
            }

        bucket = self._leaky_buckets[client_id]

        # Calculate leak (requests drain from bucket over time)
        time_passed = now - bucket["last_leak"]
        leak_rate = self.max_requests / self.window_seconds  # leaks per second
        leaked = time_passed * leak_rate

        # Update bucket level
        bucket["level"] = max(0, bucket["level"] - leaked)
        bucket["last_leak"] = now

        # Check if bucket is full
        if bucket["level"] >= self.max_requests:
            # Calculate retry after
            overflow = bucket["level"] - self.max_requests + 1
            retry_after = int(overflow / leak_rate) + 1
            return False, retry_after

        # Add request to bucket
        bucket["level"] += 1

        return True, 0

    def reset_client(self, client_id: str):
        """Reset rate limit for a client."""
        if client_id in self._fixed_window_counts:
            del self._fixed_window_counts[client_id]
        if client_id in self._sliding_window_timestamps:
            del self._sliding_window_timestamps[client_id]
        if client_id in self._token_buckets:
            del self._token_buckets[client_id]
        if client_id in self._leaky_buckets:
            del self._leaky_buckets[client_id]

        logger.info(f"Reset rate limit for client {client_id}")

    def get_stats(self, client_id: str) -> Dict[str, any]:
        """
        Get rate limit statistics for a client.

        Returns:
            Dict with current usage stats
        """
        now = time.time()

        if self.strategy == RateLimitStrategy.SLIDING_WINDOW:
            timestamps = self._sliding_window_timestamps[client_id]
            window_start = now - self.window_seconds

            # Count requests in current window
            current_count = sum(1 for ts in timestamps if ts >= window_start)

            return {
                "requests_used": current_count,
                "requests_remaining": max(0, self.max_requests - current_count),
                "requests_limit": self.max_requests,
                "window_seconds": self.window_seconds,
                "reset_at": int(timestamps[0] + self.window_seconds) if timestamps else int(now + self.window_seconds)
            }

        elif self.strategy == RateLimitStrategy.TOKEN_BUCKET:
            if client_id in self._token_buckets:
                bucket = self._token_buckets[client_id]
                return {
                    "tokens_available": int(bucket["tokens"]),
                    "tokens_max": self.max_requests,
                    "refill_rate": self.max_requests / self.window_seconds
                }

        return {
            "requests_limit": self.max_requests,
            "window_seconds": self.window_seconds
        }


class RateLimitMiddleware:
    """
    Middleware for rate limiting FastAPI endpoints.

    Usage:
        >>> app = FastAPI()
        >>> app.add_middleware(RateLimitMiddleware, max_requests=100, window_seconds=60)
    """

    def __init__(
        self,
        app,
        max_requests: int = 100,
        window_seconds: int = 60,
        get_client_id: Optional[Callable] = None
    ):
        self.app = app
        self.rate_limiter = RateLimiter(max_requests, window_seconds)
        self.get_client_id = get_client_id or self._default_get_client_id

    def _default_get_client_id(self, request) -> str:
        """Default client ID extraction (use IP address)."""
        return request.client.host if request.client else "unknown"

    async def __call__(self, request, call_next):
        """Process request with rate limiting."""
        client_id = self.get_client_id(request)

        # Check rate limit
        allowed, retry_after = self.rate_limiter.check_rate_limit(client_id)

        if not allowed:
            # Return 429 Too Many Requests
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "retry_after": retry_after
                },
                headers={"Retry-After": str(retry_after)}
            )

        # Add rate limit headers
        stats = self.rate_limiter.get_stats(client_id)

        response = await call_next(request)

        # Add rate limit info to response headers
        response.headers["X-RateLimit-Limit"] = str(stats.get("requests_limit", ""))
        response.headers["X-RateLimit-Remaining"] = str(stats.get("requests_remaining", ""))
        if "reset_at" in stats:
            response.headers["X-RateLimit-Reset"] = str(stats["reset_at"])

        return response


def rate_limit(
    max_requests: int = 100,
    window_seconds: int = 60,
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW
):
    """
    Decorator for rate limiting functions.

    Args:
        max_requests: Maximum requests allowed
        window_seconds: Time window in seconds
        strategy: Rate limiting strategy

    Example:
        >>> @rate_limit(max_requests=10, window_seconds=60)
        ... def expensive_operation(user_id):
        ...     pass
    """
    limiter = RateLimiter(max_requests, window_seconds, strategy)

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Use first argument as client_id (or extract from kwargs)
            client_id = args[0] if args else kwargs.get("client_id", "default")

            allowed, retry_after = limiter.check_rate_limit(str(client_id))

            if not allowed:
                raise Exception(f"Rate limit exceeded. Retry after {retry_after} seconds")

            return func(*args, **kwargs)

        return wrapper

    return decorator
