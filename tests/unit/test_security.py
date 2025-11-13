"""
Unit Tests for Security Features

Tests authentication, authorization, and rate limiting.
"""

import pytest
import time
from src.security.auth import AuthManager, APIKey, User, Permission, Role
from src.security.rate_limiter import RateLimiter, RateLimitStrategy
from src.security.secrets import SecretsManager, SecretsBackend


class TestAuthManager:
    """Test authentication and authorization."""

    def test_create_api_key(self):
        """Test creating API key."""
        auth = AuthManager()
        raw_key, key_id = auth.create_api_key(
            user_id="user123",
            name="Test Key",
            permissions=[Permission.TRAINING_READ]
        )

        assert raw_key.startswith("dlp_")
        assert key_id.startswith("key_")

    def test_verify_api_key(self):
        """Test verifying API key."""
        auth = AuthManager()
        raw_key, key_id = auth.create_api_key(
            user_id="user123",
            name="Test Key"
        )

        # Verify with correct key
        verified = auth.verify_api_key(raw_key)
        assert verified is not None
        assert verified.user_id == "user123"
        assert verified.key_id == key_id

        # Verify with wrong key
        wrong_verified = auth.verify_api_key("wrong_key")
        assert wrong_verified is None

    def test_revoke_api_key(self):
        """Test revoking API key."""
        auth = AuthManager()
        raw_key, key_id = auth.create_api_key(
            user_id="user123",
            name="Test Key"
        )

        # Should work before revoke
        assert auth.verify_api_key(raw_key) is not None

        # Revoke
        auth.revoke_api_key(key_id)

        # Should not work after revoke
        assert auth.verify_api_key(raw_key) is None

    def test_list_api_keys(self):
        """Test listing API keys."""
        auth = AuthManager()

        # Create multiple keys
        auth.create_api_key("user1", "Key 1")
        auth.create_api_key("user1", "Key 2")
        auth.create_api_key("user2", "Key 3")

        # List all
        all_keys = auth.list_api_keys()
        assert len(all_keys) == 3

        # List for user
        user1_keys = auth.list_api_keys(user_id="user1")
        assert len(user1_keys) == 2

    def test_create_user(self):
        """Test creating user."""
        auth = AuthManager()
        user = auth.create_user(
            user_id="user123",
            email="user@example.com",
            name="Test User",
            roles=[Role.DEVELOPER]
        )

        assert user.user_id == "user123"
        assert user.email == "user@example.com"
        assert Role.DEVELOPER in user.roles

    def test_user_permissions(self):
        """Test user permission checking."""
        user = User(
            user_id="user123",
            email="user@example.com",
            name="Test User",
            permissions=[Permission.TRAINING_READ, Permission.TRAINING_WRITE]
        )

        assert user.has_permission(Permission.TRAINING_READ)
        assert user.has_permission(Permission.TRAINING_WRITE)
        assert not user.has_permission(Permission.ADMIN)

    def test_admin_has_all_permissions(self):
        """Test that admin role has all permissions."""
        user = User(
            user_id="admin123",
            email="admin@example.com",
            name="Admin User",
            roles=[Role.ADMIN]
        )

        # Admin should have any permission
        assert user.has_permission(Permission.TRAINING_READ)
        assert user.has_permission(Permission.MODEL_DELETE)
        assert user.has_permission("any_permission")

    def test_jwt_creation_and_verification(self):
        """Test JWT token creation and verification."""
        auth = AuthManager(secret_key="test_secret")

        payload = {"user_id": "user123", "role": "developer"}
        token = auth.create_jwt(payload, expires_in_seconds=3600)

        # Verify token
        decoded = auth.verify_jwt(token)
        assert decoded is not None
        assert decoded["user_id"] == "user123"
        assert decoded["role"] == "developer"
        assert "exp" in decoded
        assert "iat" in decoded

    def test_jwt_expiration(self):
        """Test that expired JWT tokens are rejected."""
        auth = AuthManager(secret_key="test_secret")

        payload = {"user_id": "user123"}
        token = auth.create_jwt(payload, expires_in_seconds=1)

        # Should work immediately
        assert auth.verify_jwt(token) is not None

        # Wait for expiration
        time.sleep(1.1)

        # Should not work after expiration
        assert auth.verify_jwt(token) is None

    def test_jwt_invalid_signature(self):
        """Test that tokens with invalid signature are rejected."""
        auth1 = AuthManager(secret_key="secret1")
        auth2 = AuthManager(secret_key="secret2")

        # Create token with one secret
        payload = {"user_id": "user123"}
        token = auth1.create_jwt(payload)

        # Try to verify with different secret
        assert auth2.verify_jwt(token) is None


class TestRateLimiter:
    """Test rate limiting."""

    def test_sliding_window_allows_requests(self):
        """Test that sliding window allows requests within limit."""
        limiter = RateLimiter(
            max_requests=10,
            window_seconds=60,
            strategy=RateLimitStrategy.SLIDING_WINDOW
        )

        # Make 10 requests
        for i in range(10):
            allowed, retry_after = limiter.check_rate_limit("client1")
            assert allowed is True
            assert retry_after == 0

    def test_sliding_window_blocks_excess(self):
        """Test that sliding window blocks requests over limit."""
        limiter = RateLimiter(
            max_requests=5,
            window_seconds=60,
            strategy=RateLimitStrategy.SLIDING_WINDOW
        )

        # Make 5 requests (should all succeed)
        for i in range(5):
            allowed, retry_after = limiter.check_rate_limit("client1")
            assert allowed is True

        # 6th request should be blocked
        allowed, retry_after = limiter.check_rate_limit("client1")
        assert allowed is False
        assert retry_after > 0

    def test_multiple_clients(self):
        """Test that different clients have separate limits."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)

        # Client 1 makes 5 requests
        for i in range(5):
            allowed, _ = limiter.check_rate_limit("client1")
            assert allowed is True

        # Client 1 is blocked
        allowed, _ = limiter.check_rate_limit("client1")
        assert allowed is False

        # But client 2 should still work
        allowed, _ = limiter.check_rate_limit("client2")
        assert allowed is True

    def test_rate_limit_reset(self):
        """Test resetting rate limit for a client."""
        limiter = RateLimiter(max_requests=3, window_seconds=60)

        # Use up limit
        for i in range(3):
            limiter.check_rate_limit("client1")

        # Should be blocked
        allowed, _ = limiter.check_rate_limit("client1")
        assert allowed is False

        # Reset
        limiter.reset_client("client1")

        # Should work again
        allowed, _ = limiter.check_rate_limit("client1")
        assert allowed is True

    def test_token_bucket_strategy(self):
        """Test token bucket rate limiting."""
        limiter = RateLimiter(
            max_requests=10,
            window_seconds=10,
            strategy=RateLimitStrategy.TOKEN_BUCKET
        )

        # Initial burst should work
        for i in range(10):
            allowed, _ = limiter.check_rate_limit("client1")
            assert allowed is True

        # Next request should fail (tokens depleted)
        allowed, retry_after = limiter.check_rate_limit("client1")
        assert allowed is False
        assert retry_after > 0

    def test_get_stats(self):
        """Test getting rate limit statistics."""
        limiter = RateLimiter(max_requests=10, window_seconds=60)

        # Make some requests
        for i in range(3):
            limiter.check_rate_limit("client1")

        stats = limiter.get_stats("client1")
        assert "requests_used" in stats or "tokens_available" in stats
        assert "requests_limit" in stats or "tokens_max" in stats


class TestSecretsManager:
    """Test secrets management."""

    def test_env_backend(self):
        """Test environment variable backend."""
        import os

        # Set env var
        os.environ["TEST_SECRET"] = "secret_value"

        manager = SecretsManager(backend=SecretsBackend.ENV)
        value = manager.get_secret("TEST_SECRET")

        assert value == "secret_value"

        # Clean up
        del os.environ["TEST_SECRET"]

    def test_get_secret_with_default(self):
        """Test getting secret with default value."""
        manager = SecretsManager(backend=SecretsBackend.ENV)
        value = manager.get_secret("NONEXISTENT_SECRET", default="default_value")

        assert value == "default_value"

    def test_file_backend(self):
        """Test file-based secrets backend."""
        import tempfile
        import os

        # Create temp secrets file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            secrets_file = f.name

        try:
            manager = SecretsManager(
                backend=SecretsBackend.FILE,
                config={"secrets_file": secrets_file}
            )

            # Set secret
            manager.set_secret("test_key", "test_value")

            # Get secret
            value = manager.get_secret("test_key")
            assert value == "test_value"

            # Create new manager to test persistence
            manager2 = SecretsManager(
                backend=SecretsBackend.FILE,
                config={"secrets_file": secrets_file}
            )
            value2 = manager2.get_secret("test_key")
            assert value2 == "test_value"

        finally:
            os.unlink(secrets_file)

    def test_secret_caching(self):
        """Test that secrets are cached."""
        import os

        os.environ["CACHED_SECRET"] = "original_value"

        manager = SecretsManager(backend=SecretsBackend.ENV)

        # First access
        value1 = manager.get_secret("CACHED_SECRET")
        assert value1 == "original_value"

        # Change env var
        os.environ["CACHED_SECRET"] = "new_value"

        # Should still return cached value
        value2 = manager.get_secret("CACHED_SECRET")
        assert value2 == "original_value"  # From cache

        # Clean up
        del os.environ["CACHED_SECRET"]
