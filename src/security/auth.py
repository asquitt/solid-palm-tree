"""
Authentication and Authorization

This module provides API key management, JWT tokens, and role-based access control.
"""

import secrets
import hashlib
import hmac
import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json


logger = logging.getLogger(__name__)


@dataclass
class APIKey:
    """API key with metadata."""
    key_id: str
    key_hash: str  # Hashed API key for secure storage
    user_id: str
    name: str
    permissions: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_used: Optional[str] = None
    is_active: bool = True
    rate_limit: Optional[int] = None  # Requests per minute

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without sensitive data)."""
        return {
            "key_id": self.key_id,
            "user_id": self.user_id,
            "name": self.name,
            "permissions": self.permissions,
            "created_at": self.created_at,
            "last_used": self.last_used,
            "is_active": self.is_active,
            "rate_limit": self.rate_limit
        }


@dataclass
class User:
    """User with permissions."""
    user_id: str
    email: str
    name: str
    roles: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    is_active: bool = True

    def has_permission(self, permission: str) -> bool:
        """Check if user has permission."""
        return permission in self.permissions or "admin" in self.roles

    def has_role(self, role: str) -> bool:
        """Check if user has role."""
        return role in self.roles


class AuthManager:
    """
    Authentication and Authorization Manager.

    Handles API keys, JWT tokens, and permission checking.
    """

    def __init__(self, secret_key: Optional[str] = None):
        """
        Args:
            secret_key: Secret key for signing tokens
        """
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.api_keys: Dict[str, APIKey] = {}  # key_hash -> APIKey
        self.users: Dict[str, User] = {}  # user_id -> User
        self.key_id_to_hash: Dict[str, str] = {}  # key_id -> key_hash for lookup

    def create_api_key(
        self,
        user_id: str,
        name: str,
        permissions: Optional[List[str]] = None,
        rate_limit: Optional[int] = None
    ) -> tuple:
        """
        Create new API key.

        Args:
            user_id: User ID
            name: Descriptive name for the key
            permissions: List of permissions
            rate_limit: Requests per minute limit

        Returns:
            Tuple of (api_key, key_id)

        Note:
            The raw API key is returned only once and should be stored securely by the user.
        """
        # Generate API key with prefix for easy identification
        raw_key = f"dlp_{secrets.token_urlsafe(32)}"

        # Hash the key for storage
        key_hash = self._hash_key(raw_key)

        # Generate unique key ID
        key_id = f"key_{secrets.token_hex(8)}"

        # Create API key object
        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            user_id=user_id,
            name=name,
            permissions=permissions or [],
            rate_limit=rate_limit
        )

        # Store
        self.api_keys[key_hash] = api_key
        self.key_id_to_hash[key_id] = key_hash

        logger.info(f"Created API key {key_id} for user {user_id}")

        return raw_key, key_id

    def verify_api_key(self, raw_key: str) -> Optional[APIKey]:
        """
        Verify API key and return associated metadata.

        Args:
            raw_key: The raw API key string

        Returns:
            APIKey object if valid, None otherwise
        """
        if not raw_key:
            return None

        # Hash the provided key
        key_hash = self._hash_key(raw_key)

        # Look up
        api_key = self.api_keys.get(key_hash)

        if api_key and api_key.is_active:
            # Update last used
            api_key.last_used = datetime.now().isoformat()
            logger.debug(f"API key {api_key.key_id} verified for user {api_key.user_id}")
            return api_key

        logger.warning(f"Invalid or inactive API key provided")
        return None

    def revoke_api_key(self, key_id: str):
        """
        Revoke (deactivate) an API key.

        Args:
            key_id: Key ID to revoke
        """
        key_hash = self.key_id_to_hash.get(key_id)
        if key_hash and key_hash in self.api_keys:
            self.api_keys[key_hash].is_active = False
            logger.info(f"Revoked API key {key_id}")
        else:
            logger.warning(f"Attempted to revoke non-existent key {key_id}")

    def delete_api_key(self, key_id: str):
        """
        Permanently delete an API key.

        Args:
            key_id: Key ID to delete
        """
        key_hash = self.key_id_to_hash.get(key_id)
        if key_hash and key_hash in self.api_keys:
            del self.api_keys[key_hash]
            del self.key_id_to_hash[key_id]
            logger.info(f"Deleted API key {key_id}")

    def list_api_keys(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List API keys.

        Args:
            user_id: Optional user ID to filter by

        Returns:
            List of API key dictionaries (without sensitive data)
        """
        keys = self.api_keys.values()

        if user_id:
            keys = [k for k in keys if k.user_id == user_id]

        return [k.to_dict() for k in keys]

    def create_user(
        self,
        user_id: str,
        email: str,
        name: str,
        roles: Optional[List[str]] = None,
        permissions: Optional[List[str]] = None
    ) -> User:
        """
        Create a new user.

        Args:
            user_id: Unique user ID
            email: User email
            name: User name
            roles: User roles (e.g., ['admin', 'developer'])
            permissions: Direct permissions

        Returns:
            Created User object
        """
        user = User(
            user_id=user_id,
            email=email,
            name=name,
            roles=roles or [],
            permissions=permissions or []
        )

        self.users[user_id] = user
        logger.info(f"Created user {user_id}")

        return user

    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self.users.get(user_id)

    def check_permission(self, user_id: str, permission: str) -> bool:
        """
        Check if user has permission.

        Args:
            user_id: User ID
            permission: Permission to check

        Returns:
            True if user has permission
        """
        user = self.users.get(user_id)
        if not user or not user.is_active:
            return False

        return user.has_permission(permission)

    def _hash_key(self, raw_key: str) -> str:
        """Hash API key using SHA-256."""
        return hashlib.sha256(raw_key.encode()).hexdigest()

    def create_jwt(
        self,
        payload: Dict[str, Any],
        expires_in_seconds: int = 3600
    ) -> str:
        """
        Create a simple JWT-like token (simplified implementation).

        Args:
            payload: Data to encode
            expires_in_seconds: Token expiration time

        Returns:
            Signed token string

        Note:
            This is a simplified JWT implementation.
            For production, use python-jose or PyJWT library.
        """
        # Add expiration
        payload["exp"] = time.time() + expires_in_seconds
        payload["iat"] = time.time()

        # Encode payload
        import base64
        payload_json = json.dumps(payload, sort_keys=True)
        payload_encoded = base64.urlsafe_b64encode(payload_json.encode()).decode()

        # Create signature
        signature = hmac.new(
            self.secret_key.encode(),
            payload_encoded.encode(),
            hashlib.sha256
        ).hexdigest()

        # Combine
        token = f"{payload_encoded}.{signature}"
        return token

    def verify_jwt(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verify and decode JWT token.

        Args:
            token: Token to verify

        Returns:
            Decoded payload if valid, None otherwise
        """
        try:
            import base64

            # Split token
            parts = token.split(".")
            if len(parts) != 2:
                return None

            payload_encoded, signature = parts

            # Verify signature
            expected_signature = hmac.new(
                self.secret_key.encode(),
                payload_encoded.encode(),
                hashlib.sha256
            ).hexdigest()

            if not hmac.compare_digest(signature, expected_signature):
                logger.warning("Invalid token signature")
                return None

            # Decode payload
            payload_json = base64.urlsafe_b64decode(payload_encoded.encode()).decode()
            payload = json.loads(payload_json)

            # Check expiration
            if "exp" in payload:
                if time.time() > payload["exp"]:
                    logger.warning("Token expired")
                    return None

            return payload

        except Exception as e:
            logger.error(f"Error verifying token: {str(e)}")
            return None


class APIKeyAuth:
    """
    FastAPI dependency for API key authentication.

    Usage:
        >>> from fastapi import Depends
        >>> @app.get("/protected")
        >>> def protected_endpoint(api_key: APIKey = Depends(api_key_auth)):
        >>>     return {"user": api_key.user_id}
    """

    def __init__(self, auth_manager: AuthManager):
        self.auth_manager = auth_manager

    def __call__(self, api_key: Optional[str] = None) -> APIKey:
        """
        Verify API key from request.

        Args:
            api_key: API key from header or query parameter

        Returns:
            Verified APIKey object

        Raises:
            Exception if API key is invalid
        """
        if not api_key:
            raise Exception("API key required")

        verified = self.auth_manager.verify_api_key(api_key)

        if not verified:
            raise Exception("Invalid or inactive API key")

        return verified


# Permission constants

class Permission:
    """Standard permissions."""
    # Training
    TRAINING_READ = "training:read"
    TRAINING_WRITE = "training:write"
    TRAINING_DELETE = "training:delete"

    # Models
    MODEL_READ = "model:read"
    MODEL_WRITE = "model:write"
    MODEL_DEPLOY = "model:deploy"
    MODEL_DELETE = "model:delete"

    # Experiments
    EXPERIMENT_READ = "experiment:read"
    EXPERIMENT_WRITE = "experiment:write"
    EXPERIMENT_DELETE = "experiment:delete"

    # Administration
    ADMIN = "admin"
    USER_MANAGE = "user:manage"
    API_KEY_MANAGE = "apikey:manage"


class Role:
    """Standard roles."""
    ADMIN = "admin"
    DEVELOPER = "developer"
    DATA_SCIENTIST = "data_scientist"
    VIEWER = "viewer"


# Default permissions for roles
ROLE_PERMISSIONS = {
    Role.ADMIN: [Permission.ADMIN],  # Admin has all permissions
    Role.DEVELOPER: [
        Permission.TRAINING_READ,
        Permission.TRAINING_WRITE,
        Permission.MODEL_READ,
        Permission.MODEL_WRITE,
        Permission.MODEL_DEPLOY,
        Permission.EXPERIMENT_READ,
        Permission.EXPERIMENT_WRITE,
    ],
    Role.DATA_SCIENTIST: [
        Permission.TRAINING_READ,
        Permission.TRAINING_WRITE,
        Permission.MODEL_READ,
        Permission.EXPERIMENT_READ,
        Permission.EXPERIMENT_WRITE,
    ],
    Role.VIEWER: [
        Permission.TRAINING_READ,
        Permission.MODEL_READ,
        Permission.EXPERIMENT_READ,
    ],
}
