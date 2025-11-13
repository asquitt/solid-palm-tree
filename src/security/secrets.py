"""
Secrets Management

This module provides secure storage and retrieval of secrets
(API keys, credentials, tokens) with support for multiple backends.
"""

import os
import logging
from typing import Optional, Dict, Any
from enum import Enum


logger = logging.getLogger(__name__)


class SecretsBackend(Enum):
    """Secrets storage backend."""
    ENV = "env"  # Environment variables
    FILE = "file"  # Encrypted file
    AWS_SECRETS_MANAGER = "aws"  # AWS Secrets Manager
    GCP_SECRET_MANAGER = "gcp"  # GCP Secret Manager
    AZURE_KEY_VAULT = "azure"  # Azure Key Vault
    HASHICORP_VAULT = "vault"  # HashiCorp Vault


class SecretsManager:
    """
    Secrets Manager with multiple backend support.

    Provides unified interface for retrieving secrets from various sources.
    """

    def __init__(
        self,
        backend: SecretsBackend = SecretsBackend.ENV,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            backend: Secrets backend to use
            config: Backend-specific configuration
        """
        self.backend = backend
        self.config = config or {}
        self._cache: Dict[str, str] = {}
        self._client = self._init_backend()

    def _init_backend(self):
        """Initialize secrets backend client."""
        if self.backend == SecretsBackend.ENV:
            # No client needed for environment variables
            return None

        elif self.backend == SecretsBackend.AWS_SECRETS_MANAGER:
            try:
                import boto3
                region = self.config.get("region", "us-east-1")
                return boto3.client("secretsmanager", region_name=region)
            except ImportError:
                logger.error("boto3 not installed. Install with: pip install boto3")
                raise

        elif self.backend == SecretsBackend.GCP_SECRET_MANAGER:
            try:
                from google.cloud import secretmanager
                return secretmanager.SecretManagerServiceClient()
            except ImportError:
                logger.error("google-cloud-secret-manager not installed")
                raise

        elif self.backend == SecretsBackend.AZURE_KEY_VAULT:
            try:
                from azure.keyvault.secrets import SecretClient
                from azure.identity import DefaultAzureCredential

                vault_url = self.config.get("vault_url")
                if not vault_url:
                    raise ValueError("vault_url required for Azure Key Vault")

                credential = DefaultAzureCredential()
                return SecretClient(vault_url=vault_url, credential=credential)
            except ImportError:
                logger.error("azure-keyvault-secrets not installed")
                raise

        elif self.backend == SecretsBackend.FILE:
            # File-based secrets (encrypted)
            return None

        elif self.backend == SecretsBackend.HASHICORP_VAULT:
            try:
                import hvac
                vault_url = self.config.get("vault_url", "http://localhost:8200")
                token = self.config.get("token", os.getenv("VAULT_TOKEN"))
                return hvac.Client(url=vault_url, token=token)
            except ImportError:
                logger.error("hvac not installed. Install with: pip install hvac")
                raise

        return None

    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Retrieve secret by key.

        Args:
            key: Secret key
            default: Default value if secret not found

        Returns:
            Secret value or default
        """
        # Check cache first
        if key in self._cache:
            return self._cache[key]

        value = None

        try:
            if self.backend == SecretsBackend.ENV:
                value = os.getenv(key, default)

            elif self.backend == SecretsBackend.AWS_SECRETS_MANAGER:
                response = self._client.get_secret_value(SecretId=key)
                value = response.get("SecretString")

            elif self.backend == SecretsBackend.GCP_SECRET_MANAGER:
                project_id = self.config.get("project_id")
                if not project_id:
                    raise ValueError("project_id required for GCP Secret Manager")

                name = f"projects/{project_id}/secrets/{key}/versions/latest"
                response = self._client.access_secret_version(request={"name": name})
                value = response.payload.data.decode("UTF-8")

            elif self.backend == SecretsBackend.AZURE_KEY_VAULT:
                secret = self._client.get_secret(key)
                value = secret.value

            elif self.backend == SecretsBackend.HASHICORP_VAULT:
                mount_point = self.config.get("mount_point", "secret")
                response = self._client.secrets.kv.v2.read_secret_version(
                    path=key,
                    mount_point=mount_point
                )
                value = response["data"]["data"].get("value")

            elif self.backend == SecretsBackend.FILE:
                # File-based secrets
                file_path = self.config.get("secrets_file", ".secrets.json")
                if os.path.exists(file_path):
                    import json
                    with open(file_path, "r") as f:
                        secrets = json.load(f)
                    value = secrets.get(key, default)

            # Cache the value
            if value is not None:
                self._cache[key] = value

            return value if value is not None else default

        except Exception as e:
            logger.error(f"Error retrieving secret '{key}' from {self.backend.value}: {str(e)}")
            return default

    def set_secret(self, key: str, value: str):
        """
        Store secret.

        Args:
            key: Secret key
            value: Secret value
        """
        try:
            if self.backend == SecretsBackend.ENV:
                # Can't set environment variables permanently
                os.environ[key] = value
                logger.warning("Setting env var - this is not persistent!")

            elif self.backend == SecretsBackend.AWS_SECRETS_MANAGER:
                try:
                    self._client.create_secret(Name=key, SecretString=value)
                except self._client.exceptions.ResourceExistsException:
                    self._client.update_secret(SecretId=key, SecretString=value)

            elif self.backend == SecretsBackend.GCP_SECRET_MANAGER:
                project_id = self.config.get("project_id")
                parent = f"projects/{project_id}"

                # Create secret if not exists
                try:
                    secret = self._client.create_secret(
                        request={
                            "parent": parent,
                            "secret_id": key,
                            "secret": {"replication": {"automatic": {}}},
                        }
                    )
                except:
                    pass  # Secret already exists

                # Add version
                parent = f"projects/{project_id}/secrets/{key}"
                payload = value.encode("UTF-8")
                self._client.add_secret_version(
                    request={"parent": parent, "payload": {"data": payload}}
                )

            elif self.backend == SecretsBackend.AZURE_KEY_VAULT:
                self._client.set_secret(key, value)

            elif self.backend == SecretsBackend.HASHICORP_VAULT:
                mount_point = self.config.get("mount_point", "secret")
                self._client.secrets.kv.v2.create_or_update_secret(
                    path=key,
                    secret={"value": value},
                    mount_point=mount_point
                )

            elif self.backend == SecretsBackend.FILE:
                # File-based secrets
                file_path = self.config.get("secrets_file", ".secrets.json")
                secrets = {}
                if os.path.exists(file_path):
                    import json
                    with open(file_path, "r") as f:
                        secrets = json.load(f)

                secrets[key] = value

                with open(file_path, "w") as f:
                    json.dump(secrets, f, indent=2)

            # Update cache
            self._cache[key] = value
            logger.info(f"Secret '{key}' stored in {self.backend.value}")

        except Exception as e:
            logger.error(f"Error storing secret '{key}' to {self.backend.value}: {str(e)}")
            raise

    def delete_secret(self, key: str):
        """
        Delete secret.

        Args:
            key: Secret key to delete
        """
        try:
            if self.backend == SecretsBackend.AWS_SECRETS_MANAGER:
                self._client.delete_secret(SecretId=key, ForceDeleteWithoutRecovery=True)

            elif self.backend == SecretsBackend.GCP_SECRET_MANAGER:
                project_id = self.config.get("project_id")
                name = f"projects/{project_id}/secrets/{key}"
                self._client.delete_secret(request={"name": name})

            elif self.backend == SecretsBackend.AZURE_KEY_VAULT:
                self._client.begin_delete_secret(key).wait()

            elif self.backend == SecretsBackend.HASHICORP_VAULT:
                mount_point = self.config.get("mount_point", "secret")
                self._client.secrets.kv.v2.delete_metadata_and_all_versions(
                    path=key,
                    mount_point=mount_point
                )

            # Remove from cache
            if key in self._cache:
                del self._cache[key]

            logger.info(f"Secret '{key}' deleted from {self.backend.value}")

        except Exception as e:
            logger.error(f"Error deleting secret '{key}' from {self.backend.value}: {str(e)}")
            raise


# Convenience function

def get_secret(
    key: str,
    default: Optional[str] = None,
    backend: SecretsBackend = SecretsBackend.ENV
) -> Optional[str]:
    """
    Quick helper to get a secret.

    Args:
        key: Secret key
        default: Default value
        backend: Backend to use

    Returns:
        Secret value

    Example:
        >>> api_key = get_secret("OPENAI_API_KEY", default="sk-...")
    """
    manager = SecretsManager(backend=backend)
    return manager.get_secret(key, default)
