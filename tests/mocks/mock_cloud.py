"""
Mock Cloud Clients for Testing

These mocks simulate AWS S3, GCP Storage, and Azure Blob Storage
without requiring actual cloud credentials or network access.
"""

import os
from typing import Optional, Dict, Any, List
from pathlib import Path
import json


class MockS3Client:
    """
    Mock AWS S3 client for testing.

    Simulates boto3 S3 client interface with local storage.
    """

    def __init__(self):
        self.buckets: Dict[str, Dict[str, bytes]] = {}
        self.upload_count = 0
        self.download_count = 0

    def create_bucket(self, Bucket: str, **kwargs):
        """Create a mock bucket."""
        if Bucket not in self.buckets:
            self.buckets[Bucket] = {}
        return {"ResponseMetadata": {"HTTPStatusCode": 200}}

    def list_buckets(self):
        """List all buckets."""
        return {
            "Buckets": [{"Name": name} for name in self.buckets.keys()],
            "ResponseMetadata": {"HTTPStatusCode": 200}
        }

    def upload_file(self, Filename: str, Bucket: str, Key: str, **kwargs):
        """Upload file to mock S3."""
        if Bucket not in self.buckets:
            self.create_bucket(Bucket=Bucket)

        # Read file content
        try:
            with open(Filename, "rb") as f:
                content = f.read()
            self.buckets[Bucket][Key] = content
            self.upload_count += 1
            return {"ResponseMetadata": {"HTTPStatusCode": 200}}
        except FileNotFoundError:
            raise Exception(f"File not found: {Filename}")

    def upload_fileobj(self, Fileobj, Bucket: str, Key: str, **kwargs):
        """Upload file object to mock S3."""
        if Bucket not in self.buckets:
            self.create_bucket(Bucket=Bucket)

        content = Fileobj.read()
        if isinstance(content, str):
            content = content.encode()

        self.buckets[Bucket][Key] = content
        self.upload_count += 1
        return {"ResponseMetadata": {"HTTPStatusCode": 200}}

    def download_file(self, Bucket: str, Key: str, Filename: str, **kwargs):
        """Download file from mock S3."""
        if Bucket not in self.buckets:
            raise Exception(f"Bucket not found: {Bucket}")

        if Key not in self.buckets[Bucket]:
            raise Exception(f"Key not found: {Key}")

        # Write content to file
        os.makedirs(os.path.dirname(Filename) or ".", exist_ok=True)
        with open(Filename, "wb") as f:
            f.write(self.buckets[Bucket][Key])

        self.download_count += 1
        return {"ResponseMetadata": {"HTTPStatusCode": 200}}

    def download_fileobj(self, Bucket: str, Key: str, Fileobj, **kwargs):
        """Download file object from mock S3."""
        if Bucket not in self.buckets:
            raise Exception(f"Bucket not found: {Bucket}")

        if Key not in self.buckets[Bucket]:
            raise Exception(f"Key not found: {Key}")

        Fileobj.write(self.buckets[Bucket][Key])
        self.download_count += 1
        return {"ResponseMetadata": {"HTTPStatusCode": 200}}

    def list_objects_v2(self, Bucket: str, Prefix: str = "", **kwargs):
        """List objects in bucket."""
        if Bucket not in self.buckets:
            return {"Contents": [], "ResponseMetadata": {"HTTPStatusCode": 200}}

        objects = [
            {"Key": key, "Size": len(content)}
            for key, content in self.buckets[Bucket].items()
            if key.startswith(Prefix)
        ]

        return {
            "Contents": objects,
            "ResponseMetadata": {"HTTPStatusCode": 200}
        }

    def delete_object(self, Bucket: str, Key: str, **kwargs):
        """Delete object from bucket."""
        if Bucket in self.buckets and Key in self.buckets[Bucket]:
            del self.buckets[Bucket][Key]
        return {"ResponseMetadata": {"HTTPStatusCode": 204}}

    def head_object(self, Bucket: str, Key: str, **kwargs):
        """Get object metadata."""
        if Bucket not in self.buckets:
            raise Exception(f"Bucket not found: {Bucket}")

        if Key not in self.buckets[Bucket]:
            raise Exception(f"Key not found: {Key}")

        content = self.buckets[Bucket][Key]
        return {
            "ContentLength": len(content),
            "ResponseMetadata": {"HTTPStatusCode": 200}
        }


class MockGCSClient:
    """
    Mock Google Cloud Storage client for testing.

    Simulates google.cloud.storage interface.
    """

    def __init__(self, project: Optional[str] = None):
        self.project = project or "mock-project"
        self.buckets: Dict[str, "MockGCSBucket"] = {}

    def bucket(self, bucket_name: str) -> "MockGCSBucket":
        """Get or create mock bucket."""
        if bucket_name not in self.buckets:
            self.buckets[bucket_name] = MockGCSBucket(bucket_name, self)
        return self.buckets[bucket_name]

    def list_buckets(self):
        """List all buckets."""
        return list(self.buckets.values())


class MockGCSBucket:
    """Mock GCS bucket."""

    def __init__(self, name: str, client: MockGCSClient):
        self.name = name
        self.client = client
        self.blobs: Dict[str, bytes] = {}

    def blob(self, blob_name: str) -> "MockGCSBlob":
        """Get mock blob."""
        return MockGCSBlob(blob_name, self)

    def list_blobs(self, prefix: str = ""):
        """List blobs in bucket."""
        matching_blobs = [
            MockGCSBlob(name, self)
            for name in self.blobs.keys()
            if name.startswith(prefix)
        ]
        return matching_blobs

    def exists(self) -> bool:
        """Check if bucket exists."""
        return True


class MockGCSBlob:
    """Mock GCS blob."""

    def __init__(self, name: str, bucket: MockGCSBucket):
        self.name = name
        self.bucket = bucket

    def upload_from_filename(self, filename: str):
        """Upload file to blob."""
        with open(filename, "rb") as f:
            content = f.read()
        self.bucket.blobs[self.name] = content

    def download_to_filename(self, filename: str):
        """Download blob to file."""
        if self.name not in self.bucket.blobs:
            raise Exception(f"Blob not found: {self.name}")

        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        with open(filename, "wb") as f:
            f.write(self.bucket.blobs[self.name])

    def exists(self) -> bool:
        """Check if blob exists."""
        return self.name in self.bucket.blobs

    def delete(self):
        """Delete blob."""
        if self.name in self.bucket.blobs:
            del self.bucket.blobs[self.name]


class MockAzureBlobClient:
    """
    Mock Azure Blob Storage client for testing.

    Simulates azure.storage.blob.BlobServiceClient interface.
    """

    def __init__(self, connection_string: Optional[str] = None):
        self.connection_string = connection_string or "DefaultEndpointsProtocol=https;..."
        self.containers: Dict[str, "MockBlobContainer"] = {}

    @classmethod
    def from_connection_string(cls, conn_str: str):
        """Create client from connection string."""
        return cls(connection_string=conn_str)

    def get_container_client(self, container: str) -> "MockBlobContainer":
        """Get container client."""
        if container not in self.containers:
            self.containers[container] = MockBlobContainer(container, self)
        return self.containers[container]

    def create_container(self, container: str):
        """Create container."""
        if container not in self.containers:
            self.containers[container] = MockBlobContainer(container, self)
        return self.containers[container]

    def list_containers(self):
        """List all containers."""
        return [{"name": name} for name in self.containers.keys()]


class MockBlobContainer:
    """Mock Azure blob container."""

    def __init__(self, name: str, client: MockAzureBlobClient):
        self.name = name
        self.client = client
        self.blobs: Dict[str, bytes] = {}

    def get_blob_client(self, blob: str) -> "MockBlobClient":
        """Get blob client."""
        return MockBlobClient(blob, self)

    def list_blobs(self, name_starts_with: str = ""):
        """List blobs in container."""
        return [
            {"name": name}
            for name in self.blobs.keys()
            if name.startswith(name_starts_with)
        ]

    def exists(self) -> bool:
        """Check if container exists."""
        return True


class MockBlobClient:
    """Mock Azure blob client."""

    def __init__(self, name: str, container: MockBlobContainer):
        self.name = name
        self.container = container

    def upload_blob(self, data, overwrite: bool = False):
        """Upload blob."""
        if isinstance(data, str):
            data = data.encode()
        self.container.blobs[self.name] = data

    def download_blob(self):
        """Download blob."""
        if self.name not in self.container.blobs:
            raise Exception(f"Blob not found: {self.name}")

        return MockBlobDownload(self.container.blobs[self.name])

    def exists(self) -> bool:
        """Check if blob exists."""
        return self.name in self.container.blobs

    def delete_blob(self):
        """Delete blob."""
        if self.name in self.container.blobs:
            del self.container.blobs[self.name]


class MockBlobDownload:
    """Mock Azure blob download."""

    def __init__(self, content: bytes):
        self.content = content

    def readall(self) -> bytes:
        """Read all content."""
        return self.content


# Helper functions for easy mocking

def mock_boto3_client(service_name: str, **kwargs):
    """Mock boto3.client()."""
    if service_name == "s3":
        return MockS3Client()
    else:
        raise NotImplementedError(f"Mock not implemented for service: {service_name}")


def mock_storage_client(project: Optional[str] = None):
    """Mock google.cloud.storage.Client()."""
    return MockGCSClient(project=project)


def mock_blob_service_client_from_connection_string(conn_str: str):
    """Mock BlobServiceClient.from_connection_string()."""
    return MockAzureBlobClient(connection_string=conn_str)
