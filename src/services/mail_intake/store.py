"""Write-once, content-addressed object storage.

Two properties matter here and both are structural rather than procedural.

**Immutability.** The key is derived from the SHA-256 of the content, so the
same bytes always land on the same key and different bytes never can. Writing
different content to an existing key raises rather than overwriting —
reprocessing appends a derivation record, it does not mutate the original.

**Quarantine separation.** Quarantined bytes go under a different prefix from
cleared artifacts. That is what lets the extraction path be granted a key that
covers artifacts and cannot reach quarantine — the gate stops being a
convention the code must remember and becomes a permission it does not hold.

Spec §2.1 item 3, §6 stage 8.
"""
from __future__ import annotations

import hashlib
from typing import Any, Optional

ARTIFACT_PREFIX = "mail-intake/artifacts/"
QUARANTINE_PREFIX = "mail-intake/quarantine/"
RAW_MIME_PREFIX = "mail-intake/raw/"


class ImmutabilityViolation(RuntimeError):
    """An attempt to write different content to an existing key."""


def sha256_hex(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _fanned(digest: str) -> str:
    """Two-level fan-out keeps any one prefix from growing unboundedly."""
    return f"{digest[:2]}/{digest[2:4]}/{digest}"


def artifact_key(owner_ref: str, digest: str) -> str:
    return f"{ARTIFACT_PREFIX}{owner_ref}/{_fanned(digest)}"


def quarantine_key(owner_ref: str, digest: str) -> str:
    return f"{QUARANTINE_PREFIX}{owner_ref}/{_fanned(digest)}"


def raw_mime_key(owner_ref: str, bundle_id: str) -> str:
    return f"{RAW_MIME_PREFIX}{owner_ref}/{bundle_id}.eml"


class ObjectStore:
    """Thin write-once wrapper. Holds no parsing or interpretation logic."""

    def __init__(self, bucket: str, client: Optional[Any] = None) -> None:
        self._bucket = bucket
        self._client = client

    def _resolve_client(self) -> Any:
        if self._client is None:
            import boto3

            self._client = boto3.client("s3")
        return self._client

    def exists(self, key: str) -> bool:
        try:
            self._resolve_client().head_object(Bucket=self._bucket, Key=key)
            return True
        except Exception:
            return False

    def get(self, key: str) -> Optional[bytes]:
        try:
            response = self._resolve_client().get_object(
                Bucket=self._bucket, Key=key
            )
            return response["Body"].read()
        except Exception:
            return None

    def put_immutable(self, key: str, content: bytes) -> str:
        """Write content once. Identical re-writes are no-ops; different ones raise."""
        client = self._resolve_client()
        if self.exists(key):
            existing = self.get(key)
            if existing is not None and existing != content:
                raise ImmutabilityViolation(key)
            if existing == content:
                return key
            raise ImmutabilityViolation(key)
        client.put_object(Bucket=self._bucket, Key=key, Body=content)
        return key
