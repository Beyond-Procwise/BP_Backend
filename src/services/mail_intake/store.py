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


def _is_not_found(exc: Exception) -> bool:
    """True only for "this key does not exist", never for any other failure.

    The distinction is the whole write-once guarantee. An earlier version caught
    every exception and reported "absent", so a throttle, a timeout or a
    permission change made exists() return False and put_immutable overwrote
    genuinely different content under the same key — silently defeating the one
    property this class exists to provide. Not knowing whether an object exists
    is not the same as knowing it does not.
    """
    code = getattr(exc, "response", {}).get("Error", {}).get("Code")
    if code in ("404", "NoSuchKey", "NotFound"):
        return True
    status = getattr(exc, "response", {}).get(
        "ResponseMetadata", {}
    ).get("HTTPStatusCode")
    return status == 404


class StorageUnavailable(RuntimeError):
    """The store could not answer. Never silently treated as "absent"."""


class ObjectStore:
    """Thin write-once wrapper. Holds no parsing or interpretation logic."""

    def __init__(self, bucket: str, client: Optional[Any] = None) -> None:
        self._bucket = bucket
        self._client = client

    def _resolve_client(self) -> Any:
        if self._client is None:
            # Through the egress chokepoint (1.20m): the same real boto3 client, with every request
            # it makes recorded in the outbound audit under the object-storage purpose.
            from src.services import egress

            self._client = egress.aws_client("s3", purpose=egress.Purpose.OBJECT_STORAGE)
        return self._client

    def exists(self, key: str) -> bool:
        """True, False, or raise. Never "probably not"."""
        try:
            self._resolve_client().head_object(Bucket=self._bucket, Key=key)
            return True
        except Exception as exc:
            if _is_not_found(exc):
                return False
            raise StorageUnavailable(
                f"could not determine whether {key} exists"
            ) from exc

    def get(self, key: str) -> Optional[bytes]:
        try:
            response = self._resolve_client().get_object(
                Bucket=self._bucket, Key=key
            )
            return response["Body"].read()
        except Exception as exc:
            if _is_not_found(exc):
                return None
            raise StorageUnavailable(f"could not read {key}") from exc

    def put_immutable(self, key: str, content: bytes) -> str:
        """Write content once. Identical re-writes are no-ops; different ones raise.

        Every branch below either proves the write is safe or refuses. In
        particular, an existing key whose content cannot be read is NOT treated
        as a mismatch — it is a storage failure, and calling it a mismatch would
        report a fabricated integrity violation. Both are refusals; only one of
        them is true.
        """
        client = self._resolve_client()
        if not self.exists(key):
            client.put_object(Bucket=self._bucket, Key=key, Body=content)
            return key

        existing = self.get(key)
        if existing is None:
            raise StorageUnavailable(
                f"{key} exists but its content could not be read; refusing to write"
            )
        if existing != content:
            raise ImmutabilityViolation(key)
        return key
