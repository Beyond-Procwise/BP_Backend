"""Relay adapter: raw MIME delivered to object storage by our own MTA.

The default and preferred mode. Push-based, so no polling; no mailbox-wide read
scope to justify to a regulated customer's IT reviewer; and the raw MIME is
preserved exactly as delivered, which is what makes everything downstream
reconstructible.

It is also the only path where the sender-authentication verdict comes from an
MTA we control, and is therefore trustworthy (spec §4.1).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Iterator, Mapping, Optional

from src.services.mail_intake.sources.base import RawMessage


class RelayMailSource:
    def __init__(self, bucket: str, prefix: str, client: Optional[Any] = None) -> None:
        self._bucket = bucket
        self._prefix = prefix
        self._client = client

    def _resolve_client(self) -> Any:
        if self._client is None:
            import boto3

            self._client = boto3.client("s3")
        return self._client

    def iter_messages(
        self, binding: Mapping[str, Any], since: Optional[datetime]
    ) -> Iterator[RawMessage]:
        client = self._resolve_client()
        prefix = binding.get("folder_scope") or self._prefix
        paginator = client.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            for entry in page.get("Contents") or ():
                modified = entry.get("LastModified")
                if since and modified and modified < since:
                    continue
                key = entry["Key"]
                body = client.get_object(Bucket=self._bucket, Key=key)["Body"].read()
                yield RawMessage(
                    raw=body,
                    provider_message_id=key,
                    received_at=modified,
                    source_ref=f"s3://{self._bucket}/{key}",
                )
