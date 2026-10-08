"""The two liveness signals /health reports about the product's output.

Both exist because their silence has already gone unnoticed for weeks: findings
writes were rejected in bp_sqldb for 65 days, and the document vector store sat
empty for five weeks after Qdrant Cloud died. From every existing surface, both
looked exactly like "nothing to report". These helpers are pure functions over
a cursor or a client so they can be tested without a database; /health owns the
connections and turns any failure into the string "unavailable".
"""
from __future__ import annotations

import re
from typing import Any, Optional

DEFAULT_COLLECTION = "procwise_document_embeddings"


def last_finding_written(cur) -> Optional[str]:
    """ISO-8601 timestamp of the newest detection finding, or None when there
    has never been one. ``detected_at`` is the write time of a finding row."""
    cur.execute("SELECT max(detected_at) FROM proc.bp_detection_finding")
    row = cur.fetchone()
    ts = row[0] if row else None
    return ts.isoformat() if ts is not None else None


def vector_store_points(client: Any, collection: str) -> int:
    """Exact number of points in the document collection. ``exact=True`` so a
    zero is a real zero and not an estimate from a stale segment count."""
    return int(client.count(collection_name=collection, exact=True).count)


def collection_name(settings: Any) -> str:
    """The document collection's name, sanitised the same way the extraction
    agent sanitises it before writing, so /health counts the collection the
    pipeline actually fills."""
    base = getattr(settings, "qdrant_collection_name", DEFAULT_COLLECTION)
    if not isinstance(base, str) or not base.strip():
        return DEFAULT_COLLECTION
    return re.sub(r"[^A-Za-z0-9_-]", "_", base.strip()) or DEFAULT_COLLECTION
