"""Embed an extracted document into the document vector index.

Until 2026-10-08 only the legacy DataExtractionAgent wrote embeddings, so every
document that came through the live path (dispatch.py, run by the
process-monitor watcher) reached the database and never the vector index:
procwise_document_embeddings held 0 points while /ask searched it.

The text embedded is the document's own words, read back from the parser
snapshot dispatch stored on its _raw row, not the extracted field values.
That is why a document is embedded whether or not it was promoted: its text
is true even when a field read from it still waits for review.

Payload shape follows the legacy writer, because that is what rag_service
reads (record_id, document_type, chunk_id, content, summary), minus s3_key,
which DISALLOWED_METADATA_KEYS keeps out of the index.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Callable, Optional, Sequence

from qdrant_client import models

from src.services.extraction.persistence import _DOC_PK_FIELD, _RAW_TABLES
from src.services.health_signals import collection_name

log = logging.getLogger(__name__)

# bge-large-en-v1.5 truncates input at 512 tokens, and English runs about 1.3
# tokens a word, so 350 words (~455 tokens) is embedded whole. The legacy
# writer's 720-token window was silently cut short by the model.
CHUNK_WORDS = 350
CHUNK_OVERLAP = 50

_SUMMARY_CHARS = 240


def chunk_text(text: Optional[str]) -> list[str]:
    """Overlapping word windows over whitespace-collapsed text."""
    words = re.sub(r"\s+", " ", text or "").strip().split(" ")
    words = [w for w in words if w]
    if not words:
        return []
    step = CHUNK_WORDS - CHUNK_OVERLAP
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + CHUNK_WORDS]))
        if start + CHUNK_WORDS >= len(words):
            break
    return chunks


def _point_id(doc_type: str, doc_pk: str, idx: int) -> str:
    """Stable per (type, key, chunk), so re-embedding overwrites in place."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"procwise-doc:{doc_type}:{doc_pk}:{idx}"))


def build_points(
    chunks: Sequence[str], *, doc_type: str, doc_pk: str,
    encode: Callable[..., Any], model_name: Optional[str] = None,
) -> list[models.PointStruct]:
    vectors = encode(list(chunks), normalize_embeddings=True, show_progress_bar=False)
    points = []
    for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
        payload = {
            "record_id": doc_pk,
            "document_type": doc_type,
            "chunk_id": idx,
            "chunk_index": idx,
            "content": chunk,
            "summary": chunk.strip()[:_SUMMARY_CHARS],
        }
        if model_name:
            payload["embedding_model"] = model_name
        vec = vector.tolist() if hasattr(vector, "tolist") else list(vector)
        points.append(models.PointStruct(id=_point_id(doc_type, doc_pk, idx),
                                         vector=vec, payload=payload))
    return points


def _full_text(snapshot: Any) -> str:
    if isinstance(snapshot, str):
        try:
            snapshot = json.loads(snapshot)
        except Exception:  # noqa: BLE001
            return ""
    if isinstance(snapshot, dict):
        return snapshot.get("full_text") or ""
    return ""


def embed_document(agent_nick: Any, cur, *, doc_type: str, doc_pk: Any, raw_id: Any) -> int:
    """Replace this document's points with fresh ones from its _raw text.

    Returns the number of points written; 0 when there is no key or no text.
    Raises on a write failure: the caller decides whether that may fail the
    extraction (the watcher logs it and carries on)."""
    table = _RAW_TABLES[doc_type]
    if not doc_pk:
        return 0
    doc_pk = str(doc_pk)
    cur.execute(f"SELECT parser_snapshot FROM {table} WHERE raw_id = %s", (raw_id,))
    row = cur.fetchone()
    chunks = chunk_text(_full_text(row[0] if row else None))
    if not chunks:
        return 0

    settings = getattr(agent_nick, "settings", None)
    collection = collection_name(settings)
    points = build_points(
        chunks, doc_type=doc_type, doc_pk=doc_pk,
        encode=agent_nick.embedding_model.encode,
        model_name=getattr(settings, "embedding_model", None),
    )
    client = agent_nick.qdrant_client
    # Delete first: a re-read that yields fewer chunks must not leave the old
    # tail behind as a second, stale copy of the document.
    client.delete(
        collection_name=collection,
        points_selector=models.FilterSelector(filter=models.Filter(must=[
            models.FieldCondition(key="record_id", match=models.MatchValue(value=doc_pk)),
            models.FieldCondition(key="document_type", match=models.MatchValue(value=doc_type)),
        ])),
        wait=True,
    )
    client.upsert(collection_name=collection, points=points, wait=True)
    return len(points)


def latest_raw_rows(cur, doc_type: str) -> list[tuple[str, Any]]:
    """(doc_pk, raw_id) of the newest _raw row per document that carries text.

    For the backfill: a document re-read three times has three _raw rows, and
    only the newest is what the live path would have embedded last."""
    table, pk = _RAW_TABLES[doc_type], _DOC_PK_FIELD[doc_type]
    cur.execute(
        f"SELECT DISTINCT ON ({pk}) {pk}, raw_id FROM {table}"
        f" WHERE parser_snapshot ? 'full_text' AND coalesce({pk}, '') <> ''"
        f" ORDER BY {pk}, raw_id DESC"
    )
    return [(r[0], r[1]) for r in cur.fetchall()]
