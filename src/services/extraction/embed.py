"""Queue an extracted document for the document vector index.

Until 2026-10-08 only the legacy DataExtractionAgent wrote embeddings, so every document that came through the live path (dispatch.py, run by the
process-monitor watcher) reached the database and never the vector index: the document collection held 0 vectors while /ask searched it.

The text that is embedded is the document's own words, read back from the parser snapshot dispatch stored on its _raw row, not the extracted field
values. That is why a document is embedded whether or not it was promoted: its text is true even when a field read from it still waits for review.

Nothing is embedded here. ``embed_document`` puts a job on the embedding outbox and the vector worker re-reads the _raw
text and embeds it (src/services/vector_store/worker.py). The usual path queues it already, in the same transaction as the _raw row
(persistence.write_raw); this call is the safety net for a document that missed it.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from src.services.extraction.persistence import _DOC_PK_FIELD, _RAW_TABLES
from src.services.vector_store.chunking import chunk_text  # noqa: F401  (the one chunker; re-exported for the callers that used it from here)

log = logging.getLogger(__name__)


def _full_text(snapshot: Any) -> str:
    if isinstance(snapshot, str):
        try:
            snapshot = json.loads(snapshot)
        except Exception:  # noqa: BLE001
            return ""
    if isinstance(snapshot, dict):
        return snapshot.get("full_text") or ""
    return ""


def _queue_connect():
    from src.services.db import get_conn
    return get_conn()


def embed_document(agent_nick: Any, cur, *, doc_type: str, doc_pk: Any, raw_id: Any) -> int:
    """Queue this document for embedding. Returns how many chunks it will become; 0 when there is no key or no text.

    Raises on a write failure: the caller decides whether that may fail the extraction (the watcher logs it and carries on)."""
    table = _RAW_TABLES[doc_type]
    if not doc_pk:
        return 0
    doc_pk = str(doc_pk)
    cur.execute(f"SELECT parser_snapshot FROM {table} WHERE raw_id = %s", (raw_id,))
    row = cur.fetchone()
    chunks = chunk_text(_full_text(row[0] if row else None))
    if not chunks:
        return 0
    from src.services.tenant_context import company_label
    from src.services.vector_store import outbox
    with _queue_connect() as qconn, qconn.cursor() as qcur:
        outbox.enqueue(qcur, company_label(), "documents", f"{doc_type}:{doc_pk}")
    return len(chunks)


def latest_raw_rows(cur, doc_type: str) -> list[tuple[str, Any]]:
    """(doc_pk, raw_id) of the newest _raw row per document that carries text.

    A document re-read three times has three _raw rows, and only the newest is what the live path would have embedded last."""
    table, pk = _RAW_TABLES[doc_type], _DOC_PK_FIELD[doc_type]
    cur.execute(
        f"SELECT DISTINCT ON ({pk}) {pk}, raw_id FROM {table}"
        f" WHERE parser_snapshot ? 'full_text' AND coalesce({pk}, '') <> ''"
        f" ORDER BY {pk}, raw_id DESC"
    )
    return [(r[0], r[1]) for r in cur.fetchall()]
