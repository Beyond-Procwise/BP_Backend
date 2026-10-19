"""Where each collection's text comes from: always a Postgres row, so the vectors are rebuildable.

  documents -> the latest ``_raw`` row's parser_snapshot full_text (the extraction pipeline's own record; nothing is copied).
  anything else (uploaded, policy, learning, memory, ...) -> proc.bp_vector_source.
A loader returns None when the source no longer exists; the worker then removes the vectors, so a deleted source cannot leave a vector behind.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional

DEFAULT_TENANT = "default"     # the extraction tables carry no tenant column: one tenant today


@dataclass
class SourceDoc:
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def _full_text(snapshot: Any) -> str:
    if isinstance(snapshot, str):
        try:
            snapshot = json.loads(snapshot)
        except Exception:  # noqa: BLE001
            return ""
    return (snapshot.get("full_text") or "") if isinstance(snapshot, dict) else ""


class DocumentsLoader:
    collection = "documents"

    @staticmethod
    def _tables():
        from src.services.extraction.persistence import _DOC_PK_FIELD, _RAW_TABLES
        return _RAW_TABLES, _DOC_PK_FIELD

    def load(self, cur: Any, tenant_id: str, source_id: str) -> Optional[SourceDoc]:
        raw_tables, pk_fields = self._tables()
        doc_type, _, doc_pk = source_id.partition(":")
        if tenant_id != DEFAULT_TENANT or doc_type not in raw_tables or not doc_pk:
            return None
        cur.execute(f"SELECT parser_snapshot FROM {raw_tables[doc_type]} WHERE {pk_fields[doc_type]} = %s AND parser_snapshot ? 'full_text' "
                    "ORDER BY raw_id DESC LIMIT 1", (doc_pk,))
        row = cur.fetchone()
        text = _full_text(row[0]) if row else ""
        return SourceDoc(text, {"record_id": doc_pk, "document_type": doc_type}) if text.strip() else None

    def enumerate(self, cur: Any, tenant_id: str) -> Iterable[str]:
        if tenant_id != DEFAULT_TENANT:
            return
        raw_tables, pk_fields = self._tables()
        for doc_type, table in raw_tables.items():
            pk = pk_fields[doc_type]
            cur.execute(f"SELECT DISTINCT ON ({pk}) {pk} FROM {table} WHERE parser_snapshot ? 'full_text' AND coalesce({pk}, '') <> '' "
                        f"ORDER BY {pk}, raw_id DESC")
            for (doc_pk,) in cur.fetchall():
                yield f"{doc_type}:{doc_pk}"


class GenericSourceLoader:
    def __init__(self, collection: str) -> None:
        self.collection = collection

    def load(self, cur: Any, tenant_id: str, source_id: str) -> Optional[SourceDoc]:
        cur.execute("SELECT content, metadata FROM proc.bp_vector_source WHERE tenant_id = %s AND collection = %s AND source_id = %s",
                    (tenant_id, self.collection, source_id))
        row = cur.fetchone()
        if not row or not (row[0] or "").strip():
            return None
        meta = row[1]
        if isinstance(meta, str):
            meta = json.loads(meta)
        return SourceDoc(row[0], dict(meta or {}))

    def enumerate(self, cur: Any, tenant_id: str) -> Iterable[str]:
        cur.execute("SELECT source_id FROM proc.bp_vector_source WHERE tenant_id = %s AND collection = %s ORDER BY source_id",
                    (tenant_id, self.collection))
        for (sid,) in cur.fetchall():
            yield sid


def default_loaders() -> Dict[str, Any]:
    out: Dict[str, Any] = {"documents": DocumentsLoader()}
    for name in ("uploaded", "policy", "learning", "memory"):
        out[name] = GenericSourceLoader(name)
    return out


def source_collections(cur: Any, tenant_id: str) -> Iterable[str]:
    """Collections that have source rows for this tenant in bp_vector_source (so rebuild covers ones not named in default_loaders)."""
    cur.execute("SELECT DISTINCT collection FROM proc.bp_vector_source WHERE tenant_id = %s", (tenant_id,))
    return [r[0] for r in cur.fetchall()]
