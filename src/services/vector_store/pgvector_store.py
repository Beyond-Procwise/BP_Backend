"""PgVectorStore: the pgvector adapter. All pgvector SQL in the application lives in this file.

SCALING NOTE: this is one shared table with a tenant_id column, filtered on every query. The next step as tenants and volume grow is
to PARTITION the table by tenant_id (LIST or HASH), which gives each tenant physically separate rows and HNSW indexes. The queries
below do not change when that happens. After that, a dedicated vector DB is a new adapter for ``VectorStore`` plus a backfill.

Filtered HNSW search needs care: the index returns its nearest candidates first and the tenant/collection/filter predicates then
discard most of them, so a plain filtered search can return fewer than k rows. pgvector 0.8+ has iterative index scans, which keep
scanning until k rows survive the filter; they are enabled per session whenever the installed extension supports them.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .base import VectorStore
from .profiles import EmbeddingProfile, active_profile
from .types import TenantRequired, VectorHit

logger = logging.getLogger(__name__)

# Metadata that must never be indexed (kept identical to the old document writer's list; a test pins the equality).
DISALLOWED_METADATA_KEYS = frozenset({"uploaded_by", "s3_key"})
# The only columns a caller may filter on. tenant_id is deliberately absent: a filter can never choose the tenant.
FILTER_COLUMNS = ("document_type", "document_id", "record_id", "supplier_id", "session_id", "product_type", "source_type")
MAX_K = 200
_ITERATIVE_MIN = (0, 8, 0)
_COLS = "collection, source_id, chunk_idx, content, document_type, document_id, record_id, supplier_id, session_id, product_type, source_type, metadata"


def _tenant(tenant_id: Any) -> str:
    if not isinstance(tenant_id, str) or not tenant_id.strip():
        raise TenantRequired("a tenant_id is required for every vector operation")
    return tenant_id


def _vec(embedding: Sequence[float], dim: int) -> str:
    if len(embedding) != dim:
        raise ValueError(f"embedding dimension {len(embedding)} does not match this profile's dimension {dim}")
    return "[" + ",".join(repr(float(x)) for x in embedding) + "]"


def _version(text: Optional[str]) -> Tuple[int, ...]:
    return tuple(int(p) for p in re.findall(r"\d+", text or "")[:3])


class PgVectorStore(VectorStore):
    def __init__(self, connect: Optional[Callable[[], Any]] = None, *, profile: Optional[EmbeddingProfile] = None,
                 ef_search: int = 80) -> None:
        self._connect = connect
        self.profile = profile or active_profile()
        self._ef_search = ef_search
        self._iterative: Optional[bool] = None

    # -- plumbing ---------------------------------------------------------------------------------------------------------
    def _conn(self):
        if self._connect is not None:
            return self._connect()
        from src.services.db import get_conn
        return get_conn()

    def _base_where(self, tenant_id: str, collection: Optional[str], filters: Optional[Mapping[str, Any]]) -> Tuple[str, List[Any]]:
        """The WHERE clause. Tenant and model version are always present; every value is a bound parameter."""
        parts = ["tenant_id = %s", "embedding_model = %s", "embedding_version = %s"]
        params: List[Any] = [tenant_id, self.profile.model, self.profile.version]
        if collection is not None:
            parts.append("collection = %s")
            params.append(collection)
        for key, value in (filters or {}).items():
            if key == "metadata":
                if not isinstance(value, Mapping):
                    raise ValueError("metadata filter must be a mapping")
                parts.append("metadata @> %s::jsonb")
                params.append(json.dumps(dict(value), sort_keys=True, default=str))
            elif key in FILTER_COLUMNS:
                if value is None:
                    raise ValueError(f"filter {key!r} has no value")
                if isinstance(value, (list, tuple, set, frozenset)):
                    parts.append(f"{key} = ANY(%s)")
                    params.append(list(value))
                else:
                    parts.append(f"{key} = %s")
                    params.append(value)
            else:
                raise ValueError(f"unknown filter {key!r}; allowed: {list(FILTER_COLUMNS) + ['metadata']}")
        return " AND ".join(parts), params

    def _enable_iterative(self, cur: Any, k: int) -> None:
        if self._iterative is None:
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            row = cur.fetchone()
            self._iterative = bool(row) and _version(row[0]) >= _ITERATIVE_MIN
            if not self._iterative:
                logger.warning("pgvector %s has no iterative index scans (0.8.0+); filtered searches may return fewer than k rows",
                               row[0] if row else "?")
        cur.execute("SELECT set_config('hnsw.ef_search', %s, false)", (str(max(self._ef_search, min(k * 2, 1000))),))
        if self._iterative:
            cur.execute("SELECT set_config('hnsw.iterative_scan', 'strict_order', false)")

    @staticmethod
    def _hit(row: Tuple[Any, ...], score: Optional[float]) -> VectorHit:
        (collection, source_id, chunk_idx, content, document_type, document_id, record_id, supplier_id, session_id, product_type,
         source_type, metadata) = row[:12]
        payload: Dict[str, Any] = dict(metadata or {})
        for key, val in (("content", content), ("document_type", document_type), ("document_id", document_id), ("record_id", record_id),
                         ("supplier_id", supplier_id), ("session_id", session_id), ("product_type", product_type),
                         ("source_type", source_type)):
            if val is not None:
                payload[key] = val
        return VectorHit(id=f"{collection}:{source_id}:{chunk_idx}", source_id=source_id, chunk_idx=chunk_idx, collection=collection,
                         score=None if score is None else float(score), content=content, payload=payload)

    # -- VectorStore -------------------------------------------------------------------------------------------------------
    def _upsert_stmt(self, tenant_id: str, source_id: str, embedding: Sequence[float], metadata: Mapping[str, Any], collection: str,
                     chunk_idx: int, content: Optional[str]) -> Tuple[str, tuple]:
        meta = {k: v for k, v in dict(metadata or {}).items() if k not in DISALLOWED_METADATA_KEYS}
        cols = {c: (str(meta.pop(c)) if meta.get(c) is not None else None) for c in FILTER_COLUMNS}
        for c in FILTER_COLUMNS:
            meta.pop(c, None)
        meta.pop("content", None)
        vec = _vec(embedding, self.profile.dim)
        sql = (f"INSERT INTO {self.profile.table} (tenant_id, collection, source_id, chunk_idx, embedding_model, embedding_version, embedding, "
               "content, document_type, document_id, record_id, supplier_id, session_id, product_type, source_type, metadata) "
               "VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb) "
               "ON CONFLICT (tenant_id, collection, source_id, chunk_idx) DO UPDATE SET "
               "embedding = EXCLUDED.embedding, content = EXCLUDED.content, document_type = EXCLUDED.document_type, "
               "document_id = EXCLUDED.document_id, record_id = EXCLUDED.record_id, supplier_id = EXCLUDED.supplier_id, "
               "session_id = EXCLUDED.session_id, product_type = EXCLUDED.product_type, source_type = EXCLUDED.source_type, "
               "metadata = EXCLUDED.metadata, updated_at = now()")
        params = (tenant_id, collection, str(source_id), int(chunk_idx), self.profile.model, self.profile.version, vec, content,
                  cols["document_type"], cols["document_id"], cols["record_id"], cols["supplier_id"], cols["session_id"],
                  cols["product_type"], cols["source_type"], json.dumps(meta, sort_keys=True, default=str))
        return sql, params

    def upsert(self, tenant_id, source_id, embedding, metadata, *, collection, chunk_idx=0, content=None) -> None:
        tenant_id = _tenant(tenant_id)
        sql, params = self._upsert_stmt(tenant_id, source_id, embedding, metadata, collection, chunk_idx, content)
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(sql, params)

    def replace(self, tenant_id, source_id, chunks, *, collection) -> int:
        tenant_id = _tenant(tenant_id)
        stmts = [self._upsert_stmt(tenant_id, source_id, c["embedding"], c.get("metadata") or {}, collection, i, c.get("content"))
                 for i, c in enumerate(chunks)]                                   # all validated before any SQL runs
        with self._conn() as conn:
            own = bool(getattr(conn, "autocommit", False))                        # a connection we opened: make it one transaction
            if own:
                conn.autocommit = False
            try:
                with conn.cursor() as cur:
                    cur.execute(f"DELETE FROM {self.profile.table} WHERE tenant_id = %s AND collection = %s AND source_id = %s AND chunk_idx >= %s",
                                (tenant_id, collection, str(source_id), len(stmts)))
                    for sql, params in stmts:
                        cur.execute(sql, params)
                if own:
                    conn.commit()
            except Exception:
                if own:
                    conn.rollback()
                raise
        return len(stmts)

    def delete(self, tenant_id, source_id, *, collection=None) -> int:
        tenant_id = _tenant(tenant_id)
        sql, params = f"DELETE FROM {self.profile.table} WHERE tenant_id = %s AND source_id = %s", [tenant_id, str(source_id)]
        if collection is not None:
            sql += " AND collection = %s"
            params.append(collection)
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            return cur.rowcount

    def search(self, tenant_id, query_embedding, filters, k, *, collection=None) -> List[VectorHit]:
        tenant_id = _tenant(tenant_id)
        if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= MAX_K:
            raise ValueError(f"k must be between 1 and {MAX_K}")
        vec = _vec(query_embedding, self.profile.dim)
        where, params = self._base_where(tenant_id, collection, filters)
        sql = (f"SELECT {_COLS}, 1 - (embedding <=> %s::vector) AS score FROM {self.profile.table} WHERE {where} "
               "ORDER BY embedding <=> %s::vector LIMIT %s")
        with self._conn() as conn, conn.cursor() as cur:
            self._enable_iterative(cur, k)
            cur.execute(sql, tuple([vec] + params + [vec, k]))
            return [self._hit(r, r[12]) for r in cur.fetchall()]

    def list(self, tenant_id, *, collection, filters=None, limit=100, offset=0) -> List[VectorHit]:
        tenant_id = _tenant(tenant_id)
        if not 1 <= int(limit) <= 1000 or int(offset) < 0:
            raise ValueError("limit must be 1..1000 and offset >= 0")
        where, params = self._base_where(tenant_id, collection, filters)
        sql = f"SELECT {_COLS} FROM {self.profile.table} WHERE {where} ORDER BY vector_id LIMIT %s OFFSET %s"
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(sql, tuple(params + [int(limit), int(offset)]))
            return [self._hit(r, None) for r in cur.fetchall()]

    def count(self, tenant_id, *, collection=None) -> int:
        tenant_id = _tenant(tenant_id)
        where, params = self._base_where(tenant_id, collection, None)
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {self.profile.table} WHERE {where}", tuple(params))
            return int(cur.fetchone()[0])

    def clear(self, tenant_id, *, collection=None) -> int:
        tenant_id = _tenant(tenant_id)
        sql, params = f"DELETE FROM {self.profile.table} WHERE tenant_id = %s", [tenant_id]
        if collection is not None:
            sql += " AND collection = %s"
            params.append(collection)
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            return cur.rowcount
