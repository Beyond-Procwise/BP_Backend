"""The one vector-search interface. Nothing else touches the vector store.

Every method takes ``tenant_id`` first and has no default for it, and ``search`` has no cross-tenant form. A backend (pgvector today,
a dedicated vector DB later) is a new adapter implementing this class plus a backfill from source rows; callers do not change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .types import VectorHit


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, tenant_id: str, source_id: str, embedding: Sequence[float], metadata: Mapping[str, Any], *,
               collection: str, chunk_idx: int = 0, content: Optional[str] = None) -> None:
        """Insert or replace one chunk's vector. Idempotent on (tenant, collection, source, chunk)."""

    @abstractmethod
    def delete(self, tenant_id: str, source_id: str, *, collection: Optional[str] = None) -> int:
        """Remove every vector derived from a source row. Returns how many were removed."""

    @abstractmethod
    def search(self, tenant_id: str, query_embedding: Sequence[float], filters: Optional[Mapping[str, Any]], k: int, *,
               collection: Optional[str] = None) -> List[VectorHit]:
        """The k nearest chunks within ONE tenant, best first. ``filters`` is a whitelist; an unknown key is refused."""

    @abstractmethod
    def list(self, tenant_id: str, *, collection: str, filters: Optional[Mapping[str, Any]] = None, limit: int = 100,
             offset: int = 0) -> List[VectorHit]:
        """Stored chunks without similarity (what a Qdrant scroll was used for), in a stable order."""

    @abstractmethod
    def count(self, tenant_id: str, *, collection: Optional[str] = None) -> int:
        """How many vectors a tenant holds."""

    @abstractmethod
    def clear(self, tenant_id: str, *, collection: Optional[str] = None) -> int:
        """Remove a tenant's vectors (used by rebuild). There is no clear-everything call."""
