"""Qdrant-shaped hits from the VectorStore, for the few callers that only need ``.id`` / ``.score`` / ``.payload`` (migration window; goes with Qdrant)."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List, Mapping, Optional, Sequence


def search_hits(query_embedding: Sequence[float], collection: str, k: int, *, filters: Optional[Mapping[str, Any]] = None,
                store: Any = None) -> List[SimpleNamespace]:
    """Nearest chunks for the one tenant there is today, as simple objects with ``id``, ``score``, ``payload`` and ``vector`` (always None here).

    Raises on a store failure; callers keep their own best-effort handling.
    """
    from . import DEFAULT_TENANT, get_vector_store
    store = store or get_vector_store()
    hits = store.search(DEFAULT_TENANT, [float(x) for x in query_embedding], dict(filters or {}), int(k), collection=collection)
    return [SimpleNamespace(id=h.id, score=h.score, payload=dict(h.payload or {}), vector=None) for h in hits]
