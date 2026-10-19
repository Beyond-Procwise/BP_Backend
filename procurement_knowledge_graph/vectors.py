"""The knowledge graph's vector search, on the application's own Postgres (pgvector) through ``src.services.vector_store``.

It replaces the Qdrant client this package used. Vectors are embedded with the store's active profile model (not Ollama), because the
store holds one embedding dimension per profile. The three collections are prefixed ``kg_`` so they never mix with chat/RAG vectors.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.services.vector_store import DEFAULT_TENANT, active_profile, get_vector_store
from src.services.vector_store.embedder import SentenceTransformerEmbedder

COLLECTIONS = {"suppliers": "kg_suppliers", "contracts": "kg_contracts", "negotiations": "kg_negotiations"}


class KnowledgeGraphVectors:
    def __init__(self, store: Any = None, embedder: Any = None, tenant_id: str = DEFAULT_TENANT) -> None:
        self._store = store if store is not None else get_vector_store()
        self._embedder = embedder if embedder is not None else SentenceTransformerEmbedder(active_profile().model)
        self._tenant = tenant_id

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        return self._embedder.encode(list(texts))

    def upsert(self, name: str, points: Sequence[Mapping[str, Any]]) -> None:
        """``points``: dicts with ``id``, ``text`` and ``payload``. Embeds the texts, then stores each under its id."""
        if not points:
            return
        vectors = self.embed([p["text"] for p in points])
        for point, vector in zip(points, vectors):
            self._store.upsert(self._tenant, str(point["id"]), vector, point["payload"], collection=COLLECTIONS[name],
                               content=point["text"])

    def search(self, name: str, text: str, limit: int = 5) -> List[Dict[str, Any]]:
        """The nearest stored payloads to ``text``, best first, each with its ``score``."""
        hits = self._store.search(self._tenant, self.embed([text])[0], None, limit, collection=COLLECTIONS[name])
        return [dict(hit.payload, score=hit.score) for hit in hits]

    def count(self, name: Optional[str] = None) -> int:
        return self._store.count(self._tenant, collection=COLLECTIONS[name] if name else None)
