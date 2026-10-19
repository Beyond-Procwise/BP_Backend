"""Vector search for chat and RAG. Callers use ``VectorStore`` only; ``get_vector_store()`` returns the configured adapter."""
from .base import VectorStore
from .pgvector_store import PgVectorStore
from .profiles import EmbeddingProfile, PROFILES, active_profile
from .types import TenantRequired, VectorHit

DEFAULT_TENANT = "default"      # one tenant today (the same constant the sell-side tables use); the seam where a real tenant is resolved


def get_vector_store() -> VectorStore:
    return PgVectorStore(profile=active_profile())


__all__ = ["DEFAULT_TENANT", "EmbeddingProfile", "PROFILES", "PgVectorStore", "TenantRequired", "VectorHit", "VectorStore",
           "active_profile", "get_vector_store"]
