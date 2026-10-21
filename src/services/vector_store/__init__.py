"""Vector search for chat and RAG. Callers use ``VectorStore`` only; ``get_vector_store()`` returns the configured adapter."""
from .base import VectorStore
from .pgvector_store import PgVectorStore
from .profiles import EmbeddingProfile, PROFILES, active_profile
from .types import TenantRequired, VectorHit

from ..tenant_context import LEGACY_LABEL as DEFAULT_TENANT  # old name, kept for imports; code calls company_label()


def get_vector_store() -> VectorStore:
    return PgVectorStore(profile=active_profile())


__all__ = ["DEFAULT_TENANT", "EmbeddingProfile", "PROFILES", "PgVectorStore", "TenantRequired", "VectorHit", "VectorStore",
           "active_profile", "get_vector_store"]
