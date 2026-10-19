"""Embedding profiles: one per embedding model VERSION, each with its own table.

A vector's dimension is fixed by its model, so a different model cannot share a table with this one. Switching models is: add a
profile here, create its table from the migration template with the new dimension, backfill it with the rebuild command while
the old profile keeps serving, then cut over by setting ``VECTOR_PROFILE``. A search only ever reads the active profile's table,
and also filters on its model and version, so vectors from different model versions are never compared.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict

DEFAULT_PROFILE = "bge-large-en-v1.5@1"
_TABLE = re.compile(r"^proc\.[a-z0-9_]+$")


@dataclass(frozen=True)
class EmbeddingProfile:
    name: str
    model: str
    version: str
    dim: int
    table: str

    def __post_init__(self) -> None:
        if not _TABLE.match(self.table):                    # the table name is interpolated into SQL, so it must be a plain identifier
            raise ValueError(f"profile table must be a plain proc.<identifier>, got {self.table!r}")


PROFILES: Dict[str, EmbeddingProfile] = {
    DEFAULT_PROFILE: EmbeddingProfile(DEFAULT_PROFILE, "BAAI/bge-large-en-v1.5", "1", 1024, "proc.bp_vector_chunk_bge_large_v1"),
}


def active_profile() -> EmbeddingProfile:
    name = os.environ.get("VECTOR_PROFILE", "").strip() or DEFAULT_PROFILE
    if name not in PROFILES:
        raise ValueError(f"unknown VECTOR_PROFILE {name!r}; known: {sorted(PROFILES)}")
    return PROFILES[name]
