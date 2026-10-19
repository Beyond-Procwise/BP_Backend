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


def table_ddl(profile: EmbeddingProfile, *, hnsw_m: int = 16, hnsw_ef_construction: int = 64) -> str:
    """The CREATE TABLE and indexes for a profile's table: the template for a model switch.

    Create the new table from this (in a reviewed migration), add the profile to ``PROFILES``, backfill it with
    ``python -m src.services.vector_store.rebuild --profile NAME`` while the old profile serves, then cut over with ``VECTOR_PROFILE``.
    The v1 table was created by deploy/sql/2026-10-19_vector_store.sql; a test pins that this template produces the same columns and indexes.
    """
    t = profile.table
    base = t.split(".", 1)[1]
    model, version = profile.model.replace("'", "''"), profile.version.replace("'", "''")
    return f"""CREATE TABLE IF NOT EXISTS {t} (
    vector_id          bigserial PRIMARY KEY,
    tenant_id          text        NOT NULL CHECK (tenant_id <> ''),
    collection         text        NOT NULL,
    source_id          text        NOT NULL,
    chunk_idx          integer     NOT NULL DEFAULT 0,
    embedding_model    text        NOT NULL CHECK (embedding_model = '{model}'),
    embedding_version  text        NOT NULL CHECK (embedding_version = '{version}'),
    embedding          vector({int(profile.dim)}) NOT NULL,
    content            text,
    document_type      text,
    document_id        text,
    record_id          text,
    supplier_id        text,
    session_id         text,
    product_type       text,
    source_type        text,
    metadata           jsonb       NOT NULL DEFAULT '{{}}'::jsonb,
    created_at         timestamptz NOT NULL DEFAULT now(),
    updated_at         timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT uq_{base} UNIQUE (tenant_id, collection, source_id, chunk_idx)
);
CREATE INDEX IF NOT EXISTS ix_{base}_hnsw ON {t} USING hnsw (embedding vector_cosine_ops) WITH (m = {hnsw_m}, ef_construction = {hnsw_ef_construction});
CREATE INDEX IF NOT EXISTS ix_{base}_tenant_coll ON {t} (tenant_id, collection);
CREATE INDEX IF NOT EXISTS ix_{base}_tenant_src  ON {t} (tenant_id, source_id);
CREATE INDEX IF NOT EXISTS ix_{base}_doctype     ON {t} (tenant_id, document_type);
CREATE INDEX IF NOT EXISTS ix_{base}_supplier    ON {t} (tenant_id, supplier_id);
CREATE INDEX IF NOT EXISTS ix_{base}_session     ON {t} (tenant_id, session_id);
CREATE INDEX IF NOT EXISTS ix_{base}_docid       ON {t} (tenant_id, document_id);
"""
