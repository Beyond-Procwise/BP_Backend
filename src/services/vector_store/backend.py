"""Which vector backend the application reads and writes: ``VECTOR_BACKEND`` = ``qdrant`` (default, until parity is shown) or ``pgvector``.

This switch exists only for the migration window; it goes away, with Qdrant, once the cut-over is complete.
"""
from __future__ import annotations

import os

BACKENDS = ("qdrant", "pgvector")


def vector_backend() -> str:
    name = os.environ.get("VECTOR_BACKEND", "").strip().lower() or "qdrant"
    if name not in BACKENDS:
        raise ValueError(f"unknown VECTOR_BACKEND {name!r}; expected one of {BACKENDS}")
    return name


def use_pgvector() -> bool:
    return vector_backend() == "pgvector"
