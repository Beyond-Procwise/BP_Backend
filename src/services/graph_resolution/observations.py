"""What a signal actually looked at.

An observation is one (document_id, field) pair. Two signals that read the same
observation are not independent evidence, however different their arithmetic
looks -- and the whole composition model depends on noticing that. See
docs/superpowers/specs/2026-09-10-graph-entity-resolution-design.md section 4.4.
"""
from __future__ import annotations

import hashlib
from typing import Iterable, Tuple

Observation = Tuple[str, str]


def observation_digest(obs: Iterable[Observation]) -> str:
    """Stable digest over an observation set, independent of iteration order."""
    items = sorted(f"{doc}\x1f{field}" for doc, field in obs)
    joined = "\x1e".join(items).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()[:32]


def intersects(a: frozenset, b: frozenset) -> bool:
    """True when two signals drew on at least one identical observation."""
    return not a.isdisjoint(b)
