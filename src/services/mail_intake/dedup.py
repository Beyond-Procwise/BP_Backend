"""Duplicate detection at two levels.

The content hash catches byte-identical arrivals. It does not catch the more
common procurement case: the same quote forwarded three times by three people,
re-wrapped by three mail clients, arriving with three different hashes. That
needs a logical key — supplier, document type, document reference — and all
three parts must be present, because two of them are not enough to be sure.

Nothing is ever deleted. A duplicate is marked `duplicate_of` and its arrival
is recorded, because two suppliers sending the same document is information
worth keeping.

Spec §6 stages 8 and 9.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class DedupResult:
    is_duplicate: bool
    duplicate_of: Optional[str]
    matched_on: Optional[str]


ExistingLookup = Callable[[str, Optional[str]], Optional[str]]


def logical_key(
    supplier_id: Optional[str],
    doc_type: Optional[str],
    doc_reference: Optional[str],
) -> Optional[str]:
    """A stable key for "the same document, differently wrapped".

    Returns None unless all three parts are present — a partial key would
    collapse unrelated documents together.
    """
    if not (supplier_id and doc_type and doc_reference):
        return None
    parts = (
        str(supplier_id).strip().lower(),
        str(doc_type).strip().lower(),
        str(doc_reference).strip().lower(),
    )
    if not all(parts):
        return None
    return "|".join(parts)


def resolve(
    digest: str, logical: Optional[str], existing_lookup: ExistingLookup
) -> DedupResult:
    """Identify whether this artifact already exists within the dedup scope."""
    existing = existing_lookup(digest, None)
    if existing:
        return DedupResult(True, str(existing), "content_hash")

    if logical:
        existing = existing_lookup(digest, logical)
        if existing:
            return DedupResult(True, str(existing), "logical_key")

    return DedupResult(False, None, None)
