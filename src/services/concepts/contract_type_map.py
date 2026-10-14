"""Read the existing corpus's free-text contract_type as a structure.

proc.bp_contract_master holds 3,051 contracts whose contract_type is free text:
'Consulting', 'NDA', 'SLA', 'Master Agreement' and ten more. The relationship
maths needs candidate parents, and a candidate set drawn only from newly
uploaded documents would be empty for a long time -- so these 3,051 rows have to
be readable as structures.

Two decisions worth knowing:

  * NOTHING IS WRITTEN. contract_type is source data and is never modified
    (see the extraction-accuracy standing rule). This is a read-time function,
    so confirming a concept takes effect with no backfill and leaves no second
    copy of the answer to drift from the first.
  * IT GOES THROUGH THE ALIAS INDEX, not a hand-written dictionary. A dictionary
    here would be a fourth place the vocabulary lives, and the three that already
    exist (table, seed, alias index) are kept in step only by a drift test. The
    consequence is that a 'proposed' concept does not resolve -- which is the
    mechanism working, not a gap: 'Policy' matches doctype.policy_document's
    alias but that concept awaits confirmation.

Measured 2026-10-02: 9 of the 13 distinct values resolve, covering 3,016 of the
3,051 rows. The other 35 are NULL (29), 'Service' (2), 'Indirect Procurement'
(1) and 'Policy' (3).
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping, Optional

from .vocabulary import Vocabulary, ensure_vocabulary, resolve_alias


def structure_for_contract_type(
    value: Optional[str],
    *,
    vocabulary: Optional[Vocabulary] = None,
) -> Optional[str]:
    """The concept_code this free-text contract_type names, or None.

    None for a blank value, for a value nothing claims, and for a value TWO
    structures claim. The last of those is the one worth stating: picking the
    first owner would make the answer depend on row order, and the structure it
    chose would then be recorded as a fact and linked on.
    """
    raw = (value or "").strip()
    if not raw:
        return None
    vocab = vocabulary if vocabulary is not None else ensure_vocabulary()
    # A 'proposed' concept is absent from the alias index by design, so e.g.
    # 'Policy' (claimed by the proposed doctype.policy_document) yields no owner
    # here. That is the status rule working, not a vocabulary gap.
    owners = resolve_alias(raw, vocab)
    return owners[0] if len(owners) == 1 else None


def coverage(
    rows: Iterable[Mapping],
    *,
    vocabulary: Optional[Vocabulary] = None,
) -> dict:
    """How much of the corpus maps, and what the remainder actually is.

    ``rows`` are mappings with ``contract_type`` and ``n`` (a row count).

    The unmapped values are returned, not just counted. 'unmapped: 35' invites
    the assumption that the vocabulary is short of 35 documents' worth of
    structures; naming them shows that 29 are NULL in the source and 3 are a
    concept awaiting confirmation, which are three different problems.
    """
    vocab = vocabulary if vocabulary is not None else ensure_vocabulary()
    mapped = 0
    unmapped = 0
    unmapped_values: Counter = Counter()
    for row in rows:
        value = row.get("contract_type")
        count = int(row.get("n") or 0)
        if structure_for_contract_type(value, vocabulary=vocab):
            mapped += count
        else:
            unmapped += count
            unmapped_values[(value or "").strip()] += count
    return {
        "mapped": mapped,
        "unmapped": unmapped,
        "unmapped_values": dict(unmapped_values),
    }


__all__ = ["structure_for_contract_type", "coverage"]
