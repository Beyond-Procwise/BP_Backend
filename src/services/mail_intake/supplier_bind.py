"""Match a sender domain to a supplier record. Never guess.

Three outcomes, all recorded explicitly: matched, unmatched, ambiguous.
Ambiguity is a state to carry forward, not a problem to solve by picking the
closest candidate — attributing a quote to the wrong supplier is worse than
attributing it to none.

That is not a hypothetical here. bp_supplier holds 5,028 rows sharing 288
distinct domains, so ambiguity is the ordinary path in this corpus. Downstream
code must treat `ambiguous` as normal.

Spec §6 stage 4, §7.3.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

MATCHED = "matched"
UNMATCHED = "unmatched"
AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class BindResult:
    state: str
    supplier_id: Optional[str]
    candidates: Tuple[str, ...]


CandidateLookup = Callable[[str], Sequence[Tuple[str, str]]]


def bind(sender_domain: Optional[str], candidate_lookup: CandidateLookup) -> BindResult:
    """Resolve a sender domain against supplier records."""
    if not sender_domain:
        return BindResult(UNMATCHED, None, ())

    domain = sender_domain.strip().lower()
    try:
        rows = list(candidate_lookup(domain) or ())
    except Exception:
        # A lookup failure is not evidence of a match. Fail to unmatched and
        # let the caller's Finding record that provenance is incomplete.
        return BindResult(UNMATCHED, None, ())

    supplier_ids = tuple(dict.fromkeys(str(row[0]) for row in rows))
    if not supplier_ids:
        return BindResult(UNMATCHED, None, ())
    if len(supplier_ids) == 1:
        return BindResult(MATCHED, supplier_ids[0], supplier_ids)
    return BindResult(AMBIGUOUS, None, supplier_ids)
