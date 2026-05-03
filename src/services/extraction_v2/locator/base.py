"""Locator protocol — what a per-field extraction strategy looks like.

Every locator implements ``locate(doc) -> Optional[LocatorOutput]``,
returning either a typed candidate value with provenance, or None
(this strategy abstains).

Multiple locators run concurrently per field; their outputs are voted on
by the consensus runner. No single locator is load-bearing — a buggy or
faulty locator can be outvoted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple, Optional, Protocol, runtime_checkable


__all__ = ["AnchorRef", "LocatorOutput", "Locator"]


@dataclass(frozen=True)
class AnchorRef:
    """Where a locator found its evidence in the source document."""
    page: Optional[int] = None
    bbox: Optional[tuple[float, float, float, float]] = None  # (x0, y0, x1, y1)
    char_offset: Optional[int] = None
    raw_text: Optional[str] = None  # the substring as it appeared in source


class LocatorOutput(NamedTuple):
    """A single locator's proposal for a field's value.

    Attributes:
        value:      typed value (Money, IsoDate, str, ...) — already validated
        confidence: this locator's own confidence in [0.0, 1.0]
        evidence:   AnchorRef — where in the source this came from
        why:        human-readable rationale (kept short for review-queue UI)
    """
    value: Any
    confidence: float
    evidence: AnchorRef
    why: str


@runtime_checkable
class Locator(Protocol):
    """A single per-field extraction strategy.

    Implementations should be PURE FUNCTIONS of the input document:
    no side effects, no network calls, no mutable state. This makes
    them concurrency-safe and trivially testable.
    """
    field: str   # which schema field this locator targets
    name: str    # short identifier for logs / consensus-result ledger

    def locate(self, doc: Any) -> Optional[LocatorOutput]:
        """Return a candidate value or None to abstain."""
        ...
