"""Shared types for the renovation extraction pipeline.

Candidates flow from L1 (regex) → L2 (engineered) → L3 (judge) → persist.
Every value carries an evidence Span so the L3 substring grounding gate
can verify it appears in the source ParsedDocument.full_text.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Optional

CandidateSource = Literal[
    "regex",       # L1 PatternRegistry
    "table",       # L2 table_extractor
    "ner",         # L2 ner_validator (rarely emits; mostly demotes)
    "address",     # L2 address_parser
    "date",        # L2 date_normaliser
    "bbox",        # L2 bbox_proximity
    "judge",       # L3 grounded_last_resort
    "hitl",        # post-HITL fix
]


@dataclass(frozen=True)
class Span:
    """The exact source evidence for a candidate value."""

    page: int
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1) in page coords
    text: str   # the literal source substring (substring-of-full_text invariant)


@dataclass(frozen=True)
class Candidate:
    """One proposed (field, value) pair with its source evidence."""

    field: str
    value: str
    span: Span
    source: CandidateSource
    pattern_name: Optional[str]
    confidence: float

    def with_confidence(self, new_confidence: float) -> "Candidate":
        return replace(self, confidence=new_confidence)
