"""Grounding guard for obligation quotes.

Why this is not ``extraction_v3.grounding.is_value_grounded``:

That guard grounds a *field value* — and one of its tolerances is that a value is
grounded if its digit signature (>=3 digits) appears anywhere in the document. For an
invoice total that is sensible. For a *sentence* it is a hole: the fabricated clause

    "The Contractor shall indemnify the Authority in full under conditions 27.1
     and 27.2 for all consequential loss."

passes it, because the digits 271272 occur in the document's digit stream. A wholly
invented indemnity would be persisted as grounded fact.

So we reuse only its normalisation, and require whole-quote containment. No digit
fallback, no date fallback, and no "cannot verify -> allow".
"""
from __future__ import annotations

from src.services.extraction_v3.grounding import _norm

# A real contract sentence is never this short. Bare clause refs ("27.4") and
# fragments ("the Goods") die here.
MIN_QUOTE_WORDS = 8


def is_quote_grounded(quote: str, full_text: str) -> bool:
    """True only if ``quote`` appears verbatim (modulo case/whitespace) in ``full_text``."""
    q = _norm(quote)
    if len(q.split()) < MIN_QUOTE_WORDS:
        return False
    if not _norm(full_text):
        # No document means no proof, and no proof means no obligation.
        return False
    return q in _norm(full_text)
