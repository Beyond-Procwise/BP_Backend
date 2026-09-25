"""What a contract line allows: a rate, a cap, or an item included at no charge.

The line-item extractor reads a contract's rate card as description / price / unit rows,
the same as a quote. Whether a row is a RATE (charged per unit), a CAP (the most that may
be charged) or an INCLUDED item (covered, no extra charge) is stated in the row's own words
— "not to exceed", "capped at", "included", "no charge" — so it is read deterministically
from them here, not left to the model. A row that says none of those and carries a price is
a rate; a row with no price and no inclusion wording is left unclassified (None) rather than
guessed, and the gateway ignores it.

Called from dispatch.py for doc_type == "contract" immediately before
persistence.write_line_items_raw(), on each extracted line dict.
"""
from __future__ import annotations

import re
from typing import Any

_INCLUDED = re.compile(
    r"\b(?:included|inclusive|no\s+(?:additional\s+|extra\s+)?charge|at\s+no\s+cost|"
    r"free\s+of\s+charge|foc|n/?c|bundled)\b", re.IGNORECASE)
_CAP = re.compile(
    r"\b(?:not\s+to\s+exceed|nte|capped?\s+at|cap(?:ped)?|maximum|max\.?|up\s+to|ceiling|limit(?:ed)?\s+to)\b",
    re.IGNORECASE)


def classify_term_basis(description: Any, unit_price: Any, qualifier_text: Any = None) -> str | None:
    """'rate' | 'cap' | 'included' | None, from the row's own words and whether it has a price."""
    text = " ".join(str(x) for x in (description, qualifier_text) if x)
    if _INCLUDED.search(text):
        return "included"
    has_price = unit_price not in (None, "") and _to_number(unit_price) is not None
    if _CAP.search(text):
        return "cap" if has_price else None
    return "rate" if has_price else None


def qualifier_for(basis: str | None) -> str | None:
    """The short word the UI shows beside the allowed figure."""
    return {"cap": "price cap", "included": "included"}.get(basis or "")


def classify_contract_lines(line_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Set term_basis (and a default qualifier) on each extracted contract line, in place."""
    for row in line_items:
        basis = classify_term_basis(row.get("item_description"), row.get("unit_price"), row.get("qualifier"))
        row["term_basis"] = basis
        if basis == "included" and row.get("unit_price") in (None, ""):
            row["unit_price"] = 0
        if not row.get("qualifier"):
            row["qualifier"] = qualifier_for(basis)
    return line_items


def _to_number(v: Any) -> float | None:
    try:
        return float(str(v).replace(",", "").replace("£", "").replace("$", "").replace("€", "").strip())
    except (TypeError, ValueError):
        return None
