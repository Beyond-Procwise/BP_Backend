"""Header locator: scans the top of the document for a supplier name.

The supplier (the issuer of an invoice/quote/PO) is usually the first
significant non-label, non-doc-type line at the document head. This
locator captures that line when it looks like a company name.

Confidence is moderate (0.70) because the heuristic can fire on a buyer
when the layout puts the buyer first. The consensus runner is
responsible for cross-checking against other supplier locators.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from src.services.extraction_v2.locator.base import AnchorRef, Locator, LocatorOutput
from src.services.structural_extractor.parsing.model import ParsedDocument


__all__ = ["HeaderLocator"]


# Stop scanning once we hit one of these doc-type words — anything
# meaningful for the supplier comes BEFORE the document body.
_DOC_TYPE_WORDS = re.compile(
    r"^\s*(invoice|purchase\s*order|p\.?o\.?|quote|quotation|tax\s+invoice|bill)\s*\.?\s*$",
    re.IGNORECASE,
)

# Skip lines that look like labels, headers, or buyer-section markers.
_SKIP_LINE = re.compile(
    r"^\s*("
    r"item|description|qty|quantity|price|amount|total|subtotal|tax|vat|"
    r"invoice|bill|due|payment|po|order|quote|valid|delivery|customer|"
    r"date|number|reference|ship\s*to|bill\s*to|invoice\s*to|to|from|"
    r"vendor|supplier"
    r")\b",
    re.IGNORECASE,
)

# Reject lines that are pure numbers, addresses (postcodes), or money.
_REJECT_VALUE = re.compile(
    r"^\s*("
    r"[\d\W]+|"                                # mostly digits/punct
    r".*\b\d{4,}\b.*|"                         # contains a long number
    r".*[£$€¥₹]\s*\d.*|"                       # contains a money symbol
    r"[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}.*"     # UK postcode
    r")$",
    re.IGNORECASE,
)


@dataclass
class HeaderLocator(Locator):
    """Find the supplier name by scanning the document header."""

    field: str
    parser: Callable[[str], Optional[object]]
    name: str = ""
    max_lines: int = 8                         # how far down to scan

    def __post_init__(self):
        if not self.name:
            self.name = f"header:{self.field}"

    def locate(self, doc: ParsedDocument) -> Optional[LocatorOutput]:
        text = doc.full_text or ""
        if not text:
            return None

        for raw_line in text.split("\n")[: self.max_lines]:
            line = raw_line.strip().rstrip(".,;:")
            if not line:
                continue
            if _DOC_TYPE_WORDS.match(line):
                # Doc-type marker (e.g. "Invoice.") — skip but keep scanning.
                continue
            if _SKIP_LINE.match(line):
                continue
            if _REJECT_VALUE.match(line):
                continue
            if len(line) < 3 or len(line) > 80:
                continue
            parsed = self.parser(line)
            if parsed is None:
                continue
            return LocatorOutput(
                value=parsed,
                confidence=0.70,
                evidence=AnchorRef(raw_text=line),
                why="document header scan",
            )
        return None
