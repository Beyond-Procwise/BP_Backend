"""Format locator: scan the document for any token that matches a known
format pattern for the field type. Useful as a corroborating vote when
labels are missing or ambiguous.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional, Pattern

from src.services.extraction_v2.locator.base import AnchorRef, Locator, LocatorOutput
from src.services.structural_extractor.parsing.model import ParsedDocument


__all__ = ["FormatLocator"]


@dataclass
class FormatLocator(Locator):
    """Find values by raw format match (regex) in the document text."""

    field: str
    pattern: Pattern
    parser: Callable[[str], Optional[object]]
    name: str = ""
    base_confidence: float = 0.55  # format alone is weak signal — needs corroboration

    def __post_init__(self):
        if not self.name:
            self.name = f"format:{self.field}"

    def locate(self, doc: ParsedDocument) -> Optional[LocatorOutput]:
        text = doc.full_text or ""
        if not text:
            return None
        m = self.pattern.search(text)
        if not m:
            return None
        raw = m.group(0)
        parsed = self.parser(raw)
        if parsed is None:
            return None
        return LocatorOutput(
            value=parsed,
            confidence=self.base_confidence,
            evidence=AnchorRef(char_offset=m.start(), raw_text=raw),
            why=f"format match {self.pattern.pattern!r}",
        )


# Common format patterns — used by the registry to wire up format locators
INVOICE_ID_PATTERN = re.compile(r"\bINV[\-]?[\w\-]+\b", re.I)
PO_ID_PATTERN      = re.compile(r"\bPO[\-]?\d{4,}\b", re.I)
QUOTE_ID_PATTERN   = re.compile(r"\b(?:QUT|QTE|QUOTE)[\-]?[\w\-]+\b", re.I)
ISO_DATE_PATTERN   = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
UK_DATE_PATTERN    = re.compile(r"\b\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}\b")
UK_POSTCODE_PATTERN = re.compile(
    r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b", re.I,
)
