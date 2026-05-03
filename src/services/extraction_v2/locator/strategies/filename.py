"""Filename locator: extract the field's value from the source filename.

For PO/Quote/Invoice IDs, filenames in the convention
"{SUPPLIER} PO{num} for QUT{num}.pdf" carry the IDs reliably. Useful
as a fallback when the body extractor fails or as a corroborating vote.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Callable, Optional, Pattern

from src.services.extraction_v2.locator.base import AnchorRef, Locator, LocatorOutput
from src.services.structural_extractor.parsing.model import ParsedDocument


__all__ = ["FilenameLocator"]


@dataclass
class FilenameLocator(Locator):
    """Extract a value from the document filename via a regex pattern."""

    field: str
    pattern: Pattern
    parser: Callable[[str], Optional[object]]
    name: str = ""
    base_confidence: float = 0.75   # filenames are moderately reliable

    def __post_init__(self):
        if not self.name:
            self.name = f"filename:{self.field}"

    def locate(self, doc: ParsedDocument) -> Optional[LocatorOutput]:
        filename = os.path.basename(doc.filename or "")
        if not filename:
            return None
        m = self.pattern.search(filename)
        if not m:
            return None
        raw = m.group(0)
        parsed = self.parser(raw)
        if parsed is None:
            return None
        return LocatorOutput(
            value=parsed,
            confidence=self.base_confidence,
            evidence=AnchorRef(raw_text=raw),
            why=f"filename match {self.pattern.pattern!r}",
        )
