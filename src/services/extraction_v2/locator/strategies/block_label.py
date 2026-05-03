"""Block-label locator: find the value on the LINE FOLLOWING a known
label (rather than on the same line).

Use case: name/address fields commonly look like
    Bill To:
    Acme Corp
    123 Main St

The plain LabelAnchoredLocator pulls "Acme" as the next non-whitespace
token after "Bill To:", which is OK for IDs but cannot capture multi-
word names. This locator instead reads the entire next non-empty line.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from src.services.extraction_v2.locator.base import AnchorRef, Locator, LocatorOutput
from src.services.extraction_v2.locator.strategies.label_anchored import LABEL_VOCAB
from src.services.structural_extractor.parsing.model import ParsedDocument


__all__ = ["BlockLabelLocator"]


@dataclass
class BlockLabelLocator(Locator):
    """Reads the line that follows a known label."""

    field: str
    parser: Callable[[str], Optional[object]]
    name: str = ""

    _MIN_CONFIDENCE = 0.78
    _MAX_CONFIDENCE = 0.92

    def __post_init__(self):
        if not self.name:
            self.name = f"block_label:{self.field}"

    def locate(self, doc: ParsedDocument) -> Optional[LocatorOutput]:
        labels = LABEL_VOCAB.get(self.field, [])
        if not labels:
            return None

        text = doc.full_text or ""
        if not text:
            return None

        # Match "label[:]?" at end of a line; capture whole next line(s)
        # up to the first blank line. Multi-word capture is fine since
        # parser validates.
        for label in sorted(labels, key=len, reverse=True):
            pattern = re.compile(
                r"\b" + re.escape(label) + r"\b\s*[:#]?\s*\n\s*([^\n]+)",
                re.IGNORECASE,
            )
            for m in pattern.finditer(text):
                raw = m.group(1).strip().rstrip(".,;:")
                # Reject the line if it is itself another label, a doc-type
                # header, or a date/amount.
                if not raw or _looks_like_another_label(raw):
                    continue
                parsed = self.parser(raw)
                if parsed is None:
                    continue
                conf = min(
                    self._MIN_CONFIDENCE + 0.04 * len(label.split()),
                    self._MAX_CONFIDENCE,
                )
                return LocatorOutput(
                    value=parsed,
                    confidence=conf,
                    evidence=AnchorRef(
                        char_offset=m.start(1),
                        raw_text=raw,
                    ),
                    why=f"block-after label {label!r}",
                )
        return None


_LABEL_TERMINATORS = re.compile(
    r"^\s*("
    r"invoice|bill|due|payment|po|order|quote|valid|delivery|"
    r"subtotal|total|tax|vat|amount|currency|terms|date|number|reference"
    r")\s*[:#]",
    re.IGNORECASE,
)


def _looks_like_another_label(line: str) -> bool:
    return bool(_LABEL_TERMINATORS.match(line))
