"""Template locator: replays a vendor-template field hint as a strong vote.

When the document's layout fingerprint matches a stored vendor template
and that template carries a hint for the field, this locator emits the
hint's value as a high-confidence candidate. Coupled with at least one
of the generic locators (label_anchored / structural / filename), it
puts agreement at 2-3 voters and gets the field committed.

The locator is bound to a (template, field) pair at construction. The
pipeline builds these dynamically per-doc once the fingerprint is
known.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.services.extraction_v2.locator.base import AnchorRef, Locator, LocatorOutput
from src.services.extraction_v2.template_store import FieldHint
from src.services.structural_extractor.parsing.model import ParsedDocument


__all__ = ["TemplateLocator"]


@dataclass
class TemplateLocator(Locator):
    """Replays a stored FieldHint as a locator output."""
    field: str
    hint: FieldHint
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = f"template:{self.field}"

    def locate(self, doc: ParsedDocument) -> Optional[LocatorOutput]:
        # The template hint is content-bound — value is what the user
        # corrected. We trust that the same fingerprint = same template,
        # so we replay the value at the hint's confidence.
        return LocatorOutput(
            value=self.hint.value,
            confidence=self.hint.confidence,
            evidence=AnchorRef(
                raw_text=str(self.hint.value),
            ),
            why=f"vendor template hint (label={self.hint.label!r})",
        )
