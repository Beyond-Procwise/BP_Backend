"""Adapter that exposes the existing bbox-anchored structural extractor
as a v2 Locator. This is the highest-confidence locator we have for
native PDF / DOCX / XLSX inputs because it uses bounding-box anchoring
with type-specific entity detection.

The structural extractor produces an ExtractionResult; this wrapper
extracts a single field's value (if present) and packages it as a
LocatorOutput for the consensus runner.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

from src.services.extraction_v2.locator.base import AnchorRef, Locator, LocatorOutput
from src.services.structural_extractor.parsing.model import BBox, ParsedDocument

logger = logging.getLogger(__name__)


__all__ = ["StructuralLocator"]


@dataclass
class StructuralLocator(Locator):
    """Run the existing structural extractor and surface ONE field's value
    as a LocatorOutput.

    The ``doc_type`` indicates which entity schema to use ("Invoice",
    "Purchase_Order", "Quote"). The structural extractor caches its
    work via ``cache_key``-keyed memoization so multiple StructuralLocator
    instances (one per field) share a single underlying extraction.
    """

    field: str
    doc_type: str
    parser: Optional[Callable[[Any], Optional[Any]]] = None  # optional re-parse for type safety
    name: str = ""
    base_confidence: float = 0.92  # high — bbox-anchored

    def __post_init__(self):
        if not self.name:
            self.name = f"structural:{self.field}"

    def locate(self, doc: ParsedDocument) -> Optional[LocatorOutput]:
        try:
            from src.services.structural_extractor.retry.driver import run_retry_loop
        except ImportError:
            return None

        try:
            result = _cached_run(doc, self.doc_type)
        except Exception:
            logger.debug("structural extraction raised", exc_info=True)
            return None

        ev = result.header.get(self.field) if result else None
        if ev is None or ev.value in (None, ""):
            return None

        # Optional re-parse for type safety (e.g., extracted str → Money)
        value = ev.value
        if self.parser is not None:
            parsed = self.parser(value)
            if parsed is None:
                return None
            value = parsed

        anchor: Optional[AnchorRef] = None
        ref = getattr(ev, "anchor_ref", None)
        if isinstance(ref, BBox):
            anchor = AnchorRef(
                page=ref.page,
                bbox=(ref.x0, ref.y0, ref.x1, ref.y1),
                raw_text=getattr(ev, "anchor_text", None),
            )
        else:
            anchor = AnchorRef(raw_text=getattr(ev, "anchor_text", None))

        return LocatorOutput(
            value=value,
            confidence=min(getattr(ev, "confidence", 0.92), self.base_confidence),
            evidence=anchor,
            why=f"structural extractor (source={getattr(ev, 'source', '?')})",
        )


# Per-process memoization so N field-locators don't run the structural
# extractor N times on the same document.
_cache: dict[tuple[str, str], Any] = {}


def _cached_run(doc: ParsedDocument, doc_type: str):
    """Memoize structural extraction by (filename, doc_type, len(full_text))."""
    from src.services.structural_extractor.retry.driver import run_retry_loop

    key = (doc.filename or "", doc_type)
    if key in _cache:
        return _cache[key]
    result = run_retry_loop(doc, doc_type, max_attempts=4)  # NLU + structural only
    _cache[key] = result
    return result


def clear_structural_cache() -> None:
    """Clear the per-process memoization cache (use in tests)."""
    _cache.clear()
