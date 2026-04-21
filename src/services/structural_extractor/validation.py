"""Validation gates for extracted values: anchor verification, math reconciliation,
cross-field consistency."""
from __future__ import annotations

from dataclasses import dataclass, field

from src.services.structural_extractor.parsing.model import ParsedDocument
from src.services.structural_extractor.types import ExtractedValue


@dataclass
class ValidationReport:
    passed: bool
    failures: list[str] = field(default_factory=list)


def verify_anchors(header: dict, doc: ParsedDocument) -> ValidationReport:
    """Ensure every extracted (non-derived/inferred/lookup) value has an
    anchor that can be traced back to the parsed source."""
    failures: list[str] = []
    norm_full = doc.full_text.lower().replace(" ", "") if doc.full_text else ""
    for field_name, ev in header.items():
        if not isinstance(ev, ExtractedValue):
            continue
        if ev.provenance != "extracted":
            # derived / inferred / lookup: no anchor to verify
            continue
        if ev.anchor_ref is None:
            failures.append(f"{field_name}: no anchor_ref for extracted value")
            continue
        if ev.anchor_text:
            if ev.anchor_text.lower().replace(" ", "") not in norm_full:
                failures.append(
                    f"{field_name}: anchor_text {ev.anchor_text!r} not in doc.full_text"
                )
    return ValidationReport(passed=not failures, failures=failures)
