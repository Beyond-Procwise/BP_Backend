"""Vendor template store.

A vendor template caches the layout-fingerprint → field-locator map for
documents the system has already seen. Once a fingerprint is known and
its fields confirmed (either by high-confidence consensus or by a human
review/correction), subsequent documents with the same fingerprint can
reuse the stored locator hints — committing fields that pure
multi-strategy consensus would have abstained on.

The store is the bridge between the extraction pipeline and the vendor
onboarding UI: corrections from the UI write into the store; the
pipeline reads from it on every extract.

This module ships with an :class:`InMemoryTemplateStore` for tests and
a stable abstract interface (:class:`TemplateStore`) so a Postgres
backend can be plugged in without touching pipeline code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Optional


__all__ = [
    "FieldHint", "VendorTemplate", "TemplateStore", "InMemoryTemplateStore",
]


@dataclass
class FieldHint:
    """A learned hint for one field within a vendor template.

    `value` is the observed/corrected value. `label` (optional) is the
    text label that anchored it. `anchor` (optional) is a serialised
    AnchorRef pointing at where in the doc the value was found — used
    by the TemplateLocator to re-ground in the new doc.
    """
    field: str
    value: Any
    confidence: float = 0.95
    label: Optional[str] = None
    anchor: Optional[dict] = None


@dataclass
class VendorTemplate:
    fingerprint: str
    vendor_name: Optional[str]
    doc_type: str
    field_hints: dict[str, FieldHint] = field(default_factory=dict)
    success_count: int = 0
    correction_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_used_at: Optional[datetime] = None


class TemplateStore(ABC):
    """Abstract store for vendor templates."""

    @abstractmethod
    def get(self, fingerprint: str) -> Optional[VendorTemplate]: ...

    @abstractmethod
    def upsert(self, template: VendorTemplate) -> None: ...

    @abstractmethod
    def record_success(self, fingerprint: str,
                       fields_committed: Iterable[str]) -> None: ...

    @abstractmethod
    def record_correction(self, fingerprint: str, field: str, value: Any,
                          confidence: float = 0.95,
                          label: Optional[str] = None,
                          anchor: Optional[dict] = None,
                          doc_type: Optional[str] = None,
                          vendor_name: Optional[str] = None) -> None: ...


class InMemoryTemplateStore(TemplateStore):
    """Process-local store. Suitable for tests and dev runs."""

    def __init__(self):
        self._templates: dict[str, VendorTemplate] = {}

    def get(self, fingerprint: str) -> Optional[VendorTemplate]:
        return self._templates.get(fingerprint)

    def upsert(self, template: VendorTemplate) -> None:
        self._templates[template.fingerprint] = template

    def record_success(self, fingerprint: str,
                       fields_committed: Iterable[str]) -> None:
        t = self._templates.get(fingerprint)
        if t is None:
            return
        t.success_count += 1
        t.last_used_at = datetime.now(timezone.utc)

    def record_correction(self, fingerprint: str, field: str, value: Any,
                          confidence: float = 0.95,
                          label: Optional[str] = None,
                          anchor: Optional[dict] = None,
                          doc_type: Optional[str] = None,
                          vendor_name: Optional[str] = None) -> None:
        t = self._templates.get(fingerprint)
        if t is None:
            t = VendorTemplate(
                fingerprint=fingerprint,
                vendor_name=vendor_name,
                doc_type=doc_type or "Unknown",
                field_hints={},
            )
            self._templates[fingerprint] = t
        t.field_hints[field] = FieldHint(
            field=field, value=value, confidence=confidence,
            label=label, anchor=anchor,
        )
        t.correction_count += 1
