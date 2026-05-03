"""Tests for the vendor template store."""
from __future__ import annotations

from src.services.extraction_v2.template_store import (
    InMemoryTemplateStore, VendorTemplate, FieldHint,
)


class TestVendorTemplateBasics:
    def test_field_hint_defaults(self):
        h = FieldHint(field="invoice_id", value="INV-X", confidence=0.95)
        assert h.field == "invoice_id"
        assert h.value == "INV-X"
        assert h.confidence == 0.95
        assert h.label is None
        assert h.anchor is None

    def test_template_round_trip(self):
        t = VendorTemplate(
            fingerprint="abc123",
            vendor_name="Acme",
            doc_type="Invoice",
            field_hints={"invoice_id": FieldHint("invoice_id", "INV-X", 0.95)},
        )
        assert t.fingerprint == "abc123"
        assert t.field_hints["invoice_id"].value == "INV-X"


class TestInMemoryTemplateStore:
    def test_get_unknown_returns_none(self):
        store = InMemoryTemplateStore()
        assert store.get("nope") is None

    def test_upsert_then_get(self):
        store = InMemoryTemplateStore()
        t = VendorTemplate(
            fingerprint="fp1", vendor_name="V", doc_type="Invoice",
            field_hints={"invoice_id": FieldHint("invoice_id", "X", 0.95)},
        )
        store.upsert(t)
        got = store.get("fp1")
        assert got is not None
        assert got.vendor_name == "V"

    def test_upsert_overwrites_existing(self):
        store = InMemoryTemplateStore()
        store.upsert(VendorTemplate(
            fingerprint="fp", vendor_name="A", doc_type="Invoice",
            field_hints={},
        ))
        store.upsert(VendorTemplate(
            fingerprint="fp", vendor_name="B", doc_type="Invoice",
            field_hints={},
        ))
        assert store.get("fp").vendor_name == "B"

    def test_record_success_increments_count(self):
        store = InMemoryTemplateStore()
        store.upsert(VendorTemplate(
            fingerprint="fp", vendor_name="A", doc_type="Invoice",
            field_hints={}, success_count=0,
        ))
        store.record_success("fp", fields_committed=("invoice_id", "tax_amount"))
        assert store.get("fp").success_count == 1

    def test_record_success_unknown_fingerprint_is_noop(self):
        store = InMemoryTemplateStore()
        # Must not raise
        store.record_success("missing", fields_committed=())
        assert store.get("missing") is None

    def test_record_correction_adds_field_hint(self):
        store = InMemoryTemplateStore()
        store.upsert(VendorTemplate(
            fingerprint="fp", vendor_name="A", doc_type="Invoice",
            field_hints={},
        ))
        store.record_correction(
            "fp", field="supplier_name", value="Acme Ltd",
            label="From:", confidence=0.95,
        )
        t = store.get("fp")
        assert "supplier_name" in t.field_hints
        assert t.field_hints["supplier_name"].value == "Acme Ltd"
        assert t.field_hints["supplier_name"].label == "From:"

    def test_record_correction_creates_template_if_missing(self):
        store = InMemoryTemplateStore()
        store.record_correction(
            "newfp", field="invoice_id", value="X", confidence=0.95,
            doc_type="Invoice",
        )
        t = store.get("newfp")
        assert t is not None
        assert t.field_hints["invoice_id"].value == "X"
