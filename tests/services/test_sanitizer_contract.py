"""Contract tests: end-to-end behaviors that the sanitizer + orchestrator
path must guarantee, based on real production failures. These run fast
(pure Python, no DB or LLM) and serve as a regression harness — if any
of these break, we have reintroduced a known data-quality bug.
"""
import pytest

from services.extraction_sanitizer import (
    ExtractionSanitizer,
    canonical_payment_terms,
    is_garbage_party_name,
    looks_like_address,
    looks_like_doc_id,
)


# ----------------------------------------------------------------------
# Bugs we refuse to bring back
# ----------------------------------------------------------------------

class TestRegressionGuards:
    """Each test corresponds to a specific production bug. Names reference
    the bug report (commit history / conversation log) for context.
    """

    def setup_method(self):
        self.s = ExtractionSanitizer()

    def test_BUG_invoice_label_as_supplier(self):
        """Observed: 'Invoice', 'Property Invoice' stored as supplier_id."""
        h, _, r = self.s.sanitize(
            {"invoice_id": "X", "supplier_id": "Property Invoice"},
            [], "Invoice",
        )
        assert h["supplier_id"] is None
        assert any(x.field == "supplier_id" for x in r)

    def test_BUG_filename_as_supplier(self):
        """Observed: 'Quote_Wsg100025_Watermark_Split Tables' as supplier."""
        for bad in (
            "Quote_Wsg100025_Watermark_Split Tables",
            "Po5205561_Watermark_Less Items_No Vat_Duplicate",
            "Quote_Scenario_",
            "Office Clean_",
        ):
            h, _, _ = self.s.sanitize(
                {"invoice_id": "X", "supplier_id": bad}, [], "Invoice",
            )
            assert h["supplier_id"] is None, f"filename not rejected: {bad!r}"

    def test_BUG_INVOICENO_label_mash(self):
        """Observed: 'Duncan LLC INVOICENO: 132666' → 'SUP-DuncanLLCINVOICENO13'."""
        h, _, _ = self.s.sanitize(
            {"invoice_id": "X", "supplier_id": "INVOICENO: 132666"},
            [], "Invoice",
        )
        assert h["supplier_id"] is None

    def test_BUG_po_number_as_buyer(self):
        """Observed: buyer_id='PO526809' on INV304056."""
        for bad in ("PO526809", "508084", "438295"):
            h, _, _ = self.s.sanitize(
                {"invoice_id": "X", "buyer_id": bad}, [], "Invoice",
            )
            assert h["buyer_id"] is None, f"PO-as-buyer not rejected: {bad!r}"

    def test_BUG_address_as_buyer(self):
        """Observed: buyer_id='10 Redkiln Way Horsham RH13 5QH'."""
        for bad in (
            "10 Redkiln Way Horsham RH13 5QH",
            "10 Redkiln Way Horsham",
            "RH13 5QH",
            "3rd Floor, Regent House, Birmingham",
        ):
            h, _, _ = self.s.sanitize(
                {"invoice_id": "X", "buyer_id": bad}, [], "Invoice",
            )
            assert h["buyer_id"] is None, f"address not rejected: {bad!r}"

    def test_BUG_person_with_title_as_buyer(self):
        """Observed: buyer_id='Dana Parker DDS' on INV610366."""
        h, _, _ = self.s.sanitize(
            {"invoice_id": "X", "buyer_id": "Dana Parker DDS"}, [], "Invoice",
        )
        assert h["buyer_id"] is None

    def test_BUG_digits_only_payment_terms(self):
        """Observed: payment_terms='90', '236', '&', 'Payments'."""
        for bad in ("90", "236", "&", "Payments", "Full", "Net", "3"):
            h, _, _ = self.s.sanitize(
                {"invoice_id": "X", "payment_terms": bad}, [], "Invoice",
            )
            assert h["payment_terms"] is None, f"garbage terms accepted: {bad!r}"

    def test_BUG_tax_equals_subtotal(self):
        """Observed: INV-005-39 had tax=subtotal=675 stored as valid."""
        h, _, r = self.s.sanitize(
            {"invoice_id": "X", "invoice_amount": 675,
             "tax_amount": 675, "tax_percent": 100,
             "invoice_total_incl_tax": 1350},
            [], "Invoice",
        )
        assert h["tax_amount"] is None
        assert h["tax_percent"] is None
        # And it's logged so humans can see it
        assert any(x.field == "tax_amount" and x.severity == "error" for x in r)

    def test_BUG_far_future_date(self):
        """Observed: some docs stored invoice_date=2099-12-31 from OCR errors."""
        h, _, _ = self.s.sanitize(
            {"invoice_id": "X", "invoice_date": "2099-12-31"}, [], "Invoice",
        )
        assert h["invoice_date"] is None

    def test_BUG_po_id_self_reference(self):
        """Observed: my manual insert had po_id=invoice_id on INV-2026-01602."""
        h, _, _ = self.s.sanitize(
            {"invoice_id": "INV-2026-01602", "po_id": "INV-2026-01602"},
            [], "Invoice",
        )
        assert h["po_id"] is None

    def test_BUG_bare_numeric_po_normalized(self):
        """Observed: po_id='526809' (missing PO prefix) — now auto-fixed."""
        h, _, _ = self.s.sanitize(
            {"invoice_id": "X", "po_id": "526809"}, [], "Invoice",
        )
        assert h["po_id"] == "PO526809"

    def test_BUG_truncated_po_id_kept_for_review(self):
        """Observed: 'PO40586' was a truncated 'PO405867'. We accept it
        (it's validly-formatted as a PO) — the discrepancy is elsewhere."""
        h, _, _ = self.s.sanitize(
            {"invoice_id": "X", "po_id": "PO40586"}, [], "Invoice",
        )
        assert h["po_id"] == "PO40586"  # format-valid; downstream review


# ----------------------------------------------------------------------
# Known-good documents must pass through untouched
# ----------------------------------------------------------------------

class TestDoesNotCorruptCleanData:
    """The sanitizer must be a zero-op on well-formed records."""

    def setup_method(self):
        self.s = ExtractionSanitizer()

    @pytest.mark.parametrize("good_invoice", [
        {
            "invoice_id": "INV600820", "po_id": "PO507222",
            "supplier_id": "SUP-DixonReynoldsandSolomon",
            "buyer_id": "SUP-AssurityLtd",
            "invoice_date": "2024-03-15", "due_date": "2024-04-14",
            "payment_terms": "Net 30", "currency": "GBP",
            "invoice_amount": 1170.64, "tax_amount": 234.13,
            "invoice_total_incl_tax": 1404.77,
            "country": "United Kingdom", "region": "West Sussex",
        },
        {
            "invoice_id": "INV-2026-01602",
            "supplier_id": "SUP-PeopleFirstHRSolutions",
            "buyer_id": "SUP-HorizonRetailGroup",
            "invoice_date": "2025-10-10", "due_date": "2025-11-09",
            "payment_terms": "Due on Receipt", "currency": "GBP",
            "invoice_amount": 100000, "tax_amount": 20000,
            "invoice_total_incl_tax": 120000,
            "country": "United Kingdom", "region": "West Midlands",
        },
    ])
    def test_clean_invoice_untouched(self, good_invoice):
        h, _, r = self.s.sanitize(good_invoice, [], "Invoice")
        assert h == good_invoice, f"clean invoice mutated: diff={set(h.items())^set(good_invoice.items())}"
        assert r == []

    def test_clean_po_untouched(self):
        good_po = {
            "po_id": "PO526702", "supplier_id": "SUP-DuncanLLC",
            "buyer_id": "SUP-AssurityLtd",
            "order_date": "2024-09-28", "currency": "GBP",
            "total_amount": 2131.05, "tax_amount": 426.21,
            "tax_percent": 20, "total_amount_incl_tax": 2557.26,
            "payment_terms": "Net 30",
            "ship_to_country": "United Kingdom",
        }
        h, _, r = self.s.sanitize(good_po, [], "Purchase_Order")
        assert h == good_po
        assert r == []


# ----------------------------------------------------------------------
# Payment-terms normalization produces canonical strings
# ----------------------------------------------------------------------

class TestPaymentTermsCanonical:
    @pytest.mark.parametrize("raw,expected", [
        ("Net 30", "Net 30"),
        ("net 30", "Net 30"),
        ("NET 30", "Net 30"),
        ("Net30", "Net 30"),
        ("Net  30", "Net 30"),
        ("Net 30 days", "Net 30"),
        ("30 days net", "Net 30"),
        ("within 30 days", "Net 30"),
        ("Payment due within 30 days of receiving the invoice.", "Net 30"),
        ("Due on Receipt", "Due on Receipt"),
        ("due on receipt", "Due on Receipt"),
        ("Upon Receipt", "Due on Receipt"),
        ("upon receipt", "Due on Receipt"),
        ("On Receipt", "Due on Receipt"),
        ("COD", "COD"),
        ("Cash on Delivery", "COD"),
    ])
    def test_canonicalizes(self, raw, expected):
        assert canonical_payment_terms(raw) == expected
