"""End-to-end regression tests for the recovery layers.

These pin the behaviour observed in production after tonight's fixes:
  - field_recovery + line_recovery work together on real-world doc layouts
  - Mismatched / fabricated values are corrected from postcode evidence
  - The no-fabrication contract holds (NULL when truly absent)
  - Multi-column-table layouts (I-37) flag mismatches but don't fabricate

Run with: pytest tests/extraction_v2/test_recovery_integration.py
"""
from __future__ import annotations

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.services.extraction_v2.field_recovery import recover_fields  # noqa: E402
from src.services.extraction_v2.line_recovery import recover_line_items  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures: real production-like document layouts
# ---------------------------------------------------------------------------

# Layout A: City of Newport PO 502004 — line items AFTER totals,
# unusual delivery anchor. Two "Tier 3 Marketing" lines summing to £100k.
PO_502004_TEXT = """Assurity Ltd
PO Number:
PURCHASE
ORDER
PO Date:
502004
08 Jul 2019
City of Newport Council
45 Riverfront Plaza, Civic Centre
Building, Newport, NP20 4EG,
United Kingdom
DESCRIPTION
SUBTOTAL
10 Redkiln Way Horsham West Sussex RH13 5QH
Recipient:
City of Newport
+44-7955-405-4956
info@assurity.co.uk
10 Redkiln Way Horsham
West Sussex RH13 5QH
Assurity Ltd
Quote Number:
599390
Sub-total:
£100,000
Discount:
£0.00
Tax (20%):
£20,000
Total:
£120,000
Tier 3 Marketing Services
(Months 1-10) 3-5 Posts Per Week
£83,330
Tier 3 Marketing Services
(Months 10-12) 3-5 Posts Per Week
£16,670
"""

# Layout B: NEXASPARK invoice — single line item that EQUALS the subtotal
NEXASPARK_INVOICE_TEXT = """MARKETING
SERVICE
TOTAL
INVOICE NUMBER: 4759276
nexasparkideas.com
info@nexasparkideas.com
+374-586-245
NEXTSPARK DIGITAL MARKETING PACKAGE
(FINAL PAYMENT)
£15,000
BILLED TO
Assurity Ltd
+44-7955-405-4956
info@assurity.co.uk
10 Redkiln Way Horsham
West Sussex RH13 5QH
SUB TOTAL
£15,000
TAX
£3,000
TOTAL
£18,000
DATE: 12 DEC 2025
PO REF: 506789
"""

# Layout C: Quote with "Total" column header BETWEEN description and amount
# (label-tolerance walk required)
QUOTE_599390_TEXT = """Quote.
City of Newport
Quote No :
Quote Date:
QUT599390
July 1, 2019
Quote To :
Assurity Ltd
10 Redkiln Way
Horsham West
Sussex RH13 5QH
Description
Tier 3 Marketing Services
(Months 1-10) 3-5 Posts Per Week
Total
£83,330
£120,000
Subtotal
Tax (20%)
£20,000
Subtotal
£100,000
Tier 3 Marketing Services
(Months 10-12) 3-5 Posts Per Week
£16,670
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_po_502004_layout_recovers_two_line_items():
    """The canonical 'line items after totals' layout. The recovery must
    pull both Tier 3 entries (£83,330 + £16,670 = £100,000)."""
    header = {
        "total_amount": 100000.0, "tax_amount": 20000.0,
        "total_amount_incl_tax": 120000.0,
    }
    r = recover_line_items(header, parsed_text=PO_502004_TEXT,
                            doc_type="Purchase_Order")
    assert r.items_recovered == 2, (
        f"Expected 2 items, got {r.items_recovered}: {r.skipped_reason}"
    )
    amounts = [it["line_amount"] for it in r.items]
    assert 83330.0 in amounts
    assert 16670.0 in amounts


def test_quote_599390_layout_walks_past_total_column_header():
    """Quote with 'Total' label appearing BETWEEN description and amount
    (column header in original PDF, flattened by parser). The label-
    tolerance walk must find the description above and emit 2 items."""
    header = {
        "total_amount": 100000.0, "tax_amount": 20000.0,
        "total_amount_incl_tax": 120000.0,
    }
    r = recover_line_items(header, parsed_text=QUOTE_599390_TEXT,
                            doc_type="Quote")
    assert r.items_recovered == 2, (
        f"Expected 2 items, got {r.items_recovered}: {r.skipped_reason}"
    )


def test_nexaspark_single_item_recovered_without_subtotal_match_rejection():
    """Single-line-item invoice where the line amount equals the subtotal.
    We must NOT blanket-reject 'amount equals subtotal' or we lose the
    only line."""
    header = {
        "invoice_amount": 15000.0, "tax_amount": 3000.0,
        "invoice_total_incl_tax": 18000.0,
    }
    r = recover_line_items(header, parsed_text=NEXASPARK_INVOICE_TEXT,
                            doc_type="Invoice")
    assert r.items_recovered == 1
    assert abs(r.sum_recovered - 15000.0) < 0.01
    assert "MARKETING" in r.items[0]["item_description"].upper()


def test_field_recovery_fills_postcode_when_only_one_in_text():
    """Single-postcode docs are unambiguous → fill postal_code, derive
    region. (Procurement docs with both supplier AND buyer postcodes
    handled by the multi-postcode-NULL test in the main suite.)"""
    h = {}
    text = "Ship To: 1 High St, Cambridge CB2 1FD\n"
    recover_fields(h, parsed_text=text, doc_type="Purchase_Order")
    assert h["postal_code"] == "CB2 1FD"
    assert h["delivery_region"] == "Cambridgeshire"
    assert h["delivery_city"] == "Cambridge"
    assert h["ship_to_country"] == "United Kingdom"


def test_field_recovery_no_overwrite_of_present_postcode():
    """LLM-supplied postal_code wins over text scan. Confirms we don't
    second-guess a non-empty value (the post-revert behaviour)."""
    h = {"postal_code": "RH13 5QH"}
    # Text has a different first postcode (M1 7HY)
    text = "Bill From: ACME, Manchester M1 7HY\nShip To: Assurity, Horsham RH13 5QH\n"
    recover_fields(h, parsed_text=text, doc_type="Purchase_Order")
    assert h["postal_code"] == "RH13 5QH", "must trust LLM's existing postcode"
    assert h["delivery_region"] == "West Sussex"


def test_field_recovery_regions_consistent_with_postcode():
    """Critical no-fabrication invariant: region/country must derive
    from the canonical header.postal_code, never from a text scan that
    could pick up a different postcode."""
    h = {
        "postal_code": "RH13 5QH",
        "delivery_region": "Greater Manchester",  # contradicts!
    }
    # Text contains M1 (Manchester) — the region in header agrees with
    # this text-scan postcode but DISAGREES with header.postal_code.
    text = "Bill From: ACME, Manchester M1 7HY\n"
    recover_fields(h, parsed_text=text, doc_type="Purchase_Order")
    # Region MUST be derived from RH13 (canonical) → West Sussex
    assert h["delivery_region"] == "West Sussex"


def test_no_fabrication_when_no_evidence():
    """Hard contract: no postcode in text + no header.postal_code →
    region/postcode/city stay NULL."""
    h = {}
    text = "Some random invoice text. Total: £100. Thanks for your business."
    recover_fields(h, parsed_text=text, doc_type="Purchase_Order")
    # No address evidence anywhere → all delivery fields NULL
    assert h.get("postal_code") in (None, "")
    assert h.get("delivery_region") in (None, "")
    assert h.get("delivery_city") in (None, "")


def test_line_recovery_refuses_when_sum_doesnt_match():
    """No-fabrication: if recovered amounts don't sum to a plausible
    subtotal, return empty rather than insert wrong data."""
    header = {"total_amount": 1000.0}
    text = (
        "Random Service A\n"
        "£10\n"
    )  # only £10, target £1000 — way off
    r = recover_line_items(header, parsed_text=text,
                            doc_type="Purchase_Order")
    assert r.items_recovered == 0
    assert "sum_mismatch" in (r.skipped_reason or "")


def test_line_recovery_skips_obvious_totals_rows():
    """Ensure 'Sub-total: £100' / 'Tax: £20' / 'Total: £120' lines are
    NOT picked up as line items even when desc walking would find text."""
    text = (
        "Some Real Service\n"
        "£100\n"          # the actual line item
        "Sub-total:\n"
        "£100\n"          # would equal subtotal
        "Tax:\n"
        "£20\n"
        "Total:\n"
        "£120\n"
    )
    header = {"total_amount": 100.0, "tax_amount": 20.0,
              "total_amount_incl_tax": 120.0}
    r = recover_line_items(header, parsed_text=text,
                            doc_type="Purchase_Order")
    assert r.items_recovered == 1
    assert "Real Service" in r.items[0]["item_description"]
