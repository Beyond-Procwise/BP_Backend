"""Tests for the line-items recovery fallback."""
from __future__ import annotations

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.services.extraction_v2.line_recovery import (  # noqa: E402
    LineRecoveryReport, recover_line_items,
)


def test_recovers_two_items_when_sum_matches_subtotal():
    """The PO 502004-style layout: line items appear AFTER totals.
    Description lines + amount, summing to the header subtotal."""
    text = (
        "Sub-total: £100,000\n"
        "Tax (20%): £20,000\n"
        "Total: £120,000\n"
        "Tier 3 Marketing Services\n"
        "(Months 1-10) 3-5 Posts Per Week\n"
        "£83,330\n"
        "Tier 3 Marketing Services\n"
        "(Months 10-12) 3-5 Posts Per Week\n"
        "£16,670\n"
    )
    header = {"total_amount": 100000.0, "tax_amount": 20000.0,
              "total_amount_incl_tax": 120000.0}
    r = recover_line_items(header, parsed_text=text, doc_type="Purchase_Order")
    assert isinstance(r, LineRecoveryReport)
    assert r.items_recovered == 2
    assert abs(r.sum_recovered - 100000.0) < 0.01
    assert r.skipped_reason is None
    descs = [it["item_description"] for it in r.items]
    assert any("Months 1-10" in d for d in descs)
    assert any("Months 10-12" in d for d in descs)
    amounts = [it["line_amount"] for it in r.items]
    assert 83330.0 in amounts
    assert 16670.0 in amounts


def test_recovers_single_item_when_sum_matches_subtotal():
    """NEXASPARK-style: one line item matching the subtotal."""
    text = (
        "INVOICE NUMBER: 4759276\n"
        "NEXTSPARK DIGITAL MARKETING PACKAGE\n"
        "(FINAL PAYMENT)\n"
        "£15,000\n"
        "BILLED TO\n"
        "Assurity Ltd\n"
        "SUB TOTAL\n"
        "£15,000\n"
        "TAX\n"
        "£3,000\n"
        "TOTAL\n"
        "£18,000\n"
    )
    header = {"invoice_amount": 15000.0, "tax_amount": 3000.0,
              "invoice_total_incl_tax": 18000.0}
    r = recover_line_items(header, parsed_text=text, doc_type="Invoice")
    assert r.items_recovered == 1
    assert abs(r.sum_recovered - 15000.0) < 0.01
    assert "MARKETING" in r.items[0]["item_description"].upper()
    assert r.items[0]["line_amount"] == 15000.0


def test_skips_when_sum_does_not_match():
    """If recovered amounts don't sum to a plausible subtotal, return
    empty — better NULL than wrong."""
    text = (
        "Sub-total: £100,000\n"
        "Total: £120,000\n"
        "Random Service A\n"
        "£50\n"  # tiny amount, won't sum to 100k
    )
    header = {"total_amount": 100000.0}
    r = recover_line_items(header, parsed_text=text, doc_type="Purchase_Order")
    assert r.items_recovered == 0
    assert r.skipped_reason and "sum_mismatch" in r.skipped_reason


def test_skips_when_no_target_total():
    """If the header has no subtotal/total to verify against, skip."""
    text = (
        "Service A\n"
        "£100\n"
    )
    header = {}  # no totals
    r = recover_line_items(header, parsed_text=text, doc_type="Purchase_Order")
    assert r.items_recovered == 0
    assert r.skipped_reason == "no_target_total"


def test_skips_total_label_lines_as_descriptions():
    """An amount line preceded by 'Sub-total:' / 'Tax:' / 'Total:' must
    NOT be treated as a line-item amount."""
    text = (
        "Sub-total:\n"
        "£100\n"  # this is the subtotal, not a line item
        "Tax:\n"
        "£20\n"
        "Total:\n"
        "£120\n"
        "Some Real Service\n"
        "£100\n"  # actual line item
    )
    header = {"total_amount": 100.0}
    r = recover_line_items(header, parsed_text=text, doc_type="Purchase_Order")
    assert r.items_recovered == 1
    assert "Real Service" in r.items[0]["item_description"]


def test_no_parsed_text_returns_empty():
    r = recover_line_items({"total_amount": 100.0}, parsed_text="",
                            doc_type="Purchase_Order")
    assert r.items_recovered == 0
    assert r.skipped_reason == "no_parsed_text"


def test_amount_matching_total_excluded():
    """An amount line that matches the header subtotal exactly is the
    subtotal itself, not a line item — exclude it."""
    text = (
        "£100,000\n"  # this looks like an amount but is the subtotal
        "Sub-total: £100,000\n"
    )
    header = {"total_amount": 100000.0}
    r = recover_line_items(header, parsed_text=text, doc_type="Purchase_Order")
    assert r.items_recovered == 0
