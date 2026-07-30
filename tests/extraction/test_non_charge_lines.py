"""Footer/terms rows must not raise line findings.

Table extraction captures document furniture — 'Commercial terms',
'Payment terms', the supplier footer — as line items. Each then produced a
line_missing_numbers warning (201 of the 276 findings in the
Test Data_300726 session) and, worse, a CRITICAL line_not_on_po claiming
'Commercial terms' is *charged* on the document. A row with no money is not
a charge, and a row that is document furniture is not a line.
"""
from src.services.extraction import three_way_match as twm
from src.services.extraction.three_way_match import is_non_charge_line


def test_terms_and_footer_rows_are_non_charge():
    assert is_non_charge_line("Commercial terms")
    assert is_non_charge_line("Payment terms")
    assert is_non_charge_line("Price validity")
    assert is_non_charge_line("Term")
    assert is_non_charge_line(
        "Orbis Platform Solutions Ltd  ·  ORB-INV-9901  ·  Commercial-in-confidence")


def test_real_charge_descriptions_are_not_non_charge():
    assert not is_non_charge_line("Fuel surcharge @5.0%")
    assert not is_non_charge_line("Platform licence — Enterprise (240 seats)")
    assert not is_non_charge_line("Service Desk — quarterly")
    assert not is_non_charge_line("")


def _findings(monkeypatch, line_items):
    po = {"po_id": "PO-1", "total_amount": "1000", "currency": "GBP"}
    po_lines = [{"line_number": 1, "item_description": "Widgets",
                 "quantity": 1, "unit_price": 1000, "line_total": 1000}]
    monkeypatch.setattr(twm, "_load_po", lambda po_id: (po, po_lines))
    return twm.check_against_po(
        "invoice", {"po_id": "PO-1", "invoice_amount": "500"}, line_items)


def test_moneyless_unmatched_line_is_not_reported_as_charged(monkeypatch):
    out = _findings(monkeypatch, [
        {"item_description": "Commercial terms"},          # furniture, no money
        {"item_description": "Something unrecognised"},    # no money either
    ])
    assert not any(d.issue_type == "line_not_on_po" for d in out)


def test_unmatched_line_with_money_is_still_flagged(monkeypatch):
    out = _findings(monkeypatch, [
        {"item_description": "Fuel surcharge @5.0%", "line_amount": "50.00"},
    ])
    flagged = [d for d in out if d.issue_type == "line_not_on_po"]
    assert len(flagged) == 1
    assert "Fuel surcharge" in flagged[0].raw_value
