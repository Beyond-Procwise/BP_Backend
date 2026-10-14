"""Footer/terms rows must not raise line findings.

Table extraction captures document furniture — 'Commercial terms',
'Payment terms', the supplier footer — as line items. Each then produced a
line_missing_numbers warning (201 of the 276 findings in the
Test Data_300726 session) and, worse, a CRITICAL line_not_on_po claiming
'Commercial terms' is *charged* on the document. A row with no money is not
a charge, and a row that is document furniture is not a line.
"""
from src.services.extraction import two_way_match as twm
from src.services.extraction.two_way_match import is_non_charge_line


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


# ---- Which moneyless rows are flagged --------------------------------------------------
# FSM-Q-9912 as extracted: four priced lines, then its terms block. The old word list knew
# "Commercial terms" and "Payment terms" but not "Lead time / service" or a bullet, so each
# version of each of three quotes raised ~9 warnings — 87 on one deal.
from src.services.extraction.two_way_match import missing_number_lines

_KEYS = {"quantity", "unit_price", "line_amount", "line_total"}
_FSM = [
    {"item_description": "Service Desk (24x7, 3,000 users) — annual", "quantity": 1, "unit_price": 628000, "line_total": 628000},
    {"item_description": "Infrastructure monitoring & patching — annual", "quantity": 1, "unit_price": 388000, "line_total": 388000},
    {"item_description": "Endpoint management (3,000 seats) — annual", "quantity": 1, "unit_price": 294000, "line_total": 294000},
    {"item_description": "On-call / out-of-hours engineering", "quantity": 52, "unit_price": 860, "line_total": 44720},
    {"item_description": "Commercial terms"}, {"item_description": "Payment terms"},
    {"item_description": "Lead time / service"}, {"item_description": "Validity"},
    {"item_description": "Service credit pool (at risk)"}, {"item_description": "Scope & assumptions"},
    {"item_description": "•  3-year MSA."}, {"item_description": "•  SLA P1 15-min response."},
    {"item_description": "Fortis Service Management Group Ltd  ·  FSM-Q-9912  (V1)  ·  Commercial-in-confidence."},
]


def test_terms_block_after_the_priced_lines_is_not_flagged():
    assert missing_number_lines(_FSM, _KEYS) == ([], False)


def test_a_moneyless_row_among_the_priced_lines_is_still_flagged():
    lines = [_FSM[0], {"item_description": "Change management — annual"}, *_FSM[1:]]
    assert missing_number_lines(lines, _KEYS) == ([1], False)


def test_known_furniture_among_the_priced_lines_is_not_flagged():
    lines = [_FSM[0], {"item_description": "Payment terms"}, *_FSM[1:]]
    assert missing_number_lines(lines, _KEYS) == ([], False)


def test_no_priced_line_at_all_is_reported_once_for_the_document():
    lines = [{"item_description": d["item_description"]} for d in _FSM[:4]]
    assert missing_number_lines(lines, _KEYS) == ([], True)
