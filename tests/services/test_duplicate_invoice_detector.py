"""The duplicate-invoice rule.

Duplication is scored by the SAME relationship math that links an invoice to its PO and
clusters rival quotes (linking_engine.score_link, profile `invoice_duplicate`), not by a
bespoke boolean rule. So these tests assert two things: that the two new invoice-to-invoice
signals grade evidence correctly, and that the engine's bands land real pairs where they
belong — a re-issued invoice above the raise band, a recurring charge nowhere near it.

The cost of a false positive is high: accusing a supplier of double-billing a monthly
retainer damages a relationship over nothing. Everything below is calibration against that.
"""

from datetime import datetime, timezone, timedelta

from src.services.duplicate_invoice_detector import (
    CERTAIN_BAND, RAISE_BAND, cmp_date_prox, cmp_ref_prox, find_duplicates, score_pair,
)

LINES = [{"item_id": "A", "item_description": "Widget", "quantity": 2,
          "unit_price": 10, "line_amount": 20}]


def _inv(iid, supplier="Techworld", total=1000.0, po="PO-1", days_ago=0, ref=None,
         lines=None):
    return {"invoice_id": iid, "supplier_name": supplier, "total_amount": total,
            "po_id": po, "invoice_ref": ref or iid, "currency": "GBP",
            "lines": LINES if lines is None else lines,
            "invoice_date": datetime.now(timezone.utc) - timedelta(days=days_ago)}


# ---- the rule ------------------------------------------------------------

def test_same_supplier_total_and_po_within_window_flags_later():
    dups = find_duplicates([_inv("INV-A", days_ago=30), _inv("INV-B", days_ago=1)])
    assert len(dups) == 1
    assert dups[0]["later"]["invoice_id"] == "INV-B"
    assert dups[0]["amount"] == 1000.0
    assert dups[0]["score"] >= RAISE_BAND


def test_a_reissued_invoice_is_the_engine_s_strongest_case():
    # Same reference, same day, same lines — nothing else it could be.
    dups = find_duplicates([_inv("INV-9", lines=LINES),
                            _inv("INV-9-DUP", ref="INV-9", lines=LINES)])
    assert len(dups) == 1
    assert dups[0]["score"] >= CERTAIN_BAND
    assert dups[0]["band"] == "auto_link"


def test_a_numbered_series_on_one_po_is_not_a_duplicate():
    # Measured against the live corpus on 2026-07-31: same supplier + same total + same PO
    # flagged 4,974 of 12,408 invoices, and every one was a member of an INV<n>-1/-2/-3
    # series billed against one PO — identical amounts AND identical line items, so no
    # signal but the reference can tell them apart. The reference is what settles it: the
    # documents number themselves as different members of a set. A Tier-1 conflict, so the
    # engine's cap holds the pair at 60 however perfectly everything else agrees.
    series = [_inv("INV000469-1", po="PO000469", total=2663.12, days_ago=30),
              _inv("INV000469-2", po="PO000469", total=2663.12, days_ago=28),
              _inv("INV000469-3", po="PO000469", total=2663.12, days_ago=27)]
    assert find_duplicates(series) == []
    link = score_pair(series[0], series[1])
    assert link["F_cap"] == 0.60
    assert link["decision"] == "weak_relation"


def test_near_identical_ref_counts_even_across_pos():
    a, b = _inv("INV-100", po="PO-1"), _inv("INV-100A", po="PO-2")
    assert find_duplicates([a, b])
    assert cmp_ref_prox("INV-100", "INV-100A")[1] == "OK"
    # Two differently-numbered references from one numbering scheme are two documents.
    assert cmp_ref_prox("INV-100", "INV-200")[1] == "CONFLICT"


def test_recurring_charge_not_flagged():
    # same supplier + same amount but different PO and unrelated refs (monthly fee)
    a = _inv("INV-JAN", po="PO-1", ref="SVC-JAN", days_ago=60, lines=[])
    b = _inv("INV-FEB", po="PO-2", ref="SVC-FEB", days_ago=30, lines=[])
    assert find_duplicates([a, b]) == []
    assert score_pair(a, b)["decision"] == "block_or_exception"


def test_outside_90_days_not_flagged():
    # Even on an identical reference: months apart is a repeat purchase. date_prox is a
    # Tier-1 conflict there, and its cap (0.55) holds the pair below the raise band.
    far = [_inv("INV-A", days_ago=125), _inv("INV-B", ref="INV-A", days_ago=1)]
    assert find_duplicates(far) == []
    assert score_pair(far[0], far[1])["F_cap"] == 0.55


def test_different_supplier_or_amount_not_flagged():
    assert find_duplicates([_inv("INV-A"), _inv("INV-B", supplier="Acme")]) == []
    assert find_duplicates([_inv("INV-A"), _inv("INV-B", total=999.99)]) == []


# ---- the rule's edges ----------------------------------------------------

def test_supplier_is_matched_on_a_normalised_name():
    # "Techworld Ltd." and "TECHWORLD LTD" are the same company billing twice.
    a = _inv("INV-A", supplier="Techworld Ltd.", days_ago=10)
    b = _inv("INV-B", supplier="  TECHWORLD  LTD ", days_ago=1)
    assert len(find_duplicates([a, b])) == 1


def test_a_missing_amount_or_supplier_is_never_a_duplicate():
    # Absent data is not agreement. Two invoices with no total are not "the same total".
    assert find_duplicates([_inv("INV-A", total=None), _inv("INV-B", total=None)]) == []
    assert find_duplicates([_inv("INV-A", supplier=None), _inv("INV-B", supplier=None)]) == []


def test_a_missing_date_is_never_compared():
    a, b = _inv("INV-A"), _inv("INV-B")
    a["invoice_date"] = None
    assert find_duplicates([a, b]) == []
    assert cmp_date_prox(None, b["invoice_date"]) == (0.5, "MISSING")


def test_totals_agree_to_the_penny_not_approximately():
    # 2-dp equality: 1000.004 and 1000.001 are the same billed amount; 1000.01 is not.
    assert find_duplicates([_inv("INV-A", total=1000.004), _inv("INV-B", total=1000.001)])
    assert find_duplicates([_inv("INV-A", total=1000.00), _inv("INV-B", total=1000.01)]) == []


def test_credit_notes_are_not_duplicate_invoices():
    # The live corpus carries credit notes as negative-total rows (…-CN). Two of them
    # agreeing is money coming BACK, not money paid twice — this detector's whole claim is
    # "you may have paid this twice", so it must stay silent on anything not billed.
    a = _inv("CN-ALPHA", total=-1321.06, days_ago=30)
    b = _inv("CN-BETA", total=-1321.06, days_ago=1)
    assert find_duplicates([a, b]) == []
    assert find_duplicates([_inv("INV-A", total=0.0), _inv("INV-B", total=0.0)]) == []


def test_the_same_invoice_row_twice_is_not_a_duplicate_of_itself():
    a = _inv("INV-A")
    assert find_duplicates([a, dict(a)]) == []


def test_the_later_invoice_is_the_one_flagged_regardless_of_input_order():
    later = _inv("INV-LATE", days_ago=1)
    earlier = _inv("INV-EARLY", days_ago=40)
    for order in ([later, earlier], [earlier, later]):
        dup = find_duplicates(order)[0]
        assert dup["later"]["invoice_id"] == "INV-LATE"
        assert dup["earlier"]["invoice_id"] == "INV-EARLY"


def test_one_invoice_is_reported_once_against_its_strongest_match():
    # Three identical invoices raise two findings, not three overlapping pairs — and each is
    # paired with the invoice it most strongly duplicates, which is the NEAREST in time, not
    # the earliest: the closer two identical bills sit, the more they look like one bill.
    rows = [_inv("INV-A", days_ago=40), _inv("INV-B", days_ago=20), _inv("INV-C", days_ago=1)]
    dups = find_duplicates(rows)
    paired = {d["later"]["invoice_id"]: d["earlier"]["invoice_id"] for d in dups}
    assert paired == {"INV-B": "INV-A", "INV-C": "INV-B"}


def test_ref_prox_grades_an_insertion_apart_from_a_substitution():
    assert cmp_ref_prox("INV-100", "INV-100")[0] == 1.0       # identical: strongest
    assert cmp_ref_prox("INV-100", "INV-100A")[1] == "OK"    # a re-issue marker appended
    assert cmp_ref_prox("INV-9", "INV-9-DUP")[1] == "MISSING"  # >1 edit; other signals decide
    assert cmp_ref_prox("INV-100", "INV-1000")[1] == "CONFLICT"  # a different number
    assert cmp_ref_prox("INV-100", "INV-10")[1] == "CONFLICT"    # also a different number
    # A different trailing number names a different document in the same series.
    assert cmp_ref_prox("INV-100", "INV-200")[1] == "CONFLICT"
    assert cmp_ref_prox("INV-100", "INV-10000")[1] == "CONFLICT"
    assert cmp_ref_prox("inv-100", "INV-100 ")[0] == 1.0      # same ref, normalised
    assert cmp_ref_prox(None, "INV-100")[1] == "MISSING"
    assert cmp_ref_prox("", "")[1] == "MISSING"


def test_date_prox_falls_away_over_a_quarter():
    today = datetime.now(timezone.utc)
    assert cmp_date_prox(today, today)[1] == "OK"
    assert cmp_date_prox(today, today - timedelta(days=5))[1] == "OK"
    assert cmp_date_prox(today, today - timedelta(days=20))[1] == "WEAK"
    assert cmp_date_prox(today, today - timedelta(days=120))[1] == "CONFLICT"


def test_identical_lines_carry_a_pair_that_a_bare_reference_could_not():
    # The line set is one of the heaviest signals: two invoices billing the same items are
    # the same bill even when their references are unrelated.
    same = find_duplicates([_inv("AAA-77", po="PO-9", lines=LINES),
                            _inv("BBB-31", po="PO-9", lines=LINES)])
    different = find_duplicates([
        _inv("AAA-77", po="PO-9", lines=LINES),
        _inv("BBB-31", po="PO-9",
             lines=[{"item_id": "Z", "item_description": "Something else",
                     "quantity": 9, "unit_price": 3, "line_amount": 27}]),
    ])
    assert same and same[0]["score"] > (different[0]["score"] if different else 0)


def test_scales_past_a_pairwise_scan():
    # 12,000 live invoices would be 72M scored pairs. Bucketing on the rule's own
    # prerequisites (supplier + total) keeps it near-linear; this asserts the result is
    # still right at a size that would be slow if it were not.
    rows = [_inv(f"INV-{i}", supplier=f"S{i % 500}", total=100.0 + i, days_ago=5)
            for i in range(6000)]
    rows.append(_inv("INV-DUP", supplier="S7", total=107.0, days_ago=5))
    dups = find_duplicates(rows)
    assert [d["later"]["invoice_id"] for d in dups] == ["INV-DUP"]
    assert dups[0]["earlier"]["invoice_id"] == "INV-7"
