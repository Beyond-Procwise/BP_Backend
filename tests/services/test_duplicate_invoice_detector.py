"""The duplicate-invoice rule.

Paying the same invoice twice is money already out of the door, so this detector is
deliberately conservative: it only speaks when the supplier, the amount AND the paper trail
(same PO, or two references one character apart) all agree, inside a 90-day window. A
recurring monthly charge looks superficially identical — same supplier, same amount — and
must never be called a duplicate.
"""

from datetime import datetime, timezone, timedelta

from src.services.duplicate_invoice_detector import find_duplicates, _refs_near


def _inv(iid, supplier="Techworld", total=1000.0, po="PO-1", days_ago=0, ref=None):
    return {"invoice_id": iid, "supplier_name": supplier, "total_amount": total,
            "po_id": po, "invoice_ref": ref or iid,
            "invoice_date": datetime.now(timezone.utc) - timedelta(days=days_ago)}


# ---- the rule ------------------------------------------------------------

def test_same_supplier_total_and_po_within_window_flags_later():
    # Refs that are NOT a numbered series (see the sequence test below) — this exercises the
    # same-purchase-order branch on its own.
    dups = find_duplicates([_inv("INV-A", days_ago=30), _inv("INV-B", days_ago=1)])
    assert len(dups) == 1
    assert dups[0]["later"]["invoice_id"] == "INV-B"
    assert dups[0]["amount"] == 1000.0


def test_a_numbered_series_on_one_po_is_not_a_duplicate():
    # Measured against the live corpus on 2026-07-31: same supplier + same total + same PO
    # flagged 4,974 of 12,408 invoices, and every one was a member of an INV<n>-1/-2/-3
    # series billed against one PO. Several invoices against one purchase order is ordinary;
    # calling it duplicate billing would have put millions of fabricated pounds into the
    # Value Found headline.
    series = [_inv("INV000469-1", ref="INV000469-1", po="PO000469", total=2663.12, days_ago=30),
              _inv("INV000469-2", ref="INV000469-2", po="PO000469", total=2663.12, days_ago=28),
              _inv("INV000469-3", ref="INV000469-3", po="PO000469", total=2663.12, days_ago=27)]
    assert find_duplicates(series) == []


def test_near_identical_ref_counts_even_across_pos():
    a, b = _inv("INV-100", po="PO-1"), _inv("INV-100A", po="PO-2")
    assert find_duplicates([a, b])          # ref edit distance 1
    assert _refs_near("INV-100", "INV-100A") is True
    assert _refs_near("INV-100", "INV-200") is False


def test_recurring_charge_not_flagged():
    # same supplier + same amount but different PO and unrelated refs (monthly fee)
    a = _inv("INV-JAN", po="PO-1", ref="SVC-JAN", days_ago=60)
    b = _inv("INV-FEB", po="PO-2", ref="SVC-FEB", days_ago=30)
    assert find_duplicates([a, b]) == []


def test_outside_90_days_not_flagged():
    assert find_duplicates([_inv("INV-A", days_ago=120), _inv("INV-B", days_ago=1)]) == []


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


def test_a_missing_date_is_not_inside_the_window():
    a, b = _inv("INV-A"), _inv("INV-B")
    a["invoice_date"] = None
    assert find_duplicates([a, b]) == []


def test_totals_agree_to_the_penny_not_approximately():
    # 2-dp equality: 1000.004 and 1000.001 are the same billed amount; 1000.01 is not.
    assert find_duplicates([_inv("INV-A", total=1000.004), _inv("INV-B", total=1000.001)])
    assert find_duplicates([_inv("INV-A", total=1000.00), _inv("INV-B", total=1000.01)]) == []


def test_credit_notes_are_not_duplicate_invoices():
    # The live corpus carries credit notes as negative-total rows (…-CN). Two of them
    # agreeing is money coming BACK, not money paid twice — this detector's whole claim is
    # "you may have paid this twice", so it must stay silent on anything not billed.
    a = _inv("INV000001-1-CN", total=-1321.06, days_ago=30)
    b = _inv("INV000002-1-CN", total=-1321.06, days_ago=1)
    assert find_duplicates([a, b]) == []
    assert find_duplicates([_inv("INV-1", total=0.0), _inv("INV-2", total=0.0)]) == []


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


def test_one_invoice_is_reported_once_even_with_several_earlier_matches():
    # Three identical invoices: the two later ones are each flagged against the FIRST
    # (their strongest evidence), never producing three overlapping pairs for one document.
    rows = [_inv("INV-A", days_ago=40), _inv("INV-B", days_ago=20), _inv("INV-C", days_ago=1)]
    dups = find_duplicates(rows)
    assert sorted(d["later"]["invoice_id"] for d in dups) == ["INV-B", "INV-C"]
    assert all(d["earlier"]["invoice_id"] == "INV-A" for d in dups)


def test_refs_near_is_a_distance_of_one_not_a_prefix_test():
    assert _refs_near("INV-100", "INV-1000") is True      # one insertion
    assert _refs_near("INV-100", "INV-10") is True        # one deletion
    assert _refs_near("INV-100", "INV-200") is False      # one substitution... of a digit
    assert _refs_near("INV-100", "INV-10000") is False    # two insertions
    assert _refs_near("inv-100", "INV-100 ") is True      # same ref, normalised
    assert _refs_near(None, "INV-100") is False
    assert _refs_near("", "") is False                    # two absent refs agree on nothing


def test_scales_past_a_pairwise_scan():
    # 12,000 live invoices would be 72M comparisons pairwise. Bucketing on the rule's own
    # mandatory conjuncts (supplier + total) keeps it near-linear; this asserts the result
    # is still right at a size that would be slow if it were not.
    rows = [_inv(f"INV-{i}", supplier=f"S{i % 500}", total=100.0 + i, days_ago=5)
            for i in range(6000)]
    rows.append(_inv("INV-DUP", supplier="S7", total=107.0, days_ago=1))
    dups = find_duplicates(rows)
    assert [d["later"]["invoice_id"] for d in dups] == ["INV-DUP"]
    assert dups[0]["earlier"]["invoice_id"] == "INV-7"
