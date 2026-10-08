"""The shared 'one bid = its latest version' helpers every analysis uses (2026-10-08)."""
from src.services.version_collapse import latest_per_family, QUOTE_BASE_SQL, QUOTE_VERSION_SQL, latest_quote_pred


def q(qid, sup, amt):
    return {"quote_id": qid, "supplier_id": sup, "total_amount": amt}


def test_one_row_per_bid_the_latest_version():
    rows = [q("A-1", "SA", 100), q("A-1 (V2)", "SA", 95), q("A-1 (V3 (BAFO))", "SA", 90),
            q("B-7", None, 110), q("B-7 (V2)", None, 105)]
    out = latest_per_family(rows)
    assert sorted(r["quote_id"] for r in out) == ["A-1 (V3 (BAFO))", "B-7 (V2)"]


def test_a_bid_with_no_supplier_is_still_a_bid():
    assert len(latest_per_family([q("B-7", None, 1), q("C-9", None, 2)])) == 2


def test_two_suppliers_using_the_same_number_are_two_bids():
    assert len(latest_per_family([q("Q-001", "SA", 1), q("Q-001", "SB", 2)])) == 2


def test_latest_is_by_version_not_by_amount_or_order():
    out = latest_per_family([q("A-1 (V3)", "SA", 125), q("A-1", "SA", 100), q("A-1 (V2)", "SA", 95)])
    assert out[0]["quote_id"] == "A-1 (V3)"


def test_the_sql_pieces_use_the_bracketed_grammar():
    assert "\\(\\s*V(\\d+)" in QUOTE_VERSION_SQL("q.quote_id")
    assert "'i'" in QUOTE_BASE_SQL("q.quote_id") and "'i'" in QUOTE_VERSION_SQL("q.quote_id")
    pred = latest_quote_pred("q")
    assert "NOT EXISTS" in pred and "q.quote_id" in pred
