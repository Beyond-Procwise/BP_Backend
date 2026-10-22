import datetime as dt
from decimal import Decimal

from src.services.analytics import similar_items as SI
from src.services.contract_tools import store


def _r(desc, item_id="ITM1", uom="each", lines=3, sups=2, buyers=1, pmin="10", pmax="12", ccy="GBP", ccys=1,
       last=dt.date(2026, 9, 1)):
    return (desc, item_id, uom, lines, sups, buyers, Decimal(pmin), Decimal(pmax), ccy, ccys, last)


PO = [_r("Advanced Laptop Computers Installation ITM003969", "ITM003969", lines=14, sups=14, buyers=3),
      _r("Standard Laptop Computers Kit ITM003967", "ITM003967", lines=12, sups=12, buyers=2),
      _r("Ergonomic Office Chair ITM000100", "ITM000100"),
      _r("Laptop Docking Station ITM003990", "ITM003990", lines=2, sups=1, buyers=1, ccy=None, ccys=2)]
INV = [_r("Advanced Laptop Computers Installation ITM003969", "ITM003969", lines=10, sups=9, buyers=3, pmin="9.5", pmax="13")]
QUOTE = [_r("laptop computers (advanced) installation", None, None, lines=1, sups=1, buyers=1)]


def _rows(po=PO, inv=INV, quote=QUOTE):
    def rows(sql, params):
        if "bp_po_line_items_trgt" in sql:
            return list(po)
        if "bp_invoice_line_items_trgt" in sql:
            return list(inv)
        if "bp_quote_line_items_trgt" in sql:
            return list(quote)
        raise AssertionError(sql)
    return rows


def test_descriptive_words_drop_codes_and_buying_words():
    assert SI.core_words("Advanced Laptop Computers Installation ITM003969") == frozenset({"laptop", "computer"})
    assert SI.item_code("Advanced Laptop Computers Installation ITM003969") == "ITM003969"
    assert SI.item_code("Ergonomic chair") is None
    assert SI.similarity("Advanced Laptop Computers Installation", "Standard Laptop Computers Kit") == 1.0
    assert SI.similarity("Laptop Docking Station", "Laptop Computers") == round(1 / 4, 3)


def test_refuses_short_or_empty_descriptions():
    assert SI.find("ab", rows=_rows())["found"] is False
    assert "no descriptive words" in SI.find("Standard Kit ITM003967", rows=_rows())["note"]


def test_matches_carry_evidence_per_source_and_are_ranked_by_similarity_then_volume():
    out = SI.find("Advanced Laptop Computers Installation ITM003969", rows=_rows())
    assert out["found"] and out["descriptive_words"] == ["computer", "laptop"]
    descs = [m["description"] for m in out["matches"]]
    assert descs[:3] == ["Advanced Laptop Computers Installation ITM003969", "Standard Laptop Computers Kit ITM003967",
                         "laptop computers (advanced) installation"]
    top = out["matches"][0]
    assert top["same_description"] and top["same_item_code"] and top["name_similarity"] == 1.0
    assert top["sources"]["po"]["suppliers"] == 14 and top["sources"]["invoice"]["price_max"] == 13.0
    assert top["lines_total"] == 24 and top["suppliers_max"] == 14
    kit = out["matches"][1]
    assert kit["same_description"] is False and kit["same_item_code"] is False and kit["item_code"] == "ITM003967"
    assert "not identity" in out["note"] and "as billed" in out["note"]


def test_the_threshold_cuts_weak_matches_and_a_chair_never_appears():
    out = SI.find("Laptop computers", rows=_rows())
    assert all(m["name_similarity"] >= 0.6 for m in out["matches"])
    assert "Laptop Docking Station ITM003990" not in [m["description"] for m in out["matches"]]
    low = SI.find("Laptop computers", min_similarity=0.2, rows=_rows())
    assert "Laptop Docking Station ITM003990" in [m["description"] for m in low["matches"]]
    assert "Ergonomic Office Chair ITM000100" not in [m["description"] for m in low["matches"]]


def test_mixed_currency_ranges_are_flagged_not_compared():
    low = SI.find("Laptop docking", min_similarity=0.2, rows=_rows())
    dock = [m for m in low["matches"] if m["description"].startswith("Laptop Docking")][0]
    assert dock["sources"]["po"]["currency"] is None and dock["sources"]["po"]["currencies"] == 2
    assert "not comparable" in dock["sources"]["po"]["price_note"]


def test_limit_is_capped_and_overflow_is_said():
    po = [_r(f"Laptop computers model {i}", f"ITM{i:06d}") for i in range(30)]
    # "model 10" carries one more descriptive word than "model 0", so the threshold is lowered to keep all 30.
    out = SI.find("Laptop computers", min_similarity=0.3, limit=5, rows=_rows(po=po, inv=[], quote=[]))
    assert len(out["matches"]) == 5 and out["match_count"] == 30 and "Showing the closest 5 of 30" in out["note"]
    assert len(SI.find("Laptop computers", min_similarity=0.3, limit=999, rows=_rows(po=po, inv=[], quote=[]))["matches"]) == 30


def test_nothing_close_says_what_was_covered():
    out = SI.find("Forklift truck", rows=_rows())
    assert out["found"] is False and "only PO, invoice and quote lines on record" in out["note"]


def test_outage_is_not_no_match():
    def down(sql, params):
        raise store.StoreUnavailable("db down")
    out = SI.find("Laptop computers", rows=down)
    assert out["found"] is False and out["available"] is False and "does not mean" in out["note"]


def test_singular_and_plural_wording_match():
    """Live, 2026-10-22: 'laptop computer' found nothing because every line says 'Laptop Computers'."""
    assert SI.similarity("laptop computer", "Standard Laptop Computers Kit ITM003967") == 1.0
    assert SI._singular("batteries") == "battery" and SI._singular("boxes") == "box" and SI._singular("glass") == "glass"
    out = SI.find("laptop computer", rows=_rows())
    assert out["found"] and out["matches"][0]["description"].startswith("Advanced Laptop Computers")
