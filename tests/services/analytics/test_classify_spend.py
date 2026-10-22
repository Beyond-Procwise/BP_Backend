from src.services.analytics import classify_spend as CS
from src.services.contract_tools import store

TAX = [
    ("C-2000", "IT & Technology", "C-2100", "Hardware", "C-2110", "End-user devices", "C-2111", "Laptop computers",
     "C-21111", "Business laptops", "43211503", "Active", "Alex Chen"),
    ("C-2000", "IT & Technology", "C-2100", "Hardware", "C-2110", "End-user devices", "C-2112", "Monitors",
     "C-21121", "Desktop monitors", "43211902", "Active", "Alex Chen"),
    ("C-3000", "Office & Administrative Supplies", "C-3100", "Stationery", "C-3110", "Writing Instruments", "C-3111", "Pens",
     "C-31111", "Ballpoint Pens", "44121704", "Pending Review", "Maria"),
]
MAP = [("staedtler ballpoint pen black, 10 per pack", "Office & Administrative Supplies", "Stationery",
        "Writing Instruments", "Pens", "Ballpoint Pens")]


def _rows(tax=TAX, mapping=MAP):
    seen = []

    def rows(sql, params):
        seen.append(sql)
        if "bp_category_master" in sql:
            return list(tax)
        if "bp_category_product_map" in sql:
            return list(mapping)
        raise AssertionError(sql)
    rows.seen = seen
    return rows


def test_needs_items_or_a_query_and_caps_items():
    assert CS.classify(rows=_rows())["found"] is False
    assert "At most" in CS.classify(items=["x"] * (CS.MAX_ITEMS + 1), rows=_rows())["note"]


def test_a_mapped_item_is_recorded_and_an_unmapped_one_gets_draft_candidates_only():
    out = CS.classify(items=["Staedtler Ballpoint Pen Black, 10 per pack", "Advanced Laptop Computers Installation ITM003969"],
                      rows=_rows())
    assert out["found"] and out["written"] is False and out["recorded_mappings"] == 1 and out["taxonomy_rows"] == 3
    pen, laptop = out["items"]
    assert pen["status"] == "recorded" and pen["classification"]["category_level_5"] == "Ballpoint Pens"
    assert laptop["status"] == "unclassified" and laptop["classification"] is None
    assert [c["category_level_4"] for c in laptop["candidates"]] == ["Laptop computers"]
    assert laptop["candidates"][0]["matched_words"] == ["computers", "laptop"]
    assert "draft" in laptop["candidates"][0]["basis"] and "not a classification" in laptop["candidates"][0]["basis"]
    assert out["classified"] == 1 and out["coverage_pct"] == 50.0 and "never applied as fact" in out["note"]


def test_buying_words_alone_never_match_a_category():
    tax = TAX + [("C-4000", "Facilities", "C-4100", "Services", "C-4110", "Installation services", "C-4111",
                  "Standard installation", "C-41111", "Bundle installation", "7", "Active", "A")]
    out = CS.classify(items=["Advanced Standard Bundle Installation"], rows=_rows(tax=tax))
    assert out["items"][0]["status"] == "unclassified" and out["items"][0]["candidates"] == []


def test_candidates_are_capped_and_best_first():
    tax = TAX + [("C-2000", "IT & Technology", "C-2100", "Hardware", "C-2110", "End-user devices", "C-2113",
                  "Laptop docking stations", "C-21131", "Laptop docks", "4", "Active", "A"),
                 ("C-2000", "IT & Technology", "C-2100", "Hardware", "C-2110", "End-user devices", "C-2114",
                  "Laptop bags", "C-21141", "Laptop bags", "5", "Active", "A"),
                 ("C-2000", "IT & Technology", "C-2100", "Hardware", "C-2110", "End-user devices", "C-2115",
                  "Laptop stands", "C-21151", "Laptop stands", "6", "Active", "A")]
    cands = CS.classify(items=["Laptop computers"], rows=_rows(tax=tax))["items"][0]["candidates"]
    assert len(cands) == CS.MAX_CANDIDATES and cands[0]["category_level_4"] == "Laptop computers"


def test_no_mappings_at_all_is_said_and_coverage_is_zero():
    out = CS.classify(items=["Laptop computers"], rows=_rows(mapping=[]))
    assert out["coverage_pct"] == 0.0 and "No product-to-category mappings are recorded" in out["note"]


def test_a_query_browses_the_taxonomy_without_classifying():
    out = CS.classify(query="pens", rows=_rows())
    assert out["found"] and out["category_count"] == 1 and out["categories"][0]["unspsc_code"] == "44121704"
    assert "items" not in out and "Not a classification" in out["note"]
    rows = _rows()
    CS.classify(query="pens", rows=rows)
    assert not any("bp_category_product_map" in s for s in rows.seen)


def test_items_and_query_together_give_both():
    out = CS.classify(items=["Laptop computers"], query="monitors", rows=_rows())
    assert out["category_count"] == 1 and out["items"][0]["status"] == "unclassified"


def test_empty_taxonomy_and_outage_are_told_apart():
    out = CS.classify(items=["x"], rows=_rows(tax=[]))
    assert out["found"] is False and out["loaded"] is False and "draft classification" in out["note"]

    def down(sql, params):
        raise store.StoreUnavailable("db down")
    out = CS.classify(items=["x"], rows=down)
    assert out["found"] is False and out["available"] is False and "does not mean" in out["note"]


def test_a_single_string_is_one_item():
    assert len(CS.classify(items="Laptop computers", rows=_rows())["items"]) == 1
