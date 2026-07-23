# tests/services/test_requirement_similarity.py
from src.services import requirement_similarity as rs
from src.services import linking_engine as le

def test_desc_overlap_high_for_same_products_low_across_categories():
    freight = [{"item_description": "London Heathrow to Edinburgh FTL freight lane"}]
    it = [{"item_description": "Managed endpoint device support annual licence seats"}]
    s_same, _ = rs.cmp_desc_overlap(freight, freight)
    s_diff, _ = rs.cmp_desc_overlap(freight, it)
    assert s_same == 1.0
    assert s_diff < 0.2

def test_desc_overlap_missing_when_no_lines():
    assert rs.cmp_desc_overlap([], [{"item_description": "x"}]) == (0.5, "MISSING")

def test_price_proximity_graded_no_cutoff():
    near, _ = rs.cmp_price_prox({"converted_amount_usd": 900}, {"converted_amount_usd": 972})   # 1.08x
    far, _ = rs.cmp_price_prox({"converted_amount_usd": 900}, {"converted_amount_usd": 281000})  # 312x
    assert near > 0.9
    assert far < 0.05          # contributes almost nothing, but is NOT a hard reject

def test_volume_matches_on_equal_quantities():
    a = [{"quantity": 100}]; b = [{"quantity": 100}]
    s, status = rs.cmp_volume(a, b)
    assert s == 1.0 and status == "OK"

def test_buyer_corroborates_when_equal():
    assert rs.cmp_buyer({"buyer_id": "Assurity Ltd"}, {"buyer_id": "Assurity Ltd"})[0] == 1.0

def test_linking_engine_registry_dispatches_unknown_kind():
    le.register_signal("desc_overlap",
                       lambda src, tgt, sl, tl: rs.cmp_desc_overlap(sl, tl))
    s, status = le._signal_match("desc_overlap", {}, {},
                                 [{"item_description": "a b c"}], [{"item_description": "a b c"}],
                                 "quote_date")
    assert s == 1.0
