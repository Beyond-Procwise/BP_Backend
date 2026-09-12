from decimal import Decimal

from src.services import catalog_match as cm

HISTORY = [
    {"item_id": "CISCO-C9200-24T", "item_description": "Cisco Catalyst 9200 24 port switch"},
    {"item_id": "IN-1002", "item_description": "HP LaserJet toner black"},
    {"item_id": "ITM000160", "item_description": "Premium Furniture Unit ITM000160"},
]


def _item(sku, desc, mpn=None):
    return {"distributor_sku": sku, "item_description": desc, "mpn": mpn}


def test_an_mpn_equal_to_a_history_item_id_is_an_exact_match_with_no_confidence():
    (p,) = cm.propose([_item("X1", "whatever", mpn=" cisco-c9200-24t ")], HISTORY, fuzzy_min=88)
    assert (p.item_id, p.match_method, p.confidence) == ("CISCO-C9200-24T", "mpn_exact", None)


def test_a_sku_equal_to_a_history_item_id_is_sku_exact():
    (p,) = cm.propose([_item("in-1002", "whatever")], HISTORY, fuzzy_min=88)
    assert (p.item_id, p.match_method) == ("IN-1002", "sku_exact")


def test_mpn_and_sku_naming_the_same_item_propose_it_once():
    got = cm.propose([_item("IN-1002", "x", mpn="IN-1002")], HISTORY, fuzzy_min=88)
    assert [(p.item_id, p.match_method) for p in got] == [("IN-1002", "mpn_exact")]


def test_a_close_description_is_proposed_fuzzy_with_its_score():
    # Word order differs; token_sort_ratio is chosen to absorb exactly that.
    # (Measured: "24-port" vs "24 port" scores 85.3 -- below 88, so NOT used here.)
    (p,) = cm.propose([_item("Z9", "Catalyst 9200 Cisco 24 port switch")], HISTORY, fuzzy_min=88)
    assert p.match_method == "description_fuzzy"
    assert p.item_id == "CISCO-C9200-24T"
    assert Decimal("0.88") <= p.confidence <= Decimal("1")


def test_below_the_threshold_nothing_is_proposed():
    assert cm.propose([_item("Z9", "Office chair mesh")], HISTORY, fuzzy_min=88) == []


def test_an_exact_match_suppresses_fuzzy_guessing_for_that_sku():
    got = cm.propose([_item("IN-1002", "Cisco Catalyst 9200 24 port switch")], HISTORY, fuzzy_min=88)
    assert [p.match_method for p in got] == ["sku_exact"]


def test_no_history_proposes_nothing():
    assert cm.propose([_item("A", "b")], [], fuzzy_min=88) == []
