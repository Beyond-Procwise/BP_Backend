import pytest

from src.services import catalog_match as cm
from src.services.sell_side._db import StateConflict, dict_cursor
from tests.sell_side.conftest import live, seed_item

pytestmark = live


def _a_real_history_item(conn):
    cur = dict_cursor(conn)
    cur.execute("SELECT item_id, item_description FROM proc.bp_invoice_line_items_trgt "
                "WHERE item_id IS NOT NULL ORDER BY item_id LIMIT 1")
    return cur.fetchone()


def test_proposing_twice_proposes_nothing_the_second_time(live_db):
    conn, dist = live_db
    real = _a_real_history_item(conn)
    seed_item(conn, dist, "LIVETEST-M1", mpn=real["item_id"])
    first = cm.propose_matches(conn, dist)
    second = cm.propose_matches(conn, dist)
    assert first["exact"] >= 1
    assert second["proposed"] == 0 and second["already_known"] >= 1


def test_a_decision_is_final_and_attributed(live_db):
    conn, dist = live_db
    real = _a_real_history_item(conn)
    seed_item(conn, dist, "LIVETEST-M2", mpn=real["item_id"])
    cm.propose_matches(conn, dist)
    (m,) = [r for r in cm.list_matches(conn, distributor_id=dist)
            if r["distributor_sku"] == "LIVETEST-M2"]
    done = cm.confirm_match(conn, m["match_id"], "sub-reviewer")
    assert (done["status"], done["confirmed_by"]) == ("confirmed", "sub-reviewer")
    with pytest.raises(StateConflict):
        cm.reject_match(conn, m["match_id"], "sub-other")


def test_a_human_match_is_confirmed_with_no_confidence(live_db):
    conn, dist = live_db
    seed_item(conn, dist, "LIVETEST-M3")
    row = cm.record_human_match(conn, distributor_id=dist, distributor_sku="LIVETEST-M3",
                                item_id="ANY-ID", reviewer="sub-reviewer")
    assert (row["match_method"], row["status"], row["confidence"]) == ("human", "confirmed", None)


def test_a_human_match_cannot_overwrite_a_confirmed_one(live_db):
    conn, dist = live_db
    seed_item(conn, dist, "LIVETEST-M4")
    cm.record_human_match(conn, distributor_id=dist, distributor_sku="LIVETEST-M4",
                          item_id="ANY-ID", reviewer="sub-first")
    with pytest.raises(StateConflict, match="sub-first"):
        cm.record_human_match(conn, distributor_id=dist, distributor_sku="LIVETEST-M4",
                              item_id="ANY-ID", reviewer="sub-second")
    cur = dict_cursor(conn)
    cur.execute("SELECT confirmed_by FROM proc.bp_catalog_item_match WHERE distributor_id = %s "
                "AND distributor_sku = %s AND item_id = %s", (dist, "LIVETEST-M4", "ANY-ID"))
    assert cur.fetchone()["confirmed_by"] == "sub-first"


def test_a_human_match_overrides_a_rejection(live_db):
    conn, dist = live_db
    real = _a_real_history_item(conn)
    seed_item(conn, dist, "LIVETEST-M5", mpn=real["item_id"])
    cm.propose_matches(conn, dist)
    (m,) = [r for r in cm.list_matches(conn, distributor_id=dist)
            if r["distributor_sku"] == "LIVETEST-M5"]
    cm.reject_match(conn, m["match_id"], "sub-reviewer")
    row = cm.record_human_match(conn, distributor_id=dist, distributor_sku="LIVETEST-M5",
                                item_id=real["item_id"], reviewer="sub-second")
    assert (row["status"], row["match_method"], row["confirmed_by"]) == (
        "confirmed", "human", "sub-second")
