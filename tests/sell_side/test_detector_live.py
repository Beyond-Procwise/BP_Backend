"""The opportunity detector: only what the distributor states, only what we sold."""
import datetime as dt
from decimal import Decimal as D

import pytest

from src.services.sell_side import accounts, detector
from tests.sell_side.conftest import live, seed_item

pytestmark = live
PAST = dt.date.today() - dt.timedelta(days=30)


def _acct(conn, suffix="D"):
    return accounts.create_account(conn, account_id=f"LIVETEST-{suffix}",
                                   account_name="Livetest Ltd", default_currency="gbp")["account_id"]


def _quote(conn, account_id, item_id, *, qty="3", won=True):
    from src.services.sell_side import quotes
    q = quotes.create_draft(conn, account_id=account_id, currency="GBP",
                            valid_until=dt.date.today() + dt.timedelta(days=30), created_by="LIVETEST",
                            lines=[{"catalog_item_id": item_id, "quantity": qty, "unit_price": "15"}])
    qid = q["sales_quote_id"]
    if won:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO proc.bp_sales_quote_outcome (sales_quote_id, outcome, "
                        "outcome_date) VALUES (%s, 'won', CURRENT_DATE)", (qid,))
        conn.commit()
    return qid


def _relate(conn, dist, frm, to, kind):
    with conn.cursor() as cur:
        cur.execute("INSERT INTO proc.bp_catalog_item_relation (from_sku, to_sku, distributor_id, "
                    "relation_type, asserted_by) VALUES (%s, %s, %s, %s, 'distributor_feed') "
                    "RETURNING relation_id", (frm, to, dist, kind))
        rid = cur.fetchone()[0]
    conn.commit()
    return rid


def _world(conn, dist, kind="replaced_by", **a_kw):
    acct = _acct(conn)
    a = seed_item(conn, dist, "LIVETEST-A", **a_kw)
    b = seed_item(conn, dist, "LIVETEST-B")
    _relate(conn, dist, "LIVETEST-A", "LIVETEST-B", kind)
    return acct, a, b


def test_a_replaced_item_proposes_an_upgrade_with_its_evidence(live_db):
    conn, dist = live_db
    acct, a, b = _world(conn, dist)
    _quote(conn, acct, a)
    made = detector.detect(conn, account_id=acct)
    assert len(made) == 1
    o = made[0]
    assert (o["opportunity_type"], o["catalog_item_id"]) == ("upgrade", b)
    assert o["reason_codes"] == ["replaced_by"] and o["detector_type"] == "catalog_relation"
    assert o["win_probability"] is None and o["expected_quantity"] == D("3")
    assert [j["kind"] for j in o["justifications"]] == ["coverage_gap"]


def test_end_of_life_adds_its_date_as_evidence(live_db):
    conn, dist = live_db
    acct, a, b = _world(conn, dist)
    with conn.cursor() as cur:
        cur.execute("UPDATE proc.bp_catalog_item SET lifecycle_status='end_of_life', "
                    "end_of_life_date=%s WHERE catalog_item_id=%s", (PAST, a))
    conn.commit()
    _quote(conn, acct, a)
    (o,) = detector.detect(conn, account_id=acct)
    assert o["opportunity_type"] == "upgrade"
    assert sorted(o["reason_codes"]) == ["end_of_life", "replaced_by"]
    kinds = {j["kind"]: j for j in o["justifications"]}
    assert str(PAST) in kinds["end_of_life"]["claim"]


@pytest.mark.parametrize("kind, expected", [("refill_of", "refill"), ("accessory_of", "cross_sell"),
                                            ("requires", "cross_sell"), ("upgrade_of", "upgrade")])
def test_each_relation_maps_to_its_type(live_db, kind, expected):
    conn, dist = live_db
    acct, a, b = _world(conn, dist, kind)
    _quote(conn, acct, a)
    (o,) = detector.detect(conn, account_id=acct)
    assert (o["opportunity_type"], o["reason_codes"]) == (expected, [kind])


def test_no_relation_means_nothing_is_proposed(live_db):
    conn, dist = live_db
    acct = _acct(conn)
    _quote(conn, acct, seed_item(conn, dist, "LIVETEST-A"))
    assert detector.detect(conn, account_id=acct) == []


def test_an_item_the_account_already_bought_is_not_cross_sold(live_db):
    conn, dist = live_db
    acct, a, b = _world(conn, dist, "accessory_of")
    _quote(conn, acct, a)
    _quote(conn, acct, b)
    assert detector.detect(conn, account_id=acct) == []


def test_a_quote_that_was_not_won_is_not_history(live_db):
    conn, dist = live_db
    acct, a, b = _world(conn, dist)
    _quote(conn, acct, a, won=False)
    assert detector.detect(conn, account_id=acct) == []


def test_the_same_item_on_two_won_quotes_gives_one_opportunity(live_db):
    conn, dist = live_db
    acct, a, b = _world(conn, dist)
    _quote(conn, acct, a, qty="2")
    _quote(conn, acct, a, qty="5")
    assert len(detector.detect(conn, account_id=acct)) == 1


def test_a_target_missing_or_unpriced_is_skipped_without_error(live_db):
    conn, dist = live_db
    acct, a, b = _world(conn, dist)
    _relate(conn, dist, "LIVETEST-A", "LIVETEST-NOWHERE", "refill_of")
    with conn.cursor() as cur:
        cur.execute("UPDATE proc.bp_catalog_item SET list_price=NULL WHERE catalog_item_id=%s", (b,))
    conn.commit()
    _quote(conn, acct, a)
    assert detector.detect(conn, account_id=acct) == []


def test_a_second_run_adds_nothing_but_a_lost_one_does_not_block(live_db):
    conn, dist = live_db
    acct, a, b = _world(conn, dist)
    _quote(conn, acct, a)
    (first,) = detector.detect(conn, account_id=acct)
    assert detector.detect(conn, account_id=acct) == []
    with conn.cursor() as cur:
        cur.execute("UPDATE proc.bp_sales_opportunity SET outcome='lost' "
                    "WHERE sales_opportunity_id=%s", (first["sales_opportunity_id"],))
    conn.commit()
    assert len(detector.detect(conn, account_id=acct)) == 1
