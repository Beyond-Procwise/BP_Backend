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
    (o,) = detector.detect(conn, account_id=acct)
    assert o["expected_quantity"] == D("5")
    assert (o["phase_id"], o["subprocess_id"]) == ("sales.opportunity", "sales.opportunity.qualified")
    assert all(j["customer_safe"] is True for j in o["justifications"])


def _close(conn, item_id):
    with conn.cursor() as cur:
        cur.execute("UPDATE proc.bp_catalog_item SET valid_to=now() WHERE catalog_item_id=%s", (item_id,))
    conn.commit()


def test_a_target_sku_missing_from_the_catalog_is_skipped(live_db):
    conn, dist = live_db
    acct = _acct(conn)
    a = seed_item(conn, dist, "LIVETEST-A")
    _relate(conn, dist, "LIVETEST-A", "LIVETEST-NOWHERE", "refill_of")
    _quote(conn, acct, a)
    assert detector.detect(conn, account_id=acct) == []


def test_an_unpriced_target_is_skipped(live_db):
    conn, dist = live_db
    acct, a, b = _world(conn, dist)
    with conn.cursor() as cur:
        cur.execute("UPDATE proc.bp_catalog_item SET list_price=NULL WHERE catalog_item_id=%s", (b,))
    conn.commit()
    _quote(conn, acct, a)
    assert detector.detect(conn, account_id=acct) == []


def test_a_target_with_only_a_closed_version_is_skipped(live_db):
    conn, dist = live_db
    acct, a, b = _world(conn, dist)
    _close(conn, b)
    _quote(conn, acct, a)
    assert detector.detect(conn, account_id=acct) == []


def test_a_current_version_that_is_active_overrides_an_old_end_of_life_date(live_db):
    conn, dist = live_db
    acct, a, b = _world(conn, dist)
    with conn.cursor() as cur:
        cur.execute("UPDATE proc.bp_catalog_item SET lifecycle_status='end_of_life', "
                    "end_of_life_date=%s WHERE catalog_item_id=%s", (PAST, a))
    conn.commit()
    _quote(conn, acct, a)
    _close(conn, a)
    with conn.cursor() as cur:
        cur.execute("UPDATE proc.bp_catalog_item SET lifecycle_status='active' "
                    "WHERE catalog_item_id=%s", (seed_item(conn, dist, "LIVETEST-A"),))
    conn.commit()
    (o,) = detector.detect(conn, account_id=acct)
    assert o["reason_codes"] == ["replaced_by"]
    assert [j["kind"] for j in o["justifications"]] == ["coverage_gap"]


def test_evidence_ref_names_the_bought_version_when_no_current_one_exists(live_db):
    conn, dist = live_db
    acct, a, b = _world(conn, dist)
    with conn.cursor() as cur:
        cur.execute("UPDATE proc.bp_catalog_item SET lifecycle_status='end_of_life' "
                    "WHERE catalog_item_id=%s", (a,))
    conn.commit()
    _quote(conn, acct, a)
    _close(conn, a)
    (o,) = detector.detect(conn, account_id=acct)
    eol = [j for j in o["justifications"] if j["kind"] == "end_of_life"]
    assert [j["evidence_ref"] for j in eol] == [f"catalog_item:{a}"]


@pytest.mark.parametrize("status, words", [("end_of_life", "reached end of life"),
                                           ("end_of_sale", "reached end of sale")])
def test_the_claim_says_which_end_was_reached(live_db, status, words):
    conn, dist = live_db
    acct, a, b = _world(conn, dist)
    with conn.cursor() as cur:
        cur.execute("UPDATE proc.bp_catalog_item SET lifecycle_status=%s WHERE catalog_item_id=%s",
                    (status, a))
    conn.commit()
    _quote(conn, acct, a)
    (o,) = detector.detect(conn, account_id=acct)
    (j,) = [j for j in o["justifications"] if j["kind"] == "end_of_life"]
    assert words in j["claim"]


def test_a_failed_justification_withdraws_the_opportunity_and_a_rerun_recovers(live_db, monkeypatch):
    from src.services.sell_side import opportunities
    conn, dist = live_db
    acct, a, b = _world(conn, dist)
    _quote(conn, acct, a)
    real = opportunities.add_justification

    def boom(*args, **kw):
        raise RuntimeError("boom")
    monkeypatch.setattr(opportunities, "add_justification", boom)
    assert detector.detect(conn, account_id=acct) == []
    with conn.cursor() as cur:
        cur.execute("SELECT outcome FROM proc.bp_sales_opportunity WHERE account_id=%s", (acct,))
        assert [r[0] for r in cur.fetchall()] == ["withdrawn"]
    monkeypatch.setattr(opportunities, "add_justification", real)
    (o,) = detector.detect(conn, account_id=acct)
    assert o["justifications"]


def test_a_won_opportunity_blocks_a_rerun(live_db):
    conn, dist = live_db
    acct, a, b = _world(conn, dist)
    _quote(conn, acct, a)
    (first,) = detector.detect(conn, account_id=acct)
    with conn.cursor() as cur:
        cur.execute("UPDATE proc.bp_sales_opportunity SET outcome='won' "
                    "WHERE sales_opportunity_id=%s", (first["sales_opportunity_id"],))
    conn.commit()
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


def test_a_concurrent_run_is_refused_while_another_holds_the_lock(live_db):
    import os
    import psycopg2
    conn, dist = live_db
    acct, a, b = _world(conn, dist)
    _quote(conn, acct, a)
    other = psycopg2.connect(
        host=os.environ["DB_HOST"], port=os.getenv("DB_PORT", 5432),
        dbname=os.environ["DB_NAME"], user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"], connect_timeout=10)
    try:
        with other.cursor() as oc:
            oc.execute("SELECT pg_try_advisory_lock(%s)", (detector.ADVISORY_LOCK_KEY,))
            assert oc.fetchone()[0] is True
        assert detector.detect(conn, account_id=acct) == []
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM proc.bp_sales_opportunity WHERE account_id=%s", (acct,))
            assert cur.fetchone()[0] == 0
        with other.cursor() as oc:
            oc.execute("SELECT pg_advisory_unlock(%s)", (detector.ADVISORY_LOCK_KEY,))
    finally:
        other.close()
    assert len(detector.detect(conn, account_id=acct)) == 1


def test_a_missing_target_is_skipped_not_fatal(live_db, monkeypatch):
    from src.services.sell_side import opportunities
    from src.services.sell_side._db import NotFound
    conn, dist = live_db
    acct, a, b = _world(conn, dist)
    _quote(conn, acct, a)

    def gone(*a, **k):
        raise NotFound("gone")
    monkeypatch.setattr(opportunities, "create_opportunity", gone)
    assert detector.detect(conn, account_id=acct) == []


def test_the_claim_follows_the_prompt_not_the_code(live_db, monkeypatch):
    """Reword the prompt (stubbed, so no shared row is edited) and the stored claim is reworded."""
    from src.services.sell_side import reason_prompts
    monkeypatch.setattr(reason_prompts, "_load", lambda conn, name: (
        "PROMPTED {bought_sku} -> {target_sku} via {relation_type}",
        {"relation_type": {"replaced_by": "a replacement"}}))
    conn, dist = live_db
    acct, a, b = _world(conn, dist)
    _quote(conn, acct, a)
    (o,) = detector.detect(conn, account_id=acct)
    (j,) = [j for j in o["justifications"] if j["kind"] == "coverage_gap"]
    assert j["claim"] == "PROMPTED LIVETEST-A -> LIVETEST-B via a replacement"
