"""A quote's number: minted at issue, the next in the series, none missing (QDA B6).

Live database only -- what is being tested is a row lock and a rollback, and a fake has
neither. Every quote here is a LIVETEST one and takes its number from a LIVETEST series
(see conftest), so a run uses up no real number.
"""
import datetime as dt
import os
import threading
from decimal import Decimal as D

import psycopg2
import psycopg2.extras
import pytest

from src.services import agent_actions
from src.services.sell_side import accounts, numbering, quotes
from src.services.sell_side._db import StateConflict
from tests.sell_side.conftest import SENTINEL, live, seed_item

pytestmark = live
TODAY = dt.date.today()
YEAR = TODAY.year
SERIES = f"{SENTINEL}_sales_quote"


def _connect():
    return psycopg2.connect(
        host=os.environ["DB_HOST"], port=os.getenv("DB_PORT", 5432),
        dbname=os.environ["DB_NAME"], user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"], connect_timeout=10)


def _approved(conn, dist, n=1):
    """n approved quotes, ready to issue."""
    accounts.create_account(conn, account_id="LIVETEST-N", account_name="Livetest Numbers Ltd")
    item = seed_item(conn, dist, "LIVETEST-N1")
    ids = []
    for _ in range(n):
        q = quotes.create_draft(
            conn, account_id="LIVETEST-N", currency="GBP", created_by="sub-author",
            valid_until=TODAY + dt.timedelta(days=30),
            lines=[{"catalog_item_id": item, "quantity": D("4"), "unit_price": D("14.00")}])
        quotes.submit(conn, q["sales_quote_id"], actor="sub-author")
        quotes.approve(conn, q["sales_quote_id"], approver="sub-approver")
        ids.append(q["sales_quote_id"])
    return ids


def _number(n):
    return f"{SENTINEL}-Q-{YEAR}-{n:04d}"


def _last(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT last_value FROM proc.bp_doc_number_counter "
                    "WHERE doc_type = %s AND year = %s", (SERIES, YEAR))
        row = cur.fetchone()
    conn.rollback()
    return row[0] if row else None


def test_a_quote_has_no_number_until_it_is_issued(live_db):
    conn, dist = live_db
    (qid,) = _approved(conn, dist)
    assert quotes.get_quote(conn, qid)["quote_number"] is None
    assert _last(conn) is None
    issued = quotes.issue(conn, qid, actor="sub-issuer")
    assert issued["quote_number"] == _number(1) and issued["status"] == "issued"
    assert issued["quote_ref"].startswith("SQ-")          # the draft reference is kept


def test_quotes_issued_one_after_another_take_consecutive_numbers(live_db):
    conn, dist = live_db
    ids = _approved(conn, dist, 3)
    got = [quotes.issue(conn, qid, actor="sub-issuer")["quote_number"] for qid in ids]
    assert got == [_number(1), _number(2), _number(3)]


def test_the_issue_event_carries_the_number(live_db):
    conn, dist = live_db
    (qid,) = _approved(conn, dist)
    quotes.issue(conn, qid, actor="sub-issuer")
    with conn.cursor() as cur:
        cur.execute("SELECT details FROM proc.bp_agent_actions WHERE doc_type = 'sales_quote' "
                    "AND doc_pk = %s AND action_type = 'sales_quote.issued'", (str(qid),))
        (details,) = cur.fetchone()
    conn.rollback()
    assert details["quote_number"] == _number(1) and details["actor"] == "sub-issuer"


def test_an_issue_that_fails_gives_its_number_back(live_db, monkeypatch):
    """The audit row for an issue is written or the issue does not happen. Here it cannot be
    written -- after the number was taken -- so the whole issue rolls back, number included."""
    conn, dist = live_db
    first, second = _approved(conn, dist, 2)
    quotes.issue(conn, first, actor="sub-issuer")

    real = agent_actions.record_action_or_fail
    taken = []

    def _refuse(**kw):
        taken.append(kw["details"]["quote_number"])
        raise RuntimeError("the audit spine is down")

    monkeypatch.setattr(agent_actions, "record_action_or_fail", _refuse)
    with pytest.raises(RuntimeError, match="audit spine"):
        quotes.issue(conn, second, actor="sub-issuer")
    assert taken == [_number(2)]                           # it had been handed number 2
    after = quotes.get_quote(conn, second)
    assert (after["status"], after["quote_number"]) == ("approved", None)
    assert _last(conn) == 1                                # and the series is back at 1

    monkeypatch.setattr(agent_actions, "record_action_or_fail", real)
    assert quotes.issue(conn, second, actor="sub-issuer")["quote_number"] == _number(2)


def test_a_quote_refused_at_issue_takes_no_number(live_db):
    conn, dist = live_db
    (qid,) = _approved(conn, dist)
    with conn.cursor() as cur:
        cur.execute("UPDATE proc.bp_sales_quote SET quote_date = %s, valid_until = %s "
                    "WHERE sales_quote_id = %s",
                    (TODAY - dt.timedelta(days=9), TODAY - dt.timedelta(days=1), qid))
    conn.commit()
    with pytest.raises(StateConflict, match="expired"):
        quotes.issue(conn, qid, actor="sub-issuer")
    assert _last(conn) is None


def test_a_quote_is_numbered_once(live_db):
    conn, dist = live_db
    (qid,) = _approved(conn, dist)
    quotes.issue(conn, qid, actor="sub-issuer")
    with pytest.raises(StateConflict):
        quotes.issue(conn, qid, actor="sub-issuer")
    assert quotes.get_quote(conn, qid)["quote_number"] == _number(1) and _last(conn) == 1


def test_an_approved_quote_that_somehow_has_a_number_is_not_given_another(live_db):
    conn, dist = live_db
    (qid,) = _approved(conn, dist)
    with conn.cursor() as cur:
        cur.execute("UPDATE proc.bp_sales_quote SET quote_number = %s WHERE sales_quote_id = %s",
                    (f"{SENTINEL}-Q-HAND", qid))
    conn.commit()
    with pytest.raises(StateConflict, match="already has the number"):
        quotes.issue(conn, qid, actor="sub-issuer")
    assert _last(conn) is None


def test_two_quotes_cannot_hold_the_same_number(live_db):
    conn, dist = live_db
    first, second = _approved(conn, dist, 2)
    quotes.issue(conn, first, actor="sub-issuer")
    with pytest.raises(psycopg2.errors.UniqueViolation):
        with conn.cursor() as cur:
            cur.execute("UPDATE proc.bp_sales_quote SET quote_number = %s "
                        "WHERE sales_quote_id = %s", (_number(1), second))
    conn.rollback()


def test_quotes_issued_at_the_same_moment_take_consecutive_numbers(live_db):
    """Eight issues released together on eight connections: eight numbers, 1 to 8, no gap
    and no repeat."""
    conn, dist = live_db
    ids = _approved(conn, dist, 8)
    start, got, errors = threading.Barrier(len(ids)), [], []

    def _issue(qid):
        own = _connect()
        try:
            start.wait(timeout=20)
            got.append(quotes.issue(own, qid, actor="sub-issuer")["quote_number"])
        except Exception as exc:                            # pragma: no cover - reported below
            errors.append(exc)
        finally:
            own.close()

    threads = [threading.Thread(target=_issue, args=(qid,)) for qid in ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert errors == []
    assert sorted(got) == [_number(n) for n in range(1, 9)]
    assert _last(conn) == 8


def test_a_second_allocation_waits_for_the_first_and_follows_it(live_db):
    """The lock itself. A takes a number and has not committed; B must not be given one
    until A is done, and is then given the next."""
    conn, _ = live_db
    a, b = _connect(), _connect()
    result, done = [], threading.Event()
    try:
        cur_a = a.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        assert numbering.allocate(cur_a, SERIES) == (YEAR, 1)

        def _second():
            result.append(numbering.allocate(
                b.cursor(cursor_factory=psycopg2.extras.RealDictCursor), SERIES))
            done.set()

        t = threading.Thread(target=_second)
        t.start()
        assert not done.wait(timeout=1.0), "B was given a number while A still held the series"
        a.commit()
        assert done.wait(timeout=20)
        t.join()
        assert result == [(YEAR, 2)]
        b.commit()
    finally:
        a.close()
        b.close()
    assert _last(conn) == 2


def test_a_rolled_back_allocation_is_handed_out_again(live_db):
    conn, _ = live_db
    a = _connect()
    try:
        cur = a.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        assert numbering.allocate(cur, SERIES) == (YEAR, 1)
        a.commit()
        assert numbering.allocate(cur, SERIES) == (YEAR, 2)
        a.rollback()
        assert numbering.allocate(cur, SERIES) == (YEAR, 2)
        a.commit()
    finally:
        a.close()
    assert _last(conn) == 2


def test_each_series_counts_on_its_own(live_db):
    conn, _ = live_db
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    assert numbering.allocate(cur, SERIES) == (YEAR, 1)
    assert numbering.allocate(cur, f"{SENTINEL}_other") == (YEAR, 1)
    assert numbering.allocate(cur, SERIES) == (YEAR, 2)
    conn.commit()


def test_issue_on_an_autocommit_connection_is_refused(live_db):
    """Under autocommit the increment would commit on its own and a failed issue would
    burn its number. The service will not run that way."""
    conn, dist = live_db
    (qid,) = _approved(conn, dist)
    own = _connect()
    own.autocommit = True
    try:
        with pytest.raises(RuntimeError, match="transactional"):
            quotes.issue(own, qid, actor="sub-issuer")
    finally:
        own.close()
    assert _last(conn) is None


def test_a_new_quote_belongs_to_the_one_tenant(live_db):
    conn, dist = live_db
    """And so a reference cannot be used twice: with no tenant on the row, it could."""
    qid, other = _approved(conn, dist, 2)
    assert quotes.get_quote(conn, qid)["tenant_id"] == "default"
    with pytest.raises(psycopg2.errors.UniqueViolation):
        with conn.cursor() as cur:
            cur.execute("UPDATE proc.bp_sales_quote SET quote_ref = "
                        "(SELECT quote_ref FROM proc.bp_sales_quote WHERE sales_quote_id = %s) "
                        "WHERE sales_quote_id = %s", (qid, other))
    conn.rollback()


def test_the_number_is_written_one_way():
    assert numbering.formatted("Q", 2026, 7) == "Q-2026-0007"
    assert numbering.formatted("Q", 2026, 12345) == "Q-2026-12345"


def test_the_customers_page_shows_the_number_once_issued_and_passes_its_gate(live_db):
    from src.services.rga import quote_pipeline

    conn, dist = live_db
    (qid,) = _approved(conn, dist)
    before = quote_pipeline.produce(quotes.get_quote(conn, qid),
                                    quotes.get_quote_internal(conn, qid))
    assert b"Quote number" not in before.content
    quotes.issue(conn, qid, actor="sub-issuer")
    after = quote_pipeline.produce(quotes.get_quote(conn, qid),
                                   quotes.get_quote_internal(conn, qid))
    assert f"Quote number {_number(1)}".encode() in after.content
