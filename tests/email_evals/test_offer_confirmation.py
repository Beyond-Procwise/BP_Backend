"""A person confirming (or rejecting) a price read from an email, on a real Postgres: it names the figures they saw, it is once only,
and it never touches a row that is not the one they looked at."""

import importlib
from contextlib import contextmanager

import pytest

from tests.email_evals.test_learning import db  # noqa: F401  (fixture)

repo = importlib.import_module("repositories.supplier_response_repo")


@pytest.fixture
def conn(db, monkeypatch):
    with db.cursor() as cur:
        cur.execute("TRUNCATE proc.supplier_response")

    @contextmanager
    def get_conn():
        yield db
    monkeypatch.setattr(repo, "get_conn", get_conn)
    return db


def insert(conn, status="extracted_unverified", price=47.5, lead=14, uid="U-1"):
    with conn.cursor() as cur:
        cur.execute("INSERT INTO proc.supplier_response (workflow_id, unique_id, supplier_id, response_message_id, price, lead_time, extraction_status) "
                    "VALUES ('wf-1', %s, 'S-1', %s, %s, %s, %s) RETURNING id", (uid, f"<{uid}>", price, lead, status))
        return cur.fetchone()[0]


def state(conn, rid):
    with conn.cursor() as cur:
        cur.execute("SELECT extraction_status, confirmed_by, confirmed_at IS NOT NULL FROM proc.supplier_response WHERE id = %s", (rid,))
        return cur.fetchone()


def decide(rid, action="confirm", by="buyer@acme.test", price=47.5, lead_time=14):
    return repo.decide_extraction(rid, by, action, seen_price=price, seen_lead_time=lead_time)


def test_confirming_what_you_saw_marks_the_row_confirmed_by_you(conn):
    rid = insert(conn)
    assert decide(rid) == {"ok": True}
    assert state(conn, rid) == ("confirmed", "buyer@acme.test", True)


def test_rejecting_marks_it_rejected_and_needs_no_figures(conn):
    rid = insert(conn)
    assert repo.decide_extraction(rid, "buyer@acme.test", "reject") == {"ok": True}
    assert state(conn, rid) == ("rejected", "buyer@acme.test", True)


def test_a_row_of_unrecorded_origin_can_be_vouched_for(conn):
    rid = insert(conn, status=None)
    assert decide(rid)["ok"] is True and state(conn, rid)[0] == "confirmed"


@pytest.mark.parametrize("seen", [dict(price=47.6), dict(price=None), dict(lead_time=15), dict(lead_time=None)])
def test_figures_that_differ_from_the_row_are_refused_and_nothing_changes(conn, seen):
    rid = insert(conn)
    out = decide(rid, **seen)
    assert out["ok"] is False and "changed" in out["error"] and state(conn, rid)[0] == "extracted_unverified"


def test_a_decision_is_once_only(conn):
    rid = insert(conn)
    assert decide(rid)["ok"] is True
    again = decide(rid, by="someone.else@acme.test")
    assert again["ok"] is False and "already" in again["error"] and state(conn, rid)[1] == "buyer@acme.test"
    assert repo.decide_extraction(rid, "x@acme.test", "reject")["ok"] is False and state(conn, rid)[0] == "confirmed"


def test_only_the_named_row_is_touched(conn):
    a, b = insert(conn, uid="U-1"), insert(conn, uid="U-2")
    decide(a)
    assert state(conn, b)[0] == "extracted_unverified"


def test_a_missing_row_and_bad_requests_are_refused(conn):
    rid = insert(conn)
    assert "no such" in decide(rid + 999)["error"]
    assert decide(rid, by="  ")["ok"] is False and decide(rid, by=None)["ok"] is False
    assert decide(rid, action="approve")["ok"] is False
    assert state(conn, rid)[0] == "extracted_unverified"


def test_a_row_with_nothing_extracted_has_nothing_to_confirm(conn):
    rid = insert(conn, price=None, lead=None)
    out = decide(rid, price=None, lead_time=None)
    assert out["ok"] is False and "nothing" in out["error"]


def test_a_database_without_the_columns_says_so_and_does_not_raise(conn):
    rid = insert_plain(conn)
    with conn.cursor() as cur:
        cur.execute("ALTER TABLE proc.supplier_response DROP COLUMN confirmed_by")
    try:
        out = decide(rid)
    finally:
        with conn.cursor() as cur:
            cur.execute("ALTER TABLE proc.supplier_response ADD COLUMN IF NOT EXISTS confirmed_by TEXT")
    assert out["ok"] is False and "not available" in out["error"]


def insert_plain(conn):
    with conn.cursor() as cur:
        cur.execute("INSERT INTO proc.supplier_response (workflow_id, unique_id, supplier_id, response_message_id, price, lead_time) "
                    "VALUES ('wf-1', 'U-9', 'S-1', '<9>', 47.5, 14) RETURNING id")
        return cur.fetchone()[0]



def test_the_name_is_stored_trimmed(conn):
    rid = insert(conn)
    assert decide(rid, by="  buyer@acme.test  ")["ok"] is True and state(conn, rid)[1] == "buyer@acme.test"
