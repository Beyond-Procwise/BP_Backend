"""Stamping where an extracted value came from, against a real Postgres: the writer is best-effort, never overwrites a person's word,
and a database without the columns costs nothing."""

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


def insert(conn, mid="<m1>", status=None, price=47.5, lead=14, uid="U-1"):
    # a distinct dispatch id per row: production's table is UNIQUE (workflow_id, unique_id), which the generated test schema omits
    with conn.cursor() as cur:
        cur.execute("INSERT INTO proc.supplier_response (workflow_id, unique_id, supplier_id, response_message_id, price, lead_time, extraction_status) "
                    "VALUES ('wf-1', %s, 'S-1', %s, %s, %s, %s) RETURNING id", (uid, mid, price, lead, status))
        return cur.fetchone()[0]


def read(conn, rid):
    with conn.cursor() as cur:
        cur.execute("SELECT extraction_status, extraction_method, extraction_model, extraction_prompt_version, extraction_confidence, "
                    "extracted_at IS NOT NULL, confirmed_by FROM proc.supplier_response WHERE id = %s", (rid,))
        return cur.fetchone()


def stamp(**over):
    base = dict(workflow_id="wf-1", unique_id="U-1", response_message_id="<m1>", method="regex_first_number", model=None,
                prompt_version="inline:supplier_interaction_agent._analyze_response_with_llm")
    base.update(over)
    return repo.record_extraction(**base)


def test_an_extracted_row_is_stamped_with_how_it_was_read(conn):
    rid = insert(conn)
    assert stamp() is True
    status, method, model, prompt, conf, at, who = read(conn, rid)
    assert (status, method, model, conf, who) == ("extracted_unverified", "regex_first_number", None, None, None) and at is True
    assert prompt.startswith("inline:")


def test_a_model_read_records_which_model(conn):
    rid = insert(conn)
    assert stamp(method="llm", model="agentnick:unified") is True
    assert read(conn, rid)[:3] == ("extracted_unverified", "llm", "agentnick:unified")


@pytest.mark.parametrize("status", ["confirmed", "rejected"])
def test_a_persons_word_is_never_overwritten(conn, status):
    rid = insert(conn, status=status)
    assert stamp() is False and read(conn, rid)[0] == status and read(conn, rid)[1] is None


def test_restamping_an_unconfirmed_row_updates_it(conn):
    rid = insert(conn, status="extracted_unverified")
    assert stamp(method="llm", model="m2") is True and read(conn, rid)[1:3] == ("llm", "m2")


def test_a_row_with_nothing_extracted_is_not_stamped(conn):
    rid = insert(conn, price=None, lead=None)
    assert stamp() is False and read(conn, rid)[0] is None


def test_the_row_is_found_by_message_id_or_failing_that_by_unique_id(conn):
    a = insert(conn, mid="<a>")
    assert stamp(response_message_id=None, unique_id="U-1") is True and read(conn, a)[0] == "extracted_unverified"
    assert stamp(response_message_id="<nobody>", unique_id="U-9") is False


def test_other_rows_are_untouched(conn):
    a, b = insert(conn, mid="<a>", uid="U-A"), insert(conn, mid="<b>", uid="U-B")
    stamp(response_message_id="<a>")
    assert read(conn, a)[0] == "extracted_unverified" and read(conn, b)[0] is None


def test_a_database_without_the_columns_costs_nothing(conn, monkeypatch):
    with conn.cursor() as cur:
        cur.execute("ALTER TABLE proc.supplier_response RENAME COLUMN extraction_status TO extraction_status_away")
    try:
        assert stamp() is False                                                    # no exception, no change
    finally:
        with conn.cursor() as cur:
            cur.execute("ALTER TABLE proc.supplier_response RENAME COLUMN extraction_status_away TO extraction_status")


def test_a_failure_connecting_never_raises(monkeypatch):
    def boom():
        raise RuntimeError("database down")
    monkeypatch.setattr(repo, "get_conn", boom)
    assert stamp() is False


def test_the_migration_adds_only_nullable_columns_and_backfills_nothing(db):
    with db.cursor() as cur:
        cur.execute("SELECT column_name, is_nullable, column_default FROM information_schema.columns WHERE table_schema = 'proc' "
                    "AND table_name = 'supplier_response' AND column_name LIKE 'extract%' OR column_name LIKE 'confirmed_%' ORDER BY 1")
        rows = cur.fetchall()
    assert len(rows) >= 8 and all(r[1] == "YES" and r[2] is None for r in rows)          # nothing forced onto existing rows


def test_an_unknown_status_is_refused_by_the_database(conn):
    with pytest.raises(Exception):
        insert(conn, status="probably-fine")
