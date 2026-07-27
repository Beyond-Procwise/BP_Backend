"""Deal summaries must be written to the database the caller was working in.

sync_deal_summaries runs each deal on its own connection, because psycopg2
connections cannot be shared across threads. Those worker connections were
opened from the environment DSN, so a caller working against any other database
-- a test dataset, a restored snapshot, a per-tenant database -- had its
summaries written to whatever the environment pointed at instead. For the
seeder that meant AI summaries landing in live bp_sqldb, breaking the isolation
guarantee.
"""
from __future__ import annotations

from contextlib import contextmanager

import pytest

from src.services import deal_analysis_service as das


class _FakeCursor:
    def __init__(self, conn):
        self.conn = conn

    def execute(self, *args, **kwargs):
        return None

    def fetchall(self):
        return []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeConn:
    def __init__(self, label: str):
        self.label = label

    def cursor(self):
        return _FakeCursor(self)

    def get_dsn_parameters(self):
        return {
            "dbname": self.label, "host": "db.example", "port": "5432",
            "user": "someone",
        }


def _factory_for(label: str, opened: list):
    @contextmanager
    def factory():
        conn = _FakeConn(label)
        opened.append(conn)
        yield conn

    return factory


def test_workers_use_the_injected_factory_and_never_the_environment(monkeypatch):
    opened: list = []
    seen: list = []
    monkeypatch.setattr(das, "generate_for_deal", lambda d, c: seen.append((d, c.label)))

    def forbidden():
        raise AssertionError("get_conn must not be used when a factory is supplied")

    monkeypatch.setattr(das, "get_conn", forbidden)

    result = das.sync_deal_summaries(
        deal_ids=["DEAL-1", "DEAL-2"],
        connect=_factory_for("bp_testdb", opened),
        max_workers=1,
    )

    assert result["processed"] == 2
    assert {label for _, label in seen} == {"bp_testdb"}
    assert len(opened) == 2


def test_a_passed_connection_targets_its_own_database_without_an_explicit_factory(monkeypatch):
    """The dangerous default: a caller hands in a connection and gets summaries
    written somewhere else entirely."""
    seen: list = []
    monkeypatch.setattr(das, "generate_for_deal", lambda d, c: seen.append(c))

    built: list = []

    def fake_connect(**kwargs):
        built.append(kwargs)
        return _FakeConn(kwargs["dbname"])

    monkeypatch.setattr(das, "_connect_like", lambda params: fake_connect(**params))

    def forbidden():
        raise AssertionError("get_conn must not be used when a connection was passed")

    monkeypatch.setattr(das, "get_conn", forbidden)

    das.sync_deal_summaries(
        _FakeConn("bp_testdb"), deal_ids=["DEAL-1"], max_workers=1
    )

    assert built and built[0]["dbname"] == "bp_testdb"


def test_no_connection_and_no_factory_still_uses_the_environment(monkeypatch):
    """Production behaviour is unchanged when nothing is supplied."""
    opened: list = []
    monkeypatch.setattr(das, "generate_for_deal", lambda d, c: None)
    monkeypatch.setattr(das, "get_conn", _factory_for("from_environment", opened))

    das.sync_deal_summaries(deal_ids=["DEAL-1"], max_workers=1)

    assert [c.label for c in opened] == ["from_environment"]


def test_assign_deals_hands_its_connection_to_the_summary_step(monkeypatch):
    from src.services import deal_assignment_service as das_assign

    monkeypatch.setattr(das_assign, "_run", lambda cur: {"linked": 0})

    received: dict = {}

    def fake_sync(conn=None, deal_ids=None, max_workers=2, connect=None):
        received["conn"] = conn
        received["connect"] = connect
        return {"processed": 0}

    monkeypatch.setattr(
        "src.services.deal_analysis_service.sync_deal_summaries", fake_sync
    )

    caller_conn = _FakeConn("bp_testdb")
    das_assign.assign_deals(caller_conn)

    assert received["conn"] is caller_conn


def test_assign_deals_forwards_an_explicit_factory(monkeypatch):
    from src.services import deal_assignment_service as das_assign

    monkeypatch.setattr(das_assign, "_run", lambda cur: {"linked": 0})
    received: dict = {}

    def fake_sync(conn=None, deal_ids=None, max_workers=2, connect=None):
        received["connect"] = connect
        return {"processed": 0}

    monkeypatch.setattr(
        "src.services.deal_analysis_service.sync_deal_summaries", fake_sync
    )

    marker = object()
    das_assign.assign_deals(_FakeConn("bp_testdb"), connect=marker)

    assert received["connect"] is marker
