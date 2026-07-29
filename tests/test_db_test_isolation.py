"""The unit suite must not reach the live database by default.

proc.bp_opportunity gained four fixture rows ('Supplier One'/'Supplier Two')
every time tests/test_opportunity_miner_agent.py ran, because the miner persists
findings through src.services.db.get_conn — a chokepoint that built its DSN from
the environment and so bypassed the stub agent_nick the tests inject entirely.

The offline stand-in already existed; it only engaged when the database happened
to be unreachable. So the safety net worked on a laptop with no VPN and did
nothing on a developer machine with a working .env. These tests pin the inverted
default: under pytest the stand-in is used unless a run explicitly opts in.
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from src.services import db as db_mod


def test_live_db_is_off_by_default_under_pytest(monkeypatch):
    monkeypatch.delenv("PROCWISE_TEST_LIVE_DB", raising=False)
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "some::test")

    assert db_mod._use_fake_conn("host=real port=5432") is True


def test_a_run_can_opt_in_to_the_live_database(monkeypatch):
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "some::test")
    monkeypatch.setenv("PROCWISE_TEST_LIVE_DB", "1")

    assert db_mod._use_fake_conn("host=real port=5432") is False


def test_production_is_untouched(monkeypatch):
    """Outside pytest nothing changes — this must never gate the real app."""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("PROCWISE_TEST_LIVE_DB", raising=False)

    assert db_mod._use_fake_conn("host=real port=5432") is False


@pytest.mark.parametrize("value,expected", [
    ("1", False), ("true", False), ("TRUE", False), ("yes", False),
    ("0", True), ("false", True), ("", True), ("  ", True),
])
def test_opt_in_flag_parsing(monkeypatch, value, expected):
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "some::test")
    monkeypatch.setenv("PROCWISE_TEST_LIVE_DB", value)

    assert db_mod._use_fake_conn("host=real port=5432") is expected


def test_miner_file_outputs_stay_out_of_the_repo(monkeypatch):
    """Running the suite rewrote the checked-in opportunity_findings.json/.xlsx.

    Both writers used a bare relative filename, which resolves against the process
    CWD — the repo root for the service AND for pytest.
    """
    from agents import opportunity_miner_agent as oma

    monkeypatch.setenv("PYTEST_CURRENT_TEST", "some::test")
    under_test = oma._output_path("opportunity_findings.json")
    assert os.path.isabs(under_test)
    assert "opportunity_findings.json" in under_test
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    assert not under_test.startswith(repo_root), under_test

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    assert oma._output_path("opportunity_findings.json") == "opportunity_findings.json"


def test_get_conn_yields_the_stand_in_not_a_socket(monkeypatch):
    """The whole point: a test that persists must not reach a real server."""
    monkeypatch.delenv("PROCWISE_TEST_LIVE_DB", raising=False)
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "some::test")

    def _explode(*a, **k):
        raise AssertionError("psycopg2.connect must not be called under pytest")

    if db_mod.psycopg2 is not None:
        monkeypatch.setattr(db_mod.psycopg2, "connect", _explode)

    with db_mod.get_conn() as conn:
        assert isinstance(conn, db_mod._FakeConnection)
