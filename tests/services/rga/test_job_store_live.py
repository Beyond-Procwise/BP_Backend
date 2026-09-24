"""The report job table, against Postgres. The rules the table itself holds.

    PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
        tests/services/rga/test_job_store_live.py

Skipped by default: the suite runs against an in-memory fake database, and
what is under test here is two constraints only Postgres enforces. Every row a
test writes carries a report_type unique to that test and is deleted after it.
"""
from __future__ import annotations

import os
import uuid

import pytest

from src.services.rga import job_store

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in (
    "1", "true", "yes", "on")
pytestmark = pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")

SCOPE = {"period_start": "2026-01-01", "period_end": "2026-03-31",
         "period_label": "2026 Q1", "currency": "GBP"}
AS_OF = "2026-09-24"


@pytest.fixture
def rtype():
    name = f"test_{uuid.uuid4().hex[:10]}"
    yield name
    from src.services.db import get_conn

    with get_conn() as c, c.cursor() as cur:
        cur.execute("DELETE FROM proc.bp_report_job WHERE report_type = %s", (name,))


def _new(rtype, **k):
    return job_store.create(rtype, scope=SCOPE, as_of=AS_OF, requested_by="t", **k)


def test_a_new_job_is_queued_and_owned_by_this_process(rtype):
    job, created = _new(rtype)
    assert created is True
    assert job["status"] == "queued"
    assert job["job_id"].startswith("rpt-")
    assert job["requested_by"] == "t"
    assert job_store.get(job["job_id"])["status"] == "queued"


def test_the_same_request_while_active_returns_the_same_job(rtype):
    first, _ = _new(rtype)
    again, created = _new(rtype)
    assert created is False
    assert again["job_id"] == first["job_id"]


def test_a_different_period_is_a_different_job(rtype):
    first, _ = _new(rtype)
    other, created = job_store.create(rtype, scope={**SCOPE, "period_end": "2026-02-28"},
                                      as_of=AS_OF, requested_by="t")
    assert created is True and other["job_id"] != first["job_id"]


def test_once_finished_the_same_request_starts_a_fresh_job(rtype):
    first, _ = _new(rtype)
    assert job_store.claim(first["job_id"])
    job_store.finish_failed(first["job_id"], "boom")
    second, created = _new(rtype)
    assert created is True and second["job_id"] != first["job_id"]


def test_a_job_is_claimed_once(rtype):
    job, _ = _new(rtype)
    assert job_store.claim(job["job_id"]) is True
    assert job_store.claim(job["job_id"]) is False
    assert job_store.get(job["job_id"])["status"] == "running"


def test_a_released_job_carries_its_deck(rtype):
    job, _ = _new(rtype)
    job_store.claim(job["job_id"])
    job_store.finish_released(job["job_id"], run_id="FP-x", stage_reached="RELEASE",
                              deck=b"PK\x03\x04deck", media_type="application/x",
                              filename="r.pptx")
    got = job_store.get(job["job_id"])
    assert got["status"] == "released" and got["run_id"] == "FP-x"
    assert "deck" not in got                     # status reads never carry the bytes
    assert job_store.deck(job["job_id"]) == (b"PK\x03\x04deck", "application/x", "r.pptx")


def test_a_blocked_job_keeps_its_reasons_and_has_no_deck(rtype):
    job, _ = _new(rtype)
    job_store.claim(job["job_id"])
    job_store.finish_blocked(job["job_id"], run_id="FP-x", stage_reached="POST_CHECK",
                             blocking=[{"finding_id": "F1", "detail": "untraced"}])
    got = job_store.get(job["job_id"])
    assert got["status"] == "blocked"
    assert got["blocking"] == [{"finding_id": "F1", "detail": "untraced"}]
    assert job_store.deck(job["job_id"]) is None


def test_the_table_refuses_a_deck_on_a_job_that_was_not_released(rtype):
    """The code never writes one; this proves the table would stop it if it did."""
    from src.services.db import get_conn

    job, _ = _new(rtype)
    with pytest.raises(Exception, match="ck_bp_report_job_deck"):
        with get_conn() as c, c.cursor() as cur:
            cur.execute("UPDATE proc.bp_report_job SET status='blocked', deck=%s "
                        "WHERE job_id=%s", (b"x", job["job_id"]))


def test_a_job_stranded_by_a_restart_is_healed_to_failed(rtype, monkeypatch):
    job, _ = _new(rtype)
    job_store.claim(job["job_id"])
    monkeypatch.setattr(job_store, "OWNER", "a-later-process")
    got = job_store.get(job["job_id"])
    assert got["status"] == "failed"
    assert "restart" in got["error"]


def test_a_stranded_job_does_not_block_a_new_request(rtype, monkeypatch):
    first, _ = _new(rtype)
    monkeypatch.setattr(job_store, "OWNER", "a-later-process")
    second, created = _new(rtype)
    assert created is True and second["job_id"] != first["job_id"]
    assert job_store.get(first["job_id"])["status"] == "failed"


def test_an_unknown_job_is_none():
    assert job_store.get("rpt-does-not-exist") is None
    assert job_store.deck("rpt-does-not-exist") is None
