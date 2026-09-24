"""A report's sign-off decisions in proc.bp_approval, against Postgres.

    PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest tests/services/rga/test_signoff_store_live.py

Every row a test writes is keyed by a job id unique to that test and deleted after it
(bp_approval is not append-only; bp_agent_actions is, and nothing here writes to it).
"""
from __future__ import annotations

import os
import uuid

import pytest

from src.services import approval_store as store
from src.services.db import get_conn

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in ("1", "true", "yes", "on")
pytestmark = pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")


@pytest.fixture
def job():
    jid = f"rpt-test-{uuid.uuid4().hex[:8]}"
    yield jid
    with get_conn() as c, c.cursor() as cur:
        cur.execute("DELETE FROM proc.bp_approval WHERE grounding->>'report_job_id' = %s", (jid,))


def _sign(job, by="approver-1"):
    return store.record_approval(
        rfq_id=None, workflow_id=job, unique_id=None, supplier_id=None, actioned_by=by,
        policy_name="ReportSignoffAuthorityPolicy",
        grounding_extra={"report_job_id": job, "run_id": "FP-x", "deck_sha256": "h", "reason": None})


def test_no_decision_yet(job):
    assert store.find_report_decision(job) is None


def test_a_sign_off_is_found_with_its_deck_hash(job):
    aid = _sign(job)
    d = store.find_report_decision(job)
    assert (d["approval_id"], d["status"], d["actioned_by"]) == (aid, "approved", "approver-1")
    assert d["grounding"]["deck_sha256"] == "h"


def test_the_newest_decision_wins(job):
    _sign(job)
    store.record_report_refusal(job_id=job, run_id="FP-x", actioned_by="approver-2",
                                reason="figures wrong", policy_name="ReportSignoffAuthorityPolicy")
    d = store.find_report_decision(job)
    assert (d["status"], d["decision"], d["actioned_by"]) == ("refused", "deny", "approver-2")
    assert d["grounding"]["reason"] == "figures wrong"


def test_another_job_s_decision_is_not_this_one_s(job):
    _sign(job + "-other")
    try:
        assert store.find_report_decision(job) is None
    finally:
        with get_conn() as c, c.cursor() as cur:
            cur.execute("DELETE FROM proc.bp_approval WHERE grounding->>'report_job_id' = %s",
                        (job + "-other",))


def test_a_refusal_must_name_a_person_and_a_reason(job):
    with pytest.raises(ValueError):
        store.record_report_refusal(job_id=job, run_id=None, actioned_by=" ", reason="x",
                                    policy_name=None)
    with pytest.raises(ValueError):
        store.record_report_refusal(job_id=job, run_id=None, actioned_by="a", reason=" ",
                                    policy_name=None)
    assert store.find_report_decision(job) is None
