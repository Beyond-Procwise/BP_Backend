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


# ---------------------------------------------------------------------------
# decide(): one decision per awaiting deck, under a row lock
# ---------------------------------------------------------------------------
from src.services.rga import job_store, signoff  # noqa: E402


@pytest.fixture
def released(monkeypatch):
    monkeypatch.setattr(job_store, "_on_healed", lambda rows: None)
    rtype = f"test_{uuid.uuid4().hex[:10]}"
    # Clean up whatever setup managed to write, even if setup itself fails: pytest runs a
    # fixture's teardown only after a successful yield, and a half-built job otherwise stays
    # behind (2026-09-24: seven did, and the running API then healed them to failed).
    try:
        job, _ = job_store.create(rtype, scope={"period_start": "2026-01-01", "period_end": "2026-03-31"},
                                  as_of="2026-09-24", requested_by="buyer-1", entitlement=None)
        job_store.claim(job["job_id"])
        job_store.finish_released(job["job_id"], run_id=job["run_id"], stage_reached="RELEASE",
                                  deck=b"PK-live-deck", media_type="application/x", filename="d.pptx",
                                  page=b"<html>live page</html>", page_media_type="text/html; charset=utf-8")
        yield job
    finally:
        with get_conn() as c, c.cursor() as cur:
            cur.execute("DELETE FROM proc.bp_approval WHERE grounding->>'report_job_id' IN "
                        "(SELECT job_id FROM proc.bp_report_job WHERE report_type = %s)", (rtype,))
            cur.execute("DELETE FROM proc.bp_report_job WHERE report_type = %s", (rtype,))


def test_a_sign_off_is_bound_to_the_stored_deck(released):
    s = signoff.decide(released["job_id"], verdict="sign_off", by="approver-1",
                       reason=None, policy_name="ReportSignoffAuthorityPolicy")
    assert s["state"] == "signed_off" and s["by"] == "approver-1"
    assert s["deck_sha256"] == signoff.deck_hash(b"PK-live-deck")


def test_a_decided_deck_cannot_be_decided_again(released):
    signoff.decide(released["job_id"], verdict="sign_off", by="approver-1", reason=None,
                   policy_name="ReportSignoffAuthorityPolicy")
    with pytest.raises(signoff.NotDecidable) as e:
        signoff.decide(released["job_id"], verdict="refuse", by="approver-2", reason="late",
                       policy_name="ReportSignoffAuthorityPolicy")
    assert e.value.state == "signed_off"
    assert store.find_report_decision(released["job_id"])["status"] == "approved"


def test_a_refusal_holds_the_deck_with_its_reason(released):
    s = signoff.decide(released["job_id"], verdict="refuse", by="approver-2",
                       reason="figures wrong", policy_name="ReportSignoffAuthorityPolicy")
    assert (s["state"], s["reason"]) == ("refused", "figures wrong")


def test_awaiting_and_refused_decks_need_attention_signed_off_ones_do_not(released):
    rtype = released["report_type"]
    listed = lambda: {j["job_id"] for j in job_store.needs_attention(200) if j["report_type"] == rtype}
    assert released["job_id"] in listed()                       # awaiting
    signoff.decide(released["job_id"], verdict="refuse", by="a", reason="no",
                   policy_name="ReportSignoffAuthorityPolicy")
    assert released["job_id"] in listed()                       # refused: still needs a person


def test_a_signed_off_deck_leaves_attention(released):
    signoff.decide(released["job_id"], verdict="sign_off", by="a", reason=None,
                   policy_name="ReportSignoffAuthorityPolicy")
    rtype = released["report_type"]
    assert released["job_id"] not in {j["job_id"] for j in job_store.needs_attention(200)
                                      if j["report_type"] == rtype}


def test_an_unknown_job_cannot_be_decided():
    with pytest.raises(LookupError):
        signoff.decide("rpt-does-not-exist", verdict="sign_off", by="a", reason=None,
                       policy_name=None)


def test_many_jobs_decisions_are_read_in_one_go(job):
    other = job + "-b"
    try:
        _sign(job)
        store.record_report_refusal(job_id=job, run_id=None, actioned_by="ap-2", reason="late",
                                    policy_name=None)
        _sign(other, by="ap-3")
        got = store.find_report_decisions([job, other, job + "-none"])
        assert set(got) == {job, other}
        assert got[job]["status"] == "refused"          # newest wins, per job
        assert got[other]["actioned_by"] == "ap-3"
        assert store.find_report_decisions([]) == {}
    finally:
        with get_conn() as c, c.cursor() as cur:
            cur.execute("DELETE FROM proc.bp_approval WHERE grounding->>'report_job_id' = %s", (other,))



def test_a_sign_off_binds_the_page_too(released):
    s = signoff.decide(released["job_id"], verdict="sign_off", by="approver-1",
                       reason=None, policy_name="ReportSignoffAuthorityPolicy")
    assert s["page_sha256"] == signoff.deck_hash(b"<html>live page</html>")
    assert job_store.page(released["job_id"]) == (b"<html>live page</html>",
                                                  "text/html; charset=utf-8")
    assert job_store.get(released["job_id"])["has_page"] is True


def test_a_page_is_stored_only_for_a_released_job(released):
    """The table refuses a page on a job that is not released (ck_bp_report_job_page)."""
    job, _ = job_store.create(released["report_type"], scope={"period_start": "2026-04-01"},
                              as_of="2026-09-24", requested_by="t", entitlement=None)
    with pytest.raises(Exception, match="ck_bp_report_job_page"):
        with get_conn() as c, c.cursor() as cur:
            cur.execute("UPDATE proc.bp_report_job SET page = %s WHERE job_id = %s",
                        (b"<html>", job["job_id"]))



def test_a_release_with_its_pack_writes_version_one(released):
    """The fixture releases without a pack (legacy); this one releases as the worker now does."""
    rtype = released["report_type"]
    job, _ = job_store.create(rtype, scope={"period_start": "2026-07-01"}, as_of="2026-09-24",
                              requested_by="buyer-1", entitlement=None)
    job_store.claim(job["job_id"])
    pack = {"pack_id": "FP-x", "facts": [{"fact_id": "F0001"}]}
    ast = {"sections": [{"id": "s", "title": "S", "blocks": []}]}
    job_store.finish_released(job["job_id"], run_id="FP-x", stage_reached="RELEASE",
                              deck=b"PK-v1", media_type="application/x", filename="d.pptx",
                              page=b"<html>v1</html>", page_media_type="text/html",
                              fact_pack=pack, ast=ast, title="My title")
    got = job_store.get(job["job_id"])
    assert (got["current_version"], got["editable"], got["title"]) == (1, True, "My title")
    draft = job_store.draft(job["job_id"])
    assert draft["version"] == 1 and draft["ast"] == ast and draft["fact_pack"] == pack
    assert draft["title"] == "My title"
    assert job_store.get(released["job_id"])["editable"] is False     # released without a pack


def test_a_sign_off_records_its_version_and_an_edit_voids_it(released):
    from src.services.db import get_conn as _conn
    with _conn() as c, c.cursor() as cur:
        cur.execute("UPDATE proc.bp_report_job SET current_version = 1 WHERE job_id = %s",
                    (released["job_id"],))
    s = signoff.decide(released["job_id"], verdict="sign_off", by="approver-1", reason=None,
                       policy_name="ReportSignoffAuthorityPolicy")
    assert s["state"] == "signed_off"
    assert store.find_report_decision(released["job_id"])["grounding"]["version"] == 1
    with _conn() as c, c.cursor() as cur:        # an edit saved version 2
        cur.execute("UPDATE proc.bp_report_job SET current_version = 2 WHERE job_id = %s",
                    (released["job_id"],))
    assert signoff.state(job_store.get(released["job_id"]))["state"] == "awaiting"
