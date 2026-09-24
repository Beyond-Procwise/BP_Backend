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


# ---------------------------------------------------------------------------
# save_version(): an edit becomes the next version, under a row lock
# ---------------------------------------------------------------------------

def _v1(released):
    job, _ = job_store.create(released["report_type"], scope={"period_start": "2026-08-01"},
                              as_of="2026-09-24", requested_by="buyer-1", entitlement=None)
    job_store.claim(job["job_id"])
    job_store.finish_released(job["job_id"], run_id="FP-x", stage_reached="RELEASE",
                              deck=b"PK-v1", media_type="application/x", filename="d.pptx",
                              page=b"<html>v1</html>", page_media_type="text/html",
                              fact_pack={"pack_id": "FP-x"}, ast={"sections": []}, title="T1")
    return job["job_id"]


def _save(jid, base, **kw):
    args = dict(base_version=base, title="T2", ast={"sections": [{"id": "s"}]},
                deck=b"PK-v2", page=b"<html>v2</html>", by="editor-1", summary="words")
    args.update(kw)
    return job_store.save_version(jid, **args)


def test_an_edit_becomes_the_next_version_and_the_current_files(released):
    jid = _v1(released)
    seen = []
    assert _save(jid, 1, before_commit=seen.append) == 2
    assert seen == [2]
    got = job_store.get(jid)
    assert (got["current_version"], got["last_edited_by"], got["title"]) == (2, "editor-1", "T2")
    assert job_store.deck(jid)[0] == b"PK-v2" and job_store.page(jid)[0] == b"<html>v2</html>"
    d = job_store.draft(jid)
    assert d["version"] == 2 and d["ast"] == {"sections": [{"id": "s"}]}
    assert d["fact_pack"] == {"pack_id": "FP-x"}                        # the pack never changes
    with get_conn() as c, c.cursor() as cur:
        cur.execute("SELECT version, edited_by, summary, deck_sha256 FROM proc.bp_report_version "
                    "WHERE job_id = %s ORDER BY version", (jid,))
        rows = cur.fetchall()
    assert [r[:3] for r in rows] == [(1, None, "released by the reporting agent"),
                                     (2, "editor-1", "words")]
    assert rows[1][3] == signoff.deck_hash(b"PK-v2")


def test_a_stale_base_version_is_refused_and_nothing_changes(released):
    jid = _v1(released)
    _save(jid, 1)
    with pytest.raises(job_store.StaleVersion) as exc:
        _save(jid, 1, deck=b"PK-lost", by="editor-2")
    assert exc.value.current == 2
    got = job_store.get(jid)
    assert (got["current_version"], got["last_edited_by"]) == (2, "editor-1")
    assert job_store.deck(jid)[0] == b"PK-v2"


def test_an_edit_that_cannot_be_audited_is_not_saved(released):
    jid = _v1(released)

    def refuse(version):
        raise RuntimeError("audit down")

    with pytest.raises(RuntimeError):
        _save(jid, 1, before_commit=refuse)
    got = job_store.get(jid)
    assert got["current_version"] == 1 and job_store.deck(jid)[0] == b"PK-v1"
    assert job_store.draft(jid)["version"] == 1


def test_a_job_that_is_not_released_cannot_take_an_edit(released):
    job, _ = job_store.create(released["report_type"], scope={"period_start": "2026-09-01"},
                              as_of="2026-09-24", requested_by="buyer-1", entitlement=None)
    with pytest.raises(job_store.StaleVersion) as exc:
        _save(job["job_id"], 1)
    assert exc.value.current is None


# ---------------------------------------------------------------------------
# final review: the version a sign-off saw, the last editor under the lock, the attention list
# ---------------------------------------------------------------------------

def _set(job_id, **cols):
    sets = ", ".join(f"{k} = %s" for k in cols)
    with get_conn() as c, c.cursor() as cur:
        cur.execute(f"UPDATE proc.bp_report_job SET {sets} WHERE job_id = %s",
                    (*cols.values(), job_id))


def test_a_sign_off_of_a_version_that_has_since_been_replaced_is_refused(released):
    """I2: B reviews version 2, C saves version 3, B clicks Sign off -- that must not sign 3."""
    _set(released["job_id"], current_version=3)
    with pytest.raises(signoff.NotDecidable) as exc:
        signoff.decide(released["job_id"], verdict="sign_off", by="approver-1", reason=None,
                       policy_name="P", seen_version=2)
    assert exc.value.state == "edited"
    assert store.find_report_decision(released["job_id"]) is None
    s = signoff.decide(released["job_id"], verdict="sign_off", by="approver-1", reason=None,
                       policy_name="P", seen_version=3)
    assert s["state"] == "signed_off"


@pytest.mark.parametrize("who", ["buyer-1", "editor-1"])     # the requester; the last editor
def test_self_approval_is_decided_under_the_lock(released, who):
    """I3: the router's check reads the job unlocked; a save committing between that check and
    the lock made the signer the last editor. decide() re-checks on the locked row."""
    _set(released["job_id"], last_edited_by="editor-1", current_version=2)
    with pytest.raises(signoff.SelfApproval) as exc:
        signoff.decide(released["job_id"], verdict="sign_off", by=who, reason=None,
                       policy_name="P", self_approval_denied=True)
    assert exc.value.role == ("requester" if who == "buyer-1" else "editor")
    assert store.find_report_decision(released["job_id"]) is None
    s = signoff.decide(released["job_id"], verdict="sign_off", by=who, reason=None,
                       policy_name="P", self_approval_denied=False)
    assert s["state"] == "signed_off"


def test_an_edited_report_that_was_signed_off_needs_attention_again(released):
    """I5: the attention list dropped any job whose newest decision was 'approved', whatever
    version it was for -- so an edit after sign-off left no approver prompted."""
    listed = lambda: {j["job_id"] for j in job_store.needs_attention(200)
                      if j["report_type"] == released["report_type"]}
    _set(released["job_id"], current_version=1)
    signoff.decide(released["job_id"], verdict="sign_off", by="approver-1", reason=None,
                   policy_name="P")
    assert released["job_id"] not in listed()
    _set(released["job_id"], current_version=2)
    assert released["job_id"] in listed()
