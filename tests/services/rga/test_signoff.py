"""Report sign-off: which reports need it and who may review one first -- all policy.

A fake policy engine stands in for proc.bp_policy; the roles table is the shape
RoleDefinitionPolicy carries. Every unreadable answer must be the cautious one.
"""
from types import SimpleNamespace

import pytest

from src.services.rga import signoff


class Engine:
    def __init__(self, policies):
        self.policies = policies

    def get_policy(self, slug):
        return self.policies.get(slug)


ROLES = {"Viewer": {"rank": 1, "allow": ["read"]},
         "Buyer": {"rank": 2, "allow": ["read", "compute", "write"]},
         "Approver": {"rank": 3, "allow": ["read", "compute", "write", "transact"]},
         "Admin": {"rank": 4, "allow": ["read", "compute", "write", "transact", "share"]}}


def eng(signoff_rules=None, authority=True, required_role="Approver"):
    p = {}
    if signoff_rules is not None:
        p["report_signoff"] = {"details": {"rules": signoff_rules}}
    if authority:
        p["report_signoff_authority"] = {"details": {
            "required_role": required_role, "applies_to": ["report.signoff"],
            "rules": {"effect": "allow"}}}
    return Engine(p)


@pytest.fixture
def roles(monkeypatch):
    def use(role):
        monkeypatch.setattr(signoff.rbac, "effective_role", lambda p, policy_engine=None: role)
        monkeypatch.setattr(signoff.rbac, "_roles_table", lambda policy_engine=None: ROLES)
    return use


def test_all_reports_need_sign_off_when_the_policy_says_star():
    e = eng({"requires_signoff": ["*"], "self_approval": "deny"})
    assert signoff.required("exec_procurement_summary", engine=e) is True


def test_a_listed_type_needs_it_and_an_unlisted_one_does_not():
    e = eng({"requires_signoff": ["board_paper"], "self_approval": "deny"})
    assert signoff.required("board_paper", engine=e) is True
    assert signoff.required("exec_procurement_summary", engine=e) is False


def test_an_unreadable_policy_holds_every_report():
    for e in (eng(None), eng({"self_approval": "deny"}), eng({"requires_signoff": "yes"}),
              eng({"requires_signoff": [1, 2]})):
        assert signoff.required("anything", engine=e) is True
        assert signoff.policy(engine=e)["readable"] is False


def test_an_engine_that_raises_holds_every_report():
    class Broken:
        def get_policy(self, slug):
            raise RuntimeError("database down")
    assert signoff.required("anything", engine=Broken()) is True


def test_self_approval_is_denied_unless_the_policy_says_allow():
    assert signoff.self_approval_denied(engine=eng({"requires_signoff": ["*"]})) is True
    assert signoff.self_approval_denied(
        engine=eng({"requires_signoff": ["*"], "self_approval": "allow"})) is False
    assert signoff.self_approval_denied(engine=eng(None)) is True


@pytest.mark.parametrize("role, ok", [("Viewer", False), ("Buyer", False),
                                      ("Approver", True), ("Admin", True)])
def test_who_may_review_before_sign_off_follows_the_authority_policy(roles, role, ok):
    roles(role)
    assert signoff.may_sign_off(SimpleNamespace(subject="u"),
                                engine=eng({"requires_signoff": ["*"]})) is ok


def test_raising_the_required_role_in_policy_moves_it(roles):
    """Nothing about who is in code: an Admin-only authority row locks Approvers out."""
    roles("Approver")
    e = eng({"requires_signoff": ["*"]}, required_role="Admin")
    assert signoff.may_sign_off(SimpleNamespace(subject="u"), engine=e) is False


def test_no_authority_policy_means_nobody_reviews_early(roles):
    roles("Admin")
    assert signoff.may_sign_off(SimpleNamespace(subject="u"),
                                engine=eng({}, authority=False)) is False


# ---------------------------------------------------------------------------
# a job's sign-off state: policy + the newest decision
# ---------------------------------------------------------------------------
def job(status="released", rtype="exec_procurement_summary"):
    return {"job_id": "rpt-1", "report_type": rtype, "status": status, "requested_by": "buyer-1"}


def test_a_released_report_with_no_decision_awaits_sign_off():
    s = signoff.state(job(), engine=eng({"requires_signoff": ["*"]}), decision=None)
    assert (s["required"], s["state"]) == (True, "awaiting")


def test_the_newest_decision_decides():
    e = eng({"requires_signoff": ["*"]})
    s = signoff.state(job(), engine=e, decision={
        "approval_id": 7, "status": "approved", "actioned_by": "ap-1", "actioned_at": None,
        "grounding": {"deck_sha256": "h"}})
    assert (s["state"], s["by"], s["approval_id"], s["deck_sha256"]) == ("signed_off", "ap-1", 7, "h")
    r = signoff.state(job(), engine=e, decision={
        "approval_id": 8, "status": "refused", "actioned_by": "ap-2", "actioned_at": None,
        "grounding": {"reason": "figures wrong"}})
    assert (r["state"], r["reason"]) == ("refused", "figures wrong")


def test_an_unknown_decision_status_is_not_a_sign_off():
    s = signoff.state(job(), engine=eng({"requires_signoff": ["*"]}), decision={
        "approval_id": 9, "status": "revoked", "actioned_by": "ap-1", "actioned_at": None,
        "grounding": {}})
    assert s["state"] == "awaiting"


def test_a_report_that_is_not_released_has_nothing_to_sign():
    for status in ("queued", "running", "blocked", "failed"):
        assert signoff.state(job(status), engine=eng({"requires_signoff": ["*"]}),
                             decision=None)["state"] == "not_released"


def test_a_type_the_policy_does_not_list_needs_no_sign_off():
    s = signoff.state(job(), engine=eng({"requires_signoff": ["board_paper"]}), decision=None)
    assert (s["required"], s["state"]) == (False, "not_required")


def test_the_deck_hash_is_sha256():
    import hashlib
    assert signoff.deck_hash(b"PK") == hashlib.sha256(b"PK").hexdigest()


def test_a_list_of_jobs_reads_its_decisions_in_one_lookup():
    """Final review: one connection per job view -- up to 100 per /attention call -- and each
    a scan of bp_approval. A list reads every decision in a single query."""
    calls = []

    def fetch(ids):
        calls.append(list(ids))
        return {"rpt-2": {"approval_id": 1, "status": "approved", "actioned_by": "ap",
                          "actioned_at": None, "grounding": {"deck_sha256": "h"}}}

    jobs = [dict(job(), job_id="rpt-1"), dict(job(), job_id="rpt-2"),
            dict(job(status="running"), job_id="rpt-3")]
    got = signoff.states_for(jobs, engine=eng({"requires_signoff": ["*"]}), fetch=fetch)
    assert calls == [["rpt-1", "rpt-2"]]              # one lookup, released jobs only
    assert [got[j]["state"] for j in ("rpt-1", "rpt-2", "rpt-3")] == ["awaiting", "signed_off", "not_released"]


def test_a_list_that_needs_no_sign_off_looks_nothing_up():
    calls = []
    got = signoff.states_for([job()], engine=eng({"requires_signoff": []}),
                             fetch=lambda ids: calls.append(ids) or {})
    assert calls == [] and got["rpt-1"]["state"] == "not_required"
