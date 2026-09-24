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
