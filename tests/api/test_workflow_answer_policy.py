"""D-60 / D-62: who may answer a paused workflow's question is an administrator-editable policy.

No database, GPU or model: the policy row, the run record and the audit are faked.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from api.auth import require_user
from api.routers import agent_workflows as aw
from services import permission_settings as ps
from services import rbac, workflow_answer_policy as wap

RANK = {"Viewer": 1, "Buyer": 2, "Approver": 3, "Admin": 4}


@dataclass
class P:
    subject: str
    role: str = "Buyer"
    claims: Dict[str, Any] = field(default_factory=dict)


@pytest.fixture(autouse=True)
def world(monkeypatch):
    """Roles come from the principal's `role`; the policy value from `state['mode']`."""
    state: Dict[str, Any] = {"mode": "approver_only", "unreadable": False, "started_by": "starter",
                             "kind": {"field_type": "text", "required_field": "s3_prefix", "agent_slug": "x"},
                             "audit": [], "answered": []}

    monkeypatch.setattr(rbac, "policy_engine", lambda: object())
    monkeypatch.setattr(rbac, "effective_role", lambda p, policy_engine=None: getattr(p, "role", "Viewer"))
    monkeypatch.setattr(rbac, "role_rank", lambda r, policy_engine=None: RANK.get(str(r), 0))

    def read(key, conn=None):
        if state["unreadable"]:
            raise RuntimeError("database down")
        return state["mode"]
    monkeypatch.setattr(ps, "choice_value", read)

    monkeypatch.setattr(aw.reqrepo, "initiator_for", lambda run_id: state["started_by"])
    monkeypatch.setattr(aw.reqrepo, "request_kind", lambda rid: state["kind"])
    monkeypatch.setattr(aw.reqrepo, "request_run_id", lambda rid: "run-1")
    monkeypatch.setattr(aw.reqrepo, "answer", lambda *a: state["answered"].append(a) or True)
    monkeypatch.setattr(aw.reqrepo, "open_requests", lambda run_id: [{"request_id": 2}])  # still open: stop there
    monkeypatch.setattr(aw.agent_actions, "record_action_or_fail", lambda **kw: state["audit"].append(kw))
    return state


def call(world, who: Optional[P], approval=False):
    world["kind"] = ({"field_type": "approval", "required_field": "quote_approval", "agent_slug": "x"}
                     if approval else {"field_type": "text", "required_field": "s3_prefix", "agent_slug": "x"})
    app = FastAPI()
    app.include_router(aw.router)
    app.dependency_overrides[require_user] = lambda: who
    return TestClient(app).post("/agent-workflows/runs/run-1/input", json={"request_id": 1, "answer": "yes"})


# --- approver_only (the default) -------------------------------------------

def test_approver_may_answer_an_approval(world):
    r = call(world, P("manager", "Approver"), approval=True)
    assert r.status_code == 200 and world["answered"]


def test_admin_may_answer(world):
    assert call(world, P("boss", "Admin"), approval=True).status_code == 200


def test_starter_is_refused_an_approval_even_when_an_approver(world):
    r = call(world, P("starter", "Approver"), approval=True)
    assert r.status_code == 403 and "started this run" in r.json()["detail"]
    assert not world["answered"]


def test_non_approver_is_refused(world):
    r = call(world, P("buyer2", "Buyer"), approval=True)
    assert r.status_code == 403 and "approver" in r.json()["detail"]
    assert not world["answered"]


def test_non_approval_question_still_needs_an_approver_under_approver_only(world):
    assert call(world, P("buyer2", "Buyer"), approval=False).status_code == 403
    assert call(world, P("starter", "Buyer"), approval=False).status_code == 403
    assert call(world, P("manager", "Approver"), approval=False).status_code == 200


def test_starter_approver_may_answer_a_non_approval_question(world):
    # the starter bar is for approvals only
    assert call(world, P("starter", "Approver"), approval=False).status_code == 200


# --- the other two values ---------------------------------------------------

def test_starter_or_approver(world):
    world["mode"] = "starter_or_approver"
    assert call(world, P("starter", "Buyer")).status_code == 200
    assert call(world, P("manager", "Approver")).status_code == 200
    assert call(world, P("buyer2", "Buyer")).status_code == 403
    # the starter still may not approve their own approval question
    assert call(world, P("starter", "Buyer"), approval=True).status_code == 403


def test_any_signed_in(world):
    world["mode"] = "any_signed_in"
    assert call(world, P("buyer2", "Viewer")).status_code == 200
    assert call(world, P("buyer2", "Viewer"), approval=True).status_code == 200
    assert call(world, P("starter", "Buyer"), approval=True).status_code == 403


# --- fail closed -------------------------------------------------------------

def test_unreadable_policy_means_admin_only(world):
    world["unreadable"] = True
    assert call(world, P("manager", "Approver")).status_code == 403
    assert call(world, P("starter", "Buyer")).status_code == 403
    assert call(world, P("boss", "Admin")).status_code == 200


def test_unknown_stored_value_means_admin_only(world):
    world["mode"] = "anybody_at_all"
    assert call(world, P("manager", "Approver")).status_code == 403
    assert call(world, P("boss", "Admin")).status_code == 200


def test_unreadable_policy_does_not_let_the_starter_approve(world):
    world["unreadable"] = True
    assert call(world, P("starter", "Admin"), approval=True).status_code == 403


def test_no_principal_is_refused(world):
    world["mode"] = "any_signed_in"
    assert call(world, None).status_code == 403


# --- every attempt is audited -----------------------------------------------

def test_allowed_and_refused_attempts_are_audited(world):
    call(world, P("manager", "Approver"))
    call(world, P("buyer2", "Buyer"))
    assert [a["status"] for a in world["audit"]] == ["allowed", "denied"]
    assert world["audit"][1]["details"]["principal"] == "buyer2"


# --- approval-question detection --------------------------------------------

def test_what_counts_as_an_approval_question():
    f = wap.is_approval_question
    assert f({"field_type": "approval", "required_field": "x"})
    assert f({"field_type": "text", "required_field": "quote_approval"})
    assert not f({"field_type": "text", "required_field": "s3_prefix"})
    assert f(None)   # unknown request: strict reading


# --- the setting itself -----------------------------------------------------

def test_default_is_approver_only_and_listed_for_admin_config():
    c = ps.choice_setting("workflow_question_answer")
    assert c.default == "approver_only"
    assert set(c.choices) == {"approver_only", "starter_or_approver", "any_signed_in"}


def test_the_default_choice_is_not_a_role_setting():
    assert "workflow_question_answer" not in {s.key for s in ps.CATALOGUE}


def test_current_mode_reports_admin_only_for_unreadable_or_unknown(world):
    assert wap.current_mode() == "approver_only"
    world["mode"] = "anybody_at_all"
    assert wap.current_mode() == wap.ADMIN_ONLY
    world["mode"] = "any_signed_in"
    world["unreadable"] = True
    assert wap.current_mode() == wap.ADMIN_ONLY
