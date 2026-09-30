"""The endpoints: lifecycle, the self-approval bar, and proposal decisions.

The repository is stubbed. What is being tested here is the router's contract
-- which gate it calls, which status code a refusal gets, and that approving a
proposal goes through the one run path rather than a copy of it.
"""

import os
import sys

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from api.routers import playbooks as mod  # noqa: E402


class Principal:
    def __init__(self, subject="bo"):
        self.subject = subject


@pytest.fixture
def client(monkeypatch):
    gated = []
    monkeypatch.setattr(mod, "gate", lambda action, principal, **kw: gated.append(action))
    app = FastAPI()
    app.include_router(mod.router)
    app.dependency_overrides[mod.require_user] = lambda: Principal()
    c = TestClient(app)
    c.gated = gated
    return c


def test_creating_a_playbook_gates_on_playbook_write(client, monkeypatch):
    monkeypatch.setattr(mod.repo, "create", lambda **kw: 12)
    monkeypatch.setattr(mod.repo, "get", lambda pid: {"playbook_id": 12, "playbook_status": "draft"})
    r = client.post("/playbooks", json={
        "playbook_name": "Recover the duplicate",
        "trigger_source": "detection_finding",
        "trigger_match": {"rule_id": "duplicate"},
        "agent_workflow_id": 958,
    })
    assert r.status_code == 200, r.text
    assert r.json()["playbook_id"] == 12
    assert client.gated == ["playbook.write"]


def test_an_unknown_match_key_is_a_400_naming_the_key(client, monkeypatch):
    """Not a 500, and not a stored row. The author gets told which key."""
    def boom(**kw):
        raise ValueError("sevrity is not a match field for detection_finding")
    monkeypatch.setattr(mod.repo, "create", boom)
    r = client.post("/playbooks", json={
        "playbook_name": "Typo",
        "trigger_source": "detection_finding",
        "trigger_match": {"sevrity": "high"},
        "agent_workflow_id": 958,
    })
    assert r.status_code == 400
    assert "sevrity" in r.json()["detail"]


def test_approving_a_playbook_gates_on_playbook_approve(client, monkeypatch):
    monkeypatch.setattr(mod.repo, "approve",
                        lambda pid, approver: {"playbook_id": pid, "playbook_status": "active",
                                               "approved_by": approver})
    r = client.post("/playbooks/12/approve")
    assert r.status_code == 200
    assert r.json()["playbook_status"] == "active"
    assert client.gated == ["playbook.approve"]


def test_self_approval_comes_back_as_a_403(client, monkeypatch):
    """A lifecycle refusal is not a malformed request; it is a refusal."""
    def boom(pid, approver):
        raise mod.LifecycleError("a playbook cannot be approved by its own author")
    monkeypatch.setattr(mod.repo, "approve", boom)
    r = client.post("/playbooks/12/approve")
    assert r.status_code == 403
    assert "own author" in r.json()["detail"]


def test_editing_a_retired_playbook_is_a_400(client, monkeypatch):
    def boom(pid, **kw):
        raise mod.LifecycleError("a retired playbook cannot be edited")
    monkeypatch.setattr(mod.repo, "update", boom)
    r = client.put("/playbooks/12", json={
        "playbook_name": "x", "trigger_source": "detection_finding",
        "trigger_match": {}, "agent_workflow_id": 958,
    })
    assert r.status_code == 400


def test_listing_proposals_needs_no_gate(client, monkeypatch):
    monkeypatch.setattr(mod.repo, "list_proposals",
                        lambda status=None, limit=100: [{"proposal_id": 1}])
    r = client.get("/playbooks/proposals?status=proposed")
    assert r.status_code == 200
    assert r.json()["proposals"] == [{"proposal_id": 1}]
    assert client.gated == []


def test_approving_a_proposal_gates_on_workflow_run_and_uses_the_one_run_path(
    client, monkeypatch
):
    """No new action name, and no second copy of the claim-and-execute path."""
    monkeypatch.setattr(mod.repo, "get_proposal", lambda pid: {
        "proposal_id": 5, "playbook_id": 12, "proposal_status": "proposed",
        "playbook_name": "Recover the duplicate",
        "finding_source": "detection_finding", "finding_id": "4211",
        "deal_id": "D-900", "agent_workflow_id": 958, "params": {"tier": "gold"},
    })
    monkeypatch.setattr(mod, "finding_is_open", lambda source, fid: True)
    # The playbook must still be active, and the proposal is claimed before the
    # run -- see test_a_proposal_is_claimed_before_its_workflow_runs.
    monkeypatch.setattr(mod.repo, "get", lambda pid: {"playbook_status": "active"})
    monkeypatch.setattr(mod.repo, "claim_proposal", lambda pid, by: True)
    started = {}
    def fake_start_run(request, workflow_id, payload, principal):
        started.update(workflow_id=workflow_id, payload=payload)
        return {"run_id": "awf-958-abc", "status": "completed"}
    monkeypatch.setattr(mod, "start_run", fake_start_run)
    monkeypatch.setattr(mod.repo, "mark_proposal_executed", lambda pid, run_id, by: None)

    r = client.post("/playbooks/proposals/5/approve")
    assert r.status_code == 200, r.text
    assert r.json()["run_id"] == "awf-958-abc"
    assert client.gated == ["workflow.run"]
    assert started["workflow_id"] == 958
    # The playbook's static params and the finding it was raised for both reach
    # the graph -- a strategy that does not know which finding it is answering
    # is not a strategy.
    assert started["payload"]["tier"] == "gold"
    assert started["payload"]["finding_id"] == "4211"
    assert started["payload"]["deal_id"] == "D-900"


def test_a_proposal_whose_finding_is_resolved_is_superseded_not_executed(
    client, monkeypatch
):
    """A stale queue must not act on closed work."""
    monkeypatch.setattr(mod.repo, "get_proposal", lambda pid: {
        "proposal_id": 5, "playbook_id": 12, "proposal_status": "proposed",
        "playbook_name": "Recover the duplicate",
        "finding_source": "detection_finding", "finding_id": "4211",
        "deal_id": "D-900", "agent_workflow_id": 958, "params": {},
    })
    monkeypatch.setattr(mod, "finding_is_open", lambda source, fid: False)
    superseded = {}
    monkeypatch.setattr(mod.repo, "mark_proposal_superseded",
                        lambda pid, by: superseded.update(pid=pid, by=by))
    def never(*a, **kw):
        raise AssertionError("a superseded proposal must not start a run")
    monkeypatch.setattr(mod, "start_run", never)

    r = client.post("/playbooks/proposals/5/approve")
    assert r.status_code == 200
    assert r.json()["proposal_status"] == "superseded"
    assert superseded["pid"] == 5


def test_a_retired_playbooks_existing_proposal_is_still_decidable(client, monkeypatch):
    """Retiring stops a playbook proposing. It must not strand the proposals it
    already raised -- somebody still has to say yes or no to those."""
    monkeypatch.setattr(mod.repo, "get_proposal", lambda pid: {
        "proposal_id": 5, "proposal_status": "proposed", "playbook_id": 12,
        "playbook_name": "Recover the duplicate",
        "finding_source": "detection_finding", "finding_id": "4211",
        "deal_id": None, "agent_workflow_id": 958, "params": {},
    })
    monkeypatch.setattr(mod.repo, "mark_proposal_rejected", lambda pid, by, reason: None)
    monkeypatch.setattr(mod.agent_actions, "record_action", lambda **kw: None)
    r = client.post("/playbooks/proposals/5/reject", json={"reason": "strategy retired"})
    assert r.status_code == 200


def test_a_proposal_already_decided_cannot_be_decided_again(client, monkeypatch):
    monkeypatch.setattr(mod.repo, "get_proposal", lambda pid: {
        "proposal_id": 5, "proposal_status": "executed", "playbook_id": 12,
        "finding_source": "detection_finding", "finding_id": "4211",
        "deal_id": None, "agent_workflow_id": 958, "params": {},
    })
    r = client.post("/playbooks/proposals/5/approve")
    assert r.status_code == 409
    assert "executed" in r.json()["detail"]


def test_rejecting_a_proposal_records_who_and_why_and_runs_nothing(client, monkeypatch):
    monkeypatch.setattr(mod.repo, "get_proposal", lambda pid: {
        "proposal_id": 5, "proposal_status": "proposed", "playbook_id": 12,
        "playbook_name": "Recover the duplicate",
        "finding_source": "detection_finding", "finding_id": "4211",
        "deal_id": None, "agent_workflow_id": 958, "params": {},
    })
    rejected = {}
    monkeypatch.setattr(mod.repo, "mark_proposal_rejected",
                        lambda pid, by, reason: rejected.update(pid=pid, by=by, reason=reason))
    def never(*a, **kw):
        raise AssertionError("rejecting must not start a run")
    monkeypatch.setattr(mod, "start_run", never)

    r = client.post("/playbooks/proposals/5/reject", json={"reason": "already credited"})
    assert r.status_code == 200
    assert rejected == {"pid": 5, "by": "bo", "reason": "already credited"}


def test_a_rejection_is_recorded_in_the_event_log(client, monkeypatch):
    """Rejecting is ungated, so without this the most interesting outcome -- a
    person looked at the recommendation and said no -- leaves no trail."""
    monkeypatch.setattr(mod.repo, "get_proposal", lambda pid: {
        "proposal_id": 5, "proposal_status": "proposed", "playbook_id": 12,
        "playbook_name": "Recover the duplicate",
        "finding_source": "detection_finding", "finding_id": "4211",
        "deal_id": "D-900", "agent_workflow_id": 958, "params": {},
    })
    monkeypatch.setattr(mod.repo, "mark_proposal_rejected", lambda pid, by, reason: None)
    events = []
    monkeypatch.setattr(mod.agent_actions, "record_action", lambda **kw: events.append(kw))

    client.post("/playbooks/proposals/5/reject", json={"reason": "already credited"})
    assert [e["action_type"] for e in events] == ["proposal.rejected"]
    assert events[0]["details"]["reason"] == "already credited"
    assert events[0]["details"]["decided_by"] == "bo"


def test_a_rejection_must_say_why(client, monkeypatch):
    """A rejected recommendation with no reason teaches nobody anything."""
    monkeypatch.setattr(mod.repo, "get_proposal", lambda pid: {
        "proposal_id": 5, "proposal_status": "proposed", "playbook_id": 12,
        "playbook_name": "Recover the duplicate",
        "finding_source": "detection_finding", "finding_id": "4211",
        "deal_id": None, "agent_workflow_id": 958, "params": {},
    })
    r = client.post("/playbooks/proposals/5/reject", json={"reason": "   "})
    assert r.status_code == 400


def _proposal(**over):
    base = {
        "proposal_id": 5, "playbook_id": 12, "proposal_status": "proposed",
        "playbook_name": "Recover the duplicate",
        "finding_source": "detection_finding", "finding_id": "4211",
        "deal_id": "D-900", "agent_workflow_id": 958, "params": {},
    }
    base.update(over)
    return base


def test_a_proposal_is_claimed_before_its_workflow_runs(client, monkeypatch):
    """Two approvals arriving together must not both run the graph.

    _decidable reads the status, but with get_conn() on autocommit there is no
    transaction holding it. The claim has to happen BEFORE start_run, not
    after, or a double-clicked Approve sends the supplier two emails.
    """
    monkeypatch.setattr(mod.repo, "get_proposal", lambda pid: _proposal())
    monkeypatch.setattr(mod.repo, "get", lambda pid: {"playbook_status": "active"})
    monkeypatch.setattr(mod, "finding_is_open", lambda source, fid: True)
    order = []
    monkeypatch.setattr(mod.repo, "claim_proposal",
                        lambda pid, by: order.append("claim") or True)
    monkeypatch.setattr(mod, "start_run",
                        lambda *a, **kw: order.append("run") or {"run_id": "r1", "status": "completed"})
    monkeypatch.setattr(mod.repo, "mark_proposal_executed", lambda pid, run_id, by: None)

    assert client.post("/playbooks/proposals/5/approve").status_code == 200
    assert order == ["claim", "run"], "the claim must precede the run"


def test_a_second_approval_that_loses_the_claim_runs_nothing(client, monkeypatch):
    monkeypatch.setattr(mod.repo, "get_proposal", lambda pid: _proposal())
    monkeypatch.setattr(mod.repo, "get", lambda pid: {"playbook_status": "active"})
    monkeypatch.setattr(mod, "finding_is_open", lambda source, fid: True)
    monkeypatch.setattr(mod.repo, "claim_proposal", lambda pid, by: False)

    def never(*a, **kw):
        raise AssertionError("losing the claim must not start a run")
    monkeypatch.setattr(mod, "start_run", never)

    r = client.post("/playbooks/proposals/5/approve")
    assert r.status_code == 409


def test_a_run_that_stopped_to_ask_is_not_recorded_as_executed(client, monkeypatch):
    """start_run has two outcomes. Marking awaiting_input 'executed' makes the
    proposal undecidable for ever while nothing has actually run."""
    monkeypatch.setattr(mod.repo, "get_proposal", lambda pid: _proposal())
    monkeypatch.setattr(mod.repo, "get", lambda pid: {"playbook_status": "active"})
    monkeypatch.setattr(mod, "finding_is_open", lambda source, fid: True)
    monkeypatch.setattr(mod.repo, "claim_proposal", lambda pid, by: True)
    monkeypatch.setattr(mod, "start_run", lambda *a, **kw: {
        "run_id": "r1", "status": "awaiting_input", "pending": [{"node_name": "ask"}]})

    def never(pid, run_id, by):
        raise AssertionError("a run that only asked a question has not executed")
    monkeypatch.setattr(mod.repo, "mark_proposal_executed", never)
    noted = {}
    monkeypatch.setattr(mod.repo, "note_proposal_run",
                        lambda pid, run_id: noted.update(pid=pid, run_id=run_id))

    r = client.post("/playbooks/proposals/5/approve")
    assert r.status_code == 200
    assert r.json()["proposal_status"] == "approved"
    assert r.json()["status"] == "awaiting_input"
    assert noted == {"pid": 5, "run_id": "r1"}


def test_a_retired_playbooks_proposal_does_not_execute_its_workflow(client, monkeypatch):
    """Retiring is how a strategy is withdrawn. It stops the playbook
    proposing; it must also stop the proposals it already raised from running
    the graph somebody deliberately took out of service."""
    monkeypatch.setattr(mod.repo, "get_proposal", lambda pid: _proposal())
    monkeypatch.setattr(mod.repo, "get", lambda pid: {"playbook_status": "retired"})
    monkeypatch.setattr(mod, "finding_is_open", lambda source, fid: True)
    monkeypatch.setattr(mod.repo, "mark_proposal_superseded", lambda pid, by: None)

    def never(*a, **kw):
        raise AssertionError("a retired strategy must not run")
    monkeypatch.setattr(mod, "start_run", never)
    monkeypatch.setattr(mod.repo, "claim_proposal", never)

    r = client.post("/playbooks/proposals/5/approve")
    assert r.status_code == 200
    assert r.json()["proposal_status"] == "superseded"
    assert "retired" in r.json()["detail"]


def test_a_run_that_could_not_start_releases_the_claim(client, monkeypatch):
    """Found by approving a real proposal while the orchestrator was down.

    The claim is taken before the run, which is right. But if start_run then
    raises -- a 503 from an orchestrator that has not loaded, a dropped
    connection -- the proposal was left at 'approved' with no run_id, and from
    there it could not be approved (409), could not be rejected, and the unique
    index blocked a fresh proposal for that finding. One transient failure
    stranded the work for ever.
    """
    monkeypatch.setattr(mod.repo, "get_proposal", lambda pid: _proposal())
    monkeypatch.setattr(mod.repo, "get", lambda pid: {"playbook_status": "active"})
    monkeypatch.setattr(mod, "finding_is_open", lambda source, fid: True)
    monkeypatch.setattr(mod.repo, "claim_proposal", lambda pid, by: True)
    released = []
    monkeypatch.setattr(mod.repo, "release_proposal_claim",
                        lambda pid: released.append(pid))

    def orchestrator_is_down(*a, **kw):
        raise HTTPException(status_code=503, detail="Orchestrator not available")
    monkeypatch.setattr(mod, "start_run", orchestrator_is_down)

    r = client.post("/playbooks/proposals/5/approve")
    assert r.status_code == 503
    assert released == [5], "the claim must go back so the work can be retried"
