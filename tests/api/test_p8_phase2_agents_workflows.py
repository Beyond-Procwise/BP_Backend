"""P8 phase 2, agent / workflow routers: the caller reaches the handler.

Phase 1 fixed the eight writes a caller could sign in someone else's name. This
is the agent and workflow half of the 49 that were left: authenticated at the
router, never told WHO inside the handler.

Two kinds of test, for two kinds of endpoint:

* Where the endpoint writes an actor -- a created_by, an answered_by, a
  user_id carried into an agent payload -- the test asserts on what reaches the
  STORE: it is the token's subject, never the value the caller typed, and with
  no principal it is nobody rather than the typed value.

* Every endpoint, actor or not, must take the principal. That is asserted by
  request, not by inspecting dependency objects: `require_user` is overridden
  with one that answers 418, so a handler that declares it is refused with
  418 before it runs, and a handler that does not declare it runs into a
  tripwire instead. A 418 is only possible if the dependency is there.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from api.auth import require_user

CALLER = "sub-real-caller"
IMPERSONATED = "sub-someone-else"


class _Principal:
    def __init__(self, subject):
        self.subject = subject
        self.email = f"{subject}@ourcompany.com"


class _Tripwire:
    """Stands in for app.state and for every service a handler might reach.

    Touch it and the handler fails -- which is what happens when the handler
    ran without resolving the caller first."""

    def __getattr__(self, name):
        raise RuntimeError(f"handler ran without resolving the caller (touched .{name})")

    def __call__(self, *a, **k):
        raise RuntimeError("handler ran without resolving the caller")


def _app(router, *, subject=CALLER, state=None, overrides=None):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_user] = (
        (lambda: _Principal(subject)) if subject else (lambda: None)
    )
    for dep, value in (overrides or {}).items():
        app.dependency_overrides[dep] = value
    for key, value in (state or {}).items():
        setattr(app.state, key, value)
    return TestClient(app, raise_server_exceptions=False)


def _refusing_app(router, *, overrides=None):
    def _refuse():
        raise HTTPException(status_code=418, detail="principal resolved")

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_user] = _refuse
    for dep, value in (overrides or {}).items():
        app.dependency_overrides[dep] = value
    app.state.orchestrator = _Tripwire()
    app.state.agent_nick = _Tripwire()
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _no_gates(monkeypatch):
    """Authorization is not what these tests are about; attribution is."""
    for name in ("agents", "agent_workflows", "workflows"):
        module = __import__(f"api.routers.{name}", fromlist=["*"])
        if hasattr(module, "gate"):
            monkeypatch.setattr(module, "gate", lambda *a, **k: None)


# ---------------------------------------------------------------------------
# every endpoint in this group takes the principal
# ---------------------------------------------------------------------------
_ENDPOINTS = [
    ("agent_groups", "POST", "/agent-groups", {"name": "g"}),
    ("agent_groups", "PUT", "/agent-groups/1", {}),
    ("agent_groups", "DELETE", "/agent-groups/1", None),
    ("agent_workflows", "POST", "/agent-workflows/runs/R-1/input", {"request_id": 1, "answer": "x"}),
    ("agents", "POST", "/agents/execute", {"agent_type": "x"}),
    ("agents", "POST", "/agents/process-document", {}),
    ("agents", "POST", "/agents/reason", {"task": "x"}),
    ("run", "POST", "/run", {"process_id": 1}),
    ("stream", "POST", "/stream/plan", {"task": "x"}),
    ("requirements", "POST", "/requirements/message", {}),
    ("requirements", "POST", "/requirements/run-workflow", {}),
    ("workflows", "POST", "/workflows/rank", {"query": "x"}),
    ("workflows", "POST", "/workflows/quotes/evaluate", {}),
    ("workflows", "POST", "/workflows/opportunities", {"workflow": "x"}),
    ("workflows", "POST", "/workflows/extract", {}),
    ("workflows", "POST", "/workflows/negotiate", {"supplier": "s", "current_offer": 1, "target_price": 1}),
    ("workflows", "POST", "/workflows/approvals", {"amount": 1}),
    ("workflows", "POST", "/workflows/supplier-interaction", {"message": "m"}),
    ("workflows", "POST", "/workflows/discrepancy", {"extracted_docs": []}),
    ("workflows", "DELETE", "/workflows/email/U-1/attachments/0", None),
]


@pytest.mark.parametrize("module_name,method,path,body", _ENDPOINTS,
                         ids=[f"{m} {p}" for _, m, p, _ in _ENDPOINTS])
def test_the_handler_resolves_the_caller(monkeypatch, module_name, method, path, body):
    module = __import__(f"api.routers.{module_name}", fromlist=["*"])

    # Endpoints that need no body would otherwise run straight into a store.
    trip = _Tripwire()
    for target, attr in (
        ("agent_groups", "repo"),
        ("workflows", "draft_rfq_emails_repo"),
        ("requirements", "_launch_workflow"),
        ("requirements", "_run_requirements_turn"),
    ):
        if target == module_name:
            monkeypatch.setattr(module, attr, trip)

    kwargs = {} if body is None else {"json": body}
    response = _refusing_app(module.router).request(method, path, **kwargs)

    assert response.status_code == 418, (
        f"{method} {path} never asked who the caller is "
        f"({response.status_code}: {response.text[:160]})")


# ---------------------------------------------------------------------------
# agent_groups -- bp_agent_group.created_by was the literal "system"
# ---------------------------------------------------------------------------
def test_a_new_agent_group_is_created_by_the_token(monkeypatch):
    from api.routers import agent_groups

    seen = {}
    monkeypatch.setattr(agent_groups, "_validate", lambda *a, **k: None)
    monkeypatch.setattr(agent_groups.repo, "create", lambda **k: seen.update(k) or 7)

    r = _app(agent_groups.router).post("/agent-groups", json={"name": "g"})

    assert r.status_code == 200, r.text
    assert seen.get("created_by") == CALLER, (
        f"the group was not attributed to the caller: {seen}")


def test_without_a_principal_an_agent_group_is_created_by_nobody(monkeypatch):
    """Not "system": that string was written as though it named somebody."""
    from api.routers import agent_groups

    seen = {}
    monkeypatch.setattr(agent_groups, "_validate", lambda *a, **k: None)
    monkeypatch.setattr(agent_groups.repo, "create", lambda **k: seen.update(k) or 7)

    _app(agent_groups.router, subject=None).post("/agent-groups", json={"name": "g"})

    assert "created_by" in seen and seen["created_by"] is None, seen


# ---------------------------------------------------------------------------
# agent_workflows -- the HITL audit trail's answered_by was typed by the caller
# ---------------------------------------------------------------------------
def _stub_input_request(monkeypatch, seen):
    from api.routers import agent_workflows

    monkeypatch.setattr(agent_workflows.reqrepo, "request_run_id", lambda rid: "R-1")
    monkeypatch.setattr(agent_workflows.reqrepo, "answer",
                        lambda run_id, request_id, answer, answered_by:
                        seen.update(answered_by=answered_by) or True)
    # Something still outstanding, so the run does not proceed to execution.
    monkeypatch.setattr(agent_workflows.reqrepo, "open_requests", lambda run_id: [{"request_id": 2}])
    return agent_workflows


def test_an_answer_to_a_workflow_question_is_attributed_to_the_token(monkeypatch):
    seen = {}
    agent_workflows = _stub_input_request(monkeypatch, seen)

    r = _app(agent_workflows.router).post(
        "/agent-workflows/runs/R-1/input",
        json={"request_id": 1, "answer": "yes", "answered_by": IMPERSONATED})

    assert r.status_code == 200, r.text
    assert seen.get("answered_by") == CALLER, (
        f"the HITL audit trail recorded a typed name as the answerer: {seen}")


def test_without_a_principal_the_answer_is_recorded_against_nobody(monkeypatch):
    seen = {}
    agent_workflows = _stub_input_request(monkeypatch, seen)

    _app(agent_workflows.router, subject=None).post(
        "/agent-workflows/runs/R-1/input",
        json={"request_id": 1, "answer": "yes", "answered_by": IMPERSONATED})

    assert "answered_by" in seen and seen["answered_by"] is None, seen


# ---------------------------------------------------------------------------
# proc.routing.created_by was settings.script_user ("AgentNick") for every run
# a person started through these endpoints
# ---------------------------------------------------------------------------
class _Routing:
    def __init__(self):
        self.logged = {}
        self.modified_by = "unset"

    def log_process(self, **k):
        self.logged.update(k)
        return 11

    def log_action(self, **k):
        return "A-1"

    def update_process_status(self, *a, **k):
        return None

    # run.py
    def get_process_details(self, process_id, raw=False):
        return {"agents": []}

    def convert_agents_to_flow(self, details):
        return {}

    def _load_agent_links(self):
        return {}, {}, {}

    def _enrich_node(self, *a, **k):
        return None

    def update_process_details(self, process_id, details, modified_by=None):
        self.modified_by = modified_by

    def classify_completion_status(self, final):
        return 1, "completed", None


class _Orchestrator:
    def __init__(self):
        self.routing = _Routing()
        self.agent_nick = type("N", (), {"process_routing_service": self.routing})()
        self.workflows = []

    def execute_workflow(self, name, payload, *a, **k):
        self.workflows.append((name, payload))
        return {"status": "ok"}

    def execute_extraction_flow(self, *a, **k):
        return {"status": "ok"}

    def execute_agent_flow(self, *a, **k):
        return {"status": "completed"}


@pytest.mark.parametrize("module_name,path,body", [
    ("agents", "/agents/execute", {"agent_type": "opportunity_miner", "payload": {}}),
    ("workflows", "/workflows/opportunities", {"workflow": "price_variance_check"}),
    ("workflows", "/workflows/extract", {"s3_prefix": "p/"}),
])
def test_a_process_started_through_the_api_is_logged_against_the_token(
        module_name, path, body):
    module = __import__(f"api.routers.{module_name}", fromlist=["*"])
    orch = _Orchestrator()

    r = _app(module.router, overrides={module.get_orchestrator: lambda: orch}).post(path, json=body)

    assert r.status_code == 200, r.text
    assert orch.routing.logged.get("created_by") == CALLER, orch.routing.logged
    assert orch.routing.logged.get("user_id") == CALLER, orch.routing.logged


def test_without_a_principal_the_process_names_no_user():
    from api.routers import agents

    orch = _Orchestrator()
    _app(agents.router, subject=None, overrides={agents.get_orchestrator: lambda: orch}).post(
        "/agents/execute", json={"agent_type": "x", "payload": {}})

    assert orch.routing.logged.get("user_id") is None, orch.routing.logged
    assert orch.routing.logged.get("created_by") is None, orch.routing.logged


def test_a_run_started_through_the_api_is_modified_by_the_token():
    from api.routers import run

    orch = _Orchestrator()
    r = _app(run.router, overrides={run.get_orchestrator: lambda: orch}).post(
        "/run", json={"process_id": 5})

    assert r.status_code == 200, r.text
    assert orch.routing.modified_by == CALLER, orch.routing.modified_by


# ---------------------------------------------------------------------------
# workflows -- a user_id typed into the body rode into the agent payload
# ---------------------------------------------------------------------------
_TYPED_USER = [
    ("/workflows/negotiate", {"supplier": "s", "current_offer": 10, "target_price": 8}),
    ("/workflows/approvals", {"amount": 100}),
    ("/workflows/supplier-interaction", {"message": "hello"}),
    ("/workflows/discrepancy", {"extracted_docs": []}),
]


@pytest.mark.parametrize("path,body", _TYPED_USER, ids=[p for p, _ in _TYPED_USER])
def test_the_agent_payload_carries_the_token_not_the_typed_user(path, body):
    from api.routers import workflows

    orch = _Orchestrator()
    r = _app(workflows.router, overrides={workflows.get_orchestrator: lambda: orch}).post(
        path, json={**body, "user_id": IMPERSONATED})

    assert r.status_code == 200, r.text
    (_, payload), = orch.workflows
    assert payload.get("user_id") == CALLER, (
        f"{path} handed the agent a user the caller typed: {payload}")


def test_without_a_principal_the_agent_payload_names_nobody():
    from api.routers import workflows

    orch = _Orchestrator()
    _app(workflows.router, subject=None, overrides={workflows.get_orchestrator: lambda: orch}).post(
        "/workflows/approvals", json={"amount": 100, "user_id": IMPERSONATED})

    (_, payload), = orch.workflows
    assert payload.get("user_id") is None, payload


# ---------------------------------------------------------------------------
# requirements -- bp_requirement.created_by was body.created_by, else "api"
# ---------------------------------------------------------------------------
def test_a_requirement_turn_is_created_by_the_token(monkeypatch):
    from api.routers import requirements

    seen = {}
    monkeypatch.setattr(requirements, "_run_requirements_turn",
                        lambda state, payload: seen.update(payload) or {})

    r = _app(requirements.router).post(
        "/requirements/message", json={"message": "need laptops", "created_by": IMPERSONATED})

    assert r.status_code == 200, r.text
    assert seen.get("created_by") == CALLER, seen


def test_a_requirements_workflow_is_created_by_the_token(monkeypatch):
    from api.routers import requirements

    seen = {}
    monkeypatch.setattr(requirements, "_launch_workflow",
                        lambda state, payload, job_id: seen.update(payload))

    r = _app(requirements.router).post(
        "/requirements/run-workflow", json={"brief": "laptops", "created_by": IMPERSONATED})

    assert r.status_code == 200, r.text
    assert seen.get("created_by") == CALLER, seen


def test_without_a_principal_a_requirement_names_nobody(monkeypatch):
    from api.routers import requirements

    seen = {}
    monkeypatch.setattr(requirements, "_run_requirements_turn",
                        lambda state, payload: seen.update(payload) or {})

    _app(requirements.router, subject=None).post(
        "/requirements/message", json={"message": "x", "created_by": IMPERSONATED})

    assert not seen.get("created_by"), seen


def test_the_requirements_agent_is_not_told_the_caller_is_api():
    """With nobody identified the agent's context names nobody. It used to name
    "api", which the agent then wrote to bp_requirement.created_by."""
    from api.routers import requirements

    contexts = []

    class _Agent:
        def run(self, ctx):
            contexts.append(ctx)
            return type("O", (), {"data": {}})()

    state = type("S", (), {})()
    state.orchestrator = type("O", (), {})()
    state.orchestrator.agent_nick = type("N", (), {"agents": {"requirements": _Agent()}})()

    requirements._run_requirements_turn(state, {"message": "x"})

    assert contexts and contexts[0].user_id != "api", contexts[0].user_id
    assert not contexts[0].user_id, contexts[0].user_id
