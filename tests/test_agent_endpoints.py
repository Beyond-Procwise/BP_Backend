import os
import os
import sys
from typing import Any, Dict
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.auth import require_user
from api.routers.agents import router as agents_router
from api.routers.workflows import router as workflows_router


def _authorize_as_approver(app: FastAPI) -> None:
    """Stand in for an authenticated caller on ``/workflows/email``.

    Task 7 gated the send path on a real principal (guardrail check 4). These
    tests stub `EmailDispatchService` itself and exist to prove the router's
    own orchestration (process/action logging, response shaping) -- not the
    guard, which has its own dedicated tests in tests/guardrails/. Overriding
    the auth dependency is the standard FastAPI mechanism for supplying that
    precondition without touching the guard.
    """

    app.dependency_overrides[require_user] = lambda: SimpleNamespace(
        subject="test-approver", claims={"cognito:groups": ["bp-approvers"]}
    )


class DummyPRS:
    def __init__(self):
        self.logged = []
        self.updated_details = None

    def log_process(self, **kwargs):
        return 1

    def log_action(self, **kwargs):
        self.logged.append(kwargs)
        return kwargs.get("action_id", "a1")

    def log_run_detail(self, **kwargs):
        return kwargs.get("run_id", "r1")

    def validate_workflow_id(self, *_args, **_kwargs):
        return True

    def update_process_status(self, *args, **kwargs):
        pass

    def update_process_details(self, process_id, process_details, **kwargs):
        self.updated_details = process_details


class DummyOrchestrator:
    def __init__(self):
        self.agent_nick = SimpleNamespace(process_routing_service=DummyPRS())

    def execute_workflow(self, workflow_name, input_data):
        if workflow_name == "email_drafting":
            output = {
                **input_data,
                "action_id": input_data.get("action_id", "a1"),
                "body": input_data.get("body", "<p>generated</p>"),
                "sent": False,
                "drafts": [
                    {
                        "rfq_id": "RFQ-123",
                        "action_id": input_data.get("action_id", "a1"),
                        "sent_status": False,
                    }
                ],
            }
            return {
                "status": "completed",
                "workflow_id": "wf",
                "result": {"email_drafting": output},
            }
        return {
            "status": "completed",
            "workflow_id": "wf",
            "result": {"echo": input_data},
        }



def test_agent_execute_endpoint():
    app = FastAPI()
    app.include_router(agents_router)
    orchestrator = DummyOrchestrator()
    app.state.orchestrator = orchestrator
    app.state.agent_nick = orchestrator.agent_nick
    client = TestClient(app)

    resp = client.post(
        "/agents/execute",
        json={"agent_type": "test_agent", "payload": {"foo": "bar"}},
    )

    assert resp.status_code == 200
    assert resp.json()["result"]["echo"]["foo"] == "bar"
    prs = orchestrator.agent_nick.process_routing_service
    assert len(prs.logged) == 2
    assert prs.logged[0]["status"] == "started"
    assert prs.logged[1]["status"] == "completed"


def test_workflow_types_endpoint():
    app = FastAPI()
    app.include_router(workflows_router)
    client = TestClient(app)

    resp = client.get("/workflows/types")
    assert resp.status_code == 200
    body = resp.json()

    # The catalogue is keyed by slug. It used to also carry `agentType` — the Python class
    # name — which nothing consumed: the UI looks agents up by slug and the gateway never
    # reads it. It was shipping our class names to the browser for no one.
    slugs = [item["slug"] for item in body]
    assert "opportunity_miner" in slugs
    assert "discrepancy_detection" in slugs

    # And the class names must not come back.
    assert "OpportunityMinerAgent" not in resp.text
    assert "DiscrepancyDetectionAgent" not in resp.text


def test_email_workflow_returns_action_id(monkeypatch):
    app = FastAPI()
    app.include_router(workflows_router)
    _authorize_as_approver(app)
    orchestrator = DummyOrchestrator()
    app.state.orchestrator = orchestrator
    app.state.agent_nick = orchestrator.agent_nick
    client = TestClient(app)

    calls = {}

    class StubDispatch:
        def __init__(self, agent_nick):
            calls["agent_nick"] = agent_nick

        def resolve_workflow_id(self, identifier):
            return f"wf-{identifier}" if identifier else None

        def send_draft(
            self,
            identifier,
            recipients=None,
            sender=None,
            subject_override=None,
            body_override=None,
            attachments=None,
            **kwargs,
        ):
            calls["args"] = (
                identifier,
                recipients,
                sender,
                subject_override,
                body_override,
            )
            calls["kwargs"] = kwargs
            unique_id = f"PROC-WF-{identifier}"
            return {
                "unique_id": unique_id,
                "sent": True,
                "recipients": recipients or ["r1", "r2"],
                "sender": sender or "sender@example.com",
                "subject": subject_override or "s",
                "body": body_override or "<p>generated</p>",
                "thread_index": 1,
                "draft": {
                    "rfq_id": identifier,
                    "unique_id": unique_id,
                    "sent_status": True,
                    "dispatch_metadata": {"unique_id": unique_id},
                },
            }

    monkeypatch.setattr("api.routers.workflows.EmailDispatchService", StubDispatch)

    resp = client.post(
        "/workflows/email",
        data={
            "rfq_id": "RFQ-123",
            "subject": "s",
            "recipients": "r1,r2",
            "action_id": "a1",
            "body": "<p>generated</p>",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["action_id"] == "a1"
    assert data["status"] == "completed"
    assert data["result"]["sent"] is True
    assert data["result"]["recipients"] == ["r1", "r2"]
    assert data["result"]["draft"]["sent_status"] is True
    assert calls["args"][0] == "RFQ-123"

    prs = orchestrator.agent_nick.process_routing_service
    assert len(prs.logged) == 2
    assert prs.logged[0]["status"] == "started"
    assert prs.logged[1]["status"] == "completed"
    assert prs.logged[0]["action_desc"]["rfq_id"] == "RFQ-123"
    assert prs.updated_details["output"]["sent"] is True
    assert prs.updated_details["status"] == "completed"


def test_email_workflow_accepts_list_recipients(monkeypatch):
    app = FastAPI()
    app.include_router(workflows_router)
    _authorize_as_approver(app)
    orchestrator = DummyOrchestrator()
    app.state.orchestrator = orchestrator
    app.state.agent_nick = orchestrator.agent_nick
    client = TestClient(app)

    captured: Dict[str, Any] = {}

    class StubDispatch:
        def __init__(self, agent_nick):
            captured["agent_nick"] = agent_nick

        def resolve_workflow_id(self, identifier):
            return f"wf-{identifier}" if identifier else None

        def send_draft(
            self,
            identifier,
            recipients=None,
            sender=None,
            subject_override=None,
            body_override=None,
            attachments=None,
            **kwargs,
        ):
            captured["call"] = {
                "identifier": identifier,
                "recipients": recipients,
                "kwargs": kwargs,
            }
            return {
                "unique_id": identifier,
                "sent": True,
                "recipients": recipients,
                "sender": sender or "sender@example.com",
                "subject": subject_override or "subject",
            }

    monkeypatch.setattr("api.routers.workflows.EmailDispatchService", StubDispatch)

    payload = {
        "unique_id": "PROC-WF-XYZ",
        "recipients": ["quotes@example.com", "buyer@example.com"],
    }

    resp = client.post("/workflows/email", json=payload)

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["recipients"] == payload["recipients"]
    assert captured["call"]["recipients"] == payload["recipients"]


def test_email_workflow_marks_failed_dispatch(monkeypatch):
    app = FastAPI()
    app.include_router(workflows_router)
    _authorize_as_approver(app)
    orchestrator = DummyOrchestrator()
    app.state.orchestrator = orchestrator
    app.state.agent_nick = orchestrator.agent_nick
    client = TestClient(app)

    class StubDispatchFail:
        def __init__(self, agent_nick):
            self.agent_nick = agent_nick

        def resolve_workflow_id(self, identifier):
            return f"wf-{identifier}" if identifier else None

        def send_draft(
            self,
            identifier,
            recipients=None,
            sender=None,
            subject_override=None,
            body_override=None,
            attachments=None,
            **kwargs,
        ):
            return {
                "unique_id": f"PROC-WF-{identifier}",
                "sent": False,
                "recipients": recipients or [],
                "sender": sender or "sender@example.com",
                "subject": subject_override or "subject",
                "body": body_override or "<p>body</p>",
                "thread_index": 1,
                "draft": {
                    "rfq_id": identifier,
                    "unique_id": f"PROC-WF-{identifier}",
                    "sent_status": False,
                },
            }

    monkeypatch.setattr("api.routers.workflows.EmailDispatchService", StubDispatchFail)

    resp = client.post(
        "/workflows/email",
        data={
            "rfq_id": "RFQ-456",
            "subject": "supplier action",
            "recipients": "buyer@example.com",
            "action_id": "workflow-action",
        },
    )

    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "failed"
    assert data["result"]["sent"] is False

    prs = orchestrator.agent_nick.process_routing_service
    assert prs.updated_details["status"] == "failed"


def test_email_dispatch_without_workflow_is_rejected(monkeypatch):
    app = FastAPI()
    app.include_router(workflows_router)
    _authorize_as_approver(app)
    orchestrator = DummyOrchestrator()
    app.state.orchestrator = orchestrator
    app.state.agent_nick = orchestrator.agent_nick
    client = TestClient(app)

    class StubDispatch:
        def __init__(self, agent_nick):
            self.agent_nick = agent_nick

        def resolve_workflow_id(self, identifier):
            return None

        def send_draft(self, *args, **kwargs):  # pragma: no cover - should not be called
            raise AssertionError("send_draft should not be invoked when workflow is missing")

    monkeypatch.setattr("api.routers.workflows.EmailDispatchService", StubDispatch)

    resp = client.post("/workflows/email", json={"unique_id": "PROC-WF-123"})

    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["error"] == "WorkflowUnavailable"
    assert detail["identifier"] == "PROC-WF-123"


def test_email_dispatch_detects_workflow_mismatch(monkeypatch):
    app = FastAPI()
    app.include_router(workflows_router)
    _authorize_as_approver(app)
    orchestrator = DummyOrchestrator()
    app.state.orchestrator = orchestrator
    app.state.agent_nick = orchestrator.agent_nick
    client = TestClient(app)

    class StubDispatch:
        def __init__(self, agent_nick):
            self.agent_nick = agent_nick

        def resolve_workflow_id(self, identifier):
            return "wf-stored"

        def send_draft(self, *args, **kwargs):  # pragma: no cover - should not be called
            raise AssertionError("send_draft should not be invoked when workflows mismatch")

    monkeypatch.setattr("api.routers.workflows.EmailDispatchService", StubDispatch)

    resp = client.post(
        "/workflows/email",
        json={"unique_id": "PROC-WF-456", "workflow_id": "wf-request"},
    )

    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["error"] == "WorkflowMismatch"
    assert detail["request_workflow_id"] == "wf-request"
    assert detail["stored_workflow_id"] == "wf-stored"


def test_reload_governance_reloads_both_engines():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from types import SimpleNamespace
    from api.routers import agents as agents_router

    calls = {"prompts": 0, "policies": 0}

    prompt_engine = SimpleNamespace(
        refresh=lambda: calls.__setitem__("prompts", calls["prompts"] + 1),
        all_prompts=lambda: [{"promptId": 1}, {"promptId": 2}],
    )
    policy_engine = SimpleNamespace(
        reload_policies=lambda: calls.__setitem__("policies", calls["policies"] + 1),
        list_policies=lambda: [{"policyId": "a"}],
    )

    app = FastAPI()
    app.include_router(agents_router.router)
    app.state.agent_nick = SimpleNamespace(
        prompt_engine=prompt_engine,
        policy_engine=policy_engine,
        agents={},
    )

    client = TestClient(app)
    resp = client.post("/agents/reload-governance")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["prompts"] == 2
    assert body["policies"] == 1
    assert calls == {"prompts": 1, "policies": 1}


# ---------------------------------------------------------------------------
# Fix round 1: the other two send paths must also refuse an unauthenticated
# caller. /workflows/email/batch and /workflows/{id}/email/dispatch-all call
# EmailDispatchService.send_draft directly and, before this round, took no
# principal at all -- the guard would always see principal=None and always
# deny, but nothing in the router *required* a caller identity in the first
# place. These two tests assert the send itself never happens without one,
# not merely that some status code comes back.
#
# `_MISSING` matters: `kwargs.get("principal")` returns None both when the
# router passes `principal=None` (the case being tested) AND when the router
# never passes the keyword at all (the regression -- e.g. if `principal=`
# were dropped from the send_draft call again). Without the sentinel, this
# test would still pass with that keyword deleted, defeating its own point.
# ---------------------------------------------------------------------------

_MISSING = object()


def test_email_batch_dispatch_refuses_without_a_principal(monkeypatch):
    """No principal reaches send_draft -> the real guard's check 4 would deny
    (see tests/guardrails/test_send_path_gate.py::test_check_4_no_principal_is_denied
    and tests/test_email_dispatch_service.py's wiring test for that proof).
    This test is about the ROUTER: does /email/batch even ask who is calling,
    and does it thread that answer into every send_draft call in the loop.

    The stub reproduces DispatchDenied's real contract (raise when principal
    is None) rather than re-deriving the guard's own logic, so a batch of
    two drafts must show both refused and neither ever reaching SES.
    """
    app = FastAPI()
    app.include_router(workflows_router)
    # ASK_AUTH_MODE="off" in this environment: require_user returns None
    # rather than raising. Overriding it here pins that exact, documented
    # case instead of depending on api.auth's global, test-order-sensitive
    # `_mode`.
    app.dependency_overrides[require_user] = lambda: None
    orchestrator = DummyOrchestrator()
    app.state.orchestrator = orchestrator
    app.state.agent_nick = orchestrator.agent_nick
    client = TestClient(app)

    principals_seen = []

    class StubDispatch:
        def __init__(self, agent_nick):
            pass

        def send_draft(self, identifier, **kwargs):
            principals_seen.append(kwargs.get("principal", _MISSING))
            if kwargs.get("principal") is None:
                raise PermissionError(
                    "no authenticated principal: irreversible actions are refused"
                )
            raise AssertionError(  # pragma: no cover - guard path
                "this test only exercises the unauthenticated case"
            )

    monkeypatch.setattr("api.routers.workflows.EmailDispatchService", StubDispatch)

    resp = client.post(
        "/workflows/email/batch",
        json={
            "drafts": [
                {"unique_id": "PROC-WF-BATCH-1", "supplier_id": "SUP-1"},
                {"unique_id": "PROC-WF-BATCH-2", "supplier_id": "SUP-2"},
            ]
        },
    )

    assert resp.status_code == 200
    body = resp.json()

    # The router reached send_draft for both drafts (proving the dependency is
    # wired through the loop) and every single call explicitly carried
    # principal=None -- not merely omitted the keyword altogether.
    assert principals_seen == [None, None]

    # And the send itself never happened: zero successes, both refused with
    # the guard's own reason, not silently swallowed as some other error.
    assert body["sent"] == 0
    assert body["failed"] == 2
    for result in body["results"]:
        assert result["sent"] is False
        assert "no authenticated principal" in result["error"]


def test_dispatch_all_refuses_without_a_principal(monkeypatch):
    """Same proof for /workflows/{workflow_id}/email/dispatch-all.

    This endpoint queries proc.draft_rfq_emails directly rather than going
    through the orchestrator, so the fake agent_nick here answers that query
    itself instead of using DummyOrchestrator/DummyPRS.
    """

    class _Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, query, params=None):
            self._rows = [("PROC-WF-ALL-1", "SUP-1", "subject", False)]

        def fetchall(self):
            return self._rows

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def cursor(self):
            return _Cursor()

    class _AgentNick:
        def get_db_connection(self):
            return _Conn()

    app = FastAPI()
    app.include_router(workflows_router)
    app.dependency_overrides[require_user] = lambda: None
    app.state.agent_nick = _AgentNick()
    client = TestClient(app)

    principals_seen = []

    class StubDispatch:
        def __init__(self, agent_nick):
            pass

        def send_draft(self, identifier, **kwargs):
            principals_seen.append(kwargs.get("principal", _MISSING))
            if kwargs.get("principal") is None:
                raise PermissionError(
                    "no authenticated principal: irreversible actions are refused"
                )
            raise AssertionError(  # pragma: no cover - guard path
                "this test only exercises the unauthenticated case"
            )

    monkeypatch.setattr("api.routers.workflows.EmailDispatchService", StubDispatch)

    resp = client.post("/workflows/wf-1/email/dispatch-all")

    assert resp.status_code == 200
    body = resp.json()

    # Explicitly None, not merely absent -- see the module-level note on
    # `_MISSING` above.
    assert principals_seen == [None]
    assert body["sent"] == 0
    assert body["failed"] == 1
    assert "no authenticated principal" in body["results"][0]["error"]


