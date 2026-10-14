"""A workflow's user_id is the person who started it, or nobody -- never a stand-in.

``proc.workflow_execution.user_id`` held 'AgentNick' (14 rows), 'system' (5)
and 'human' (2), and never a person. The orchestrator filled AgentContext.user_id
with settings.script_user whenever it was told nobody, and nothing that starts a
workflow ever told it anybody: the routers resolved the principal (P8) and then
called execute_workflow without it. The canvas started runs as whatever
``body.user_id`` said (default "system") and resumed them as the literal "human".

That value is the input to the self-approval bar on negotiation rounds
(approval_store.workflow_initiator). A stand-in name can never equal an
approver's subject, so the bar could not fire -- and a typed one could be made
to equal someone else's. Negotiation also never reached workflow_execution at
all: only the declarative engine wrote that row, and negotiation has no graph.

One meaning now: the subject of whoever started the run, or None when no person
did (a scheduled job, a background watcher, or authentication switched off).
The service's own identity is not a user; where a record needs one it has its
own field for it.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.auth import require_user

ROOT = pathlib.Path(__file__).resolve().parents[1]
CALLER = "sub-real-caller"
IMPERSONATED = "sub-someone-else"


class _Principal:
    def __init__(self, subject):
        self.subject = subject


def _app(router, *, subject=CALLER, overrides=None, state=None):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_user] = (
        (lambda: _Principal(subject)) if subject else (lambda: None))
    for dep, value in (overrides or {}).items():
        app.dependency_overrides[dep] = value
    for key, value in (state or {}).items():
        setattr(app.state, key, value)
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# the orchestrator: no stand-in, and a non-engine run is recorded too
# ---------------------------------------------------------------------------
class _RunTrail:
    """Stands in for StateManager: what proc.workflow_execution would hold."""

    def __init__(self):
        self.created = []
        self.statuses = []

    def create_workflow_execution(self, workflow_id, workflow_name, user_id=None):
        self.created.append((workflow_id, workflow_name, user_id))
        return 41

    def update_workflow_status(self, execution_id, status, completed_at=None):
        self.statuses.append((execution_id, status))


def _orchestrator(monkeypatch, seen):
    from orchestration.orchestrator import Orchestrator

    nick = SimpleNamespace(
        settings=SimpleNamespace(script_user="AgentNick", max_workers=1),
        agents={}, policy_engine=SimpleNamespace(),
        query_engine=SimpleNamespace(),
        routing_engine=SimpleNamespace(routing_model={}),
    )
    orch = Orchestrator(nick)
    orch.manifest_service = SimpleNamespace(build_manifest=lambda name: {})
    for hook in ("_apply_authority", "_publish_workflow_complete",
                 "_finalise_workflow_memory", "_learn_from_workflow"):
        monkeypatch.setattr(orch, hook, lambda *a, **k: None)
    monkeypatch.setattr(orch, "_apply_governance_envelope", lambda *a, **k: None)
    monkeypatch.setattr(orch, "_validate_workflow", lambda *a, **k: True)

    def _generic(name, context):
        seen["context_user"] = context.user_id
        seen["workflow_id"] = context.workflow_id
        return {"ok": True}

    monkeypatch.setattr(orch, "_execute_generic_workflow", _generic)
    orch._run_trail = _RunTrail()
    return orch


def test_a_workflow_started_by_nobody_is_started_by_nobody(monkeypatch):
    seen = {}
    orch = _orchestrator(monkeypatch, seen)

    orch.execute_workflow("negotiation", {"supplier": "s"})

    assert seen["context_user"] is None, (
        f"the agent was told a stand-in started this: {seen['context_user']!r}")


def test_a_workflow_started_by_a_person_carries_that_person(monkeypatch):
    seen = {}
    orch = _orchestrator(monkeypatch, seen)

    orch.execute_workflow("negotiation", {"supplier": "s"}, user_id=CALLER)

    assert seen["context_user"] == CALLER


def test_a_workflow_with_no_graph_still_records_who_started_it(monkeypatch):
    """Negotiation has no declarative graph, so only the engine's row existed --
    and the self-approval bar on a round reads exactly that row."""
    seen = {}
    orch = _orchestrator(monkeypatch, seen)

    orch.execute_workflow("negotiation", {"supplier": "s"}, user_id=CALLER)

    assert orch._run_trail.created == [(seen["workflow_id"], "negotiation", CALLER)]
    assert orch._run_trail.statuses == [(41, "completed")]


def test_a_failed_run_is_recorded_as_failed(monkeypatch):
    seen = {}
    orch = _orchestrator(monkeypatch, seen)

    def _boom(name, context):
        raise RuntimeError("agent fell over")

    monkeypatch.setattr(orch, "_execute_generic_workflow", _boom)

    orch.execute_workflow("negotiation", {}, user_id=CALLER)

    assert orch._run_trail.statuses == [(41, "failed")]


# ---------------------------------------------------------------------------
# the engine: no "system" default, on a fresh run or a restored one
# ---------------------------------------------------------------------------
def test_the_engine_does_not_default_the_initiator():
    from orchestration.workflow_engine import WorkflowEngine

    assert inspect.signature(WorkflowEngine.execute).parameters["user_id"].default is None


def test_a_restored_run_with_no_initiator_has_none():
    from orchestration.workflow_engine import WorkflowState

    state = WorkflowState.from_dict({"workflow_id": "w", "workflow_name": "n"})

    assert state.user_id is None


# ---------------------------------------------------------------------------
# the routers hand the orchestrator the token's subject
# ---------------------------------------------------------------------------
class _Orchestrator:
    def __init__(self):
        self.calls = []
        self.agent_nick = SimpleNamespace(process_routing_service=_Routing())

    def execute_workflow(self, name, payload, user_id=None):
        self.calls.append((name, user_id))
        return {"status": "completed"}

    def execute_agent_flow(self, flow, payload=None, process_id=None, prs=None,
                           user_id=None):
        self.calls.append(("agent_flow", user_id))
        return {"status": 100}


class _Routing:
    def log_process(self, **k):
        return 5

    def log_action(self, **k):
        return "a-1"

    def get_process_details(self, pid, raw=False):
        return {"agents": []}

    def convert_agents_to_flow(self, details):
        return {"entrypoint": "x", "steps": {}}

    def _load_agent_links(self):
        return {}, {}, {}

    def _enrich_node(self, *a):
        return None

    def update_process_details(self, *a, **k):
        return None

    def classify_completion_status(self, final):
        return 100, "completed", None

    def update_process_status(self, *a, **k):
        return None


_STARTS = [
    ("workflows", "/workflows/negotiate",
     {"supplier": "s", "current_offer": 10, "target_price": 8}),
    ("workflows", "/workflows/approvals", {"amount": 100}),
    ("workflows", "/workflows/supplier-interaction", {"message": "hello"}),
    ("workflows", "/workflows/discrepancy", {"extracted_docs": []}),
    ("workflows", "/workflows/quotes/evaluate", {}),
    ("workflows", "/workflows/opportunities", {"workflow": "contract_expiry"}),
    ("agents", "/agents/execute", {"agent_type": "supplier_ranking", "payload": {}}),
    ("agents", "/agents/process-document", {"s3_prefix": "x/"}),
]


def _started(module_name, path, body, subject):
    module = __import__(f"api.routers.{module_name}", fromlist=["*"])
    orch = _Orchestrator()
    client = _app(module.router, subject=subject,
                  overrides={module.get_orchestrator: lambda: orch})
    r = client.post(path, json={**body, "user_id": IMPERSONATED})
    return r, orch


@pytest.mark.parametrize("module_name,path,body", _STARTS, ids=[p for _, p, _ in _STARTS])
def test_a_workflow_started_through_the_api_is_started_by_the_token(
        module_name, path, body):
    r, orch = _started(module_name, path, body, CALLER)

    assert orch.calls, f"{path} never reached the orchestrator ({r.status_code}: {r.text[:200]})"
    assert orch.calls[-1][1] == CALLER, (
        f"{path} started the workflow as {orch.calls[-1][1]!r}, not the caller")


@pytest.mark.parametrize("module_name,path,body", _STARTS[:2], ids=[p for _, p, _ in _STARTS[:2]])
def test_without_a_principal_the_workflow_is_started_by_nobody(module_name, path, body):
    r, orch = _started(module_name, path, body, None)

    assert orch.calls and orch.calls[-1][1] is None, orch.calls


def test_a_requirements_workflow_is_started_by_the_token(monkeypatch):
    from api.routers import requirements

    orch = _Orchestrator()
    requirements._launch_workflow(SimpleNamespace(orchestrator=orch),
                                  {"brief": "laptops", "created_by": CALLER}, "job-1")
    import time
    for _ in range(50):
        if orch.calls:
            break
        time.sleep(0.02)

    assert orch.calls == [("requirements_to_ranking", CALLER)]


def test_a_flow_run_is_started_by_the_token():
    from api.routers import run

    orch = _Orchestrator()
    client = _app(run.router, overrides={run.get_orchestrator: lambda: orch})
    r = client.post("/run", json={"process_id": 5})

    assert r.status_code in (200, 202), r.text
    import time
    for _ in range(50):
        if orch.calls:
            break
        time.sleep(0.02)
    assert orch.calls == [("agent_flow", CALLER)]


# ---------------------------------------------------------------------------
# the canvas: started by the token, resumed as whoever STARTED it
# ---------------------------------------------------------------------------
def _canvas(monkeypatch, seen, *, open_requests):
    from api.routers import agent_workflows as aw

    monkeypatch.setattr(aw, "gate", lambda *a, **k: None)
    monkeypatch.setattr(aw.repo, "get", lambda wid: {"name": "wf", "graph": {}})
    monkeypatch.setattr(aw, "pending_requests", lambda *a, **k: [])
    monkeypatch.setattr(aw.reqrepo, "answers_for", lambda run_id: {})
    monkeypatch.setattr(aw.reqrepo, "create_run",
                        lambda run_id, **k: seen.update(created=k))
    monkeypatch.setattr(aw.reqrepo, "request_run_id", lambda rid: "R-1")
    monkeypatch.setattr(aw.reqrepo, "answer", lambda *a: True)
    monkeypatch.setattr(aw.reqrepo, "open_requests", lambda run_id: open_requests)
    monkeypatch.setattr(aw.reqrepo, "workflow_id_for", lambda run_id: 3)
    monkeypatch.setattr(aw.reqrepo, "payload_for", lambda run_id: {})
    monkeypatch.setattr(aw.reqrepo, "initiator_for", lambda run_id: "sub-who-started-it",
                        raising=False)
    monkeypatch.setattr(aw, "_claim_and_execute",
                        lambda request, run_id, wf, data, user_id:
                        seen.update(user_id=user_id) or {"run_id": run_id})
    return aw


def test_a_canvas_run_is_started_by_the_token_not_the_body(monkeypatch):
    seen = {}
    aw = _canvas(monkeypatch, seen, open_requests=[])

    r = _app(aw.router).post("/agent-workflows/3/run",
                             json={"payload": {}, "user_id": IMPERSONATED})

    assert r.status_code == 200, r.text
    assert seen["user_id"] == CALLER, seen
    assert seen["created"].get("initiated_by") == CALLER, (
        f"the run row does not record who started it: {seen['created']}")


def test_a_canvas_run_with_no_principal_is_started_by_nobody(monkeypatch):
    seen = {}
    aw = _canvas(monkeypatch, seen, open_requests=[])

    _app(aw.router, subject=None).post("/agent-workflows/3/run",
                                       json={"payload": {}, "user_id": IMPERSONATED})

    assert seen["user_id"] is None, seen


def test_a_resumed_canvas_run_is_started_by_whoever_started_it(monkeypatch):
    """Not "human", and not whoever answered the last question: the person who
    answers is recorded as the answerer, the person who started is the
    initiator, and a self-approval check needs them kept apart."""
    seen = {}
    aw = _canvas(monkeypatch, seen, open_requests=[])

    r = _app(aw.router, subject="sub-the-answerer").post(
        "/agent-workflows/runs/R-1/input", json={"request_id": 1, "answer": "yes"})

    assert r.status_code == 200, r.text
    assert seen["user_id"] == "sub-who-started-it", seen


# ---------------------------------------------------------------------------
# the guard: no stand-in identity anywhere in src
# ---------------------------------------------------------------------------
def _is_standin(node) -> bool:
    """A string literal, or a script_user read, used as a user."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value:
        return True
    if isinstance(node, ast.Attribute) and node.attr == "script_user":
        return True
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
        return any(_is_standin(v) for v in node.values[1:])
    return False


def _standin_users() -> list:
    hits = []
    for path in sorted((ROOT / "src").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError):
            continue
        where = lambda n: f"{path.relative_to(ROOT)}:{n.lineno}"  # noqa: E731
        for node in ast.walk(tree):
            # AgentContext(user_id="system"), execute(user_id=x or script_user)
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg == "user_id" and _is_standin(kw.value):
                        hits.append(where(node))
                # data.get("user_id", "system")
                f = node.func
                if (isinstance(f, ast.Attribute) and f.attr == "get"
                        and len(node.args) == 2
                        and isinstance(node.args[0], ast.Constant)
                        and node.args[0].value == "user_id"
                        and _is_standin(node.args[1])):
                    hits.append(where(node))
            # def execute(..., user_id: str = "system")
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = node.args.args + node.args.kwonlyargs
                defaults = ([None] * (len(node.args.args) - len(node.args.defaults))
                            + list(node.args.defaults) + list(node.args.kw_defaults))
                for arg, default in zip(args, defaults):
                    if arg.arg == "user_id" and default is not None and _is_standin(default):
                        hits.append(where(node))
            # class RunBody: user_id: str = "system"
            if (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                    and node.target.id == "user_id" and node.value is not None
                    and _is_standin(node.value)):
                hits.append(where(node))
    return hits


def test_the_guard_can_see_a_standin():
    """A guard that finds nothing to object to passes forever. Pin that it looks."""
    probe = ast.parse('AgentContext(workflow_id="w", agent_id="a", user_id="system", input_data={})')
    call = probe.body[0].value
    assert _is_standin(call.keywords[2].value)
    assert _is_standin(ast.parse("x or self.settings.script_user").body[0].value)
    assert not _is_standin(ast.parse("principal_subject").body[0].value)


def test_no_workflow_is_started_by_a_standin():
    hits = _standin_users()
    assert not hits, (
        "a user_id that is a name rather than a person -- use the principal's "
        "subject, or None when no person started it:\n  " + "\n  ".join(hits))
