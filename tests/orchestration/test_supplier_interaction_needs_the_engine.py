"""A workflow that belongs to the declarative engine fails when it is missing.

`orchestrator.py` used to carry a second, hand-rolled implementation of
supplier_interaction (`_execute_supplier_interaction_workflow`). In normal
operation the declarative engine shadowed it, so it never ran; it ran only when
`WorkflowEngine` failed to construct and the constructor fell back to "legacy
routing". That fallback was silent, its four tests had stopped exercising it
without anyone noticing -- the stub agent was never called -- and the two
implementations had different behaviour.

Rerouting a workflow through a second, unverified implementation because
something has already gone wrong is worse than not running it. A registered
workflow with no engine now fails, and says why.
"""
from types import SimpleNamespace

import pytest

from agents.base_agent import AgentOutput, AgentStatus
from orchestration.orchestrator import Orchestrator
from orchestration.workflow_definitions import WORKFLOW_REGISTRY


class StubSettings:
    script_user = "tester"
    max_workers = 1
    email_response_poll_seconds = 1
    email_response_timeout_seconds = 5


class StubAgent:
    def __init__(self):
        self.calls = []

    def execute(self, context):
        self.calls.append(context.input_data)
        return AgentOutput(status=AgentStatus.SUCCESS, data={"processed": True})


class StubNick:
    def __init__(self, agents):
        self.settings = StubSettings()
        self.agents = agents
        self.policy_engine = SimpleNamespace(
            supplier_policies=[],
            validate_workflow=lambda *a, **k: {"allowed": True},
        )
        self.query_engine = SimpleNamespace(fetch_supplier_data=lambda *_: {})
        self.routing_engine = SimpleNamespace(
            routing_model={"global_settings": {"max_chain_depth": 3}}
        )
        self.backend_scheduler = None

    def get_db_connection(self):
        class _Cursor:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def execute(self, *a, **k):
                return None

            def fetchall(self):
                return []

            def close(self):
                return None

        class _Conn:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def cursor(self):
                return _Cursor()

        return _Conn()


@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    monkeypatch.setattr("orchestration.orchestrator.configure_gpu", lambda: "cpu")
    monkeypatch.setattr(
        "orchestration.orchestrator.BackendScheduler.ensure",
        classmethod(lambda cls, *a, **k: SimpleNamespace()),
    )
    monkeypatch.setattr(
        "orchestration.orchestrator.get_event_bus",
        lambda: SimpleNamespace(publish=lambda *a, **k: None),
    )


@pytest.fixture
def supplier_agent():
    return StubAgent()


@pytest.fixture
def orchestrator(supplier_agent):
    nick = StubNick(
        {"supplier_interaction": supplier_agent, "negotiation": StubAgent()}
    )
    orch = Orchestrator(nick)
    # What the constructor does when WorkflowEngine cannot be built
    # (orchestrator.py, "Workflow engine init failed").
    orch._workflow_engine = None
    orch._workflow_registry = {}
    return orch


def test_the_legacy_supplier_interaction_path_is_gone():
    """Two implementations of one workflow, one of which only runs when
    something has already gone wrong, is what this removes."""
    assert not hasattr(Orchestrator, "_execute_supplier_interaction_workflow")


def test_a_registered_workflow_without_its_engine_fails(orchestrator):
    assert "supplier_interaction" in WORKFLOW_REGISTRY  # premise

    response = orchestrator.execute_workflow("supplier_interaction", {})

    assert response["status"] == "failed", (
        f"expected a loud failure, got {response.get('status')!r}"
    )


def test_the_failure_says_the_engine_is_why(orchestrator):
    response = orchestrator.execute_workflow("supplier_interaction", {})
    error = (response.get("error") or "").lower()
    assert "workflow engine" in error or "engine" in error, response


def test_it_does_not_quietly_run_the_agents_instead(orchestrator, supplier_agent):
    """The harm the fallback did: a different pipeline ran and looked like a
    result. Nothing should have been executed."""
    orchestrator.execute_workflow("supplier_interaction", {})
    assert not supplier_agent.calls, (
        f"supplier_interaction ran anyway with no engine: {supplier_agent.calls}"
    )


def test_a_workflow_outside_the_registry_is_unaffected(orchestrator):
    """The guard must be narrow. `negotiation` has no declarative graph and is
    meant to reach _execute_generic_workflow, engine or no engine."""
    assert "negotiation" not in WORKFLOW_REGISTRY  # premise

    response = orchestrator.execute_workflow("negotiation", {})

    assert response["status"] == "completed", response
