"""A workflow whose governance could not be resolved does not run.

`_apply_governance_envelope` was flag-gated and fail-open, and said so: "any
error -> return None and the workflow runs exactly as before". The workflow then
ran ungoverned, and nothing anywhere reported that it had. Everywhere else in
this product fails closed -- `_apply_authority` twelve lines below it is
explicitly fail-CLOSED, and says the two must never be merged.

The hole underneath it was worse than the flag. `resolve_governance` swallowed
every exception and returned `{}` -- the same value it returns for "resolved
fine, nothing governs this workflow". A caller could not tell a governance
database that was unreachable from a workflow that legitimately has no
governance, so failing closed was not implementable at all until the resolver
stopped hiding the difference.

Three states, and they must stay distinguishable:

  error      the governance could not be read -> STOP, blocked, evidence kept
  empty      it was read, nothing governs this workflow -> run, and say so
  excluded   policy says this one runs without an envelope -> run, no envelope

`empty` is not hypothetical and must keep running: on 2026-09-10 quote_evaluation
and requirements_to_ranking both resolve to nothing at all against the live
governance tables. Turning "no governance" into "stop" would have taken them out.
"""
from types import SimpleNamespace

import pytest

from agents.base_agent import AgentOutput, AgentStatus
from orchestration.orchestrator import Orchestrator


# ---------------------------------------------------------------------------
# harness — a real Orchestrator driven through execute_workflow, after
# tests/orchestration/test_supplier_interaction_needs_the_engine.py
# ---------------------------------------------------------------------------
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


class StubPolicyEngine:
    """Stands in for PolicyEngine. ``exclusions`` is what the policy row says."""

    def __init__(self, exclusions=("document_extraction",), present=True):
        self.present = present
        self.exclusions = list(exclusions)
        self.asked = []

    def get_policy(self, slug):
        self.asked.append(slug)
        if not self.present:
            return None
        return {
            "policyName": "WorkflowGovernancePolicy",
            "details": {"rules": {"ungoverned_workflows": self.exclusions}},
        }

    supplier_policies = []

    def validate_workflow(self, *a, **k):
        return {"allowed": True}


class StubNick:
    def __init__(self, agents, policy_engine):
        self.settings = StubSettings()
        self.agents = agents
        self.policy_engine = policy_engine
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
    # No agent-action writes from a unit test.
    import src.services.agent_actions as A
    monkeypatch.setattr(A, "record_action", lambda **k: None)


@pytest.fixture
def agent():
    return StubAgent()


def _orchestrator(agent, policy_engine):
    # `negotiation` has no declarative graph, so it reaches the generic path and
    # completes -- which is what makes "it ran anyway" observable.
    nick = StubNick({"negotiation": agent, "document_extraction": agent},
                    policy_engine)
    orch = Orchestrator(nick)
    orch._workflow_engine = None
    orch._workflow_registry = {}
    return orch


def _explodes(monkeypatch, boom="governance database is unreachable"):
    """Make the resolver fail the way a dead governance database does."""
    import src.services.governance_tools.envelope as E

    def _boom(workflow_name, agent=None):
        raise RuntimeError(boom)

    monkeypatch.setattr(E, "resolve_governance", _boom)


def _resolves(monkeypatch, envelope):
    import src.services.governance_tools.envelope as E
    monkeypatch.setattr(E, "resolve_governance",
                        lambda workflow_name, agent=None: envelope)


# ---------------------------------------------------------------------------
# error -> stop
# ---------------------------------------------------------------------------
def test_an_exploding_resolver_stops_the_workflow(agent, monkeypatch):
    """The finding itself. Today this returns completed."""
    _explodes(monkeypatch)
    orch = _orchestrator(agent, StubPolicyEngine())

    response = orch.execute_workflow("negotiation", {})

    assert response["status"] == "blocked", (
        f"a workflow whose governance could not be resolved ran anyway: {response}"
    )


def test_it_does_not_run_the_agents_ungoverned(agent, monkeypatch):
    """'Blocked' has to mean nothing happened, not that a status was relabelled."""
    _explodes(monkeypatch)
    orch = _orchestrator(agent, StubPolicyEngine())

    orch.execute_workflow("negotiation", {})

    assert not agent.calls, f"the workflow ran ungoverned anyway: {agent.calls}"


def test_the_error_is_kept_as_evidence(agent, monkeypatch):
    """A refusal nobody can explain later is not reviewable."""
    _explodes(monkeypatch, boom="governance database is unreachable")
    orch = _orchestrator(agent, StubPolicyEngine())

    response = orch.execute_workflow("negotiation", {})

    blob = f"{response.get('reason', '')} {response.get('evidence', '')}".lower()
    assert "governance" in blob, response
    assert "unreachable" in blob, (
        f"the underlying error was dropped, so the block cannot be reviewed: {response}"
    )


# ---------------------------------------------------------------------------
# the exclusion list lives in policy, not in an `if`
# ---------------------------------------------------------------------------
def test_the_exclusion_list_is_read_from_policy(agent, monkeypatch):
    _explodes(monkeypatch)  # would block anything that resolves
    policies = StubPolicyEngine(exclusions=["document_extraction"])
    orch = _orchestrator(agent, policies)

    response = orch.execute_workflow("document_extraction", {})

    assert response["status"] != "blocked", (
        f"an excluded workflow was blocked: {response}"
    )
    assert policies.asked, "the exclusion list was never read from policy"


def test_a_workflow_the_policy_does_not_exclude_is_not_excluded(agent, monkeypatch):
    _explodes(monkeypatch)
    orch = _orchestrator(agent, StubPolicyEngine(exclusions=["document_extraction"]))

    response = orch.execute_workflow("negotiation", {})

    assert response["status"] == "blocked", response


def test_a_missing_exclusion_policy_excludes_nothing(agent, monkeypatch):
    """A missing rule denies. The exemption must be stated to exist."""
    _explodes(monkeypatch)
    orch = _orchestrator(agent, StubPolicyEngine(present=False))

    response = orch.execute_workflow("document_extraction", {})

    assert response["status"] == "blocked", (
        f"a workflow was exempted by a policy that does not exist: {response}"
    )


def test_the_policy_can_widen_the_exclusion_without_a_code_change(agent, monkeypatch):
    """The point of moving it out of the `if`: it is data now."""
    _explodes(monkeypatch)
    orch = _orchestrator(agent, StubPolicyEngine(exclusions=["negotiation"]))

    response = orch.execute_workflow("negotiation", {})

    assert response["status"] != "blocked", response


# ---------------------------------------------------------------------------
# empty is not an error
# ---------------------------------------------------------------------------
def test_a_workflow_with_no_governance_still_runs(agent, monkeypatch):
    """quote_evaluation and requirements_to_ranking are in this state live.
    Over-correcting here would have stopped them."""
    _resolves(monkeypatch, {"agent": "negotiation_agent", "policies": [], "prompts": []})
    orch = _orchestrator(agent, StubPolicyEngine())

    response = orch.execute_workflow("negotiation", {})

    assert response["status"] == "completed", response
    assert agent.calls, "an ungoverned-but-resolvable workflow should still run"


def test_a_resolved_envelope_is_still_injected(agent, monkeypatch):
    """The existing behaviour this must not break."""
    _resolves(monkeypatch, {"agent": "negotiation_agent",
                            "policies": [{"policy_type": "negotiation"}], "prompts": []})
    orch = _orchestrator(agent, StubPolicyEngine())

    orch.execute_workflow("negotiation", {})

    assert agent.calls, "the workflow did not run"
    assert agent.calls[0].get("governed", {}).get("agent") == "negotiation_agent"


# ---------------------------------------------------------------------------
# the resolver must stop hiding the difference
# ---------------------------------------------------------------------------
def test_the_resolver_reports_a_failure_instead_of_returning_empty(monkeypatch):
    """`{}` meant both "nothing governs this" and "the lookup blew up". While
    those were the same value, nothing above could fail closed on one and not
    the other."""
    import src.services.governance_tools as pkg
    import src.services.governance_tools.envelope as E

    class _Boom:
        @staticmethod
        def refresh():
            raise RuntimeError("governance database is unreachable")

    # Patched on the PACKAGE, not in sys.modules: `from package import tools`
    # reads the package attribute once the submodule has been imported, so a
    # sys.modules entry is ignored whenever another test got there first.
    monkeypatch.setattr(pkg, "tools", _Boom)

    with pytest.raises(E.GovernanceUnavailable):
        E.resolve_governance("supplier_ranking")


def test_a_governance_database_that_cannot_be_read_is_not_reported_as_ungoverned(
    monkeypatch,
):
    """The failure mode that makes the rest of this hollow.

    Nothing under the resolver raises. `PolicyEngine._fetch_policy_rows` logs
    "Failed to load policies from database" and returns `[]`; `tools.refresh`
    swallows and logs at debug; `tools.list_governance` returns empty lists on
    any exception. So an unreachable governance database does not arrive as an
    error at all -- it arrives as an envelope governing nothing, which reads as
    "no policy is linked to this workflow" and runs.

    A control that only fires when something remembers to raise is not a
    control. This asserts on the real fail-open path rather than on an injected
    exception.
    """
    import src.services.governance_tools.tools as GT

    class _DeadPromptEngine:
        def refresh(self):
            return None

        def all_prompts(self):
            return []

    class _DeadCursor:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, *a, **k):
            raise RuntimeError("could not read the governance database")

    class _DeadConn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def cursor(self):
            return _DeadCursor()

    from engines.policy_engine import PolicyEngine
    import src.services.governance_tools.envelope as E

    # Exactly what production holds after the governance query starts failing:
    # engines that loaded nothing and told nobody. (A factory that fails to
    # CONNECT does raise, and is already handled; this is the quieter one.)
    monkeypatch.setattr(GT, "_pol", PolicyEngine(connection_factory=_DeadConn))
    monkeypatch.setattr(GT, "_pe", _DeadPromptEngine())

    assert GT._pol.list_policies() == [], "premise: the engine loaded nothing"

    with pytest.raises(E.GovernanceUnavailable):
        E.resolve_governance("supplier_ranking")


def test_a_refresh_that_failed_does_not_resolve_on_stale_governance(monkeypatch):
    """`tools.refresh` catches its own failure and logs it at debug.

    The engines then keep whatever they last loaded, so resolution carries on
    against governance that may be arbitrarily old — with nothing in the
    envelope, the log or the run to say the reload never happened. Old rules
    quietly applied are their own kind of ungoverned.
    """
    import src.services.governance_tools.tools as GT
    import src.services.governance_tools.envelope as E

    class _StalePolicyEngine:
        def reload_policies(self):
            raise RuntimeError("could not read the governance database")

        def list_policies(self):
            return [{"policy_type": "supplier_ranking", "slug": "supplier_ranking",
                     "policy_linked_agents": "supplier_ranking_agent"}]

    class _StalePromptEngine:
        def refresh(self):
            raise RuntimeError("could not read the governance database")

        def all_prompts(self):
            return []

    monkeypatch.setattr(GT, "_pol", _StalePolicyEngine())
    monkeypatch.setattr(GT, "_pe", _StalePromptEngine())

    with pytest.raises(E.GovernanceUnavailable):
        E.resolve_governance("supplier_ranking")
