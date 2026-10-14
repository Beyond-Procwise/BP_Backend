"""A policy store that cannot be READ must not look like a store with NO policies.

``RuleBook`` already refuses that trade and its module docstring names
``PolicyEngine`` as the counter-example: "``PolicyEngine`` returns ``[]`` when
its query fails, so a governance outage there is indistinguishable from 'no
policy applies'". These tests close that gap.

Two things are deliberately NOT changed, and are asserted here so a later edit
cannot quietly take them:

* Construction still succeeds against a dead store. ``base_agent`` builds a
  ``PolicyEngine`` during API startup; raising there turns a database blip into
  a boot failure, and a degraded boot is when the audit trail matters most.
* ``policy_rows=[]`` is an explicit, legitimate "no policies" fixture and stays
  available. Only a store that could not be *read* is degraded.
"""

import pytest

from engines.policy_engine import PolicyEngine, PolicyStoreUnavailable


WEIGHT_ROW = {
    "policy_id": "weight_allocation_policy",
    "policy_name": "WeightAllocationPolicy",
    "policy_type": "supplier_ranking",
    "policy_desc": "ranking weights",
    "policy_details": {
        "policy_identifier": "weight_allocation_policy",
        "applies_to": ["rank.suppliers"],
        "rules": {"default_weights": {"price": 0.5, "lead_time": 0.5}},
    },
    "policy_linked_agents": "supplier_ranking_agent",
    "version": 1,
}


class _DeadCursor:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, *a, **k):
        raise RuntimeError("could not read the governance database")


class _LiveCursor:
    description = (
        ("policy_id",), ("policy_name",), ("policy_type",), ("policy_desc",),
        ("policy_details",), ("policy_linked_agents",), ("version",),
    )

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, *a, **k):
        return None

    def fetchall(self):
        return [tuple(WEIGHT_ROW[c[0]] for c in self.description)]


class _Conn:
    def __init__(self, cursor_cls):
        self._cursor_cls = cursor_cls

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def cursor(self):
        return self._cursor_cls()


def dead_factory():
    """What production holds once the bp_policy query starts failing."""
    return _Conn(_DeadCursor)


def live_factory():
    return _Conn(_LiveCursor)


def dead_engine():
    return PolicyEngine(connection_factory=dead_factory)


# ---------------------------------------------------------------------------
# The store reports its own health
# ---------------------------------------------------------------------------
def test_construction_survives_an_unreadable_store():
    """Startup must not die on a governance outage -- but it must not lie either."""
    engine = dead_engine()
    assert engine.policy_store_available is False
    assert "could not read" in str(engine.policy_store_error)


def test_an_explicitly_empty_fixture_is_not_a_degraded_store():
    engine = PolicyEngine(policy_rows=[])
    assert engine.policy_store_available is True
    assert engine.list_policies() == []


def test_a_readable_store_is_available():
    engine = PolicyEngine(connection_factory=live_factory)
    assert engine.policy_store_available is True
    assert len(engine.list_policies()) == 1


# ---------------------------------------------------------------------------
# The authorization read paths refuse rather than answer "nothing applies"
# ---------------------------------------------------------------------------
def test_the_action_lookup_refuses_when_the_store_is_unreadable():
    """`policies_for_action` is the gate's only lookup path. Returning [] there
    is what lets guardrail.authorize fall through to its default-allow branch."""
    engine = dead_engine()
    with pytest.raises(PolicyStoreUnavailable):
        engine.policies_for_action("email.send")


def test_a_named_policy_lookup_refuses_when_the_store_is_unreadable():
    engine = dead_engine()
    with pytest.raises(PolicyStoreUnavailable):
        engine.get_policy("weight_allocation_policy")


def test_a_named_policy_lookup_still_returns_none_when_the_policy_is_absent():
    """Absent is not the same as unreadable, and must stay distinguishable."""
    engine = PolicyEngine(policy_rows=[])
    assert engine.get_policy("weight_allocation_policy") is None


def test_supplier_ranking_is_refused_when_the_store_is_unreadable():
    """This path used to return allowed=True with "No weight policy; using agent
    defaults" -- an outage silently became permission to rank on whatever the
    agent felt like."""
    engine = dead_engine()
    verdict = engine.validate_workflow("supplier_ranking", "user-1", {})
    assert verdict["allowed"] is False
    assert "policy store" in verdict["reason"].lower()


def test_supplier_ranking_is_still_allowed_when_no_weight_policy_exists():
    """The deliberate hole stays open: a store that genuinely holds no weight
    policy still lets ranking proceed on agent defaults."""
    engine = PolicyEngine(policy_rows=[])
    verdict = engine.validate_workflow("supplier_ranking", "user-1", {})
    assert verdict["allowed"] is True


def test_a_ranking_intent_is_refused_when_the_store_is_unreadable():
    engine = dead_engine()
    allowed, reason, _ = engine.validate_and_apply(
        {"template_id": "rank_by_criteria", "parameters": {"criteria": ["price"]}}
    )
    assert allowed is False
    assert "policy store" in reason.lower()


# ---------------------------------------------------------------------------
# A transient failure must not be permanent
# ---------------------------------------------------------------------------
def test_a_degraded_store_recovers_without_rebuilding_the_engine():
    """base_agent builds ONE PolicyEngine at startup and never rebuilds it, so a
    blip during boot used to leave that agent with zero policies for the life of
    the process."""
    calls = {"n": 0}

    def flaky_factory():
        calls["n"] += 1
        return _Conn(_DeadCursor if calls["n"] == 1 else _LiveCursor)

    engine = PolicyEngine(connection_factory=flaky_factory)
    engine.retry_cooldown_seconds = 0.0
    assert engine.policy_store_available is False

    policy = engine.get_policy("weight_allocation_policy")

    assert engine.policy_store_available is True
    assert policy is not None
    assert policy["policyName"] == "WeightAllocationPolicy"


def test_a_still_dead_store_is_not_retried_on_every_read():
    """The retry must not turn each authorization decision into a fresh connect
    attempt against a dead database."""
    calls = {"n": 0}

    def counting_factory():
        calls["n"] += 1
        return _Conn(_DeadCursor)

    engine = PolicyEngine(connection_factory=counting_factory)
    assert calls["n"] == 1

    for _ in range(3):
        with pytest.raises(PolicyStoreUnavailable):
            engine.policies_for_action("email.send")

    assert calls["n"] == 1, "reads inside the cooldown must not re-query"


# ---------------------------------------------------------------------------
# End to end: the gate closes
# ---------------------------------------------------------------------------
def test_the_guardrail_denies_when_the_policy_store_is_unreadable():
    """The gate already converts an exploding engine into a denial carrying the
    real error. It never fired for this case because the engine did not explode
    -- it returned []."""
    from src.services import guardrail
    from tests.guardrails.test_rbac import FakePrincipal

    decision = guardrail.authorize(
        "email.send",
        "communicate",
        FakePrincipal("sub-approver", {"cognito:groups": ["bp-approvers"]}),
        {},
        policy_engine=dead_engine(),
    )

    assert decision.allowed is False
    assert "could not read the governance database" in decision.evidence.get("error", "")
