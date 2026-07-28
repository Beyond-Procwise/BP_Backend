"""The autonomy policy must resolve by slug, and ship with auto-reply disabled."""
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from engines.policy_engine import PolicyEngine

# The rules body exactly as the migration writes it. If the migration and this
# literal drift, the engine reads something the test never checked.
AUTONOMY_RULES = {
    "auto_reply_intents": [],
    "escalate_intents": [
        "price_change", "terms_change", "contract_variation",
        "liability", "dispute", "new_commitment",
    ],
    "defer_value_limit_to": "approval_threshold",
    "max_auto_replies_per_thread": 2,
    "min_intent_confidence": 0.8,
    "on_missing_policy": "escalate",
    "on_ungrounded_facts": "escalate",
}


def _autonomy_row():
    return {
        "policy_id": 11,
        "policy_name": "EmailReplyAutonomyPolicy",
        "policy_type": "email_autonomy",
        "policy_desc": "When the email agent may reply unattended",
        "policy_details": json.dumps(
            {"policy_identifier": "email_reply_autonomy", "rules": AUTONOMY_RULES}
        ),
        "policy_linked_agents": "email_drafting_agent, negotiation_agent, supplier_interaction_agent",
    }


def test_autonomy_policy_resolves_by_slug():
    engine = PolicyEngine(policy_rows=[_autonomy_row()])
    policy = engine.get_policy("email_reply_autonomy")
    assert policy is not None
    assert policy["policy_type"] == "email_autonomy"


def test_autonomy_policy_ships_with_auto_reply_disabled():
    engine = PolicyEngine(policy_rows=[_autonomy_row()])
    rules = engine.get_policy("email_reply_autonomy")["details"]["rules"]
    # The conservative default IS the safety property: nothing auto-sends until a
    # human widens this list in the Policies screen.
    assert rules["auto_reply_intents"] == []
    assert rules["on_missing_policy"] == "escalate"


def test_autonomy_policy_resolves_by_linked_agent_alias():
    engine = PolicyEngine(policy_rows=[_autonomy_row()])
    assert engine.get_policy("email_drafting_agent") is not None


# ---------------------------------------------------------------------------
# Task 2: resolve_authority -- per-agent authority blocks, fail-closed.
# ---------------------------------------------------------------------------

from src.services.governance_tools.authority import resolve_authority

_APPROVAL_ROW = {
    "policy_id": 10,
    "policy_name": "ApprovalThresholdPolicy",
    "policy_type": "approval",
    "policy_desc": "Spend authority",
    "policy_details": json.dumps({
        "policy_identifier": "approval_threshold",
        "rules": {"currency": "GBP", "default_threshold_gbp": 10000,
                  "on_above": "escalate", "on_at_or_below": "approve"},
    }),
    "policy_linked_agents": "approvals_agent",
}

AGENTS = ["email_drafting_agent", "negotiation_agent"]


def test_resolves_every_agent_with_the_governed_limit():
    engine = PolicyEngine(policy_rows=[_autonomy_row(), _APPROVAL_ROW])
    out = resolve_authority(engine, AGENTS)
    assert set(out) == set(AGENTS)
    for agent in AGENTS:
        block = out[agent]
        assert block["governed"] is True
        assert block["slug"] == "email_reply_autonomy"
        assert block["policy_name"] == "EmailReplyAutonomyPolicy"
        # The money limit is READ from the approval policy, never restated here.
        assert block["limit_gbp"] == "10000"
        assert block["limit_currency"] == "GBP"
        assert block["auto_intents"] == []
        assert "price_change" in block["escalate_intents"]
        assert block["max_auto_replies_per_thread"] == 2
        assert block["min_intent_confidence"] == 0.8


def test_missing_autonomy_policy_fails_closed():
    engine = PolicyEngine(policy_rows=[_APPROVAL_ROW])  # autonomy row absent
    block = resolve_authority(engine, AGENTS)["email_drafting_agent"]
    assert block["governed"] is False
    assert "email_reply_autonomy" in block["reason"]


def test_missing_approval_policy_fails_closed():
    engine = PolicyEngine(policy_rows=[_autonomy_row()])  # approval row absent
    block = resolve_authority(engine, AGENTS)["email_drafting_agent"]
    # An autonomy rule that defers its money limit to a policy that does not exist
    # has no limit at all. Ungoverned money must not be spendable.
    assert block["governed"] is False
    assert "approval_threshold" in block["reason"]


def test_unparseable_rules_fail_closed():
    bad = dict(_autonomy_row())
    bad["policy_details"] = json.dumps({"policy_identifier": "email_reply_autonomy",
                                        "rules": "not-an-object"})
    engine = PolicyEngine(policy_rows=[bad, _APPROVAL_ROW])
    block = resolve_authority(engine, AGENTS)["email_drafting_agent"]
    assert block["governed"] is False


def test_a_raising_policy_engine_fails_closed_not_open():
    class Exploding:
        def get_policy(self, slug):
            raise RuntimeError("policy table unreachable")

    out = resolve_authority(Exploding(), AGENTS)
    assert out["email_drafting_agent"]["governed"] is False
    assert out["negotiation_agent"]["governed"] is False


def test_resolution_runs_concurrently():
    # Each agent's resolution is an independent read; serial round-trips are the
    # only cost. Assert overlap rather than wall-clock, which is flaky.
    import threading
    import time

    inside = []
    barrier_hit = threading.Event()

    class Slow:
        def get_policy(self, slug):
            inside.append(slug)
            if len(inside) >= 2:
                barrier_hit.set()
            time.sleep(0.15)
            return None

    resolve_authority(Slow(), ["a_agent", "b_agent", "c_agent"])
    assert barrier_hit.is_set(), "agents were resolved one after another"
