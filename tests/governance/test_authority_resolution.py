"""The autonomy policy must resolve by slug, and ship with auto-reply disabled."""
import json

# tests/conftest.py owns sys.path (repo root + src/). This file used to append the repo
# root itself; harmless, but the per-file pattern is what produced the order-dependent
# import precedence fixed in this round, so the path lives in exactly one place now.
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


def test_deferred_approval_lookup_raising_fails_closed():
    # The autonomy lookup (first call) succeeds and defers its value limit to
    # "approval_threshold"; the SECOND call -- the approval lookup this defers
    # to -- is the one that raises. This is a different raise site from
    # test_a_raising_policy_engine_fails_closed_not_open, which never gets past
    # the first call, and it must fail closed too.
    autonomy_policy = {
        "policy_type": "email_autonomy",
        "slug": "email_reply_autonomy",
        "policyName": "EmailReplyAutonomyPolicy",
        "details": {"rules": AUTONOMY_RULES},
        "raw_row": {"policy_id": 11, "policy_name": "EmailReplyAutonomyPolicy"},
    }

    class ExplodingOnDeferredLookup:
        def get_policy(self, slug):
            if slug == "email_reply_autonomy":
                return autonomy_policy
            raise RuntimeError("approval policy table unreachable")

    out = resolve_authority(ExplodingOnDeferredLookup(), ["email_drafting_agent"])
    block = out["email_drafting_agent"]
    assert block["governed"] is False
    assert "approval_threshold" in block["reason"]


def test_resolution_runs_concurrently():
    # Each agent's resolution is an independent read; serial round-trips are the
    # only cost. Prove concurrent entry with a Barrier rather than a timing
    # heuristic: every call blocks until ALL agents' calls have arrived. A
    # serial implementation can never get more than one thread onto the
    # barrier at a time, so it cannot fill the barrier before the timeout and
    # every call raises BrokenBarrierError -- which fails the test without
    # ever asserting a wall-clock duration.
    import threading

    agents = ["a_agent", "b_agent", "c_agent"]
    barrier = threading.Barrier(len(agents), timeout=2)
    broke = {"happened": False}

    class Slow:
        def get_policy(self, slug):
            try:
                barrier.wait()
            except threading.BrokenBarrierError:
                broke["happened"] = True
            return None

    resolve_authority(Slow(), agents)
    assert not broke["happened"], "agents were not resolved concurrently (barrier never filled)"


def test_an_approval_policy_with_no_currency_yields_no_currency():
    """A denomination the policy does not state must not be invented as 'GBP'.

    `str(rules.get("currency") or "GBP")` handed the decision engine a limit that
    looked like sterling when nobody had said so, and its currency gate would then
    match a GBP reply against a currency of the resolver's own invention. None means
    "the policy does not say", and the decision engine escalates on that.

    governed stays True on purpose: the escalation belongs at decide time, where it can
    be explained to a human, not in the resolver.
    """
    approval = {**_APPROVAL_ROW, "policy_details": json.dumps({
        "policy_identifier": "approval_threshold",
        "rules": {"default_threshold_gbp": 10000},   # no currency stated
    })}
    engine = PolicyEngine(policy_rows=[_autonomy_row(), approval])
    block = resolve_authority(engine, ["email_drafting_agent"])["email_drafting_agent"]
    assert block["governed"] is True
    assert block["limit_gbp"] == "10000"
    assert block["limit_currency"] is None, "a currency nobody stated is not GBP"


def test_a_stated_non_sterling_currency_is_carried_through_unchanged():
    approval = {**_APPROVAL_ROW, "policy_details": json.dumps({
        "policy_identifier": "approval_threshold",
        "rules": {"currency": "EUR", "default_threshold_gbp": 10000},
    })}
    engine = PolicyEngine(policy_rows=[_autonomy_row(), approval])
    block = resolve_authority(engine, ["email_drafting_agent"])["email_drafting_agent"]
    assert block["limit_currency"] == "EUR"


def test_a_resolved_block_with_no_currency_escalates_at_decide_time():
    """The other half of the fix: the resolver reports it, the engine acts on it."""
    # conftest owns sys.path; this used to re-insert src/ mid-test, which reordered
    # import precedence for every module loaded after it ran.
    from engines.decision_engine import DecisionEngine, ESCALATED

    approval = {**_APPROVAL_ROW, "policy_details": json.dumps({
        "policy_identifier": "approval_threshold",
        "rules": {"default_threshold_gbp": 10000},
    })}
    engine = PolicyEngine(policy_rows=[_autonomy_row(), approval])
    block = resolve_authority(engine, ["email_drafting_agent"])["email_drafting_agent"]
    # Widen the intent list so the decision reaches the money gates at all.
    block = {**block, "auto_intents": ["acknowledge"]}

    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from src.services.email_intent import ReplyIntent

    eng = DecisionEngine(SimpleNamespace(
        policy_engine=None,
        get_db_connection=MagicMock(side_effect=RuntimeError("db off in test")),
    ))
    eng._fetch_email_reply = lambda _id: {          # type: ignore
        "id": 1, "unique_id": "wf-1-x", "supplier_id": "x",
        "response_text": "Thank you.", "price": 94000, "currency": "GBP",
        "prior_price": 96000, "round_number": 1, "prior_price_round": 1,
        "auto_replies_on_thread": 0,
    }
    eng._classify = lambda _b: ReplyIntent("acknowledge", 0.99, "Thank you.", True)  # type: ignore
    d = eng.decide_email_reply("1", authority=block)
    assert d.resolution == ESCALATED
    assert "carries no currency of its own" in d.rationale
