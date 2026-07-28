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
