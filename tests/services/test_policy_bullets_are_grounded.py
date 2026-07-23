"""A policy answer may only say what the policy says.

`_policy_section_bullets` was inventing policy in three different ways, and the
invented text was indistinguishable from the real thing on screen:

1. **Substitution.** `_expand_policy_sentence` keyword-matched each clause and
   emitted a canned sentence *instead of* it. "All expenses must be submitted
   within 30 days of the transaction date" was rendered as "Submit only
   eligible, well-documented expenses so they are accepted" — the deadline, the
   one actionable fact in the clause, was dropped.

2. **Inversion.** Restriction clauses were split on the word "include" and each
   fragment prefixed with "Do not claim". Applied to "Alcohol cannot be claimed,
   although team meals include soft drinks" it produced "Do not claim soft
   drinks" — the opposite of the policy — and lost the alcohol rule entirely.

3. **Padding.** Every section was topped up to a quota (3 concise, 5 expanded)
   from a table of hardcoded sentences, so a policy with no spending limits
   still asserted "Check the policy for specific monetary caps before spending",
   and a policy with no exceptions still told the reader to "Escalate unusual
   circumstances to Finance".

The clauses reaching this function are verbatim sentences pulled from the
retrieved policy documents by `_extract_policy_payload`. Rendering them is the
whole job.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from agents.rag_agent import RAGAgent


@pytest.fixture()
def agent() -> RAGAgent:
    return RAGAgent.__new__(RAGAgent)


def _payload(**sections):
    base = {
        "requirements": [],
        "restrictions": [],
        "spending_limits": [],
        "approval_process": [],
        "examples": [],
        "exceptions": [],
        "operational_notes": [],
    }
    base.update(sections)
    return base


# --------------------------------------------------------------------------
# 1. The clause itself is what gets rendered
# --------------------------------------------------------------------------


def test_the_source_clause_survives_instead_of_a_canned_stand_in(agent):
    clause = "All expenses must be submitted within 30 days of the transaction date."
    bullets = agent._policy_section_bullets(
        _payload(requirements=[clause]), "requirements", "concise"
    )
    assert bullets == [clause]


def test_an_approval_clause_is_not_replaced_by_a_generic_one(agent):
    clause = "Purchases above £5,000 require Finance Director approval before the order is placed."
    bullets = agent._policy_section_bullets(
        _payload(approval_process=[clause]), "approval_process", "concise"
    )
    assert bullets == [clause]
    assert not any("unusual spend" in b for b in bullets)


def test_a_policy_mention_does_not_invent_an_intranet(agent):
    clause = "This policy applies to all Group employees and contractors."
    bullets = agent._policy_section_bullets(
        _payload(requirements=[clause]), "requirements", "concise"
    )
    assert bullets == [clause]
    assert not any("intranet" in b.lower() for b in bullets)


def test_a_stated_limit_is_quoted_not_paraphrased(agent):
    clause = "Hotel accommodation is capped at £150 per night."
    bullets = agent._policy_section_bullets(
        _payload(spending_limits=[clause]), "spending_limits", "concise"
    )
    assert bullets == [clause]
    assert not any("unless you have written approval" in b for b in bullets)


# --------------------------------------------------------------------------
# 2. Nothing is inverted
# --------------------------------------------------------------------------


def test_a_permission_inside_a_restriction_is_not_turned_into_a_prohibition(agent):
    """The regression in full: soft drinks are allowed, and were forbidden."""
    clause = "Alcohol cannot be claimed, although team meals include soft drinks."
    bullets = agent._policy_section_bullets(
        _payload(restrictions=[clause]), "restrictions", "concise"
    )
    assert bullets == [clause]
    assert not any("Do not claim soft drinks" in b for b in bullets)
    # And the rule that was actually stated must still be on screen.
    assert any("Alcohol cannot be claimed" in b for b in bullets)


def test_a_restriction_does_not_manufacture_a_requirement(agent):
    clause = "First class travel is not allowed unless approved by the CFO."
    bullets = agent._policy_section_bullets(
        _payload(restrictions=[clause]), "requirements", "concise"
    )
    assert bullets == []


def test_an_amount_in_a_restriction_does_not_become_a_spending_limit(agent):
    clause = "Gifts over £50 cannot be claimed."
    bullets = agent._policy_section_bullets(
        _payload(restrictions=[clause]), "spending_limits", "concise"
    )
    assert bullets == []
    assert not any("ceiling" in b for b in bullets)


# --------------------------------------------------------------------------
# 3. Nothing is padded
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "requirements",
        "restrictions",
        "spending_limits",
        "approval_process",
        "examples",
        "exceptions",
        "operational_notes",
    ],
)
def test_a_section_the_policy_is_silent_on_stays_empty(agent, key):
    assert agent._policy_section_bullets(_payload(), key, "concise") == []


@pytest.mark.parametrize("depth", ["concise", "expanded"])
def test_one_clause_yields_one_bullet_at_any_depth(agent, depth):
    clause = "Receipts must be attached to every claim."
    bullets = agent._policy_section_bullets(
        _payload(requirements=[clause]), "requirements", depth
    )
    assert bullets == [clause]


def test_no_hardcoded_policy_assertions_remain(agent):
    """None of the invented sentences may be reachable, from any input."""
    invented = (
        "Follow this policy before and after each purchase to stay compliant.",
        "Treat anything not explicitly allowed in the policy as prohibited.",
        "Check the policy for specific monetary caps before spending.",
        "Capture written approval in advance for any exception.",
        "Model your claim on the compliant scenarios described in the policy.",
        "Escalate unusual circumstances to Finance for documented exceptions.",
        "Keep documentation tidy—attach receipts, coding, and approvals in the workflow.",
    )
    payload = _payload(
        requirements=["Expenses must be approved by a line manager."],
        restrictions=["Fines cannot be claimed unless the company accepts liability."],
        spending_limits=["Meals are limited to £30 per person."],
    )
    rendered = []
    for key in payload:
        rendered.extend(agent._policy_section_bullets(payload, key, "expanded"))

    for sentence in invented:
        assert sentence not in rendered


# --------------------------------------------------------------------------
# What must keep working
# --------------------------------------------------------------------------


def test_every_clause_is_rendered_not_just_the_first(agent):
    clauses = [
        "Receipts must be attached to every claim.",
        "Claims must be submitted within 30 days.",
        "Mileage must be recorded against a project code.",
    ]
    bullets = agent._policy_section_bullets(
        _payload(requirements=clauses), "requirements", "concise"
    )
    assert bullets == clauses


def test_duplicate_clauses_collapse(agent):
    clause = "Receipts must be attached to every claim."
    bullets = agent._policy_section_bullets(
        _payload(requirements=[clause, clause]), "requirements", "concise"
    )
    assert bullets == [clause]


def test_the_depth_cap_still_limits_a_long_section(agent):
    clauses = [f"Rule number {n} must be followed." for n in range(1, 9)]
    concise = agent._policy_section_bullets(
        _payload(requirements=clauses), "requirements", "concise"
    )
    expanded = agent._policy_section_bullets(
        _payload(requirements=clauses), "requirements", "expanded"
    )
    assert concise == clauses[:3]
    assert expanded == clauses[:5]


def test_examples_and_exceptions_keep_their_labels(agent):
    bullets = agent._policy_section_bullets(
        _payload(examples=["Taxis to client sites are claimable."]), "examples", "concise"
    )
    assert bullets == ["Example: Taxis to client sites are claimable."]

    bullets = agent._policy_section_bullets(
        _payload(exceptions=["Overnight stays are allowed when travel exceeds four hours."]),
        "exceptions",
        "concise",
    )
    assert bullets == [
        "Exception: Overnight stays are allowed when travel exceeds four hours."
    ]
