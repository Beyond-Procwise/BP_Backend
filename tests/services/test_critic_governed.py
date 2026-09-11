"""Governed inputs are fetched, never defaulted.

The approvals_agent lesson: an unresolvable governed threshold must escalate,
never fall back to a constant. A made-up materiality floor silently discards
real findings.
"""
from src.engines.policy_engine import PolicyEngine
from src.orchestration.prompt_engine import PromptEngine
from src.services.opportunity_critic.governed import (
    Thresholds,
    load_system_prompt,
    load_thresholds,
)


class _FakeEngine:
    def __init__(self, payload):
        self._payload = payload

    def get_policy(self, slug):
        return self._payload


def test_thresholds_come_from_the_governed_policy():
    engine = _FakeEngine({
        "policy_id": 42,
        "version": 3,
        "policy_details": {"rules": {
            "index_band_pp": 2.0,
            "materiality_floor_gbp": 5000.0,
            "relative_gap_floor": 0.05,
            "anchor_stale_days": 365,
            "friction_bands": {"default": 15.0, "sole_source": 40.0},
        }},
    })
    out = load_thresholds(engine)
    assert out.index_band_pp == 2.0
    assert out.materiality_floor_gbp == 5000.0
    assert out.friction_bands["sole_source"] == 40.0
    assert out.source == {"policy_id": 42, "version": 3}


def test_a_missing_policy_yields_none_not_a_default():
    out = load_thresholds(_FakeEngine(None))
    assert out.index_band_pp is None
    assert out.materiality_floor_gbp is None
    assert out.friction_bands == {}
    assert out.source is None


def test_a_partial_policy_leaves_the_missing_field_none():
    engine = _FakeEngine({"policy_id": 1, "version": 1,
                          "policy_details": {"rules": {"index_band_pp": 2.0}}})
    out = load_thresholds(engine)
    assert out.index_band_pp == 2.0
    assert out.materiality_floor_gbp is None


def test_thresholds_are_frozen_so_a_caller_cannot_edit_a_governed_rule():
    import dataclasses
    import pytest
    out = load_thresholds(_FakeEngine(None))
    with pytest.raises(dataclasses.FrozenInstanceError):
        out.index_band_pp = 99.0


# The fake engine above hands back a raw bp_policy row. The real engines do
# not: PolicyEngine normalises rules under "details" and keeps the row under
# "raw_row", and PromptEngine.get_prompt() takes a numeric id. A loader written
# against the fake alone reads nothing from either in production -- so these
# drive the real engines, from in-memory rows shaped exactly like the seed.

_POLICY_ROW = {
    "policy_id": 901,
    "policy_name": "opportunity_critic_thresholds",
    "policy_type": "critique",
    "policy_desc": "Thresholds the Opportunity Critic applies when testing a candidate.",
    "policy_details": {"rules": {
        "index_band_pp": 2.0,
        "materiality_floor_gbp": None,
        "relative_gap_floor": 0.05,
        "anchor_stale_days": 365,
        "friction_bands": {"default": 15.0, "sole_source": 40.0, "commodity": 5.0},
        "shadow_detectors": [],
    }},
    "policy_linked_agents": "opportunity_critic",
    "policy_status": 1,
    "version": 1,
}

_PROMPT_ROW = {
    "prompt_id": 77,
    "prompt_name": "opportunity_critic_system",
    "prompt_type": "critique",
    "prompt_linked_agents": "opportunity_critic",
    "prompts_desc": {"prompt_template": "# Opportunity Critic — System Prompt"},
    "prompts_status": 1,
    "version": 4,
}


def test_thresholds_resolve_through_the_real_policy_engine():
    out = load_thresholds(PolicyEngine(policy_rows=[_POLICY_ROW]))
    assert out.index_band_pp == 2.0
    assert out.relative_gap_floor == 0.05
    assert out.anchor_stale_days == 365
    assert out.friction_bands == {"default": 15.0, "sole_source": 40.0, "commodity": 5.0}
    assert out.source == {"policy_id": 901, "version": 1}


def test_a_seeded_null_floor_stays_none_through_the_real_engine():
    # Present-and-null is the governed answer "no floor chosen", not a gap to fill.
    out = load_thresholds(PolicyEngine(policy_rows=[_POLICY_ROW]))
    assert out.materiality_floor_gbp is None


def test_system_prompt_resolves_by_name_through_the_real_prompt_engine():
    text, version = load_system_prompt(PromptEngine(prompt_rows=[_PROMPT_ROW]))
    assert text == "# Opportunity Critic — System Prompt"
    assert version == 4


def test_an_absent_prompt_is_none_so_the_critic_refuses_to_run():
    assert load_system_prompt(PromptEngine(prompt_rows=[])) == (None, None)
