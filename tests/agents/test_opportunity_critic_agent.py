"""The critic agent: refuses without its governed prompt, refuses malformed output."""
import json
from unittest.mock import MagicMock, patch

from agents.base_agent import AgentContext, AgentStatus
from src.agents.opportunity_critic_agent import OpportunityCriticAgent

_FINDING = {
    "opportunity_ref_id": "ref-1",
    "detector_type": "Price Benchmark Variance",
    "supplier_id": "SUP-MeridianSystems12",
    "financial_impact_gbp": 18330.80,
    "facts_state": "RESOLVED",
    "source_records": ["PO000967"],
    "calculation_details": {"actual_price": 4998.53, "benchmark_price": 4540.26,
                            "quantity": 40.0},
}

_WELL_FORMED = {
    "verdict": "INVALID",
    "confidence": "UNASSESSED",
    "critic_claim": None,
    "negotiator_note": "The benchmark is the cheapest price ever recorded, not a baseline.",
    "tests": [{"test": "anchor_validity", "result": "INVALIDATE", "reason": "fabricated"}],
    "value": {"detector_proposed": 18330.80, "critic_addressable": None,
              "currency": "GBP", "basis": "annualised", "haircuts_applied": []},
    "lever": {"exists": False, "type": "none"},
    "gaps": [{"gap_id": "G1", "test": "anchor_validity", "type": "DETECTOR_LOGIC",
              "what_is_missing": "anchor selection", "why_it_matters": "class of error",
              "blocking": False, "resolves_to": "prevents recurrence",
              "likely_source": "detector fix", "owner_hint": "engineering",
              "effort": "LOW"}],
}


def _context():
    return AgentContext(workflow_id="wf-1", agent_id="opportunity_critic",
                        user_id=None, input_data={"finding": _FINDING})


def _agent():
    return OpportunityCriticAgent(MagicMock())


def _answers(payload, error=None):
    """Patch the critic's model call. It returns (answer, error), not a
    ToolRunResult: the critic talks to the model directly, with no tools and no
    output-safety gate (see opportunity_critic/llm.py)."""
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return patch("src.agents.opportunity_critic_agent.ask_for_critique",
                 return_value=(text, error))


def test_refuses_to_run_without_its_governed_prompt():
    # The DB prompt is the system of record. A code default that silently wins
    # is how governance stops being governance.
    with patch("src.services.opportunity_critic.governed.load_system_prompt",
               return_value=(None, None)), _answers(_WELL_FORMED):
        out = _agent().run(_context())
    assert out.status == AgentStatus.FAILED
    assert "governed prompt" in (out.error or "")


def test_a_well_formed_critique_is_recorded():
    with patch("src.services.opportunity_critic.governed.load_system_prompt",
               return_value=("PROMPT", 1)), \
         _answers(_WELL_FORMED), \
         patch("src.agents.opportunity_critic_agent.record_critique",
               return_value=99) as rec:
        out = _agent().run(_context())
    assert out.status == AgentStatus.SUCCESS
    assert out.data["critique_id"] == 99
    assert rec.called


def test_the_governed_prompt_reaches_the_model():
    # The bp_prompt row is the critic's instructions; if it never reaches the
    # model, the agent is judging by AgentNick's default character instead.
    with patch("src.services.opportunity_critic.governed.load_system_prompt",
               return_value=("THE GOVERNED PROMPT", 1)), \
         patch("src.agents.opportunity_critic_agent.ask_for_critique",
               return_value=(json.dumps(_WELL_FORMED), None)) as ask, \
         patch("src.agents.opportunity_critic_agent.record_critique", return_value=1):
        _agent().run(_context())
    system = ask.call_args.args[0]
    assert "THE GOVERNED PROMPT" in system


def test_a_critique_breaking_an_invariant_is_refused_not_repaired():
    broken = dict(_WELL_FORMED, verdict="VALID")
    broken["value"] = dict(broken["value"], critic_addressable=99999.0)
    with patch("src.services.opportunity_critic.governed.load_system_prompt",
               return_value=("PROMPT", 1)), \
         _answers(broken), \
         patch("src.agents.opportunity_critic_agent.record_critique") as rec:
        out = _agent().run(_context())
    assert out.status == AgentStatus.FAILED
    assert "above the detector" in (out.error or "")
    assert not rec.called, "a critique that breaks an invariant must not be persisted"


def test_unparseable_model_output_fails_rather_than_guessing():
    with patch("src.services.opportunity_critic.governed.load_system_prompt",
               return_value=("PROMPT", 1)), \
         _answers("I think this one is fine, really"), \
         patch("src.agents.opportunity_critic_agent.record_critique") as rec:
        out = _agent().run(_context())
    assert out.status == AgentStatus.FAILED
    assert not rec.called


def test_a_dead_model_is_reported_not_guessed_at():
    with patch("src.services.opportunity_critic.governed.load_system_prompt",
               return_value=("PROMPT", 1)), \
         patch("src.agents.opportunity_critic_agent.ask_for_critique",
               return_value=(None, "Read timed out")), \
         patch("src.agents.opportunity_critic_agent.record_critique") as rec:
        out = _agent().run(_context())
    assert out.status == AgentStatus.FAILED
    assert "Read timed out" in (out.error or "")
    assert not rec.called
