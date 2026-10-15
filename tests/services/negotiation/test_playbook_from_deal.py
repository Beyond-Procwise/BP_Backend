"""The Advisor's bucket and style reach the NegotiationAgent's plays (audit rec. 8)."""
import pytest

from agents.base_agent import AgentContext
from agents.negotiation_agent import NegotiationAgent
from src.services.negotiation_advice import advisor

_POSTURE = {
    "deal_id": "D-1", "quadrant": "Leverage", "style": "Competitive",
    "signals": {"deal_value": 200000.0, "quote_supplier_count": 1,
                "alternative_supplier_count": 120, "price_variance_pct": 8.4,
                "invoice_total": None, "po_total": None},
    "stated_facts": {},
}


def _agent():
    # The resolver uses no instance state; skip the constructor's DB/model setup.
    return NegotiationAgent.__new__(NegotiationAgent)


def _ctx(**input_data):
    return AgentContext(workflow_id="wf", agent_id="negotiation", user_id="t",
                        input_data=input_data)


@pytest.fixture
def calls(monkeypatch):
    seen = []

    def fake(**kw):
        seen.append(kw)
        return dict(_POSTURE) if kw.get("deal_id") == "D-1" else None

    monkeypatch.setattr(advisor, "read_posture", fake)
    return seen


def test_a_deal_gives_the_agent_the_advisors_bucket_style_and_grounded_plays(calls):
    out = _agent()._resolve_playbook_context(_ctx(deal_id="D-1"), {})
    assert out["plays"]
    assert out["style"] == "Competitive"
    assert out["deal_id"] == "D-1"
    assert all(p["state"] in ("ready", "groundwork") for p in out["plays"])


def test_no_deal_means_no_lookup_and_no_plays(calls):
    out = _agent()._resolve_playbook_context(_ctx(supplier="S1"), {})
    assert out["plays"] == []
    assert calls == []


def test_quote_id_is_passed_through_for_the_deal_lookup(calls):
    _agent()._resolve_playbook_context(_ctx(quote_id="Q-9"), {})
    assert calls == [{"deal_id": None, "quote_id": "Q-9"}]


def test_a_caller_supplied_bucket_wins_but_plays_are_still_grounded(calls):
    out = _agent()._resolve_playbook_context(
        _ctx(deal_id="D-1", supplier_type="Bottleneck", negotiation_style="Principled"), {})
    assert out["descriptor"].startswith("Bottleneck") or "Bottleneck" in str(out)
    assert out["plays"] and all("state" in p for p in out["plays"])


def test_a_failed_lookup_is_no_plays_not_a_crash(monkeypatch):
    def boom(**kw):
        raise RuntimeError("db down")

    monkeypatch.setattr(advisor, "read_posture", boom)
    out = _agent()._resolve_playbook_context(_ctx(deal_id="D-1"), {})
    assert out["plays"] == []


def test_the_deals_bidder_count_reaches_the_agent(calls):
    """The email may mention other proposals only on this count (lever audit rec. 5)."""
    assert _agent()._resolve_playbook_context(_ctx(deal_id="D-1"), {})["bidder_count"] == 1


def test_no_deal_no_bidder_count(calls):
    assert _agent()._resolve_playbook_context(_ctx(supplier="S1"), {}).get("bidder_count") is None
