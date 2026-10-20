"""Templates that arrive in a request payload must not be able to run code."""
import pytest
from jinja2.exceptions import SecurityError

from src.agents.email_drafting_agent import EmailDraftingAgent
from src.orchestration import orchestrator

EVIL = "{{ ''.__class__.__mro__[1].__subclasses__() }}"


def test_email_template_renders_normally():
    agent = EmailDraftingAgent.__new__(EmailDraftingAgent)
    assert agent._render_template_string("Hi {{ name }}", {"name": "Ann"}) == "Hi Ann"
    assert agent._render_template_string("Hi {name}", {"name": "Ann"}) == "Hi Ann"


def test_email_template_cannot_reach_python_internals():
    agent = EmailDraftingAgent.__new__(EmailDraftingAgent)
    with pytest.raises(SecurityError):
        agent._render_template_string(EVIL, {})


def test_flow_sandbox_cannot_reach_python_internals():
    with pytest.raises(SecurityError):
        orchestrator._FLOW_SANDBOX.from_string(EVIL).render(ctx={}, payload={})
    assert orchestrator._FLOW_SANDBOX.from_string("{{ payload.a }}").render(ctx={}, payload={"a": 7}) == "7"
