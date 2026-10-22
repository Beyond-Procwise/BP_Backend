"""1.20j: an outbound call made while an agent reasons is audited under that agent's name.

The model client, the FX lookup and every other caller inside a reasoning loop never knew which agent
they served, so the audit said "unknown" for every model call. ``egress.calling(agent)`` sets the
calling agent for everything that happens inside it (a context variable, so it follows the request
through awaits and threads started from it), and ``request``/``observe`` use it when no explicit agent
is given. The tool loop (``tool_runtime.run_tools`` / ``run_tools_stream``) enters it for the agent it
runs, so every call the model's turn makes is named. An explicit ``agent=`` still wins.
"""
from __future__ import annotations

import pytest

from src.services import egress, egress_log
from src.services import egress_policy as ep
from src.services.egress import Purpose


class _Resp:
    status_code, headers, text = 200, {}, ""


@pytest.fixture
def events(monkeypatch):
    out = []
    monkeypatch.setattr(egress.requests, "request", lambda m, u, **kw: _Resp())
    monkeypatch.setattr(egress, "_resolve", lambda host: ["127.0.0.1"] if host == "localhost" else ["93.184.216.34"])
    monkeypatch.setattr(egress_log, "record", lambda **kw: out.append(kw))
    monkeypatch.setattr(ep, "_load_row", lambda: None)
    ep.invalidate()
    yield out
    ep.invalidate(forget_last_good=True)


def test_a_call_inside_calling_is_audited_under_that_agent(events):
    with egress.calling("negotiation"):
        egress.post("http://localhost:11434/api/generate", purpose=Purpose.MODEL_INFERENCE,
                    require_global=False, json={"prompt": "x"})
    assert events[-1]["agent"] == "negotiation"
    egress.post("http://localhost:11434/api/generate", purpose=Purpose.MODEL_INFERENCE,
                require_global=False, json={"prompt": "x"})
    assert events[-1]["agent"] == "unknown", "outside the block nothing is assumed"


def test_an_explicit_agent_wins_and_blocks_nest(events):
    with egress.calling("outer"):
        with egress.calling("inner"):
            egress.get("https://example.com/", purpose=Purpose.FX_RATES)
            assert events[-1]["agent"] == "inner"
            egress.get("https://example.com/", purpose=Purpose.FX_RATES, agent="explicit")
            assert events[-1]["agent"] == "explicit"
        egress.get("https://example.com/", purpose=Purpose.FX_RATES)
        assert events[-1]["agent"] == "outer"
    assert egress.current_agent() is None


def test_observe_uses_the_calling_agent_too(events):
    with egress.calling("supplier_research_run"):
        with egress.observe(Purpose.SUPPLIER_RESEARCH, "html.duckduckgo.com", payload="q"):
            pass
    assert events[-1]["agent"] == "supplier_research_run"


def test_an_exception_inside_calling_restores_the_previous_agent(events):
    with pytest.raises(ValueError):
        with egress.calling("boom"):
            raise ValueError("x")
    assert egress.current_agent() is None


def test_the_tool_loop_runs_under_the_agents_name(monkeypatch):
    """run_tools(agent=...) enters egress.calling for the whole loop: whatever the loop does, the
    audit names the agent. The loop body is stubbed; what is asserted is the context it runs in."""
    from src.services import tool_runtime
    from orchestration import executor
    seen = {}

    def fake_run(*a, **k):
        seen["agent"] = egress.current_agent()
        return "ran"

    def fake_run_stream(*a, **k):
        # The real executor.run_stream is a plain function returning a ToolRunResult (it streams
        # through callbacks); it is NOT a generator. The wrapper must return its value as-is.
        seen["stream_agent"] = egress.current_agent()
        return "streamed-result"
    monkeypatch.setattr(executor, "run", fake_run)
    monkeypatch.setattr(executor, "run_stream", fake_run_stream)
    assert tool_runtime.run_tools("task", [], "system", agent="quote_comparison") == "ran"
    assert seen["agent"] == "quote_comparison"
    assert tool_runtime.run_tools_stream("task", [], "system", agent="negotiation") == "streamed-result"
    assert seen["stream_agent"] == "negotiation"
    assert egress.current_agent() is None


def test_the_research_loop_runs_under_its_own_name(monkeypatch):
    from src.services.supplier_enrichment import research as R
    seen = {}
    monkeypatch.setattr(R, "_run_loop_inner", lambda name: seen.update(agent=egress.current_agent()) or ("", {}))
    assert R._run_loop("Borcelle") == ("", {})
    assert seen["agent"] == "supplier_research"
