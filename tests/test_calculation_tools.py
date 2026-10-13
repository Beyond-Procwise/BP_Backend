"""The registered calculations reach an agent two ways, and they are not the same.

A calculation an agent must always have is DECLARED on it: run before it
reasons, from the inputs the run arrived with, and stated in its task. The agent
is not asked. A calculation nobody could schedule is a TOOL the agent may call.
Either way the figure comes from the register, never from the model — and a
figure in an answer that came from nowhere is reported.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from agents import base_agent
from agents.auto_registry import AutoRegistry
from agents.base_agent import AgentContext, AgentStatus
from engines.policy_engine import PolicyEngine
from orchestration import agentnick_control
from orchestration.prompt_engine import PromptEngine
from repositories import agent_run_repo
from services import calculation_tools as CT
from services.tool_runtime import ToolCall, ToolRunResult
from src.services.formulas import evaluate

TERMS = "supplier.payment_terms_score"
PCT = "deal.pct_change"
A_SET = "supplier.criterion_normalisation"


def _tools():
    nick = SimpleNamespace(agents={}, auto_registry=None)
    return {t.name: t for t in agentnick_control.build_tools(nick, workflow_id="wf-1")}


# ---------------------------------------------------------------------------
# The tools
# ---------------------------------------------------------------------------
def test_the_calculations_are_on_offer_as_two_tools_not_one_per_formula():
    names = set(_tools())
    assert {"list_calculations", "run_calculation"} <= names
    assert not any(n.startswith(("supplier.", "deal.")) for n in names)


def test_the_list_is_the_register_not_a_list_in_the_tool():
    from src.services.formulas import REGISTRY

    out = _tools()["list_calculations"].handler()

    assert out["count"] == len(REGISTRY)
    assert {c["name"] for c in out["calculations"]} == set(REGISTRY)
    by_name = {c["name"]: c for c in out["calculations"]}
    assert by_name[PCT]["inputs"] == ["new", "base"]   # the names, so none has to be guessed


def test_asking_for_one_calculation_says_what_it_needs():
    out = _tools()["list_calculations"].handler(name=TERMS)

    inputs = {i["name"]: i for i in out["calculations"][0]["inputs"]}
    assert set(inputs) == {"payment_terms_days", "min_days", "max_days"}
    assert inputs["payment_terms_days"]["unit"] == "days"


def test_a_family_or_a_name_that_does_not_exist_is_said_not_guessed():
    out = _tools()["list_calculations"].handler(family="astrology")
    assert out["found"] is False and "supplier" in out["families"]


def test_running_a_calculation_returns_the_registers_own_value():
    out = _tools()["run_calculation"].handler(name=TERMS, inputs={"payment_terms_days": 30})

    assert out["assessed"] is True
    assert out["value"] == evaluate(TERMS, {"payment_terms_days": 30}, audit=False).value


def test_inputs_an_agent_typed_in_are_marked_as_asserted():
    out = _tools()["run_calculation"].handler(name=TERMS, inputs={"payment_terms_days": 30})
    assert out["confidence"] == "asserted"


def test_a_missing_input_is_not_assessed_and_is_never_a_zero():
    out = _tools()["run_calculation"].handler(name=PCT, inputs={"new": 110})

    assert out["assessed"] is False
    assert "value" not in out
    assert "base" in out["why"]


def test_a_wrongly_named_input_is_refused_with_the_names_to_use():
    out = _tools()["run_calculation"].handler(name=PCT, inputs={"baseline": 100, "current": 110})

    assert out["assessed"] is False
    assert [i["name"] for i in out["expects"]] == ["new", "base"]
    assert "again" in out["retry"]


def test_an_input_outside_its_declared_range_is_refused():
    out = _tools()["run_calculation"].handler(name=TERMS, inputs={"payment_terms_days": -5})
    assert out["assessed"] is False and "below the declared minimum" in out["why"]


def test_a_value_computed_without_an_optional_input_says_which():
    out = _tools()["run_calculation"].handler(name=TERMS, inputs={"payment_terms_days": 30})
    assert out["not_given"] == ["min_days", "max_days"]


def test_inputs_sent_as_a_json_string_are_read_and_a_broken_one_is_refused():
    run = _tools()["run_calculation"].handler
    assert run(name=TERMS, inputs='{"payment_terms_days": 30}')["assessed"] is True
    assert run(name=TERMS, inputs="{payment_terms_days: 30")["assessed"] is False


def test_an_unknown_calculation_is_refused_with_where_to_look():
    out = _tools()["run_calculation"].handler(name="supplier.vibes", inputs={})
    assert out["assessed"] is False and "list_calculations" in out["hint"]


def test_a_set_calculation_needs_its_rows():
    out = _tools()["run_calculation"].handler(name=A_SET, inputs={})
    assert out["assessed"] is False and "rows" in out["why"]


# ---------------------------------------------------------------------------
# Declared on an agent: always run, never chosen
# ---------------------------------------------------------------------------
def test_only_a_registered_scalar_calculation_can_be_declared():
    assert CT.checked_required([TERMS, TERMS, " "]) == [TERMS]
    with pytest.raises(ValueError, match="supplier.vibes"):
        CT.checked_required(["supplier.vibes"])
    with pytest.raises(ValueError, match="whole set"):
        CT.checked_required([A_SET])


def test_a_declared_calculation_reads_its_inputs_from_the_run_by_name():
    [out] = CT.run_required([TERMS], {"payment_terms_days": 30, "deal_id": "D1"})

    assert out["assessed"] is True
    assert out["value"] == evaluate(TERMS, {"payment_terms_days": 30}, audit=False).value
    assert out["inputs"] == {"payment_terms_days": 30}     # and nothing it did not declare


def test_a_declared_calculation_without_its_inputs_is_reported_not_skipped():
    [out] = CT.run_required([PCT], {"deal_id": "D1", "new": 110})

    assert out["assessed"] is False and "base" in out["why"]
    assert "NOT ASSESSED" in CT.render_required([out])


def test_a_calculation_declared_then_unregistered_is_said():
    [out] = CT.run_required(["supplier.vibes"], {})
    assert out["assessed"] is False and "no longer registered" in out["why"]


def test_the_catalogue_stamps_the_declared_calculations_on_the_agent(tmp_path):
    import json

    path = tmp_path / "defs.json"
    path.write_text(json.dumps({"agents": [
        {"slug": "terms-checker", "class_path": "x.Y", "calculations": [TERMS]},
        {"slug": "plain", "class_path": "x.Y"},
    ]}))
    registry = AutoRegistry.from_json(str(path))

    assert registry._contracts["terms-checker"].calculations == [TERMS]
    assert registry._contracts["plain"].calculations is None


class _Registry:
    def tool_schemas(self):
        return []


class _Created(base_agent.BaseAgent):
    def run(self, *a, **k):  # pragma: no cover
        raise AssertionError("run() must not be called for a tool-loop node")


def _agent(calculations=None):
    nick = SimpleNamespace(
        settings=SimpleNamespace(script_user="tester", extraction_model="m"),
        prompt_engine=PromptEngine(prompt_rows=[]),
        policy_engine=PolicyEngine(policy_rows=[]),
        learning_repository=None, auto_registry=_Registry(), agents={},
    )
    agent = _Created(nick)
    agent.governance_slug = "terms-checker"
    if calculations is not None:
        agent.required_calculations = calculations
    return agent


def _ctx(**inputs):
    return AgentContext(workflow_id="wf-9", agent_id="terms-checker", user_id="u1",
                        input_data={"deal_id": "D1", **inputs})


@pytest.fixture()
def loop(monkeypatch):
    """The model's side of a run: what it was asked, and what it answers."""
    seen = SimpleNamespace(task=None, rows=[], answer="Noted.", calls=[])
    monkeypatch.setattr(base_agent.BaseAgent, "_record_context_example",
                        lambda self, ctx, out: None)
    monkeypatch.setattr(agent_run_repo, "record", lambda **kw: seen.rows.append(kw))

    def fake_run_tools(task, tools, system, **kw):
        seen.task = task
        return ToolRunResult(answer=seen.answer, rounds=1, calls=list(seen.calls),
                             model="m", system=system)

    monkeypatch.setattr(agentnick_control, "run_tools", fake_run_tools)
    return seen


def test_a_declared_calculation_is_run_before_the_agent_reasons_and_stated_in_its_task(loop):
    score = evaluate(TERMS, {"payment_terms_days": 30}, audit=False).value

    out = _agent([TERMS]).execute(_ctx(payment_terms_days=30), tool_loop=True)

    assert out.status == AgentStatus.SUCCESS
    assert f"- {TERMS}" in loop.task and str(score) in loop.task
    assert out.data["calculations"][0]["value"] == score
    assert "run_calculation" not in out.data["tools_used"]     # it was never asked


def test_the_task_that_is_recorded_is_the_task_the_agent_was_given(loop):
    _agent([TERMS]).execute(_ctx(payment_terms_days=30), tool_loop=True)
    assert loop.rows[0]["task"] == loop.task


def test_a_declared_calculation_that_cannot_run_tells_the_agent_so(loop):
    out = _agent([PCT]).execute(_ctx(), tool_loop=True)

    assert out.data["calculations"][0]["assessed"] is False
    assert f"- {PCT}: NOT ASSESSED" in loop.task


def test_an_agent_with_nothing_declared_is_given_the_task_it_always_was(loop):
    out = _agent().execute(_ctx(payment_terms_days=30), tool_loop=True)

    assert "Calculations already run" not in loop.task
    assert "calculations" not in out.data


# ---------------------------------------------------------------------------
# A figure in the answer that came from nowhere
# ---------------------------------------------------------------------------
def test_a_figure_read_from_what_the_run_was_given_is_traced():
    sources = ["Deal D1", {"rows": [{"invoice_total": 10812.5, "share": 0.125}]}]

    assert CT.untraced_figures("Billed GBP 10,812.50, which is 12.5% of spend.", sources) == []


def test_a_figure_found_in_nothing_the_run_was_given_is_reported_once():
    out = CT.untraced_figures("The saving is £4,300.00, i.e. £4,300.00 a year.",
                              [{"saving": 4200.0}])
    assert out == ["4,300.00"]


def test_counts_and_ordinals_are_not_figures():
    assert CT.untraced_figures("Round 2: 3 suppliers replied.", ["nothing"]) == []


def test_a_run_reports_the_figures_its_answer_did_not_get_from_anywhere(loop):
    loop.answer = "Terms score well. The saving would be GBP 9,999.00."
    out = _agent([TERMS]).execute(_ctx(payment_terms_days=30), tool_loop=True)

    assert out.status == AgentStatus.SUCCESS                 # reported, not enforced
    assert out.data["untraced_figures"] == ["9,999.00"]


def test_a_figure_a_tool_returned_is_not_reported(loop):
    loop.calls = [ToolCall(name="get_corpus_facts", arguments={}, ok=True,
                           result={"saving_gbp": 9999.0})]
    loop.answer = "The saving would be GBP 9,999.00."
    out = _agent().execute(_ctx(), tool_loop=True)

    assert out.data["untraced_figures"] == []
