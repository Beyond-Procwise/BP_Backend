"""A created agent returns the outputs it was defined with, by name.

A created agent's outputs were copied from the built-in class it was made
from — outputs it never produced, because it reasons instead of running that
class. What came back was a paragraph. Now a person names the outputs when
they create the agent, and each run fills them in: a value, the words of the
evidence it rests on, and whether those words were really in what the tools
returned. An output the evidence does not establish is reported as missing,
never filled in.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents import base_agent, named_outputs
from agents.auto_registry import AutoRegistry
from agents.base_agent import AgentContext, AgentStatus
from engines.policy_engine import PolicyEngine
from orchestration import agentnick_control
from orchestration.prompt_engine import PromptEngine
from repositories import agent_run_repo
from services.tool_runtime import ToolCall, ToolRunResult

SPEC = [
    {"name": "supplier", "description": "The supplier that overbilled."},
    {"name": "overbilled_amount", "description": "How much above the PO, with currency."},
    {"name": "contract_end", "description": "When the contract ends."},
]
CALLS = [ToolCall(name="get_corpus_facts", arguments={"query": "overbilling"}, ok=True,
                  result={"rows": [{"supplier": "Acme Ltd", "invoice_total": "GBP 10,812.50",
                                    "po_total": "GBP 10,000.00"}]})]


class _Registry:
    def tool_schemas(self):
        return []


class _Created(base_agent.BaseAgent):
    def run(self, *a, **k):  # pragma: no cover
        raise AssertionError("run() must not be called for a tool-loop node")


def _agent(spec=SPEC):
    nick = SimpleNamespace(
        settings=SimpleNamespace(script_user="tester", extraction_model="m"),
        prompt_engine=PromptEngine(prompt_rows=[]),
        policy_engine=PolicyEngine(policy_rows=[]),
        learning_repository=None, auto_registry=_Registry(), agents={},
    )
    agent = _Created(nick)
    agent.governance_slug = "overcharge-hunter"
    if spec is not None:
        agent.output_spec = spec
    return agent


def _ctx():
    return AgentContext(workflow_id="wf-9", agent_id="overcharge-hunter", user_id="u1",
                        input_data={"deal_id": "D1"})


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.setattr(base_agent.BaseAgent, "_record_context_example",
                        lambda self, ctx, out: None)
    monkeypatch.setattr(agent_run_repo, "record", lambda **kw: _quiet.rows.append(kw))
    _quiet.rows = []
    monkeypatch.setattr(
        agentnick_control, "run_tools",
        lambda task, tools, system, **kw: ToolRunResult(
            answer="Acme Ltd billed GBP 812.50 above its PO.", rounds=2, calls=list(CALLS),
            model="m", system=system),
    )


@pytest.fixture()
def model(monkeypatch):
    """What the model fills in when asked for the named outputs."""
    asked = {}

    def fake_structured(system, user, schema, **kw):
        asked.update(system=system, user=user, schema=schema)
        if isinstance(asked.get("reply"), Exception):
            raise asked["reply"]
        return asked.get("reply") or {
            "supplier": {"found": True, "value": "Acme Ltd", "evidence": '"supplier": "Acme Ltd"'},
            "overbilled_amount": {"found": True, "value": "GBP 812.50",
                                  "evidence": "GBP 10,812.50"},
            "contract_end": {"found": False, "value": "", "evidence": ""},
        }

    monkeypatch.setattr(named_outputs, "structured", fake_structured)
    return asked


def test_a_run_returns_each_named_output(model):
    out = _agent().execute(_ctx(), tool_loop=True)

    assert out.status == AgentStatus.SUCCESS
    assert out.data["outputs"]["supplier"] == {
        "value": "Acme Ltd", "status": "evidence_found", "evidence": '"supplier": "Acme Ltd"'}
    assert out.data["outputs"]["overbilled_amount"]["value"] == "GBP 812.50"
    assert out.data["answer"] == "Acme Ltd billed GBP 812.50 above its PO."


def test_an_output_the_evidence_does_not_establish_is_missing_not_invented(model):
    out = _agent().execute(_ctx(), tool_loop=True)
    assert out.data["outputs"]["contract_end"] == {
        "value": None, "status": "missing", "evidence": None}


def test_a_value_given_for_an_output_marked_not_found_is_not_kept(model):
    model["reply"] = {
        "supplier": {"found": False, "value": "Probably Acme", "evidence": ""},
        "overbilled_amount": {"found": False, "value": "", "evidence": ""},
        "contract_end": {"found": False, "value": "", "evidence": ""},
    }
    out = _agent().execute(_ctx(), tool_loop=True)
    assert out.data["outputs"]["supplier"]["value"] is None


def test_evidence_that_is_not_in_what_the_tools_returned_is_flagged(model):
    model["reply"] = {
        "supplier": {"found": True, "value": "Acme Ltd", "evidence": '"supplier": "Acme Ltd"'},
        "overbilled_amount": {"found": True, "value": "GBP 812.50", "evidence": "x"},
        "contract_end": {"found": True, "value": "31 March 2027",
                         "evidence": "The contract ends on 31 March 2027."},
    }
    out = _agent().execute(_ctx(), tool_loop=True)

    assert out.data["outputs"]["contract_end"]["status"] == "evidence_not_found"
    assert out.data["outputs"]["contract_end"]["value"] == "31 March 2027"   # shown, and flagged
    assert out.data["outputs"]["overbilled_amount"]["status"] == "evidence_not_found"  # too short to be evidence
    assert out.data["outputs"]["supplier"]["status"] == "evidence_found"


def test_evidence_may_come_from_the_inputs_the_workflow_gave_the_agent(model):
    model["reply"] = {
        "supplier": {"found": True, "value": "D1", "evidence": '"deal_id": "D1"'},
        "overbilled_amount": {"found": False, "value": "", "evidence": ""},
        "contract_end": {"found": False, "value": "", "evidence": ""},
    }
    out = _agent().execute(_ctx(), tool_loop=True)
    assert out.data["outputs"]["supplier"]["status"] == "evidence_found"


def test_each_output_is_also_published_by_name_for_the_next_step(model):
    out = _agent().execute(_ctx(), tool_loop=True)
    assert out.data["supplier"] == "Acme Ltd"
    assert out.data["overbilled_amount"] == "GBP 812.50"
    assert "contract_end" not in out.data            # a missing output publishes nothing


def test_the_model_is_asked_for_exactly_the_named_outputs_with_no_unions(model):
    _agent().execute(_ctx(), tool_loop=True)
    schema = model["schema"]

    assert schema["required"] == ["supplier", "overbilled_amount", "contract_end"]
    assert set(schema["properties"]) == {"supplier", "overbilled_amount", "contract_end"}
    assert "oneOf" not in str(schema) and "anyOf" not in str(schema)
    assert "The supplier that overbilled." in model["user"]
    assert "GBP 10,812.50" in model["user"]          # it is shown what the tools returned


def test_when_the_outputs_cannot_be_filled_every_one_is_missing_and_the_run_says_why(model):
    model["reply"] = RuntimeError("ollama unreachable")
    out = _agent().execute(_ctx(), tool_loop=True)

    assert out.status == AgentStatus.SUCCESS          # the answer still stands
    assert {o["status"] for o in out.data["outputs"].values()} == {"missing"}
    assert "ollama unreachable" in out.data["outputs_error"]


def test_an_agent_with_no_named_outputs_is_unchanged(model):
    out = _agent(spec=None).execute(_ctx(), tool_loop=True)
    assert "outputs" not in out.data
    assert model == {}                                # the model was not asked a second time


def test_the_outputs_are_part_of_the_record_of_the_run(model):
    _agent().execute(_ctx(), tool_loop=True)
    assert _quiet.rows[0]["outputs"]["supplier"]["value"] == "Acme Ltd"


def test_a_failed_run_is_not_asked_for_outputs(model, monkeypatch):
    monkeypatch.setattr(
        agentnick_control, "run_tools",
        lambda task, tools, system, **kw: ToolRunResult(answer="", rounds=1, error="boom"),
    )
    out = _agent().execute(_ctx(), tool_loop=True)
    assert out.status == AgentStatus.FAILED
    assert model == {}


# --- naming the outputs -------------------------------------------------------

@pytest.mark.parametrize("bad", [
    [{"name": "Supplier Name"}],                       # not an identifier
    [{"name": "answer"}],                              # the run's own field
    [{"name": "a"}, {"name": "a"}],                    # twice
    [],                                                # an empty list defines nothing
    [{"name": f"f{i}"} for i in range(13)],            # too many to be "a short list"
])
def test_a_bad_list_of_outputs_is_refused(bad):
    with pytest.raises(ValueError):
        named_outputs.checked_spec(bad)


def test_a_good_list_is_kept_in_order_with_its_descriptions():
    assert named_outputs.checked_spec(
        [{"name": "supplier", "description": " The supplier. "}, {"name": "amount"}]
    ) == [{"name": "supplier", "description": "The supplier."},
          {"name": "amount", "description": ""}]


# --- the outputs reach the agent from the catalogue ---------------------------

def test_the_catalogue_outputs_are_stamped_on_the_agent(tmp_path):
    import json

    path = tmp_path / "agent_definitions.json"
    path.write_text(json.dumps([{
        "agentId": 1, "slug": "overcharge-hunter", "agentType": "X",
        "class_path": f"{__name__}._Created", "derived_from": "supplier_ranking",
        "output_spec": SPEC, "outputs": [s["name"] for s in SPEC]}]))
    registry = AutoRegistry.from_json(path)
    registry.set_agent_nick(_agent().agent_nick)

    assert registry.get_agent("overcharge-hunter").output_spec == SPEC
