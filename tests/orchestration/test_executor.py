"""The executor loop with every dependency faked: it asks the model for a proposal, asks the policy
gate before each proposed tool call, runs only what the gate allows, and releases an answer only
through the output check. It owns no model, no data store and no policy of its own."""
import ast
import json
import logging
import re
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from orchestration import executor as ex


@dataclass
class Call:
    name: str
    arguments: Dict[str, Any]
    ok: bool
    result: Any = None
    error: Optional[str] = None
    duration_ms: int = 0


@dataclass
class Result:
    answer: str = ""
    calls: List[Call] = field(default_factory=list)
    rounds: int = 0
    error: Optional[str] = None
    model: str = ""
    system: str = ""
    tools_offered: List[str] = field(default_factory=list)
    safety_retry: bool = False
    safety_blocked: bool = False


@dataclass
class Precedent:
    note: str


class Tool:
    def __init__(self, name, handler):
        self.name, self.handler = name, handler

    def schema(self):
        return {"type": "function", "function": {"name": self.name}}


def tc(name, **args):
    return {"function": {"name": name, "arguments": args}}


class Script:
    """Scripted model turns, and a log of what the loop asked of each dependency."""

    def __init__(self, turns, gate=None, safety=None):
        self.turns, self.gate, self.safety_fn = list(turns), gate, safety
        self.events: List[Any] = []

    def wiring(self):
        def turn(messages, schemas, model, timeout):
            self.events.append(("propose", len(messages)))
            return self.turns.pop(0)

        def stream(messages, schemas, model, timeout):
            self.events.append(("propose", len(messages)))
            t = self.turns.pop(0)
            return SimpleNamespace(content=t.get("content", ""), tool_calls=t.get("tool_calls") or [])

        def policy(name, args, **kw):
            self.events.append(("gate", name))
            return self.gate(name, args, **kw) if self.gate else None

        def safety(content, messages, schemas, model, timeout, result):
            self.events.append(("safety", content))
            return self.safety_fn(content) if self.safety_fn else content

        return ex.Wiring(
            ToolCall=Call, ToolRunResult=Result, RanOnPrecedent=Precedent,
            propose_turn=turn, propose_stream=stream, policy_check=policy, safety_gate=safety,
            coerce_args=lambda raw: raw if isinstance(raw, dict) else {},
            render_result=lambda v: json.dumps(v, default=str),
            with_note=lambda c, allowed: c if allowed is None else f"{c}\n{allowed.note}",
            log=logging.getLogger("test"))


def tool(name, fn=None, record=None):
    def handler(**kw):
        if record is not None:
            record.append((name, kw))
        return fn(**kw) if fn else {"ok": name}
    return Tool(name, handler)


def run(script, tools, **kw):
    return ex.run(script.wiring(), "task", tools, "system", model="m", max_rounds=kw.pop("max_rounds", 6),
                  timeout_s=5, **kw)


def run_stream(script, tools, **kw):
    return ex.run_stream(script.wiring(), "task", tools, "system", model="m", max_rounds=kw.pop("max_rounds", 6),
                         timeout_s=5, **kw)


FINAL = {"role": "assistant", "content": "Done."}
MODES = [("whole", run), ("stream", run_stream)]


@pytest.mark.parametrize("mode,runner", MODES)
def test_the_gate_is_asked_before_the_handler_runs(mode, runner):
    ran = []
    s = Script([{"role": "assistant", "content": "", "tool_calls": [tc("a", x=1)]}, FINAL])
    res = runner(s, [tool("a", record=ran)])
    assert s.events[:3] == [("propose", 2), ("gate", "a"), ("propose", 4)] and ran == [("a", {"x": 1})]
    assert res.answer == "Done." and [c.name for c in res.calls] == ["a"] and res.calls[0].ok


@pytest.mark.parametrize("mode,runner", MODES)
def test_a_refused_call_never_reaches_its_handler_and_the_model_is_told_why(mode, runner):
    ran = []
    refusal = {"result": "blocked", "reason": "not allowed"}
    s = Script([{"role": "assistant", "content": "", "tool_calls": [tc("a")]}, FINAL], gate=lambda n, a, **k: refusal)
    res = runner(s, [tool("a", record=ran)])
    assert ran == [] and res.calls[0].ok is False and res.calls[0].result == refusal


@pytest.mark.parametrize("mode,runner", MODES)
def test_an_unknown_tool_is_reported_and_the_gate_is_not_asked(mode, runner):
    s = Script([{"role": "assistant", "content": "", "tool_calls": [tc("nope")]}, FINAL])
    res = runner(s, [tool("a")])
    assert res.calls[0].error == "unknown tool 'nope'" and ("gate", "nope") not in s.events


@pytest.mark.parametrize("mode,runner", MODES)
def test_a_precedent_allow_runs_the_tool_and_tells_the_model(mode, runner):
    ran, seen = [], []
    s = Script([{"role": "assistant", "content": "", "tool_calls": [tc("a")]}, FINAL],
               gate=lambda n, a, **k: Precedent("ran on precedent"))
    orig = s.wiring

    def wiring():
        w = orig()
        return ex.Wiring(**{**w.__dict__, "propose_turn": lambda m, *r: (seen.append(list(m)), w.propose_turn(m, *r))[1],
                            "propose_stream": lambda m, *r: (seen.append(list(m)), w.propose_stream(m, *r))[1]})
    s.wiring = wiring
    runner(s, [tool("a", record=ran)])
    assert ran and "ran on precedent" in seen[-1][-1]["content"]


@pytest.mark.parametrize("mode,runner", MODES)
def test_the_gate_sees_who_and_why(mode, runner):
    got = {}
    s = Script([{"role": "assistant", "content": "because", "tool_calls": [tc("a")]}, FINAL],
               gate=lambda n, a, **k: got.update(k))
    runner(s, [tool("a")], agent="ag", workflow_id="wf", user_id="u")
    assert got == {"agent": "ag", "reason": "because", "workflow_id": "wf", "user_id": "u"}


@pytest.mark.parametrize("mode,runner", MODES)
def test_the_answer_is_released_only_through_the_output_check(mode, runner):
    s = Script([{"role": "assistant", "content": "raw draft"}], safety=lambda c: "checked")
    res = runner(s, [tool("a")])
    assert res.answer == "checked" and ("safety", "raw draft") in s.events


@pytest.mark.parametrize("mode,runner", MODES)
def test_running_out_of_rounds_is_an_error_not_a_hang(mode, runner):
    s = Script([{"role": "assistant", "content": "", "tool_calls": [tc("a")]}] * 3)
    res = runner(s, [tool("a")], max_rounds=3)
    # three tool rounds, then one closing turn the script cannot answer: still an error, never a hang
    assert res.rounds == 4 and "max_rounds (3) exhausted" in res.error and len(res.calls) == 3


@pytest.mark.parametrize("mode,runner", MODES)
def test_a_handler_that_raises_is_recorded_not_propagated(mode, runner):
    def boom(**k):
        raise RuntimeError("db down")
    s = Script([{"role": "assistant", "content": "", "tool_calls": [tc("a")]}, FINAL])
    res = runner(s, [tool("a", boom)])
    assert res.calls[0].ok is False and res.calls[0].error == "db down" and res.answer == "Done."


def test_bad_arguments_get_their_own_wording_in_whole_turn_mode_only():
    def strict(x):
        return x
    turns = [{"role": "assistant", "content": "", "tool_calls": [tc("a", y=1)]}, FINAL]
    whole = run(Script(turns), [tool("a", strict)])
    streamed = run_stream(Script(turns), [tool("a", strict)])
    assert whole.calls[0].error.startswith("invalid arguments:")
    assert not streamed.calls[0].error.startswith("invalid arguments:")


def test_a_transport_failure_comes_back_as_an_error_with_what_was_gathered():
    def boom(*a):
        raise RuntimeError("ollama down")
    s = Script([])
    w = s.wiring()
    w = ex.Wiring(**{**w.__dict__, "propose_turn": boom, "propose_stream": boom})
    for fn in (ex.run, ex.run_stream):
        res = fn(w, "t", [tool("a")], "s", model="m", max_rounds=2, timeout_s=1)
        assert res.error == "ollama down" and res.answer == ""


def test_whole_turn_mode_nudges_once_when_no_tool_was_used_and_grounding_is_required():
    s = Script([{"role": "assistant", "content": "guess"}, {"role": "assistant", "content": "guess again"}])
    res = run(s, [tool("a")], require_tool_use=True)
    assert res.answer == "guess again" and [e for e in s.events if e[0] == "propose"] == [("propose", 2), ("propose", 4)]
    assert res.rounds == 2


def test_whole_turn_mode_salvages_the_last_thing_said_when_rounds_run_out():
    s = Script([{"role": "assistant", "content": "thinking", "tool_calls": [tc("a")]}] * 2)
    res = run(s, [tool("a")], max_rounds=2)
    assert res.answer == "thinking" and "exhausted" in res.error


def test_stream_mode_reports_each_call_and_releases_the_answer_through_on_delta():
    told, delta = [], []
    s = Script([{"role": "assistant", "content": "", "tool_calls": [tc("a")]}, FINAL])
    run_stream(s, [tool("a")], on_tool=told.append, on_delta=delta.append)
    assert [c.name for c in told] == ["a"] and delta == ["Done."]


def test_the_result_records_what_the_run_was_given():
    s = Script([FINAL])
    res = run(s, [tool("a"), tool("b")])
    assert (res.model, res.system, res.tools_offered) == ("m", "system", ["a", "b"])


def test_the_executor_imports_no_model_wire_data_store_or_policy_engine():
    tree = ast.parse(open(ex.__file__, encoding="utf-8").read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    bad = [m for m in imported if re.search(r"tool_runtime|agentnick_driver|egress|ollama|agent_policy|governance|"
                                            r"\bdb\b|psycopg|sqlalchemy|requests", m)]
    assert not bad, bad


ROOT = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.dirname(
    __import__("os").path.abspath(__file__))))
# Files that still run tool calls a model proposed, with their own loop and NO policy gate. Both are
# moved onto the executor in a later step. The list may only shrink: a new loop outside the executor
# fails the first test; a file that stops running proposed calls must be removed here.
OWN_LOOP_BASELINE = {
    "src/services/governance_tools/governed_reasoning.py",
    "src/services/supplier_enrichment/research.py",
}


def _files_running_proposed_tool_calls():
    import os
    found = set()
    for base, _d, files in os.walk(os.path.join(ROOT, "src")):
        if "__pycache__" in base:
            continue
        for name in files:
            rel = os.path.relpath(os.path.join(base, name), ROOT).replace(os.sep, "/")
            if not name.endswith(".py") or rel in ("src/orchestration/executor.py", "src/services/agentnick_driver.py"):
                continue
            if "tool_calls" in open(os.path.join(base, name), encoding="utf-8", errors="ignore").read():
                found.add(rel)
    return found


def test_no_new_loop_runs_a_models_tool_calls_outside_the_executor():
    new = sorted(_files_running_proposed_tool_calls() - OWN_LOOP_BASELINE)
    assert not new, "These run tool calls a model proposed. Use orchestration/executor.py:\n  " + "\n  ".join(new)


def test_the_own_loop_baseline_only_lists_files_that_still_have_one():
    gone = sorted(OWN_LOOP_BASELINE - _files_running_proposed_tool_calls())
    assert not gone, "Remove from OWN_LOOP_BASELINE (they no longer have their own loop):\n  " + "\n  ".join(gone)


def test_tool_runtime_no_longer_runs_tool_handlers_itself():
    import os
    src = open(os.path.join(ROOT, "src", "services", "tool_runtime.py"), encoding="utf-8").read()
    assert ".handler(" not in src and "range(max_rounds)" not in src


# ---- the verify step -------------------------------------------------------------------------------

def _with_verify(script, fn):
    w = script.wiring()
    return ex.Wiring(**{**w.__dict__, "verify": fn})


class V:
    def __init__(self, action, answer="", untraced=(), retry_message=None):
        self.action, self.answer, self.untraced, self.retry_message = action, answer, list(untraced), retry_message


@pytest.mark.parametrize("fn_name", ["run", "run_stream"])
def test_with_no_verifier_nothing_changes(fn_name):
    s = Script([{"role": "assistant", "content": "draft"}])
    assert getattr(ex, fn_name)(s.wiring(), "t", [], "s", model="m", max_rounds=3, timeout_s=1).answer == "draft"


@pytest.mark.parametrize("fn_name", ["run", "run_stream"])
def test_a_verifier_can_replace_the_answer_after_the_output_check(fn_name):
    s = Script([{"role": "assistant", "content": "model wording"}], safety=lambda c: "checked wording")
    seen = {}

    def verify(answer, result, retried):
        seen["answer"], seen["retried"] = answer, retried
        return V("replace", "authoritative text")

    res = getattr(ex, fn_name)(_with_verify(s, verify), "t", [], "s", model="m", max_rounds=3, timeout_s=1)
    assert res.answer == "authoritative text" and seen == {"answer": "checked wording", "retried": False}


@pytest.mark.parametrize("fn_name", ["run", "run_stream"])
def test_a_retry_sends_the_model_back_once_with_the_instruction_and_then_releases_the_second_answer(fn_name):
    s = Script([{"role": "assistant", "content": "first, with 40,000"}, {"role": "assistant", "content": "second"}])
    asked = []

    def verify(answer, result, retried):
        asked.append((answer, retried))
        return V("retry", answer, ["40,000"], "remove 40,000") if not retried else V("keep", answer)

    sent = []
    w = _with_verify(s, verify)
    orig = w.propose_turn, w.propose_stream
    w = ex.Wiring(**{**w.__dict__,
                     "propose_turn": lambda m, *r: (sent.append(list(m)), orig[0](m, *r))[1],
                     "propose_stream": lambda m, *r: (sent.append(list(m)), orig[1](m, *r))[1]})
    res = getattr(ex, fn_name)(w, "t", [], "s", model="m", max_rounds=4, timeout_s=1)
    assert res.answer == "second" and asked == [("first, with 40,000", False), ("second", True)]
    assert sent[1][-1] == {"role": "user", "content": "remove 40,000"}
    assert sent[1][-2] == {"role": "assistant", "content": "first, with 40,000"}


@pytest.mark.parametrize("fn_name", ["run", "run_stream"])
def test_the_model_is_sent_back_at_most_once(fn_name):
    s = Script([{"role": "assistant", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "assistant", "content": "c"}])
    verify = lambda answer, result, retried: V("retry", answer, ["1"], "again") if not retried else V("caution", answer + " (unverified)", ["1"])
    res = getattr(ex, fn_name)(_with_verify(s, verify), "t", [], "s", model="m", max_rounds=5, timeout_s=1)
    assert res.answer == "b (unverified)" and res.rounds == 2


@pytest.mark.parametrize("fn_name", ["run", "run_stream"])
def test_what_the_verifier_found_is_recorded_on_the_result(fn_name):
    s = Script([{"role": "assistant", "content": "x"}])
    res = getattr(ex, fn_name)(_with_verify(s, lambda a, r, t: V("caution", a, ["7"])), "t", [], "s", model="m",
                               max_rounds=3, timeout_s=1)
    assert res.verification == {"action": "caution", "untraced": ["7"]}


def test_the_stream_still_reports_the_final_answer_through_on_delta_after_verification():
    s = Script([{"role": "assistant", "content": "model wording"}])
    got = []
    ex.run_stream(_with_verify(s, lambda a, r, t: V("replace", "authoritative")), "t", [], "s", model="m",
                  max_rounds=3, timeout_s=1, on_delta=got.append)
    assert got == ["authoritative"]


@pytest.mark.parametrize("fn_name", ["run", "run_stream"])
def test_even_a_verifier_that_always_asks_for_a_retry_cannot_loop_the_model(fn_name):
    s = Script([{"role": "assistant", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "assistant", "content": "c"}])
    always = lambda answer, result, retried: V("retry", answer, ["1"], "again")
    res = getattr(ex, fn_name)(_with_verify(s, always), "t", [], "s", model="m", max_rounds=6, timeout_s=1)
    assert res.rounds == 2 and res.answer == "b"


# ---- step 3b: the figures a model passes into a tool ---------------------------------------------

def _with_args_check(script, fn):
    w = script.wiring()
    return ex.Wiring(**{**w.__dict__, "check_arguments": fn})


@pytest.mark.parametrize("fn_name", ["run", "run_stream"])
def test_a_call_with_an_unsourced_figure_is_refused_before_the_gate_and_the_handler(fn_name):
    ran = []
    refusal = {"result": "blocked", "reasonCode": "figure_not_sourced", "reason": "x was not run"}
    s = Script([{"role": "assistant", "content": "", "tool_calls": [tc("a", rate=0.08)]}, FINAL])
    res = getattr(ex, fn_name)(_with_args_check(s, lambda name, args, result: refusal), "t", [tool("a", record=ran)],
                               "s", model="m", max_rounds=4, timeout_s=1)
    assert ran == [] and ("gate", "a") not in s.events
    assert res.calls[0].ok is False and res.calls[0].result == refusal and res.answer == "Done."


@pytest.mark.parametrize("fn_name", ["run", "run_stream"])
def test_a_call_that_passes_the_check_still_goes_to_the_gate_and_runs(fn_name):
    ran = []
    s = Script([{"role": "assistant", "content": "", "tool_calls": [tc("a", x=1)]}, FINAL])
    res = getattr(ex, fn_name)(_with_args_check(s, lambda n, a, r: None), "t", [tool("a", record=ran)], "s",
                               model="m", max_rounds=4, timeout_s=1)
    assert ran == [("a", {"x": 1})] and ("gate", "a") in s.events and res.calls[0].ok


@pytest.mark.parametrize("fn_name", ["run", "run_stream"])
def test_the_check_sees_the_arguments_and_the_calls_made_so_far(fn_name):
    seen = []
    s = Script([{"role": "assistant", "content": "", "tool_calls": [tc("a", n=1)]},
                {"role": "assistant", "content": "", "tool_calls": [tc("b", n=2)]}, FINAL])

    def check(name, args, result):
        seen.append((name, dict(args), [c.name for c in result.calls]))

    getattr(ex, fn_name)(_with_args_check(s, check), "t", [tool("a"), tool("b")], "s", model="m", max_rounds=5, timeout_s=1)
    assert seen == [("a", {"n": 1}, []), ("b", {"n": 2}, ["a"])]


@pytest.mark.parametrize("fn_name", ["run", "run_stream"])
def test_an_unknown_tool_is_never_run_through_the_check(fn_name):
    seen = []
    s = Script([{"role": "assistant", "content": "", "tool_calls": [tc("nope", n=1)]}, FINAL])
    getattr(ex, fn_name)(_with_args_check(s, lambda *a: seen.append(a)), "t", [tool("a")], "s", model="m",
                         max_rounds=3, timeout_s=1)
    assert seen == []


@pytest.mark.parametrize("fn_name", ["run", "run_stream"])
def test_with_no_argument_check_nothing_changes(fn_name):
    ran = []
    s = Script([{"role": "assistant", "content": "", "tool_calls": [tc("a", rate=0.08)]}, FINAL])
    getattr(ex, fn_name)(s.wiring(), "t", [tool("a", record=ran)], "s", model="m", max_rounds=4, timeout_s=1)
    assert ran == [("a", {"rate": 0.08})]


@pytest.mark.parametrize("mode,runner", MODES)
def test_when_rounds_run_out_the_model_gets_one_closing_turn_without_tools(mode, runner):
    """Live, 2026-10-22: multi-step playbooks spent every round on tools and returned nothing."""
    seen = {}
    s = Script([{"role": "assistant", "content": "", "tool_calls": [tc("a")]}] * 2 + [{"role": "assistant", "content": "From what I gathered: ok."}])
    w = s.wiring()
    inner = w.propose_turn

    def turn(messages, schemas, model, timeout):
        seen.setdefault("schemas", []).append(len(schemas))
        seen["last_user"] = [m["content"] for m in messages if m["role"] == "user"][-1]
        return inner(messages, schemas, model, timeout)
    w = ex.Wiring(**{**w.__dict__, "propose_turn": turn})
    fn = ex.run if mode == "whole" else ex.run_stream
    res = fn(w, "task", [tool("a")], "system", model="m", max_rounds=2, timeout_s=5)
    assert res.answer == "From what I gathered: ok." and res.error is None and res.rounds == 3 and len(res.calls) == 2
    assert res.closed_without_tools is True
    assert seen["schemas"][-1] == 0 and "No more tool calls" in seen["last_user"]
    assert ("safety", "From what I gathered: ok.") in s.events
