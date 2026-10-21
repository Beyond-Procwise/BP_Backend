"""One bounded tool-calling loop for AgentNick.

There were three hand-rolled Ollama tool loops in this codebase — one in
`governance_tools/governed_reasoning.py` (governance tools), one in
`supplier_enrichment/research.py` (web tools), and a fake one behind
`/stream/plan` that never called an agent at all. Each re-implemented the same
mechanics (round cap, tool dispatch, argument coercion, error handling) and each
was reachable only from a single API router. Meanwhile `BaseAgent` had no tool
support whatsoever, so no agent in the registry could call a tool, and AgentNick
— the thing nominally in charge — had no `run()` at all.

This is that loop, extracted once. It is deliberately dumb: it does not know what
a policy or an agent is. Callers supply `Tool`s; the runtime calls them and keeps
a record. The one exception: immediately before a handler runs, the agent-policy gate
(services.agent_policy.gate) is asked whether the call may run; a refused call never reaches
its handler and the model is told why instead.

The record is the point. Every tool call and every result is captured on the
returned `ToolRunResult.calls`, so an answer or a decision can be traced back to
the facts that produced it instead of being taken on trust.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
from src.services.governed_limits import limit as _governed_limit
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.services import agentnick_driver as _driver
from src.services import egress

log = logging.getLogger(__name__)

# The wire to the model, its default model and its timeout live in the driver (the one boundary to
# AgentNick). These names stay so that modules which import them from here keep working.
_OLLAMA_CHAT = _driver.OLLAMA_CHAT
_DEFAULT_MODEL = _driver.DEFAULT_MODEL
def _DEFAULT_MAX_ROUNDS() -> int:
    """AgentReachPolicy (P9): how far one agent may go before it must answer."""
    return _governed_limit("agent_reach", "tool_runtime_max_rounds",
                           env="TOOL_RUNTIME_MAX_ROUNDS", cast=int)
_DEFAULT_TIMEOUT_S = _driver.DEFAULT_TIMEOUT_S

# Results are fed back to the model as text. A tool that returns a huge blob
# (e.g. a full document) would blow the context window and evict the task itself,
# so results are truncated. The cap is generous enough for a policy body or a
# page of corpus facts.
_MAX_RESULT_CHARS = int(os.getenv("TOOL_RUNTIME_MAX_RESULT_CHARS", "6000"))


@dataclass
class Tool:
    """A callable the model may invoke, plus the schema it is advertised under."""

    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema for the arguments object
    handler: Callable[..., Any]

    def schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolCall:
    """One invocation, and what came back. This is the audit record."""

    name: str
    arguments: Dict[str, Any]
    ok: bool
    result: Any = None
    error: Optional[str] = None
    duration_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.name,
            "arguments": self.arguments,
            "ok": self.ok,
            "result": self.result,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


@dataclass
class ToolRunResult:
    answer: str = ""
    calls: List[ToolCall] = field(default_factory=list)
    rounds: int = 0
    error: Optional[str] = None
    # What the run was actually given, for the record of it: the model that
    # answered, the system prompt as sent, and the tools that were on offer.
    # Never part of ``to_dict`` — that is what a caller is shown.
    model: str = ""
    system: str = ""
    tools_offered: List[str] = field(default_factory=list)
    # The answer tried to describe the machine and was sent back to be re-framed.
    safety_retry: bool = False
    # …and the second attempt did it too, so the user got the safe reply instead.
    safety_blocked: bool = False
    # What the orchestrator told the model (layers) and what its verify step found. Not part of
    # ``to_dict``: this is the record of a run, not what a caller is shown.
    context: List[Dict[str, Any]] = field(default_factory=list)
    verification: Optional[Dict[str, Any]] = None

    @property
    def tools_used(self) -> List[str]:
        return [c.name for c in self.calls if c.ok]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "rounds": self.rounds,
            "error": self.error,
            "tools_used": self.tools_used,
            "safety_retry": self.safety_retry,
            "safety_blocked": self.safety_blocked,
            # The full trace: what was asked of each tool and what it returned. This is raw
            # internal detail — tool arguments, SQL result rows, policy bodies — and it is
            # only safe to return because the boundary gate in main.py scrubs it on the way
            # out. Do not surface it to a user path that bypasses that gate.
            "trace": [c.to_dict() for c in self.calls],
        }


# Token counting. Whoever wants to know what a piece of work cost opens a count
# around it; every model call made inside notes what the model server reported.
# Counts opened inside another add to both, so a run that starts another run
# carries the whole cost. A call made outside any count is simply not counted.
count_tokens = _driver.count_tokens
note_usage = _driver.note_usage


def _coerce_args(raw: Any) -> Dict[str, Any]:
    """Ollama sometimes hands back arguments as a JSON string rather than a dict."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _render_result(value: Any) -> str:
    """Render a tool result for the model, bounded in size."""
    if value is None:
        return "null"
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, default=str)
        except Exception:  # noqa: BLE001
            text = str(value)
    if len(text) > _MAX_RESULT_CHARS:
        return text[:_MAX_RESULT_CHARS] + f"\n...[truncated, {len(text)} chars total]"
    return text


@dataclass
class _RanOnPrecedent:
    """The gate allowed the call on precedent: the handler runs, and the model is told so in one
    line after the tool's own result (``note``). Every other allow is ``None``, unchanged."""

    note: str


def _precedent_note(to_agent: Dict[str, Any]) -> str:
    return ("Note: this action ran without a person approving it, on precedent (decided the same way "
            f"{to_agent.get('precedentCount')} times before, case {to_agent.get('conflictCaseId')}).")


def _with_note(content: str, allowed: Optional["_RanOnPrecedent"]) -> str:
    """The tool message the model reads: the result as rendered, then the precedent note, if any."""
    return content if allowed is None else f"{content}\n{allowed.note}"


def _policy_refusal(
    name: str,
    args: Dict[str, Any],
    *,
    agent: Optional[str],
    reason: Optional[str],
    workflow_id: Optional[str],
    user_id: Optional[str],
) -> Optional[Dict[str, Any]]:
    """The agent-policy gate, consulted immediately before a handler runs.

    None means run the tool. A ``_RanOnPrecedent`` means run it too, and tell the model it ran
    on precedent. Otherwise the dict is what the model is told instead, and the handler is never
    called. The gate never raises; should it be unreachable at all, the call is refused (fail
    closed) exactly as the gate refuses when policies cannot be checked.
    """
    # The kill switch is read here, before the gate is imported, so switching enforcement off
    # works even when the gate itself cannot load. With it on, an import failure refuses.
    if os.getenv("AGENT_POLICY_ENFORCEMENT", "on").strip().lower() == "off":
        return None
    try:
        from services.agent_policy import gate
    except Exception:  # noqa: BLE001
        log.exception("agent policy gate unavailable; refusing tool %s", name)
        return {"result": "blocked", "reasonCode": "policy_check_unavailable",
                "reason": "Policy checks are unavailable, so this action was not run."}
    verdict = gate.before_tool(tool_name=name, args=args, agent=agent, reason=reason,
                               workflow_id=workflow_id, user_id=user_id)
    if verdict.allow:
        told = verdict.to_agent or {}
        return _RanOnPrecedent(_precedent_note(told)) if told.get("precedent") is True else None
    return verdict.to_agent or dict(gate.UNAVAILABLE)


def _chat(
    messages: List[Dict[str, Any]],
    schemas: List[Dict[str, Any]],
    model: str,
    timeout: int,
) -> Dict[str, Any]:
    return _driver.chat(messages, schemas, model=model, timeout=timeout).message


def structured(
    system: str,
    user: str,
    schema: Dict[str, Any],
    *,
    model: Optional[str] = None,
    timeout_s: Optional[int] = None,
) -> Dict[str, Any]:
    """One turn, no tools, answered as a JSON object in ``schema``.

    The schema is passed as the model server's ``format``, so the answer is
    constrained to it as it is generated rather than parsed out of prose
    afterwards. Raises when the model cannot be reached or answers nothing.
    """
    return _driver.structured(system, user, schema, model=model, timeout_s=timeout_s)


def _gate(
    content: str,
    messages: List[Dict[str, Any]],
    schemas: List[Dict[str, Any]],
    model: str,
    timeout_s: int,
    result: "ToolRunResult",
) -> str:
    """The answer must be about the product, never about the machine.

    The model knows how ProcWise works — that is deliberate, it is what lets it answer "why
    did my upload fail" instead of filing a ticket. But knowing is not saying. A draft that
    names a table, a path, an env var or a route, or that merely *narrates the backend*
    ("extraction is triggered before the upload finishes"), is not shown to anyone.

    It gets exactly one chance to re-frame. It has the facts already; it chose the wrong
    register. That is a cheap fix and usually produces the answer the user actually wanted.
    A second failure is not a phrasing problem, so we stop and say plainly that we could not
    help — which is the honest outcome, and it is logged for review.
    """
    from services import output_safety as osafe

    if osafe.is_safe(content):
        return content

    first = osafe.inspect(content)
    log.warning(
        "output_safety: agent draft blocked (kinds=%s); asking it to re-frame",
        sorted({v.kind for v in first}),
    )
    result.safety_retry = True

    messages.append({"role": "user", "content": osafe.RETRY_INSTRUCTION})
    try:
        retry = _chat(messages, schemas, model, timeout_s)
    except Exception as exc:  # noqa: BLE001
        log.warning("output_safety: re-frame call failed: %s", exc)
        result.safety_blocked = True
        return osafe.SAFE_REPLY

    second = retry.get("content") or ""
    if osafe.is_safe(second):
        return second

    log.warning(
        "output_safety: re-framed draft ALSO leaked (kinds=%s); refusing",
        sorted({v.kind for v in osafe.inspect(second)}),
    )
    result.safety_blocked = True
    return osafe.SAFE_REPLY


def _wiring(verify: Optional[Callable[[str, Any, bool], Any]] = None,
            check_arguments: Optional[Callable[[str, Dict[str, Any], Any], Optional[Dict[str, Any]]]] = None):
    """What the orchestrator's executor loop needs, taken from this module's own names at call time.

    The loop itself lives in ``orchestration.executor``; it owns the state of a run and the order of
    things, and imports none of the model, the policy gate or the output check. Looking the names up
    here, when a run starts, keeps them the ones this module currently has.
    """
    from orchestration import executor

    return executor.Wiring(
        ToolCall=ToolCall,
        ToolRunResult=ToolRunResult,
        RanOnPrecedent=_RanOnPrecedent,
        propose_turn=lambda messages, schemas, model, timeout: _chat(messages, schemas, model, timeout),
        propose_stream=lambda messages, schemas, model, timeout: _driver.chat_stream(
            messages, schemas, model=model, timeout=timeout),
        policy_check=lambda *a, **k: _policy_refusal(*a, **k),
        safety_gate=lambda *a, **k: _gate(*a, **k),
        coerce_args=_coerce_args,
        render_result=_render_result,
        with_note=_with_note,
        log=log,
        verify=verify,
        check_arguments=check_arguments,
    )


def run_tools_stream(
    task: str,
    tools: Sequence[Tool],
    system: str,
    *,
    model: Optional[str] = None,
    max_rounds: int = _DEFAULT_MAX_ROUNDS(),
    timeout_s: int = _DEFAULT_TIMEOUT_S,
    on_tool: Optional[Callable[[ToolCall], None]] = None,
    on_delta: Optional[Callable[[str], None]] = None,
    agent: Optional[str] = None,
    workflow_id: Optional[str] = None,
    user_id: Optional[str] = None,
    verify: Optional[Callable[[str, Any, bool], Any]] = None,
    check_arguments: Optional[Callable[[str, Dict[str, Any], Any], Optional[Dict[str, Any]]]] = None,
) -> ToolRunResult:
    """Same loop, but the FINAL answer is streamed out through ``on_delta``.

    Tool-calling rounds are not streamed. ``on_tool`` fires once per call so the UI can say what is
    being looked up. The loop is the orchestrator's (``orchestration.executor.run_stream``).
    """
    from orchestration import executor

    return executor.run_stream(
        _wiring(verify, check_arguments), task, tools, system, model=model or _DEFAULT_MODEL, max_rounds=max_rounds,
        timeout_s=timeout_s, on_tool=on_tool, on_delta=on_delta, agent=agent, workflow_id=workflow_id,
        user_id=user_id)


def run_tools(
    task: str,
    tools: Sequence[Tool],
    system: str,
    *,
    model: Optional[str] = None,
    max_rounds: int = _DEFAULT_MAX_ROUNDS(),
    timeout_s: int = _DEFAULT_TIMEOUT_S,
    require_tool_use: bool = False,
    nudge: Optional[str] = None,
    agent: Optional[str] = None,
    workflow_id: Optional[str] = None,
    user_id: Optional[str] = None,
    verify: Optional[Callable[[str, Any, bool], Any]] = None,
    check_arguments: Optional[Callable[[str, Dict[str, Any], Any], Optional[Dict[str, Any]]]] = None,
) -> ToolRunResult:
    """Run AgentNick over ``task``, letting it call ``tools`` until it answers.

    Bounded by ``max_rounds``. Never raises to the caller: transport failures come back as
    ``ToolRunResult.error`` with whatever was gathered so far, because a dead model server should
    degrade the answer, not take down the request. The loop is the orchestrator's
    (``orchestration.executor.run``); ``require_tool_use`` nudges the model once if it tries to answer
    without consulting a single tool.
    """
    from orchestration import executor

    return executor.run(
        _wiring(verify, check_arguments), task, tools, system, model=model or _DEFAULT_MODEL, max_rounds=max_rounds,
        timeout_s=timeout_s, require_tool_use=require_tool_use, nudge=nudge, agent=agent,
        workflow_id=workflow_id, user_id=user_id)
