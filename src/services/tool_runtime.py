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
a record.

The record is the point. Every tool call and every result is captured on the
returned `ToolRunResult.calls`, so an answer or a decision can be traced back to
the facts that produced it instead of being taken on trust.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.services import egress

log = logging.getLogger(__name__)

_OLLAMA_CHAT = (
    os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/") + "/api/chat"
)
# AgentNick is the only base model. Never repoint this at another family.
_DEFAULT_MODEL = os.getenv("AGENTNICK_MODEL", "BeyondProcwise/AgentNick:unified")
_DEFAULT_MAX_ROUNDS = int(os.getenv("TOOL_RUNTIME_MAX_ROUNDS", "6"))
_DEFAULT_TIMEOUT_S = int(os.getenv("TOOL_RUNTIME_TIMEOUT_S", "180"))

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
    # The answer tried to describe the machine and was sent back to be re-framed.
    safety_retry: bool = False
    # …and the second attempt did it too, so the user got the safe reply instead.
    safety_blocked: bool = False

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


def _chat(
    messages: List[Dict[str, Any]],
    schemas: List[Dict[str, Any]],
    model: str,
    timeout: int,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        # Reasoning-tuned models return an EMPTY `response`/content unless think is
        # off. Leaving this on silently yields blank answers.
        "think": False,
        "keep_alive": -1,
        "options": {"temperature": 0, "num_predict": 2048},
    }
    if schemas:
        payload["tools"] = schemas
    response = egress.post(
        _OLLAMA_CHAT,
        purpose=egress.Purpose.MODEL_INFERENCE,
        json=payload,
        timeout=timeout,
        require_global=False,   # the model daemon is on localhost by design
        raise_transport_errors=True,
    )
    if response is None:
        return {}
    response.raise_for_status()
    return response.json().get("message", {}) or {}


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


def run_tools_stream(
    task: str,
    tools: Sequence[Tool],
    system: str,
    *,
    model: Optional[str] = None,
    max_rounds: int = _DEFAULT_MAX_ROUNDS,
    timeout_s: int = _DEFAULT_TIMEOUT_S,
    on_tool: Optional[Callable[[ToolCall], None]] = None,
    on_delta: Optional[Callable[[str], None]] = None,
) -> ToolRunResult:
    """Same loop, but the FINAL answer is streamed out through ``on_delta``.

    Tool-calling rounds are not streamed — there is nothing to show while the model is
    deciding which tool to call, and a half-formed tool call is not something a user
    should ever see. ``on_tool`` fires once per call so the UI can say what is being
    looked up. Only once the model stops calling tools and starts writing prose do the
    tokens flow.
    """
    model = model or _DEFAULT_MODEL
    by_name = {t.name: t for t in tools}
    schemas = [t.schema() for t in tools]

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": task},
    ]
    result = ToolRunResult()

    for _ in range(max_rounds):
        result.rounds += 1
        try:
            payload: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "stream": True,
                "think": False,
                "keep_alive": -1,
                "options": {"temperature": 0, "num_predict": 2048},
                "tools": schemas,
            }
            response = egress.post(
                _OLLAMA_CHAT,
                purpose=egress.Purpose.MODEL_INFERENCE,
                json=payload,
                timeout=timeout_s,
                stream=True,
                require_global=False,
                raise_transport_errors=True,
            )
            if response is None:
                result.error = "egress refused the model call"
                return result
            response.raise_for_status()

            content_parts: List[str] = []
            tool_calls: List[Dict[str, Any]] = []
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                message = chunk.get("message") or {}
                if message.get("tool_calls"):
                    tool_calls.extend(message["tool_calls"])
                fragment = message.get("content")
                if fragment:
                    content_parts.append(fragment)
        except Exception as exc:  # noqa: BLE001
            log.warning("tool_runtime stream failed: %s", exc)
            result.error = str(exc)[:300]
            return result

        content = "".join(content_parts)
        if not tool_calls:
            # Tokens used to go straight out to the browser as they arrived. They no longer
            # do, and the reason is that a token cannot be un-sent: by the time a scanner
            # sees `proc.` and `process_monitor` land in two separate fragments, the user has
            # already read them. So the answer is assembled, checked, re-framed if it was
            # describing the machine, and only then released.
            #
            # The cost is the typing effect on the final answer. The user still watches the
            # tool stages tick over live, so nothing looks frozen — and an answer that
            # appears half a second later is a much better trade than one that leaks.
            result.answer = _gate(content, messages, schemas, model, timeout_s, result)
            if on_delta and result.answer:
                on_delta(result.answer)
            return result

        messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
        for call in tool_calls:
            fn = call.get("function") or {}
            name = fn.get("name") or ""
            args = _coerce_args(fn.get("arguments"))
            tool = by_name.get(name)
            started = time.monotonic()
            if tool is None:
                record = ToolCall(name=name, arguments=args, ok=False, error=f"unknown tool '{name}'")
            else:
                try:
                    record = ToolCall(
                        name=name, arguments=args, ok=True, result=tool.handler(**args)
                    )
                except Exception as exc:  # noqa: BLE001
                    log.exception("tool %s failed", name)
                    record = ToolCall(name=name, arguments=args, ok=False, error=str(exc)[:300])
            record.duration_ms = int((time.monotonic() - started) * 1000)
            result.calls.append(record)
            if on_tool:
                on_tool(record)
            messages.append(
                {
                    "role": "tool",
                    "name": name,
                    "content": _render_result(
                        record.result if record.ok else {"error": record.error}
                    ),
                }
            )

    result.error = f"max_rounds ({max_rounds}) exhausted without a final answer"
    return result


def run_tools(
    task: str,
    tools: Sequence[Tool],
    system: str,
    *,
    model: Optional[str] = None,
    max_rounds: int = _DEFAULT_MAX_ROUNDS,
    timeout_s: int = _DEFAULT_TIMEOUT_S,
    require_tool_use: bool = False,
    nudge: Optional[str] = None,
) -> ToolRunResult:
    """Run AgentNick over ``task``, letting it call ``tools`` until it answers.

    Bounded by ``max_rounds``. Never raises to the caller: transport failures come
    back as ``ToolRunResult.error`` with whatever was gathered so far, because a
    dead Ollama should degrade the answer, not take down the request.

    ``require_tool_use`` nudges the model once if it tries to answer without
    consulting a single tool. That is the anti-hallucination guard: for a grounded
    question, an answer produced without looking anything up is a guess.
    """
    model = model or _DEFAULT_MODEL
    by_name = {t.name: t for t in tools}
    schemas = [t.schema() for t in tools]

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": task},
    ]

    result = ToolRunResult()
    nudged = False
    any_tool_call = False

    for _ in range(max_rounds):
        result.rounds += 1
        try:
            message = _chat(messages, schemas, model, timeout_s)
        except Exception as exc:  # noqa: BLE001
            log.warning("tool_runtime chat failed: %s", exc)
            result.error = str(exc)[:300]
            return result

        messages.append(message)
        tool_calls = message.get("tool_calls") or []

        if not tool_calls:
            content = message.get("content") or ""
            # The model wants to answer. If it never looked anything up and the
            # caller demanded grounding, push back exactly once.
            if require_tool_use and not any_tool_call and not nudged:
                nudged = True
                messages.append(
                    {
                        "role": "user",
                        "content": nudge
                        or (
                            "You answered without calling any tool. Do not rely on "
                            "assumptions. Call the tools you need to establish the "
                            "facts, then answer from what they return."
                        ),
                    }
                )
                continue
            result.answer = _gate(content, messages, schemas, model, timeout_s, result)
            return result

        any_tool_call = True
        for call in tool_calls:
            fn = call.get("function") or {}
            name = fn.get("name") or ""
            args = _coerce_args(fn.get("arguments"))
            tool = by_name.get(name)

            started = time.monotonic()
            if tool is None:
                record = ToolCall(
                    name=name,
                    arguments=args,
                    ok=False,
                    error=f"unknown tool '{name}'",
                )
            else:
                try:
                    value = tool.handler(**args)
                    record = ToolCall(name=name, arguments=args, ok=True, result=value)
                except TypeError as exc:
                    # Bad arguments from the model — tell it, don't crash.
                    record = ToolCall(
                        name=name,
                        arguments=args,
                        ok=False,
                        error=f"invalid arguments: {exc}",
                    )
                except Exception as exc:  # noqa: BLE001
                    log.exception("tool %s failed", name)
                    record = ToolCall(
                        name=name, arguments=args, ok=False, error=str(exc)[:300]
                    )
            record.duration_ms = int((time.monotonic() - started) * 1000)
            result.calls.append(record)

            messages.append(
                {
                    "role": "tool",
                    "name": name,
                    "content": _render_result(
                        record.result if record.ok else {"error": record.error}
                    ),
                }
            )

    # Ran out of rounds. Return the last thing said rather than nothing.
    result.error = f"max_rounds ({max_rounds}) exhausted without a final answer"
    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("content"):
            result.answer = message["content"]
            break
    return result
