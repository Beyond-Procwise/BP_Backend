"""The executor loop: act, observe, verify. This is the orchestrator's, not AgentNick's.

AgentNick is a coprocessor that sits alongside the orchestrator. This loop hands it the conversation
so far and gets back a proposal: some text and any tool calls it would like made. Nothing in a
proposal has authority. For each proposed tool call this loop asks the policy gate first, runs the
tool only if the gate allows it, records what happened, and feeds the result back. A final answer is
released only after the output check has seen it.

The loop owns all the state of a run (the messages, the calls made, the result). It owns no model,
no data store and no policy: every one of those arrives through ``Wiring``, supplied by the caller,
which is also why this module imports none of them. The caller supplies its own result types as
well, so a run returns the objects that caller expects whichever way its package was imported.

Two entry points keep the behaviour of the two loops that used to live in ``tool_runtime``:
``run`` takes whole turns and can nudge a model that answers without looking anything up;
``run_stream`` reads each turn as a stream and reports each call and the final answer as they happen.
Their small differences (the wording for bad arguments, the nudge, salvaging the last thing said)
are kept as they were.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class Wiring:
    """Everything the loop needs from outside it."""

    ToolCall: Any                # the record of one call
    ToolRunResult: Any           # the result of a run
    RanOnPrecedent: Any          # what the gate returns when it allows a call on precedent
    propose_turn: Callable[..., Dict[str, Any]]    # (messages, schemas, model, timeout) -> message dict
    propose_stream: Callable[..., Any]             # (messages, schemas, model, timeout) -> .content, .tool_calls
    policy_check: Callable[..., Any]               # (name, args, *, agent, reason, workflow_id, user_id)
    safety_gate: Callable[..., str]                # (content, messages, schemas, model, timeout, result) -> answer
    coerce_args: Callable[[Any], Dict[str, Any]]
    render_result: Callable[[Any], str]
    with_note: Callable[[str, Any], str]
    log: Any
    # (answer, result, already_retried) -> something with .action .answer .untraced .retry_message.
    # Optional: with none, an answer is released as the output check left it.
    verify: Optional[Callable[[str, Any, bool], Any]] = None


def _act(w: Wiring, call: Dict[str, Any], by_name: Dict[str, Any], *, agent, reason, workflow_id, user_id,
         bad_arguments_wording: bool):
    """One proposed call: ask the gate, run the tool if allowed, return (record, refusal, allowed)."""
    fn = call.get("function") or {}
    name = fn.get("name") or ""
    args = w.coerce_args(fn.get("arguments"))
    tool = by_name.get(name)
    started = time.monotonic()
    refusal = None
    if tool is not None:
        refusal = w.policy_check(name, args, agent=agent, reason=reason, workflow_id=workflow_id, user_id=user_id)
    allowed = refusal if isinstance(refusal, w.RanOnPrecedent) else None
    if allowed is not None:
        refusal = None
    if tool is None:
        record = w.ToolCall(name=name, arguments=args, ok=False, error=f"unknown tool '{name}'")
    elif refusal is not None:
        # Refused by an agent policy: the handler never runs; the model is told why.
        record = w.ToolCall(name=name, arguments=args, ok=False, result=refusal, error=None)
    else:
        try:
            record = w.ToolCall(name=name, arguments=args, ok=True, result=tool.handler(**args))
        except TypeError as exc:
            if not bad_arguments_wording:
                w.log.exception("tool %s failed", name)
                record = w.ToolCall(name=name, arguments=args, ok=False, error=str(exc)[:300])
            else:
                # Bad arguments from the model: tell it, don't crash.
                record = w.ToolCall(name=name, arguments=args, ok=False, error=f"invalid arguments: {exc}")
        except Exception as exc:  # noqa: BLE001
            w.log.exception("tool %s failed", name)
            record = w.ToolCall(name=name, arguments=args, ok=False, error=str(exc)[:300])
    record.duration_ms = int((time.monotonic() - started) * 1000)
    return record, refusal, allowed


def _observation(w: Wiring, record: Any, refusal: Any, allowed: Any) -> Dict[str, Any]:
    """The tool message the model reads next."""
    return {
        "role": "tool",
        "name": record.name,
        "content": (
            json.dumps(refusal, default=str)
            if refusal is not None
            else w.with_note(w.render_result(record.result if record.ok else {"error": record.error}), allowed)
        ),
    }


def _release(w: Wiring, answer: str, result: Any, retried: bool):
    """Verify a final answer. Returns (answer, retry_message): a retry message means send the model back."""
    if w.verify is None:
        return answer, None
    verdict = w.verify(answer, result, retried)
    result.verification = {"action": verdict.action, "untraced": list(verdict.untraced)}
    if verdict.action == "retry" and not retried:
        return answer, verdict.retry_message
    return verdict.answer, None


def run_stream(w: Wiring, task: str, tools: Sequence[Any], system: str, *, model: str, max_rounds: int,
               timeout_s: int, on_tool: Optional[Callable[[Any], None]] = None,
               on_delta: Optional[Callable[[str], None]] = None, agent: Optional[str] = None,
               workflow_id: Optional[str] = None, user_id: Optional[str] = None) -> Any:
    """Turns read as streams; each call and the final answer are reported as they happen.

    Tool-calling rounds are not streamed: there is nothing to show while the model is deciding which
    tool to call, and a half-formed tool call is not something a user should ever see. ``on_tool``
    fires once per call so the UI can say what is being looked up.
    """
    by_name = {t.name: t for t in tools}
    schemas = [t.schema() for t in tools]
    messages: List[Dict[str, Any]] = [{"role": "system", "content": system}, {"role": "user", "content": task}]
    result = w.ToolRunResult(model=model, system=system, tools_offered=[t.name for t in tools])
    retried = False

    for _ in range(max_rounds):
        result.rounds += 1
        try:
            proposal = w.propose_stream(messages, schemas, model, timeout_s)
        except Exception as exc:  # noqa: BLE001
            w.log.warning("tool_runtime stream failed: %s", exc)
            result.error = str(exc)[:300]
            return result

        content, tool_calls = proposal.content, proposal.tool_calls
        if not tool_calls:
            # Tokens used to go straight out to the browser as they arrived. They no longer do, and
            # the reason is that a token cannot be un-sent: by the time a scanner sees `proc.` and
            # `process_monitor` land in two separate fragments, the user has already read them. So the
            # answer is assembled, checked, re-framed if it was describing the machine, and only then
            # released. The cost is the typing effect on the final answer; the user still watches the
            # tool stages tick over live, so nothing looks frozen.
            answer = w.safety_gate(content, messages, schemas, model, timeout_s, result)
            answer, again = _release(w, answer, result, retried)
            if again is not None:
                retried = True
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": again})
                continue
            result.answer = answer
            if on_delta and result.answer:
                on_delta(result.answer)
            return result

        messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
        for call in tool_calls:
            record, refusal, allowed = _act(w, call, by_name, agent=agent, reason=content,
                                            workflow_id=workflow_id, user_id=user_id, bad_arguments_wording=False)
            result.calls.append(record)
            if on_tool:
                on_tool(record)
            messages.append(_observation(w, record, refusal, allowed))

    result.error = f"max_rounds ({max_rounds}) exhausted without a final answer"
    return result


def run(w: Wiring, task: str, tools: Sequence[Any], system: str, *, model: str, max_rounds: int, timeout_s: int,
        require_tool_use: bool = False, nudge: Optional[str] = None, agent: Optional[str] = None,
        workflow_id: Optional[str] = None, user_id: Optional[str] = None) -> Any:
    """Whole turns, bounded by ``max_rounds``. Never raises: a dead model server degrades the answer.

    ``require_tool_use`` nudges the model once if it tries to answer without consulting a single tool.
    That is the anti-hallucination guard: for a grounded question, an answer produced without looking
    anything up is a guess.
    """
    by_name = {t.name: t for t in tools}
    schemas = [t.schema() for t in tools]
    messages: List[Dict[str, Any]] = [{"role": "system", "content": system}, {"role": "user", "content": task}]
    result = w.ToolRunResult(model=model, system=system, tools_offered=[t.name for t in tools])
    nudged = False
    any_tool_call = False
    retried = False

    for _ in range(max_rounds):
        result.rounds += 1
        try:
            message = w.propose_turn(messages, schemas, model, timeout_s)
        except Exception as exc:  # noqa: BLE001
            w.log.warning("tool_runtime chat failed: %s", exc)
            result.error = str(exc)[:300]
            return result

        messages.append(message)
        tool_calls = message.get("tool_calls") or []

        if not tool_calls:
            content = message.get("content") or ""
            # The model wants to answer. If it never looked anything up and the caller demanded
            # grounding, push back exactly once.
            if require_tool_use and not any_tool_call and not nudged:
                nudged = True
                messages.append({
                    "role": "user",
                    "content": nudge or (
                        "You answered without calling any tool. Do not rely on assumptions. Call the tools "
                        "you need to establish the facts, then answer from what they return."),
                })
                continue
            answer = w.safety_gate(content, messages, schemas, model, timeout_s, result)
            answer, again = _release(w, answer, result, retried)
            if again is not None:
                retried = True
                messages.append({"role": "user", "content": again})
                continue
            result.answer = answer
            return result

        any_tool_call = True
        for call in tool_calls:
            record, refusal, allowed = _act(w, call, by_name, agent=agent, reason=message.get("content"),
                                            workflow_id=workflow_id, user_id=user_id, bad_arguments_wording=True)
            result.calls.append(record)
            messages.append(_observation(w, record, refusal, allowed))

    # Ran out of rounds. Return the last thing said rather than nothing.
    result.error = f"max_rounds ({max_rounds}) exhausted without a final answer"
    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("content"):
            result.answer = message["content"]
            break
    return result
