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

import re

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
    # (tool_name, args, result_so_far) -> a refusal dict, or None to carry on. Optional. Runs BEFORE the
    # policy gate, so a figure the model invented never reaches a tool.
    check_arguments: Optional[Callable[[str, Dict[str, Any], Any], Optional[Dict[str, Any]]]] = None


def _act(w: Wiring, call: Dict[str, Any], by_name: Dict[str, Any], result: Any, *, agent, reason, workflow_id,
         user_id, bad_arguments_wording: bool):
    """One proposed call: ask the gate, run the tool if allowed, return (record, refusal, allowed)."""
    fn = call.get("function") or {}
    name = fn.get("name") or ""
    args = w.coerce_args(fn.get("arguments"))
    tool = by_name.get(name)
    started = time.monotonic()
    refusal = None
    if tool is not None and w.check_arguments is not None:
        refusal = w.check_arguments(name, args, result)
    if tool is not None and refusal is None:
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


CLOSING_INSTRUCTION = (
    "No more tool calls are available for this request. Write the final answer now from the tool "
    "results above, in the format the instructions ask for. State plainly which steps could not be "
    "done and what is still needed; do not invent a figure to fill a gap. Answer as yourself in plain "
    "text: do not narrate a screen, a button or something 'the system' did on its own."
)


_SCREEN_TALK = re.compile(r"\byou are (?:on|viewing|looking at) (?:the|a|an|your)\b"
                          r"|\bthe system (?:has (?:automatically|identified|retrieved|checked|flagged|analy[sz]ed)|shows|displays|is showing|identified|retrieved|analy[sz]ed)\b", re.IGNORECASE)
REWRITE_INSTRUCTION = (
    "That answer describes a screen or something 'the system' did, which does not exist. Rewrite it as yourself "
    "in plain text with the same facts and figures and nothing added: no screen, no buttons, no 'the system has'."
)


_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


def scrub_screen_talk(text: str) -> str:
    """Drop the sentences that narrate a screen or 'the system'; keep everything else in order. Returns the
    text unchanged when nothing would be left. Used when the model repeats the narration in its rewrite."""
    parts = [p for p in _SENTENCE.split(text or "") if p is not None]
    kept = [p for p in parts if p.strip() and not _SCREEN_TALK.search(p)]
    return "\n".join(kept).strip() if kept else text


def narrates_a_screen(text: str) -> bool:
    """True when an answer talks about a screen, page or 'the system' as if the reader were looking at a UI."""
    return bool(_SCREEN_TALK.search(text or ""))


def _unnarrated(w: Wiring, content: str, messages: List[Dict[str, Any]], model: str, timeout_s: int, result: Any) -> str:
    """An answer that narrates a screen is sent back once for a plain rewrite; the original is kept if the
    rewrite comes back empty. Runs 5 to 9 on 2026-10-22: the habit showed in ordinary final answers as well as
    closing turns, so every release path goes through here."""
    if not content.strip() or not narrates_a_screen(content):
        return content
    messages.append({"role": "assistant", "content": content})
    messages.append({"role": "user", "content": REWRITE_INSTRUCTION})
    result.rounds += 1
    try:
        redo = (w.propose_turn(messages, [], model, timeout_s) or {}).get("content") or ""
    except Exception as exc:  # noqa: BLE001
        w.log.warning("screen rewrite failed: %s", exc)
        redo = ""
    # Live, run 10: the rewrite came back narrating again ("You are on the Incoterms Selection screen"); the
    # narrating sentences are then cut out mechanically, which keeps the facts and loses only the framing.
    chosen = redo if redo.strip() else content
    return scrub_screen_talk(chosen) if narrates_a_screen(chosen) else chosen


def _closing_turn(w: Wiring, messages: List[Dict[str, Any]], model: str, timeout_s: int, result: Any,
                  retried: bool, max_rounds: int) -> Any:
    """The round budget is spent and the model is still proposing calls. Live, 2026-10-22: skills with a
    multi-step playbook spent every round on tools and returned nothing. One more turn is offered with no
    tool schemas, so the only thing the model can do is answer from what it has; the answer goes through
    the same safety gate and verification as any other. If that turn yields nothing, the last thing said
    is salvaged and the exhaustion is reported as before."""
    messages.append({"role": "user", "content": CLOSING_INSTRUCTION})
    result.rounds += 1
    try:
        message = w.propose_turn(messages, [], model, timeout_s)
    except Exception as exc:  # noqa: BLE001
        w.log.warning("closing turn failed: %s", exc)
        message = {}
    content = _unnarrated(w, (message or {}).get("content") or "", messages, model, timeout_s, result)
    if not content.strip():
        result.error = f"max_rounds ({max_rounds}) exhausted without a final answer"
        for earlier in reversed(messages):
            if earlier.get("role") == "assistant" and earlier.get("content"):
                result.answer = earlier["content"]
                break
        return result
    messages.append({"role": "assistant", "content": content})
    answer = w.safety_gate(content, messages, [], model, timeout_s, result)
    answer = scrub_screen_talk(answer) if narrates_a_screen(answer) else answer
    answer, again = _release(w, answer, result, retried)
    if again is not None:
        messages.append({"role": "user", "content": again})
        try:
            message = w.propose_turn(messages, [], model, timeout_s)
            content = (message or {}).get("content") or content
        except Exception as exc:  # noqa: BLE001
            w.log.warning("closing retry failed: %s", exc)
        content = _unnarrated(w, content, messages, model, timeout_s, result)
        answer = w.safety_gate(content, messages, [], model, timeout_s, result)
        answer = scrub_screen_talk(answer) if narrates_a_screen(answer) else answer
        answer, _ = _release(w, answer, result, True)
    result.answer = answer
    result.closed_without_tools = True
    return result


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
            content = _unnarrated(w, content, messages, model, timeout_s, result)
            answer = w.safety_gate(content, messages, schemas, model, timeout_s, result)
            answer = scrub_screen_talk(answer) if narrates_a_screen(answer) else answer
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
            record, refusal, allowed = _act(w, call, by_name, result, agent=agent, reason=content,
                                            workflow_id=workflow_id, user_id=user_id, bad_arguments_wording=False)
            result.calls.append(record)
            if on_tool:
                on_tool(record)
            messages.append(_observation(w, record, refusal, allowed))

    # Ran out of rounds with the model still asking for tools. One closing turn with no tools on
    # offer: write the answer from what was gathered, and say what could not be done.
    return _closing_turn(w, messages, model, timeout_s, result, retried, max_rounds)


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
            content = _unnarrated(w, content, messages, model, timeout_s, result)
            answer = w.safety_gate(content, messages, schemas, model, timeout_s, result)
            answer = scrub_screen_talk(answer) if narrates_a_screen(answer) else answer
            answer, again = _release(w, answer, result, retried)
            if again is not None:
                retried = True
                messages.append({"role": "user", "content": again})
                continue
            result.answer = answer
            return result

        any_tool_call = True
        for call in tool_calls:
            record, refusal, allowed = _act(w, call, by_name, result, agent=agent, reason=message.get("content"),
                                            workflow_id=workflow_id, user_id=user_id, bad_arguments_wording=True)
            result.calls.append(record)
            messages.append(_observation(w, record, refusal, allowed))

    # Ran out of rounds with the model still asking for tools. One closing turn with no tools on
    # offer: write the answer from what was gathered, and say what could not be done.
    return _closing_turn(w, messages, model, timeout_s, result, retried, max_rounds)
