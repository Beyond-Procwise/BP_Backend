"""The verify step: nothing AgentNick says becomes the answer until the orchestrator has checked it.

Two checks, in this order.

0. **An empty answer** is never released as if it had been checked: one retry, then a plain statement
   that nothing was produced.
1. **An authoritative tool's reply.** A skill can declare that one tool's ``reply_text`` is the answer
   (``authoritative_reply_tool`` in its settings). The policy checker is one: its text is computed
   from the governed policy, so the model's rewording of it can only add error. When such a call
   succeeded, its text is shown verbatim.
2. **Figures.** Every figure in the answer must appear in the request, in something a tool returned,
   or in the skill's own settings. A figure from none of those was produced by the model. It gets one
   chance to fix that; if the figure is still there, the answer stands but the figures are named as
   not verified, in plain words. What a model passed *into* a tool does not count as a source: only
   what came back does.

This module is a pure check. It calls no model and no tool and changes nothing it is given.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from typing import Any, List, Mapping, Optional, Sequence


@dataclass(frozen=True)
class Verdict:
    action: str                      # keep | replace | retry | caution
    answer: str
    untraced: List[str] = field(default_factory=list)
    retry_message: Optional[str] = None


def _authoritative(calls: Sequence[Any], tool: str) -> Optional[str]:
    text = None
    for c in calls:
        result = getattr(c, "result", None)
        if getattr(c, "name", None) == tool and getattr(c, "ok", False) and isinstance(result, Mapping):
            reply = result.get("reply_text")
            if isinstance(reply, str) and reply.strip():
                text = reply
    return text


def verify(answer: str, calls: Sequence[Any], task: str, *, skill_config: Optional[Mapping[str, Any]] = None,
           retried: bool = False, today: Optional[date] = None) -> Verdict:
    config = skill_config or {}
    tool = config.get("authoritative_reply_tool")
    if tool:
        reply = _authoritative(calls, str(tool))
        if reply is not None:
            return Verdict("replace", reply)

    if not (answer or "").strip():
        # An empty reply is not a checked answer. Ask once for a real one; if it is still empty,
        # say so plainly rather than release nothing as though it had been verified.
        if not retried:
            return Verdict("retry", answer or "", [], (
                "Your answer was empty. Write the answer now, using only the request and the tool "
                "results above. If a step could not be done, say which and why."))
        return Verdict("caution", "No answer was produced for this request, so nothing has been checked or "
                                  "decided. Please try again or ask in a different way.", [])

    from services import calculation_tools

    now = today or date.today()
    # Today's date is known to the orchestrator, so a model that says when it checked is not inventing it.
    sources: List[Any] = [task, now.isoformat(), now.strftime("%d %b %Y"), now.strftime("%d %B %Y")]
    from .argument_check import NON_SOURCE_TOOLS

    if config:
        sources.append(json.dumps(config, default=str))
    for c in calls:
        if (getattr(c, "ok", False) and getattr(c, "result", None) is not None
                and getattr(c, "name", None) not in NON_SOURCE_TOOLS):
            sources.append(_without_echoes(c, sources))
    untraced = calculation_tools.untraced_figures(answer or "", sources)
    if not untraced:
        return Verdict("keep", answer)
    listed = ", ".join(untraced)
    if not retried:
        return Verdict("retry", answer, untraced, (
            f"These figures in your answer are not in the request or in any tool result: {listed}. "
            "Remove them, or get them from a tool, and answer again. Do not estimate or work out a "
            "figure yourself."))
    return Verdict("caution", f"{answer}\n\nNote: these figures are not from a tool result or from your "
                              f"request, so they are not verified: {listed}.", untraced)



def _without_echoes(call: Any, sources: Sequence[Any]) -> str:
    """A tool's result as a source, minus the figures the model itself put into the call.

    Live, 2026-10-22: the model invented a contract value, passed it to check_policy, and the reply
    repeated it, so the invented figure traced to "a tool result". A number that was in the call's
    arguments but in nothing the run had been given before the call is the model's, and the tool
    echoing it back does not make it a fact; such numbers are blanked from the result text."""
    from services import calculation_tools as ct

    result_text = json.dumps(call.result, default=str)
    args = getattr(call, "arguments", None) or getattr(call, "args", None) or {}
    try:
        arg_numbers = ct._numbers(json.dumps(args, default=str))
    except (TypeError, ValueError):
        return result_text
    if not arg_numbers:
        return result_text
    known = ct._numbers("\n".join(s if isinstance(s, str) else json.dumps(s, default=str) for s in sources))
    invented = {n for n in arg_numbers if n not in known}
    if not invented:
        return result_text

    def _blank(m):
        value = ct._decimal(m.group(0).rstrip(","))
        return " " if value is not None and value in invented else m.group(0)

    return ct._NUMBER.sub(_blank, result_text)
