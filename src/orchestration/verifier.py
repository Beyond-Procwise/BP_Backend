"""The verify step: nothing AgentNick says becomes the answer until the orchestrator has checked it.

Two checks, in this order.

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

    from services import calculation_tools

    now = today or date.today()
    # Today's date is known to the orchestrator, so a model that says when it checked is not inventing it.
    sources: List[Any] = [task, now.isoformat(), now.strftime("%d %b %Y"), now.strftime("%d %B %Y")]
    for c in calls:
        if getattr(c, "ok", False) and getattr(c, "result", None) is not None:
            sources.append(c.result)
    if config:
        sources.append(json.dumps(config, default=str))
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
