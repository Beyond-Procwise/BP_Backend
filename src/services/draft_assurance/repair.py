"""What the repair model is told to fix: one plain instruction per failed check, never an internal code.

Measured live 2026-10-09 (AgentNick:unified, 8 failing drafts, same acceptance rule): listing the checks as codes
("ungrounded_figure: 43.10") fixed 1 draft; these instructions fixed 6. The repair cannot supply information, so a
missing deadline is repaired only with the deadline the run already holds. Asked to "add a deadline" without one, the
model invented "by the end of the week", which no check can tell from a real one. Without a held deadline the item is
left out and the draft stays failing for a person.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

_PLAIN = {
    "ungrounded_figure": "Delete the figure {d} and the words that depend on it; it is not a verified value. "
                         "Do not replace it with another number.",
    "ungrounded_date": "Delete the date {d} and the clause that depends on it; it is not a verified date. "
                       "Do not replace it with another date.",
    "unresolved_placeholder": "Delete the placeholder {d}. If the sentence needs it, delete the whole sentence. Never fill it in.",
    "internal_figure_leaked": "Delete the sentence that reveals our internal limit. Do not hint at it.",
    "forbidden_content": "Delete the sentence containing: {text}.",
}


def instructions(failed: List[Dict[str, Any]], *, deadline: Optional[str] = None) -> List[str]:
    out: List[str] = []
    for v in failed:
        kind, detail = v.get("kind"), str(v.get("detail") or "")
        if kind == "missing_required_element":
            if detail == "deadline":
                if deadline:
                    out.append(f"The email must ask for a reply by {deadline}. Add that deadline to the request, worded "
                               "naturally. Use no other date.")
                continue                                   # nothing to add it from: a person must
            if detail == "explicit_ask":
                out.append("The email must end with one clear question to the supplier, using no new figures, dates or references.")
                continue
        template = _PLAIN.get(kind)
        if template:
            text = detail.split(": ", 1)[1] if kind == "forbidden_content" and ": " in detail else detail
            out.append(template.format(d=detail, text=text))
        else:
            out.append(f"Fix this problem: {detail}")
    return out
