"""History condensing for /ask: turn "When does it expire?" into a question that stands on its own.

Retrieval sees only the text it is given. After "Pull up our contract with Ironbridge Managed IT Ltd", the
follow-up "When does it expire?" carries no supplier, so retrieval answered about the Preferred Suppliers
List instead (test C2). One short model call rewrites the latest turn, with the last few turns as context,
into a single standalone question, which retrieval then uses. The answer is still written to the user's own
words; the original question is what is kept in the history.

The model's output is only ever a search query, but it is not trusted: ``accept`` throws a rewrite away
(and retrieval uses the original) unless it is one short line, keeps every number and identifier the user
typed, and would not itself be blocked by the input guardrail - so a prompt-injection sitting in an earlier
answer cannot be laundered into the query. No history, no call.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from . import ask_exact_id, ask_input_guard

MAX_TURNS = 3
MAX_ANSWER_CHARS = 500
MAX_LEN = 300
FAILURE_TEXT = "Could not generate an answer."

SYSTEM = (
    "You rewrite a user's latest message as ONE standalone question, using the earlier conversation only to "
    "resolve what it refers to (it, they, that supplier, the same order, 'and the other one'). "
    "Keep every name, number, date and identifier exactly as written. Do not answer the question and do not add "
    "facts that are not in the conversation. If the message already stands on its own, return it unchanged. "
    'Reply with JSON only: {"answer": "<the standalone question>", "follow_ups": []}'
)

_NUMBER = re.compile(r"\d[\d,.]*\d|\d")


def build_prompt(query: str, turns: List[Dict[str, Any]]) -> str:
    """The conversation so far and the latest message, oldest first."""
    lines: List[str] = ["Earlier in the conversation:"]
    for t in turns[-MAX_TURNS:]:
        q = str(t.get("query", "")).strip()
        a = str(t.get("answer", "")).strip()[:MAX_ANSWER_CHARS]
        if q or a:
            lines.append(f"User: {q}")
            lines.append(f"Assistant: {a}")
    lines.append("")
    lines.append(f"Latest message: {query.strip()}")
    return "\n".join(lines)


def accept(raw: Any, original: str) -> Optional[str]:
    """The rewrite if it is safe to search with, else None (use the original)."""
    text = raw.get("answer") if isinstance(raw, dict) else raw
    if not isinstance(text, str):
        return None
    text = re.sub(r"\s+", " ", text).strip().strip('"\'`').strip()
    if not text or text == FAILURE_TEXT or len(text) > MAX_LEN or len(text) < 3:
        return None
    if text.lower() == re.sub(r"\s+", " ", original).strip().lower():
        return None                                            # unchanged: nothing to report
    kept_ids = ask_exact_id.extract(original)
    rewritten_ids = ask_exact_id.extract(text)
    if any(i not in rewritten_ids.get(kind, []) for kind, ids in kept_ids.items() for i in ids):
        return None
    flat = text.replace(",", "")
    if any(n.replace(",", "") not in flat for n in _NUMBER.findall(original)):
        return None                                            # a number the user typed went missing
    if ask_input_guard.screen_rules(text).blocked:
        return None
    return text
