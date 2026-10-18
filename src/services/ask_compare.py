"""Comparison questions for /ask: "compare X and Y" is two retrievals, not one.

One search for "compare Ironbridge and Meridia" returns whatever sits nearest to the whole sentence, and the
answer was "no data" for each (test B2). Here the two things being compared are pulled out of the question with
patterns (no model), each is searched for on its own, and the results are labelled with the entity they came
from before the model writes the comparison.

This is plain code rather than a step in ``orchestration``: that engine runs named agent workflows, and a
conditional edge there costs a workflow run for what is, here, two calls to the same search.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

_TAIL = r"(?:\s+(?:on|for|in\s+terms\s+of|by|regarding|about|across|with\s+respect\s+to)\s+(?P<topic>.+?))?"
_SEP = r"(?:and|with|to|against|vs\.?|versus|or)"
_PATTERNS = [
    re.compile(rf"^\s*(?:please\s+)?(?:can\s+you\s+)?compare\s+(?:the\s+)?(?P<a>.+?)\s+{_SEP}\s+(?:the\s+)?(?P<b>.+?){_TAIL}\s*[?.!]*\s*$", re.I),
    re.compile(rf"^\s*(?:what(?:'s|\s+is|\s+are)\s+)?(?:the\s+)?differences?\s+between\s+(?:the\s+)?(?P<a>.+?)\s+and\s+(?:the\s+)?(?P<b>.+?){_TAIL}\s*[?.!]*\s*$", re.I),
    re.compile(rf"^\s*how\s+does\s+(?:the\s+)?(?P<a>.+?)\s+compare\s+(?:with|to|against)\s+(?:the\s+)?(?P<b>.+?){_TAIL}\s*[?.!]*\s*$", re.I),
    re.compile(rf"^\s*(?:the\s+)?(?P<a>[A-Z][^?]+?)\s+(?:vs\.?|versus)\s+(?:the\s+)?(?P<b>.+?){_TAIL}\s*[?.!]*\s*$"),
]
_VAGUE = {"it", "them", "this", "that", "these", "those", "each other", "one another", "both", "the two", "other", "others"}
MAX_ENTITY = 60


@dataclass(frozen=True)
class Comparison:
    entities: Tuple[str, str]
    topic: str = ""

    def search_queries(self) -> List[str]:
        """One query per entity: its name plus what is being compared."""
        return [f"{e} {self.topic}".strip() for e in self.entities]


def _clean(text: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip(" ,;:.?!\"'")


def detect(query: str) -> Optional[Comparison]:
    """The two things being compared, or None when this is not a comparison of two named things."""
    text = query if isinstance(query, str) else ""
    for rx in _PATTERNS:
        m = rx.match(text)
        if not m:
            continue
        a, b, topic = _clean(m.group("a")), _clean(m.group("b")), _clean(m.groupdict().get("topic"))
        if not (2 <= len(a) <= MAX_ENTITY and 2 <= len(b) <= MAX_ENTITY):
            continue
        if a.lower() in _VAGUE or b.lower() in _VAGUE or a.lower() == b.lower():
            continue
        return Comparison((a, b), topic)
    return None


def instruction(comparison: Comparison, missing: List[str]) -> str:
    """The line added to the prompt: what is being compared, and which side has nothing."""
    a, b = comparison.entities
    line = (f"This question compares {a} with {b}. The snippets were retrieved separately and each is labelled with the "
            f"one it came from. Compare them side by side, using only what the snippets say.")
    if missing:
        line += (" No relevant records were found for " + " or ".join(missing)
                 + "; say that plainly for that side and do not fill the gap from the other side or from memory.")
    return line


# Words that do not identify a company or a document: dropped when checking whether a document is about an entity.
_GENERIC = frozenset("""ltd limited plc inc llc corp co the and of for group company services service solutions systems managed
quote quotes quotation contract contracts agreement invoice invoices po pos order orders policy policies terms payment price pricing
supplier suppliers vendor vendors record records document documents""".split())
_TEXT_KEYS = ("content", "text", "summary", "text_summary", "title", "document_name", "supplier_name", "record_id", "document_id")


def mentions(entity: str, payload: Dict[str, Any]) -> bool:
    """Does this retrieved document actually name ``entity``?

    Vector search returns the nearest text, which for "Meridia payment terms" can be a generic payment-terms policy.
    Labelling that as Meridia's data would present another document's content as the answer, so a document counts for
    a side only if one of the entity's distinctive words appears in it. An entity with no distinctive word
    (``quotes``) cannot be checked and is not filtered.
    """
    words = [w for w in re.findall(r"[A-Za-z0-9]+", entity or "") if len(w) >= 3 and w.lower() not in _GENERIC]
    if not words:
        return True
    text = " ".join(str(payload.get(k, "")) for k in _TEXT_KEYS if payload.get(k))
    return any(re.search(rf"\b{re.escape(w)}\b", text, re.I) for w in words)
