"""One narrow output check for /ask: the assistant must never approve, waive or grant anything.

``output_safety.py`` looks for leaked internals; nothing looked for the assistant speaking *as an
authority*. Only first-person approval language is caught here ("I approve", "I waive", "approved on
behalf of ..."), on purpose: a wider net replaced harmless answers before (tests D5 and F2). A refusal
("I cannot approve ...") and a third-person explanation ("the CPO can waive ...") do not match.
"""
from __future__ import annotations

import re
from typing import List, Optional

REPLY = (
    "I can explain how approvals and waivers work and who has the authority to grant them, but I "
    "can't approve, waive or grant anything myself. Tell me which policy or approval you need and I "
    "will set out the rule and who decides."
)

_VERB = r"(?:approve|waive|authori[sz]e|sign\s+off(?:\s+on)?|grant(?:\s+(?:you|an?\s+exception|approval|permission))?|override|exempt)"
_RULES = {
    "first-person-approval": re.compile(r"\bi(?:\s+hereby|\s+now|\s+am\s+happy\s+to|\s+will|\s+am\s+going\s+to)?\s+" + _VERB + r"\b", re.I),
    "first-person-continuous": re.compile(r"\bi(?:'|’)?m\s+(?:hereby\s+)?(?:approving|waiving|authori[sz]ing|granting|overriding)\b", re.I),
    "approved-on-behalf": re.compile(r"\b(?:approved|waived|authori[sz]ed|granted)\s+on\s+behalf\s+of\b", re.I),
    "as-authority-i": re.compile(r"\bas\s+the\s+(?:cpo|cfo|ceo|coo|approver|chief\s+\w+\s+officer|head\s+of\s+\w+|director)\b[^.!?]{0,40}\bi\b", re.I),
}
_TAGS = re.compile(r"<[^>]{0,200}>")
def find_violation(text: str) -> Optional[str]:
    """The id of the first rule the text trips, or None."""
    if not isinstance(text, str) or not text:
        return None
    plain = _TAGS.sub(" ", text).replace("’", "'")
    for rule_id, rx in _RULES.items():
        if rx.search(plain):
            return rule_id
    return None


class StreamGate:
    """Holds streamed answer text back to a sentence boundary, so a violating sentence is never sent.

    ``feed(text)`` returns what is safe to forward now. Once a violation is seen, ``tripped`` is set,
    ``rule`` says which, and nothing more is released. ``flush()`` releases the final partial sentence.
    """

    _BOUNDARY = re.compile(r"(?<=[.!?\n])(?=\s)")      # zero-width: no character is dropped

    def __init__(self) -> None:
        self._held = ""
        self.tripped = False
        self.rule: Optional[str] = None

    def _release(self, final: bool) -> str:
        if self.tripped:
            return ""
        parts = self._BOUNDARY.split(self._held)
        ready, self._held = (parts, "") if final else (parts[:-1], parts[-1])
        out: List[str] = []
        for sentence in ready:
            rule = find_violation(sentence)
            if rule:
                self.tripped, self.rule, self._held = True, rule, ""
                return "".join(out)
            out.append(sentence)
        return "".join(out)

    def feed(self, text: str) -> str:
        if self.tripped:
            return ""
        self._held += text or ""
        return self._release(final=False)

    def flush(self) -> str:
        return self._release(final=True)
