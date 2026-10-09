"""The frame around a model-written email: greeting and sign-off are code, never the model's.

Found live 2026-10-09: model drafts signed as invented people (a name, a job title, a company, an email address, phone numbers)
and greeted someone other than the contact on record. A deterministic check catches phone numbers and addresses, but nothing can
tell an invented name from a real one, so the model is not allowed to write either end of the email: whatever greeting and
signature it produced are removed, the greeting names the contact on record (or no one), and the sign-off is fixed.
"""

from __future__ import annotations

import re
from typing import Optional

SIGN_OFF = "Kind regards,\nProcurement Team"

# Told to every model-written path, so the stripping below has less to do. The stripping does not rely on it.
BODY_ONLY = ("Write only the body paragraphs. Do not write a greeting line, a sign-off, a signature, or any name, job title, "
             "company, email address or phone number for the sender; the greeting and sign-off are added for you.")

_GREETING = re.compile(
    r"^\s*(?:dear|hello|hi|hey|greetings|good\s+(?:morning|afternoon|evening))\b[^\n]{0,80}$", re.IGNORECASE)
_CLOSING = re.compile(
    r"^\s*(?:"
    r"(?:with\s+)?(?:kind|best|warm|warmest|many)?\s*regards"
    r"|(?:yours\s+)?(?:sincerely|faithfully|truly)(?:\s+yours)?"
    r"|yours\s+(?:sincerely|faithfully|truly)"
    r"|respectfully(?:\s+yours)?"
    r"|(?:many\s+)?thanks(?:\s+again)?(?:\s+and\s+(?:kind\s+|best\s+)?regards)?"
    r"|thank\s+you(?:\s+again)?"
    r"|cheers|best|best\s+wishes|all\s+the\s+best"
    r")\s*[,.!]?\s*$", re.IGNORECASE)


_INLINE_GREETING = re.compile(
    r"^\s*(?:dear|hello|hi|hey|greetings|good\s+(?:morning|afternoon|evening))\b[^,.!?\n]{0,60},\s+(?=[A-Z])", re.IGNORECASE)
# Only after a sentence end, and only a closing phrase followed by at most a short name: "With regards to ..." is a sentence.
_INLINE_CLOSING = re.compile(
    r"(?<=[.!?])\s+(?i:(?:kind|best|warm|warmest|many)\s+regards|regards|yours\s+(?:sincerely|faithfully|truly)|sincerely|"
    r"respectfully|best\s+wishes)\s*,?(?:\s+[A-Z][\w.'-]*){0,4}\s*[.!]?\s*$")


def strip_frame(text: str) -> str:
    """The body alone: a leading greeting line and everything from the first closing line onward are removed."""

    lines = (text or "").replace("\r\n", "\n").split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines:
        # A greeting run into the first sentence on one line ("Dear Procurement Manager, We refer to ...") loses the
        # greeting only; a line that is nothing but a greeting goes.
        rest = _INLINE_GREETING.sub("", lines[0], count=1)
        if rest != lines[0] and rest.strip():
            lines[0] = rest
        elif _GREETING.match(lines[0]):
            lines.pop(0)
    for i, line in enumerate(lines):
        if _CLOSING.match(line) and any(l.strip() for l in lines[:i]):
            lines = lines[:i]
            break
    body = "\n".join(lines).strip()
    # A sign-off run onto the end of the last sentence: "... by Friday. Kind regards, Eleanor Hartwell"
    return _INLINE_CLOSING.sub("", body).strip()


def frame(text: str, *, contact_name: Optional[str]) -> str:
    name = (contact_name or "").strip()
    greeting = f"Dear {name}," if name else "Hello,"
    return f"{greeting}\n\n{strip_frame(text)}\n\n{SIGN_OFF}"


def close(text: str) -> str:
    """The body with the fixed sign-off and no greeting: for a path that puts its own greeting above other content."""

    return f"{strip_frame(text)}\n\n{SIGN_OFF}"


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def reframe_like(original: str, repaired: str) -> str:
    """Give ``repaired`` the frame ``original`` had. A repair rewrites the body only; it never owns either end."""

    if not _squash(original).endswith(_squash(SIGN_OFF)):
        return repaired                                     # not framed by us: nothing to keep
    first = (original or "").strip().split("\n", 1)[0].strip()
    greeting = first if _GREETING.match(first) else None
    body = strip_frame(repaired)
    return (f"{greeting}\n\n" if greeting else "") + f"{body}\n\n{SIGN_OFF}"
