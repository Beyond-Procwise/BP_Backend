"""Stop invented facts reaching a draft.

The system prompt tells the model to invent nothing and to write a placeholder where the
task did not supply something. It does not obey. Asked to quote a named supplier for 25
desks, it produced a fluent, correctly-voiced email addressed to a contact who does not
exist, citing a PO number it made up, against a quote deadline nobody set — three times
running, with the instruction present and version-bumped in between.

That is the expected result, and the reason this module exists rather than a stronger
sentence in the prompt. An instruction is a request. This is the enforcement half, the
same split the platform already uses between a governed response style and the code that
enforces it.

What is checked: the classes of fact that are both easy to fabricate and expensive to get
wrong in an email a supplier will act on — reference numbers, money, dates, and the name
of the person being written to. Each is matched against the task text; anything that did
not come from there is replaced with a visible placeholder.

What is deliberately not checked: prose, tone, and ordinary words. A grounding check that
tried to verify sentences would either reject everything or, worse, pass a fabricated
clause because its words individually appeared somewhere — a failure this codebase has
already met once.

The bias is to replace. A visible ``[reference]`` is a gap someone will fill. A plausible
``PO-94103`` is a gap nobody will notice.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

PLACEHOLDER_REF = "[reference]"
PLACEHOLDER_AMOUNT = "[amount]"
PLACEHOLDER_DATE = "[date]"
PLACEHOLDER_CONTACT = "[contact name]"

# Document and order identifiers: PO-94103, INV 4490, RFQ-8812, Q-7732, QUT136586.
_REF_RE = re.compile(
    r"\b(?:PO|INV|RFQ|RFP|RFI|QUT|QT|Q|SO|CN|DN|REF)[-/ ]?\d{2,}[A-Z\d/-]*\b",
    re.IGNORECASE,
)

_CURRENCY = r"(?:[£$€¥]|\b(?:GBP|USD|EUR|CHF|AUD|CAD)\b)"
_MONEY_RE = re.compile(
    rf"{_CURRENCY}\s?\d[\d,]*(?:\.\d+)?(?:\s?[kKmM]\b)?"
    rf"|\b\d[\d,]*(?:\.\d+)?\s?{_CURRENCY}",
    re.IGNORECASE,
)

_MONTHS = (
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
)
_MONTH_ALT = "|".join(_MONTHS)
_DATE_RE = re.compile(
    rf"\b(?:\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTH_ALT})"       # 30 April
    rf"|(?:{_MONTH_ALT})\s+\d{{1,2}}(?:st|nd|rd|th)?"            # April 30
    rf"|\d{{4}}-\d{{2}}-\d{{2}}"                                  # 2026-04-30
    rf"|\d{{1,2}}/\d{{1,2}}(?:/\d{{2,4}})?)\b",                   # 30/04
    re.IGNORECASE,
)

_GREETING_RE = re.compile(
    r"^([ \t]*(?:Hi|Hello|Hey|Dear|Good morning|Good afternoon)[ \t]+)"
    r"([A-Z][\w'’-]+(?:[ \t]+[A-Z][\w'’-]+)?)",
    re.MULTILINE,
)


@dataclass
class GroundingResult:
    """A draft with its ungrounded facts replaced, and a record of what went."""

    subject: Optional[str]
    body: str
    replacements: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.replacements)

    @property
    def clean(self) -> bool:
        return not self.replacements


def _normalise(text: str) -> str:
    """Lowercase, strip separators, collapse whitespace.

    So ``PO-94103`` in a draft matches ``PO 94103`` in the task, and ``30 April`` matches
    ``30th April``. Grounding should turn on whether the fact was supplied, not on how it
    was punctuated.
    """

    lowered = (text or "").lower()
    lowered = re.sub(r"(\d)(?:st|nd|rd|th)\b", r"\1", lowered)
    lowered = re.sub(r"[-/,.\s]+", " ", lowered)
    return f" {lowered.strip()} "


def _is_grounded(token: str, task_norm: str) -> bool:
    candidate = _normalise(token).strip()
    return bool(candidate) and candidate in task_norm


def _replace_ungrounded(
    text: str,
    pattern: re.Pattern,
    placeholder: str,
    task_norm: str,
    replacements: List[Tuple[str, str]],
) -> str:
    def _sub(match: re.Match) -> str:
        token = match.group(0)
        if _is_grounded(token, task_norm):
            return token
        replacements.append((token, placeholder))
        return placeholder

    return pattern.sub(_sub, text)


def ground_draft(
    subject: Optional[str], body: str, task: str
) -> GroundingResult:
    """Replace any reference, amount, date or contact name the task did not supply."""

    task_norm = _normalise(task)
    replacements: List[Tuple[str, str]] = []

    def _clean(text: Optional[str]) -> Optional[str]:
        if not text:
            return text
        # Money before references: "$1,720" should not be read as an identifier.
        text = _replace_ungrounded(text, _MONEY_RE, PLACEHOLDER_AMOUNT, task_norm, replacements)
        text = _replace_ungrounded(text, _REF_RE, PLACEHOLDER_REF, task_norm, replacements)
        text = _replace_ungrounded(text, _DATE_RE, PLACEHOLDER_DATE, task_norm, replacements)
        return text

    cleaned_subject = _clean(subject)
    cleaned_body = _clean(body) or ""

    # The greeting is handled separately: the name is a group inside the match, and the
    # greeting word itself must survive because it is part of the style.
    def _greet(match: re.Match) -> str:
        opener, name = match.group(1), match.group(2)
        if _is_grounded(name, task_norm):
            return match.group(0)
        replacements.append((name, PLACEHOLDER_CONTACT))
        return f"{opener}{PLACEHOLDER_CONTACT}"

    cleaned_body = _GREETING_RE.sub(_greet, cleaned_body)

    if replacements:
        logger.info(
            "Grounding guard replaced %s ungrounded item(s) in a draft: %s",
            len(replacements),
            ", ".join(sorted({p for _, p in replacements})),
        )

    return GroundingResult(
        subject=cleaned_subject, body=cleaned_body, replacements=replacements
    )
