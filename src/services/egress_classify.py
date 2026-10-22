"""What an outbound payload carries, in four cheap tags (1.20 c).

Before an external call leaves, the text it carries (the URL's query string and the body) is scanned
by four detectors. The result is a list of tags, recorded on the egress event, and a LEVEL that an
allow rule in the egress policy may cap (``max_classification``):

    public        nothing found (a supplier's name, generic words)
    commercial    money amounts, or document references (invoice / PO / quote numbers)
    confidential  email addresses, or a long verbatim passage (document text, a whole prompt)

This is deliberately a small, legible rule set rather than a model: a tag must be explainable to
the administrator who reads the event row. False negatives are possible (a price written as plain
words), which is why enforcement rests on the DESTINATION rules first and this cap second.

Cost: a few regexes over at most ``SCAN_CAP_CHARS`` characters; local destinations are never scanned.
"""
from __future__ import annotations

import json
import re
from typing import Any, List
from urllib.parse import unquote_plus, urlsplit

LEVELS = ("public", "commercial", "confidential")
LONG_TEXT_CHARS = 2000        # a payload this long is a passage, not a query
SCAN_CAP_CHARS = 65_536       # scan this much, no more: the cost must stay flat

# Every pattern is LINEAR in the input: starts are anchored (a lookbehind forbids starting inside a
# run) and repeats are bounded, so a 64 KB body of digits, commas or letters costs milliseconds, not
# minutes. A pattern that backtracks across a run would stall the outbound call it guards.
_EMAIL = re.compile(r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9.-]{1,255}\.[A-Za-z]{2,}")
_CCY = r"(?:GBP|USD|EUR|INR|AED|CHF|JPY|AUD|CAD)"
_NUM = r"\d{1,12}(?:,\d{3}){0,6}(?:\.\d{1,4})?"
# £1,234.56 | GBP 15877.50 | 1234.00 EUR | $99   (currency codes are upper-case: "5 cad" is not money)
_AMOUNT = re.compile(
    r"(?:(?:[£$€]|\b" + _CCY + r")\s?" + _NUM
    + r"|(?<![\d,.])" + _NUM + r"\s?" + _CCY + r"\b)")
# INV-2024-001 | PO 4500123 | QT-77A: an upper-case prefix, then a separator or a digit, then a token
# that CONTAINS a digit. "Portsmouth", "inventory", "so good", "RFQ for widgets" do not qualify.
_DOC_REF_CODE = re.compile(
    r"\b(?:INV|PO|PUR|QUO|QT|RFQ|RFP|SO|DN|GRN|CN)(?:[-_ ]|(?=\d))(?=[A-Za-z0-9-]{0,40}\d)[A-Za-z0-9][A-Za-z0-9-]{1,40}\b")
# "invoice 01-2024-002" | "purchase order 4500123" | "quote no. 77": a document word, then a token with a digit
_DOC_REF_WORD = re.compile(
    r"\b(?:invoice|purchase\s+order|quote|quotation|credit\s+note|delivery\s+note)\s+(?:no\.?\s*|number\s*|#\s*)?"
    r"(?=[A-Za-z0-9/-]{0,40}\d)[A-Za-z0-9][A-Za-z0-9/-]{0,40}\b", re.IGNORECASE)

_TAG_LEVEL = {"amount": "commercial", "document_ref": "commercial",
              "email": "confidential", "long_text": "confidential"}


def tags_for(text: str) -> List[str]:
    """The tags found in ``text`` (first ``SCAN_CAP_CHARS`` characters), in a fixed order."""
    if not text:
        return []
    total = len(text)
    text = text[:SCAN_CAP_CHARS]
    tags: List[str] = []
    if "@" in text and _EMAIL.search(text):
        tags.append("email")
    if _AMOUNT.search(text):
        tags.append("amount")
    if _DOC_REF_CODE.search(text) or _DOC_REF_WORD.search(text):
        tags.append("document_ref")
    if total > LONG_TEXT_CHARS:
        tags.append("long_text")
    return tags


def level_of(tags: List[str]) -> str:
    best = 0
    for t in tags:
        best = max(best, LEVELS.index(_TAG_LEVEL.get(t, "public")))
    return LEVELS[best]


def exceeds(level: str, cap: str) -> bool:
    """Is ``level`` above ``cap``? Unknown values read as the strictest reading."""
    li = LEVELS.index(level) if level in LEVELS else len(LEVELS)
    ci = LEVELS.index(cap) if cap in LEVELS else -1
    return li > ci


def _text_of(body: Any) -> str:
    if body is None:
        return ""
    try:
        if isinstance(body, (bytes, bytearray)):
            return bytes(body)[:4 * SCAN_CAP_CHARS].decode("utf-8", "replace")   # slice before decoding
        if isinstance(body, str):
            return body
        return json.dumps(body, sort_keys=True, default=str, ensure_ascii=False)
    except Exception:  # noqa: BLE001 - a body we cannot read is not classified, and never fails the call
        return ""


def classify(url: str, body: Any) -> List[str]:
    """Tags for one call: the URL's query string plus the ``json=`` / ``data=`` body. Never raises.

    Not scanned: ``params=`` (no caller uses it; a future one must put the query in the URL or
    extend this) and ``files=``. A file or generator passed as ``data`` is represented, not read."""
    try:
        query = unquote_plus(urlsplit(url).query or "")
    except Exception:  # noqa: BLE001
        query = ""
    return tags_for(query + "\n" + _text_of(body))
