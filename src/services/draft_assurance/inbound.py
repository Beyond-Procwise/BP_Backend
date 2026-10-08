"""Screening an INBOUND reply before anything is drafted against it.

Today: the payment-detail-change screen. A reply that asks for NEW or CHANGED bank or payment details is the classic invoice-redirection
fraud, and it arrives by email. The screen is deterministic (no model: a model can be talked out of it by the very email it reads),
biased toward flagging (a false alarm costs a person a glance, a miss costs a payment to a criminal), and defeats the usual disguises:
markup splitting a word, zero-width characters, spaced-out letters, capitals, odd whitespace.

It returns categories and keywords only. It never returns text from the email, so a bank detail in the reply cannot leak through it.
"""

from __future__ import annotations

import html
import re
from typing import Any, Dict, List, Set, Tuple

MAX_SCAN = 60_000          # characters scanned from each end of a very long mail: a phrase hidden at the end still counts
NEAR = 100                 # characters between a payment term and a change term for them to count as one request

_ZERO_WIDTH = re.compile("[​‌‍⁠﻿­᠎]")
_BLOCK_TAG = re.compile(r"</?(?:br|p|div|li|ul|ol|tr|td|th|table|h[1-6]|blockquote)\b[^>]{0,200}>", re.I)
_ANY_TAG = re.compile(r"<[^>]{0,300}>")
_SPACED_LETTERS = re.compile(r"(?<![a-z])(?:[a-z] ){2,}[a-z](?![a-z])")

# What is being changed: a payment instrument. Bare "account" is NOT one ("account manager", "cover the account").
_NOT_A_PERSON_OR_TEAM = r"(?!\s+(?:manager|executive|director|rep|representative|team|payable|receivable|management|owner|lead|officer))"
_OBJECT = re.compile(
    r"bank(?:ing)?\s+(?:account|details|information|info|instructions|transfer\s+details)"
    r"|bank\b(?!\s+holiday)"
    r"|account\s+(?:number|details|no\b)"
    r"|sort\s*code|\biban\b|\bswift\b|\bbic\b|routing\s+number|beneficiar(?:y|ies)"
    r"|remittance\s+(?:details|instructions|information)|payment\s+(?:details|instructions|information)|\bpayee\b"
    r"|wire\s+transfer|wire\s+the\s+(?:next|payment|funds)"
    r"|(?:alternative|alternate|new|different|revised|updated|other)\s+account" + _NOT_A_PERSON_OR_TEAM +
    r"|settlement\s+account"
)
# That it is being changed.
_CHANGE = re.compile(
    r"chang(?:e|ed|es|ing)\b|\bnew\b|updat(?:e|ed|es|ing)\b|revis(?:e|ed|ion)\b|replac(?:e|ed|ing|ement)\b|switch(?:ed|ing)?\b"
    r"|migrat(?:e|ed|ing)\b|amend(?:ed)?\b|alternat(?:e|ive)\b|different\b|\binstead\b|no\s+longer|going\s+forward|from\s+now\s+on"
    r"|disregard|\bmoved\b|\bclosed\b|previous\s+details|old\s+(?:account|details)"
)
_PRESSURE = re.compile(
    r"\burgent(?:ly)?\b|immediately|\basap\b|\btoday\b|right\s+away|without\s+delay|do\s+not\s+call|don'?t\s+call|do\s+not\s+contact"
    r"|do\s+not\s+verify|phones?\s+(?:are|is)\s+down|cannot\s+be\s+reached|keep\s+this\s+(?:confidential|between)|time[- ]sensitive"
)
_BANK_DETAIL = re.compile(
    r"\b[a-z]{2}\d{2}[a-z0-9]{11,30}\b"                       # IBAN-shaped
    r"|\b\d{2}-\d{2}-\d{2}\b"                                # UK sort code
    r"|account\s*(?:number|no\.?)\s*:?\s*\d{6,12}"
)


def _normalise(text: Any) -> str:
    if not isinstance(text, str) or not text:
        return ""
    if len(text) > 2 * MAX_SCAN:
        text = text[:MAX_SCAN] + "\n" + text[-MAX_SCAN:]
    text = _ZERO_WIDTH.sub("", text)
    text = _BLOCK_TAG.sub(" ", text)
    text = _ANY_TAG.sub("", text)                              # a tag INSIDE a word is removed, so b<b></b>ank is "bank"
    text = html.unescape(text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return _SPACED_LETTERS.sub(lambda m: m.group(0).replace(" ", ""), text)      # "b a n k" -> "bank"


def _gap(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return max(0, b[0] - a[1], a[0] - b[1])


def screen_payment_change(subject: Any = "", body: Any = "") -> Dict[str, Any]:
    """{"suspected", "kinds", "terms", "bank_details_present", "pressure"}. Never raises; non-text input is not flagged.

    ``suspected`` means a person must look before anything is drafted or sent against this reply.
    """

    text = _normalise(f"{subject if isinstance(subject, str) else ''}\n{body if isinstance(body, str) else ''}")
    objects = [(m.start(), m.end(), m.group(0)) for m in _OBJECT.finditer(text)]
    changes = [(m.start(), m.end(), m.group(0)) for m in _CHANGE.finditer(text)]
    kinds: Set[str] = set()
    terms: List[str] = []
    for o in objects:
        for c in changes:
            if _gap(o[:2], c[:2]) <= NEAR:
                kinds.add("payment_detail_change")
                for t in (o[2], c[2]):
                    t = t.strip()[:30]
                    if t and t not in terms:
                        terms.append(t)
    bank = bool(_BANK_DETAIL.search(text))
    pressure = bool(_PRESSURE.search(text))
    if bank and pressure:
        kinds.add("bank_details_with_pressure")
    return {"suspected": bool(kinds), "kinds": sorted(kinds), "terms": terms[:8],
            "bank_details_present": bank, "pressure": pressure}
