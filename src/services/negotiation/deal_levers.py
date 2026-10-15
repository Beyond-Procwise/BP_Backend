"""The levers a supplier actually raised in their own reply, quoted, or nothing.

The negotiation used to reach for a fixed menu -- Net-15, a 24-month term, a
volume tier -- whatever the deal was. A five-year renewal that turned on
exclusivity, CPI indexation and service credits got the same menu as a
three-month haulage lane. This reader lists a lever only because the supplier's
words contain it, quotes the sentence it came from, and carries a figure only
when one is written beside it. Nothing is inferred: a lever the supplier did
not mention is absent, and a lever with no stated figure has figure None.

Pattern matching only -- no model -- so the same words always give the same
levers, and every figure can be found verbatim in the reply.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Pattern, Tuple

from src.services.negotiation.reply_offer import own_words

LEVER_LABELS: Dict[str, str] = {
    "term": "Contract term",
    "exclusivity": "Exclusivity",
    "indexation": "Price indexation",
    "service_levels": "Service levels / credits",
    "renewal_termination": "Renewal / termination",
    "payment_terms": "Payment terms",
    "volume": "Volume commitment",
    "lead_time": "Lead time",
    "warranty": "Warranty",
}

_I = re.IGNORECASE
_NUM = r"\d+(?:\.\d+)?"
_WORDNUM = r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve)"
_DURATION = rf"\b{_WORDNUM}[\s-]*(?:years?|yrs?|months?)\b"

# Per lever: what shows the supplier raised it, and the figure that belongs to it.
# The figure is looked for only in the same clause as the lever's own words, so
# "CPI+2% indexation and service credits capped at 5%" gives indexation CPI+2%
# and service credits capped at 5%, never the other way round.
_RULES: Tuple[Tuple[str, Pattern, Optional[Pattern]], ...] = (
    ("term",
     re.compile(rf"{_DURATION}(?![\s-]*(?:warrant|guarantee|notice|lead))"
                r"(?=.*?\b(?:term|contract|agreement|commitment|renewal|framework)\b)"
                r"|\b(?:term|contract|agreement|commitment|renewal|framework)\b"
                rf"(?=.*?{_DURATION}(?![\s-]*(?:warrant|guarantee|notice|lead)))"
                r"|\bmulti[\s-]?year\b", _I),
     re.compile(rf"{_DURATION}(?![\s-]*(?:warrant|guarantee|notice|lead))|\bmulti[\s-]?year\b", _I)),
    ("exclusivity",
     re.compile(r"\bexclusiv\w*|\bsole[\s-](?:supplier|source|provider)\b", _I),
     None),
    ("indexation",
     re.compile(r"\b(?:cpih?|rpi|indexation|indexed|index[\s-]linked|inflation)\b"
                r"|\bannual (?:price )?(?:increase|uplift|adjustment|review)\b", _I),
     re.compile(rf"\b(?:cpih?|rpi)\s*[+-]\s*{_NUM}\s*%|\b(?:capped|cap|ceiling|floor)\s+(?:at\s+|of\s+)?{_NUM}\s*%"
                rf"|{_NUM}\s*%", _I)),
    ("service_levels",
     re.compile(r"\bslas?\b|\bservice[\s-]levels?\b|\bservice credits?\b|\buptime\b"
                r"|\bavailability target\b|\bkpis?\b|\bresponse times?\b", _I),
     re.compile(rf"\b(?:capped|cap|ceiling)\s+(?:at\s+|of\s+)?{_NUM}\s*%|{_NUM}\s*%"
                rf"|\b\d+\s*(?:hours?|hrs?)\b", _I)),
    ("renewal_termination",
     re.compile(r"\b(?:auto[\s-]?)?renew\w*|\bterminat\w*|\bnotice period\b|\bbreak clause\b"
                r"|\bevergreen\b|\broll[\s-]?over\b", _I),
     re.compile(r"\b\d+\s*(?:days?|months?)(?='?s?'?\s*(?:written\s+)?notice)"
                r"|(?<=notice period of )\d+\s*(?:days?|months?)", _I)),
    ("payment_terms",
     re.compile(r"\bnet[\s-]?\d+\b|\bpayment terms?\b|\bearly[\s-]payment\b"
                r"|\b\d+\s*days?\s+(?:from|after|of)\s+(?:date of\s+)?invoice\b", _I),
     re.compile(r"\bnet[\s-]?\d+\b|\b\d+\s*days?\s+(?:from|after|of)\s+(?:date of\s+)?invoice\b"
                rf"|{_NUM}\s*%(?=\s*(?:early|discount|settlement))", _I)),
    ("volume",
     re.compile(r"\bvolumes?\b|\bmoq\b|\bminimum (?:order|quantity|purchase)\b|\btiered\b"
                r"|\bvolume (?:discount|rebate|tier)s?\b", _I),
     re.compile(r"\b\d[\d,]*\s*(?:units?|pcs|pieces|tonnes|kg|loads|pallets|items)\b", _I)),
    ("lead_time",
     re.compile(r"\blead[\s-]?times?\b|\bdeliver\w*\s+(?:with)?in\s+\d+", _I),
     re.compile(r"\b\d+\s*(?:working\s+|business\s+)?(?:days?|weeks?)\b", _I)),
    ("warranty",
     re.compile(r"\bwarrant(?:y|ies)\b|\bdefects liability\b", _I),
     re.compile(rf"{_DURATION}(?=[\s-]*(?:warrant|defects))|(?<=warranty of ){_DURATION}", _I)),
)

_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
# Where one ask ends and the next begins inside a sentence.
_CLAUSE = re.compile(r",|;|\s+and\s+|\s+but\s+", _I)


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE.split(text) if s and s.strip()]


def _clauses(sentence: str) -> List[str]:
    return [c.strip() for c in _CLAUSE.split(sentence) if c and c.strip()]


def read_levers(text: Optional[str]) -> List[dict]:
    """Each lever the supplier raised, once, in order of first mention.

    Returns dicts with ``lever`` (a key of LEVER_LABELS), ``label``, ``phrase``
    (the sentence it was raised in, verbatim) and ``figure`` (the figure written
    beside it, verbatim, or None).
    """
    words = own_words(text)
    if not words:
        return []
    found: Dict[str, dict] = {}
    order: List[Tuple[int, int, str]] = []
    for s_index, sentence in enumerate(_sentences(words)):
        for c_index, clause in enumerate(_clauses(sentence)):
            for lever, detect, figure_re in _RULES:
                if lever in found:
                    continue
                hit = detect.search(clause)
                if not hit:
                    continue
                figure = None
                if figure_re is not None:
                    m = figure_re.search(clause)
                    figure = m.group(0).strip() if m else None
                found[lever] = {"lever": lever, "label": LEVER_LABELS[lever],
                                "phrase": sentence, "figure": figure}
                order.append((s_index, c_index * 1000 + hit.start(), lever))
    return [found[lever] for _, _, lever in sorted(order)]
