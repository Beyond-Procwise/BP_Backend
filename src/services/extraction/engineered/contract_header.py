"""A contract's title, governing law and payment terms, read from its own words.

The last three fields of the pattern-less group audited on 2026-10-04: nine, four
and four `canonical_labels` between them, no `patterns`, and NULL on all 7 live
contract rows while the documents state them.

THREE TRAPS, every one of them live in those seven documents:

1. **"governed by Framework Agreement No. FA-2026-0042"** appears on two of the
   seven. That is an incorporation clause, not a choice of law -- the same
   cited-agreement trap that gave an order form its framework's start date. So
   "the laws of ..." (or "<Adjective> law") is mandatory, and `governed by` alone
   never answers this field.
2. The real Marketing Agreement states **"will provide an invoice to the Client
   every 30 days"** -- an invoicing cadence -- and **"within a period of 10
   business days"** -- a cure period. Neither is a payment term. So a day count
   only counts when it hangs off payment wording.
3. **docling dropped that document's title**: its parsed text begins
   "## PARTIES". A section heading is not a title, so a heading must name a
   contract-family thing to be one -- and the document's own opening sentence
   ("This Marketing Agreement ...") is the second source, the only one that works
   for that file.

The title is stored in Title Case because ALL CAPS in a heading is typography
rather than the contract's name, and `proc.bp_contract_master` holds Title Case.
Payment terms are stored as the document's phrasing lightly normalised, which is
the convention already in the data: `proc.bp_purchase_order_raw` holds "Annual in
advance, 30 days" and `proc.bp_invoice_stg` holds "30 days - due 30 Jul 2025", so
"30 days" is the comparable core of both.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from ..types import Candidate, Span
from .contract_parties import CONFIDENCE, _squeeze, _tidy

TITLE_FIELD = "contract_title"
LAW_FIELD = "governing_law"
PAYMENT_FIELD = "payment_terms"

#: A heading only names this document if it names a contract-family thing. Kept
#: local rather than read from proc.bp_document_type: that vocabulary exists to
#: CLASSIFY a document, this is about what it calls itself, and a reader that
#: needs no database stays testable in both venvs.
_TITLE_WORDS = (
    "agreement", "contract", "order form", "statement of work", "sow",
    "schedule", "addendum", "variation", "change control note", "ccn",
    "non-disclosure", "nda", "memorandum", "mou", "service level agreement",
    "sla", "call-off", "call off", "framework", "licence", "license", "lease",
    "deed", "terms of business", "engagement letter", "purchase order",
)

_HEADING = re.compile(r"(?m)^#{1,6}\s*(?P<text>[^\n]{3,90})\s*$")

#: The bare "title" is DELIBERATELY absent, though contract.yaml lists it: in a
#: signature block "Title:" is the job title, and that is where the word appears
#: most. promoting_signed.pdf took "Managing Director Date: 1 March 2026 CLIENT
#: Name: Tom Okafor ..." as its contract title because of it. The document's own
#: heading already answers what a bare "Title:" would have.
_TITLE_LABEL = re.compile(
    r"(?:^|[\s\n])(?i:contract\s+title|agreement\s+title|contract\s+name|"
    r"agreement\s+name|subject)\s*[:–]\s*(?P<value>[^\n]{3,90})"
)

#: The next field on the same line. The parser collapses a header block onto one
#: line, so a labelled value has to be bounded the way every other one is.
_NEXT_HEADER_FIELD = re.compile(
    r"\s+(?i:effective\s+date|start\s+date|end\s+date|expiry\s+date|date|"
    r"supplier|buyer|client|customer|vendor|contract\s+no|contract\s+number|"
    r"reference|ref|name|title|signature|signed|payment\s+terms|currency|"
    r"total|value)\s*[:–]"
)

#: "This Marketing Agreement (hereinafter ...) is entered into on ..." -- the
#: document naming itself in its operative sentence.
_SELF_NAMING = re.compile(
    r"(?i)\bthis\s+(?P<name>[A-Z][\w\-]*(?:\s+[A-Z][\w\-]*){0,4}?\s+"
    r"(?:agreement|contract|order\s+form|statement\s+of\s+work|schedule|"
    r"addendum|variation|deed|licence|license|lease))\b"
)

#: A choice of law. "the laws of X" or "<Adjective> law" -- never a bare
#: "governed by", which on this corpus points at another agreement twice.
_LAW_PROSE = re.compile(
    r"(?i)(?:governed\s+by|subject\s+to|construed\s+in\s+accordance\s+with|"
    r"in\s+accordance\s+with)\s+(?:and\s+construed\s+in\s+accordance\s+with\s+)?"
    r"(?:the\s+)?laws?\s+of\s+(?P<value>(?:the\s+)?[A-Z][\w\-]*"
    r"(?:\s+(?:of|and|the)?\s*[A-Z][\w\-]*){0,4})"
)
_LAW_ADJECTIVE = re.compile(
    r"(?i)governed\s+by\s+(?P<value>[A-Z][\w\-]*(?:\s+[A-Z][\w\-]*){0,2}\s+law)\b"
)
_LAW_LABEL = re.compile(
    r"(?:^|[\s\n])(?i:governing\s+law|applicable\s+law|choice\s+of\s+law)"
    r"\s*[:–]\s*(?P<value>[^\n]{2,60})"
)

#: A day count that hangs off PAYMENT wording. "every 30 days" (a cadence) and
#: "within a period of 10 business days" (a cure period) must not match.
_PAYMENT_DAYS = re.compile(
    r"(?i)\b(?:payment|pay|payable|invoices?)\b[^.\n]{0,60}?"
    r"\b(?:within|after|from)\s+(?P<days>\d{1,3})\s+(?:calendar\s+|working\s+|business\s+)?days"
)
_PAYMENT_LABEL = re.compile(
    r"(?:^|[\s\n])(?i:payment\s+terms|terms\s+of\s+payment|payment\s+conditions|"
    r"net\s+terms)\s*[:–]\s*(?P<value>[^\n]{2,60})"
)


@dataclass(frozen=True)
class ContractHeader:
    title: Optional[str] = None
    governing_law: Optional[str] = None
    payment_terms: Optional[str] = None
    title_text: Optional[str] = None
    law_text: Optional[str] = None
    payment_text: Optional[str] = None


def _title_case(text: str) -> str:
    """"FRAMEWORK AGREEMENT" -> "Framework Agreement"; mixed case left alone."""
    clean = _squeeze(text).strip(" .:#")
    if clean.isupper():
        return " ".join(
            "-".join(p[:1].upper() + p[1:].lower() for p in word.split("-"))
            for word in clean.split()
        )
    return clean


def _names_a_contract(text: str) -> bool:
    low = _squeeze(text).lower()
    return any(word in low for word in _TITLE_WORDS)


def _read_title(full_text: str) -> tuple[Optional[str], Optional[str]]:
    m = _TITLE_LABEL.search(full_text)
    if m:
        raw = m.group("value")
        nxt = _NEXT_HEADER_FIELD.search(raw)
        value = _tidy(raw[:nxt.start()] if nxt else raw)
        if value:
            return (_title_case(value), _squeeze(m.group(0)))

    for m in _HEADING.finditer(full_text):
        raw = m.group("text")
        # A numbered clause heading ("3. PAYMENT TERMS") is a section, not a name.
        if re.match(r"^\s*\d+[\.\)]", raw):
            continue
        if _names_a_contract(raw):
            return (_title_case(raw), _squeeze(m.group(0)))

    m = _SELF_NAMING.search(full_text)
    if m:
        return (_title_case(m.group("name")), _squeeze(m.group(0)))
    return (None, None)


def _read_law(full_text: str) -> tuple[Optional[str], Optional[str]]:
    for pattern in (_LAW_LABEL, _LAW_PROSE, _LAW_ADJECTIVE):
        m = pattern.search(full_text)
        if not m:
            continue
        value = _tidy(m.group("value"))
        if value and _names_a_contract(value):
            # "governed by Framework Agreement No. FA-1" reaching this far means
            # the pattern matched something that is a DOCUMENT, not a law.
            continue
        if value:
            return (value, _squeeze(m.group(0)))
    return (None, None)


def _read_payment(full_text: str) -> tuple[Optional[str], Optional[str]]:
    m = _PAYMENT_LABEL.search(full_text)
    if m:
        value = _tidy(m.group("value"))
        if value:
            return (value, _squeeze(m.group(0)))

    m = _PAYMENT_DAYS.search(full_text)
    if m:
        # Prose normalises to the comparable core, "N days". Only a LABELLED value
        # is kept verbatim ("Net 30", "45 days from invoice date"), because there
        # the document is filling in a field rather than writing a sentence.
        return (f"{m.group('days')} days", _squeeze(m.group(0)))
    return (None, None)


def read_header(full_text: str) -> ContractHeader:
    """Title, governing law and payment terms -- each one, or None."""
    if not full_text:
        return ContractHeader()
    title, title_text = _read_title(full_text)
    law, law_text = _read_law(full_text)
    pay, pay_text = _read_payment(full_text)
    return ContractHeader(title=title, governing_law=law, payment_terms=pay,
                          title_text=title_text, law_text=law_text,
                          payment_text=pay_text)


def header_candidates(full_text: str) -> list[Candidate]:
    h = read_header(full_text)
    out: list[Candidate] = []
    for field, value, literal in ((TITLE_FIELD, h.title, h.title_text),
                                  (LAW_FIELD, h.governing_law, h.law_text),
                                  (PAYMENT_FIELD, h.payment_terms, h.payment_text)):
        if not value:
            continue
        out.append(Candidate(
            field=field,
            value=value,
            span=Span(page=0, bbox=(0.0, 0.0, 0.0, 0.0), text=literal or value),
            source="regex",
            pattern_name="contract_header_clause",
            confidence=CONFIDENCE,
        ))
    return out


__all__ = ["ContractHeader", "read_header", "header_candidates",
           "TITLE_FIELD", "LAW_FIELD", "PAYMENT_FIELD"]


# ---------------------------------------------------------------------------
# Correcting rows stored before anything read a contract's header.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HeaderCorrection:
    title: Optional[str]
    governing_law: Optional[str]
    payment_terms: Optional[str]
    changed: bool
    reason: str


def decide_header_correction(
    *,
    full_text: str,
    stored_title: Optional[str],
    stored_law: Optional[str],
    stored_payment: Optional[str],
    provenance_source: Optional[str],
) -> HeaderCorrection:
    """Should this row's header fields change, and to what?

    The same three rules as the term and the value. These fields were never
    produced by the entity sweep either, so there is no wrong value of its to
    clear -- and a field this reader cannot find KEEPS whatever is stored,
    because the context layer may have grounded a shape these patterns miss.
    """
    from .contract_parties import HUMAN_SOURCE

    if (provenance_source or "").lower() == HUMAN_SOURCE:
        return HeaderCorrection(stored_title, stored_law, stored_payment, False,
                                "a human confirmed these values; nothing overrules that")
    if not (full_text or "").strip():
        return HeaderCorrection(stored_title, stored_law, stored_payment, False,
                                "no stored text to re-read; left as found")

    h = read_header(full_text)
    title = h.title or stored_title
    law = h.governing_law or stored_law
    payment = h.payment_terms or stored_payment
    changed = ((title != stored_title) or (law != stored_law)
               or (payment != stored_payment))
    if not changed:
        return HeaderCorrection(title, law, payment, False,
                                "read from the document and already stored correctly")
    return HeaderCorrection(title, law, payment, True,
                            "read from the document's own heading, law clause and "
                            "payment wording")


__all__ += ["HeaderCorrection", "decide_header_correction"]
