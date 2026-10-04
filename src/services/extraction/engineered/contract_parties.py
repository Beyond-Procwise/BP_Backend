"""Who the two parties to a contract are, read from the contract's own words.

WHY THIS EXISTS. On 2026-10-03 the first six contract documents this product ever
ingested all stored the WRONG party: `supplier_id` held the buyer's name on five of
them and the sentence fragment "Framework Agreement No" on the sixth, and
`buyer_org_id` held the same value as `supplier_id` on all six.

The cause was not the model. `SpacyNERExtractor.produce_candidates` has party-aware
branches for exactly two field NAMES -- `supplier_name` (header position + a
buyer-context filter) and `buyer_id` (the BILL TO block) -- both shaped for an
invoice or a purchase order. The contract schema names its party fields
`supplier_id` and `buyer_org_id`, so neither branch matched and both fields fell
through to the default path, which emits EVERY entity of the required type for ANY
field. Two fields both asking for an ORG therefore received identical candidate
lists, headed by whichever company the "between A and B" sentence named first --
normally the buyer -- and on the real Marketing Agreement the list also contained
`Services`, `LIABILITY`, `Arbitration`, `SEVERABILITY` and `Bank Transfer to`.

A contract does not have a masthead or a BILL TO block. It has a party clause, and
it says in words which party is which. So the fix is not a better guess: it is to
read the clause. Two shapes cover every contract shape seen so far:

    Supplier: NexaSpark Marketing Ltd., 125 Innovation Park, London, UK
    ... between BrightWave Digital Ltd. (hereinafter referred to as the "Client")
        and NexaSpark Marketing Ltd. (hereinafter referred to as the "Marketer") ...

The labels are not invented here: they are the `canonical_labels` that
`extraction_schemas/contract.yaml` has always declared for `supplier_id` and
`buyer_org_id` and that nothing has ever read.

WHAT IT WILL NOT DO. If the document does not say, this returns nothing, and the
field stays NULL for the context layer to ground or for `missing_required` to flag.
"The first ORG in the document" was wrong five times out of six, and two wrong
party fields are worse than two empty ones -- see
`feedback_no_fabrication_null_when_absent`. For the same reason, a document that
appears to name the SAME company as both parties yields neither: that is a read
error, not two facts.

Called from dispatch.py for doc_type == "contract", BEFORE fill_ner_gaps, whose
candidates it therefore pre-empts (the gap-filler only fills fields nothing else
claimed).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from ..types import Candidate, Span

#: Confidence: above the NER sweep's 0.7 so a read clause beats a guessed entity,
#: below an L1 regex hit so an explicit pattern still wins.
CONFIDENCE = 0.88

SUPPLIER_FIELD = "supplier_id"
BUYER_FIELD = "buyer_org_id"

# The labels extraction_schemas/contract.yaml already declares, plus the obvious
# synonyms. Longest first, so "Service Provider" is not read as "Provider".
_SUPPLIER_LABELS = (
    "service provider", "sub-contractor", "subcontractor", "supplier", "vendor",
    "contractor", "consultant", "licensor", "provider", "seller",
)
_BUYER_LABELS = (
    "buyer", "client", "customer", "purchaser", "licensee",
)

# Role words a definition clause may use. "Company" is deliberately absent: it is
# the buyer in an employment contract and the supplier in a services one, and
# guessing which is exactly the failure this module exists to stop.
_SUPPLIER_ROLES = frozenset({
    "supplier", "vendor", "contractor", "sub-contractor", "subcontractor",
    "consultant", "consultancy", "service provider", "provider", "seller",
    "licensor", "marketer", "agency", "adviser", "advisor", "contractor party",
})
_BUYER_ROLES = frozenset({
    "buyer", "client", "customer", "purchaser", "licensee",
})

#: A parenthetical that names a party's role: `("the Supplier")`,
#: `(hereinafter referred to as the "Marketer")`, `(the 'Client')`.
_ROLE_CLAUSE = re.compile(
    # `the` may sit on either side of the quote: `as the "Marketer"` and `("the Client")`.
    r"\(\s*(?:hereinafter\s+)?(?:referred\s+to\s+as\s+)?(?:the\s+)?"
    r"[\"'“”‘’]?\s*(?:the\s+)?([A-Za-z][A-Za-z \-]{1,28}?)\s*"
    r"[\"'“”‘’]?\s*\)",
    re.IGNORECASE,
)

#: The company name immediately to the LEFT of a role clause. Capitalised tokens
#: only, so "...London, UK and NexaSpark Marketing Ltd." stops at the lowercase
#: "and" and yields "NexaSpark Marketing Ltd." rather than "UK and NexaSpark...".
#: "of" and "&" are allowed through because "Bank of Scotland" is a company and
#: "and" is not part of one often enough to be worth the false positives.
_NAME_TAIL = re.compile(
    r"([A-Z][\w&.'’-]*(?:\s+(?:of|&|[A-Z][\w&.'’-]*)){0,5})\s*$"
)

#: Legal-form tokens. A comma followed by one of these is part of the name
#: ("Acme, Inc."); any other comma starts the address.
_SUFFIX_AFTER_COMMA = re.compile(
    r"^\s*(?:inc|inc\.|llc|l\.l\.c\.|ltd|ltd\.|limited|plc|p\.l\.c\.|llp|gmbh|"
    r"s\.a\.|s\.a|sa|bv|b\.v\.|nv|n\.v\.|pty|ag|oy|ab|as|aps|srl|spa)\b",
    re.IGNORECASE,
)

#: Legal-form abbreviations whose trailing period is PART OF THE NAME. "Ltd." must
#: keep its dot (it is how the company writes itself and how the supplier table
#: holds it); a sentence-ending period after "Council" must not be kept.
_LEGAL_ABBREV = frozenset({
    "ltd", "inc", "llc", "plc", "llp", "co", "corp", "gmbh", "bv", "nv", "pty",
    "ag", "oy", "ab", "as", "aps", "srl", "spa", "sa", "pte", "kk", "oyj",
})

#: Words that are never a party name on their own. Measured: every one of these was
#: emitted as a party candidate by the NER sweep on 2026-10-03.
_NOT_A_PARTY = frozenset({
    "agreement", "parties", "party", "services", "service", "liability",
    "arbitration", "severability", "confidentiality", "termination", "term",
    "payment", "fees", "schedule", "annex", "appendix", "exhibit", "effective date",
    "framework agreement no", "order form", "purchase order", "invoice", "quote",
    "bank transfer to", "client account", "the effective date",
})


@dataclass(frozen=True)
class Parties:
    """What the contract says. Either side may be None -- that is a real answer."""

    supplier: Optional[str] = None
    buyer: Optional[str] = None
    supplier_evidence: Optional[str] = None
    buyer_evidence: Optional[str] = None


#: Every party-role word, longest first so "service provider" is not read as
#: "provider". Exposed because a signature block runs its label straight into the
#: next word ("SUPPLIER Name: ...") when the parser drops the line breaks, so the
#: only reliable way to find the label is to look for these words themselves.
ROLE_WORDS: tuple[str, ...] = tuple(
    sorted(_SUPPLIER_ROLES | _BUYER_ROLES, key=len, reverse=True)
)


def side_for_role(role: str | None) -> Optional[str]:
    """``'supplier'`` / ``'buyer'`` / ``None`` for a party-role word.

    Shared with contract_signatories: a signature block labels its sections with
    exactly the words the party clause uses ("MARKETER", "CLIENT"), so one
    vocabulary serves both and they cannot drift apart.
    """
    word = _squeeze(role).lower().strip(" .:")
    if word in _SUPPLIER_ROLES:
        return "supplier"
    if word in _BUYER_ROLES:
        return "buyer"
    return None


def _squeeze(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _tidy(name: str) -> str:
    """Trim punctuation a sentence left behind, but never a legal form's own dot."""
    out = _squeeze(name).strip().strip(",;:\u2013\u2014 ")
    if out.endswith("."):
        last = out[:-1].split()[-1].lower().rstrip(".") if out[:-1].split() else ""
        if last not in _LEGAL_ABBREV:
            out = out[:-1].rstrip()
    return out


def _plausible_name(name: str) -> bool:
    """Is this a company name at all, rather than a heading or a clause word?"""
    squeezed = _tidy(name)
    if len(squeezed) < 3 or len(squeezed) > 120:
        return False
    if squeezed.lower() in _NOT_A_PARTY:
        return False
    # A name has at least one letter and is not a bare number or a date.
    if not re.search(r"[A-Za-z]", squeezed):
        return False
    return True


def _trim_address(value: str) -> str:
    """A party line usually carries the address on the same line. Cut at the first
    comma, unless what follows the comma is a legal form ("Acme, Inc.")."""
    out = value
    while True:
        i = out.find(",")
        if i < 0:
            break
        if _SUFFIX_AFTER_COMMA.match(out[i + 1:]):
            # keep this comma and look for the next one after the suffix
            nxt = out.find(",", i + 1)
            if nxt < 0:
                break
            out = out[:nxt]
            continue
        out = out[:i]
        break
    return _tidy(out) or _tidy(value)


#: The start of the NEXT field on the same line. The parser collapses a contract's
#: whole party block onto one line, so without this the value for "Supplier:" runs
#: on through the address and into "Effective Date: ...". Only KNOWN labels count:
#: a generic "Capitalised Words:" pattern cut "Westminster City Council Buyer:"
#: down to "Westminster", because "City Council Buyer:" matched it.
_NEXT_LABEL = re.compile(
    r"\s+(?i:supplier|buyer|vendor|client|customer|purchaser|contractor|seller|"
    r"licensee|licensor|service provider|consultant|effective date|end date|"
    r"start date|commencement date|date|address|registered office|contact|email|"
    r"telephone|phone|reference|ref|term|charges|total|signed|company number)"
    r"\s*:\s"
)


def _label_pattern(labels: tuple[str, ...]) -> re.Pattern:
    alts = "|".join(re.escape(lbl) for lbl in labels)
    # The COLON is the label's signature, not the line start. Measured 2026-10-03:
    # docling collapses "Buyer: X ... Supplier: Y ... Effective Date: Z" onto a
    # single line, so a line-anchored label matched nothing on a real document.
    # Prose is excluded by the colon itself -- "the Supplier may be asked to
    # provide" has none -- and a drafting colon ("If the Supplier: (a) fails") is
    # excluded by the value having to look like a name.
    return re.compile(
        rf"(?:^|[\s\n])(?:[-*\u2022]\s*)?(?i:{alts})\s*(?:\(s\))?\s*[:\u2013]\s*"
        rf"(?P<value>[A-Z0-9][^\n]*)"
    )


_SUPPLIER_LABEL_RE = _label_pattern(_SUPPLIER_LABELS)
_BUYER_LABEL_RE = _label_pattern(_BUYER_LABELS)


def _from_labels(text: str, pattern: re.Pattern) -> tuple[Optional[str], Optional[str]]:
    for m in pattern.finditer(text):
        raw = m.group("value").split("\n")[0]
        # Stop at the next field on the line, then at a column gap, then let
        # _trim_address deal with the comma that starts the postal address.
        nxt = _NEXT_LABEL.search(raw)
        if nxt:
            raw = raw[:nxt.start()]
        raw = re.split(r"\s{2,}", raw)[0]
        name = _trim_address(_squeeze(raw))
        if _plausible_name(name):
            return name, _squeeze(m.group(0))
    return None, None


def _from_role_clauses(text: str) -> dict[str, tuple[str, str]]:
    """{'supplier': (name, evidence)} for every role clause whose role is known."""
    found: dict[str, tuple[str, str]] = {}
    for m in _ROLE_CLAUSE.finditer(text):
        role = _squeeze(m.group(1)).lower().strip(" .")
        side = side_for_role(role)
        if side is None or side in found:
            continue
        # Look left for the company name. The window is collapsed first, because a
        # PDF breaks "BrightWave\nDigital Ltd." across lines.
        window = _squeeze(text[max(0, m.start() - 160):m.start()])
        tail = _NAME_TAIL.search(window)
        if not tail:
            continue
        name = _trim_address(tail.group(1))
        if _plausible_name(name):
            found[side] = (name, _squeeze(m.group(0)))
    return found


def read_parties(full_text: str) -> Parties:
    """The contract's two parties, or None for either side it does not state.

    A label wins over a definition clause: a labelled field is a statement about a
    party, while the clause is prose that happens to define one.
    """
    if not full_text:
        return Parties()

    sup, sup_ev = _from_labels(full_text, _SUPPLIER_LABEL_RE)
    buy, buy_ev = _from_labels(full_text, _BUYER_LABEL_RE)

    clauses = _from_role_clauses(full_text)
    if sup is None and "supplier" in clauses:
        sup, sup_ev = clauses["supplier"]
    if buy is None and "buyer" in clauses:
        buy, buy_ev = clauses["buyer"]

    # One company cannot be both sides of its own contract. Two wrong fields are
    # worse than two empty ones, and this is the exact symptom being fixed.
    if sup and buy and _squeeze(sup).lower() == _squeeze(buy).lower():
        return Parties()

    return Parties(supplier=sup, buyer=buy,
                   supplier_evidence=sup_ev, buyer_evidence=buy_ev)


def _span_for(name: str, full_text: str) -> Span:
    """The literal source substring for `name`, matched whitespace-flexibly so a
    name broken across lines still yields exact evidence."""
    flexible = r"\s+".join(re.escape(tok) for tok in name.split())
    m = re.search(flexible, full_text)
    literal = m.group(0) if m else name
    return Span(page=0, bbox=(0.0, 0.0, 0.0, 0.0), text=literal)


def party_candidates(full_text: str) -> list[Candidate]:
    """`supplier_id` / `buyer_org_id` candidates, by the names the CONTRACT schema
    uses -- not `supplier_name` / `buyer_id`, which are the invoice names and the
    reason the party branches never fired."""
    parties = read_parties(full_text)
    out: list[Candidate] = []
    for field, value in ((SUPPLIER_FIELD, parties.supplier),
                         (BUYER_FIELD, parties.buyer)):
        if not value:
            continue
        out.append(Candidate(
            field=field,
            value=value,
            span=_span_for(value, full_text),
            source="parties",
            pattern_name="contract_party_clause",
            confidence=CONFIDENCE,
        ))
    return out


__all__ = ["Parties", "read_parties", "party_candidates", "CONFIDENCE",
           "side_for_role", "ROLE_WORDS",
           "SUPPLIER_FIELD", "BUYER_FIELD", "Correction", "decide_correction",
           "BROKEN_SOURCE", "HUMAN_SOURCE"]


# ---------------------------------------------------------------------------
# Correcting rows that were written before the party clause was read.
#
# A backfill that cannot say WHY it changed a stored value is indistinguishable
# from a second bug, so the decision is here, next to the reader, and tested --
# not SQL inside a script.
# ---------------------------------------------------------------------------

#: The provenance source that produced the wrong values: the entity sweep. It is
#: the only source a backfill may overrule, because it is the one proven to hand
#: identical candidate lists to both party fields and to offer clause headings
#: ("Services", "LIABILITY") as companies.
BROKEN_SOURCE = "ner"

#: A human's correction outranks everything, including this.
HUMAN_SOURCE = "hitl"


@dataclass(frozen=True)
class Correction:
    """What a stored row should hold, and why."""

    supplier: Optional[str]
    buyer: Optional[str]
    changed: bool
    reason: str


def decide_correction(
    *,
    full_text: str,
    stored_supplier: Optional[str],
    stored_buyer: Optional[str],
    provenance_source: Optional[str],
) -> Correction:
    """Should this row's party fields change, and to what?

    Four rules, in order. Each one is a test in
    tests/services/extraction/test_contract_parties.py.

    1. A human-confirmed value is never touched.
    2. A row with no stored text cannot be re-read, so it is left alone. Guessing
       in that case is worse than skipping.
    3. If the document states its parties, they are the answer.
    4. If it does not, the stored value only goes when it came from the entity
       sweep -- the one path proven broken. The context layer reads the whole
       document and is grounding-checked, so a backfill has no standing to
       overrule it, and an absent provenance is not evidence of the sweep.
    """
    if (provenance_source or "").lower() == HUMAN_SOURCE:
        return Correction(stored_supplier, stored_buyer, False,
                          "a human confirmed this value; nothing overrules that")
    if not (full_text or "").strip():
        return Correction(stored_supplier, stored_buyer, False,
                          "no stored text to re-read; left as found")

    parties = read_parties(full_text)
    if parties.supplier or parties.buyer:
        changed = (parties.supplier != stored_supplier) or (parties.buyer != stored_buyer)
        return Correction(
            parties.supplier, parties.buyer, changed,
            "read from the document's own party clause"
            + ("" if changed else " and already stored correctly"),
        )

    if (provenance_source or "").lower() == BROKEN_SOURCE and (stored_supplier or stored_buyer):
        return Correction(
            None, None, True,
            "the document does not state its parties and the stored value came "
            "from the entity sweep, which had no way to tell a party from a "
            "clause heading; cleared rather than left as a fact",
        )

    return Correction(stored_supplier, stored_buyer, False,
                      "the document does not state its parties and the stored "
                      "value did not come from the sweep; left as found")
