"""Who signed the contract, read from its signature block.

WHY THIS EXISTS. `contract_signatory_name` held **"Email Marketing"** on the real
Marketing Agreement (measured 2026-10-03). Same root cause as the party fields:
`SpacyNERExtractor`'s default path emits every entity of the required type for any
field, so a PERSON-typed field got whatever spaCy first mis-tagged as a person --
here a line from the services list. One field, so no duplicate value gave it away
the way `supplier_id == buyer_org_id` did for the parties.

The document said it plainly and nothing read it:

    SIGNATURE AND DATE
    MARKETER
    Name: John Smith      Signature: ____________  Date: June 12, 2025
    CLIENT
    Name: Sarah Johnson   Signature: ____________  Date: June 12, 2025

THE RULING. `proc.bp_contracts` has ONE signatory field and a contract has two
signatories. The field holds the **supplier's** signatory: this is a procurement
system, and the question a single slot must answer is "who bound the counterparty".
The buyer's signatory is read (it is what attributes the other one) but not stored,
because the schema has nowhere to put it and adding a column is a larger change than
this. `read_signatory().buyer_name` exposes it for whoever adds that column.

Three refusals, all tested:
  * a signature RULE is not a name -- docling renders the line as escaped
    underscores, and "\\_\\_\\_\\_" was a candidate the sweep would have taken;
  * a date is not a name ("Name: June 12, 2025" happens when a block is empty);
  * a company is not a signatory ("For and on behalf of NexaSpark Marketing Ltd.").

And one more, which is the difference between this and a guess: a block naming
several signatories none of which can be attributed to a party yields NOTHING.
Ambiguity is not a fact -- see `feedback_no_fabrication_null_when_absent`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from ..types import Candidate, Span
from .contract_parties import (
    CONFIDENCE, ROLE_WORDS, _squeeze, _tidy, side_for_role,
)

NAME_FIELD = "contract_signatory_name"       # the SUPPLIER's; see the migration
ROLE_FIELD = "contract_signatory_role"
BUYER_NAME_FIELD = "buyer_signatory_name"
BUYER_ROLE_FIELD = "buyer_signatory_role"

#: Where the signatures start. A contract's signature block is always announced.
_SECTION = re.compile(
    r"(?i)\b(?:signature(?:s)?(?:\s+and\s+date)?|signed\s+(?:by|for|on\s+behalf)|"
    r"in\s+witness\s+whereof|executed\s+(?:by|as)|signatories|"
    r"agreed\s+and\s+accepted)\b"
)

#: `Name: John Smith`, and the labels extraction_schemas/contract.yaml declares.
#: The LABEL only -- no value group. A greedy `[^\\n]*` value swallowed the
#: rest of the line, so when the parser puts a whole signature block on one
#: line ("SUPPLIER Name: Priya Raman ... CLIENT Name: Tom Okafor ...")
#: finditer never saw the second signatory and the buyer's column stayed
#: empty on a document that names both. Each signatory's span is bounded by
#: the NEXT label instead, in read_signatory.
_NAME_LABEL = re.compile(
    r"(?:^|[\s\n])(?i:authorised\s+signatory|authorized\s+signatory|signed\s+by|"
    r"authorised\s+by|authorized\s+by|signatory|print\s+name|name)"
    r"\s*[:\u2013]\s*"
)

#: `Title: Managing Director` and its synonyms, for the job title.
_ROLE_LABEL = re.compile(
    r"(?:^|[\s\n])(?i:job\s+title|designation|position|title|role)"
    r"\s*[:–]\s*(?P<value>[^\n]*)"
)

#: The next field on the same line. A signature block is one line per signatory:
#: "Name: John Smith Signature: ____ Date: June 12, 2025".
_NEXT_FIELD = re.compile(
    r"\s+(?i:signature|signed|name|print name|title|role|position|designation|"
    r"date|dated|for and on behalf of|company|witness|email|telephone)\s*[:–]"
)

#: A person's name: two to four capitalised words, optional initials and particles.
_PERSON = re.compile(
    r"^(?:[A-Z][A-Za-z'’\-]+|[A-Z]\.)(?:\s+(?:de|van|von|der|den|bin|al|"
    r"[A-Z][A-Za-z'’\-]+|[A-Z]\.)){1,3}$"
)

#: Legal forms: a company is not a signatory.
_COMPANY = re.compile(
    r"(?i)\b(?:ltd|limited|llc|plc|llp|inc|incorporated|gmbh|bv|nv|pty|corp|"
    r"corporation|company|co|group|holdings|services|solutions|partners|"
    r"associates|trust|council|authority)\b\.?"
)

_MONTHS = ("january", "february", "march", "april", "may", "june", "july",
           "august", "september", "october", "november", "december")


@dataclass(frozen=True)
class Signatory:
    """Both signatories a contract carries.

    `name` / `role` are the SUPPLIER's -- the pair that lands in
    contract_signatory_name / _role, named without a supplier_ prefix for the
    historical reason set out in
    deploy/sql/2026-10-04_contract_buyer_signatory.sql. `buyer_name` / `buyer_role`
    land in buyer_signatory_name / _role, which that migration added: before it,
    the buyer's signatory was parsed and then discarded for want of a column.
    """

    name: Optional[str] = None
    role: Optional[str] = None
    party: Optional[str] = None
    buyer_name: Optional[str] = None
    buyer_role: Optional[str] = None
    evidence: Optional[str] = None


def _is_person(value: str) -> bool:
    v = _tidy(value)
    if not v or len(v) > 70:
        return False
    if "_" in v or "\\" in v:
        return False                       # a signature rule, not a name
    if _COMPANY.search(v):
        return False                       # the party, not the person
    if any(mon in v.lower() for mon in _MONTHS) or re.search(r"\d", v):
        return False                       # a date, not a name
    return bool(_PERSON.match(v))


def _cut(raw: str) -> str:
    """One signatory is one line: stop at the next field on it."""
    m = _NEXT_FIELD.search(raw)
    return raw[:m.start()] if m else raw


#: The role words as whole words, longest first. Built from contract_parties'
#: vocabulary so the two readers cannot disagree about what "Marketer" means.
_ROLE_WORD_RE = re.compile(
    r"(?i)\b(?:" + "|".join(re.escape(w) for w in ROLE_WORDS) + r")\b"
)


def _party_label_before(text: str, pos: int) -> Optional[str]:
    """The party whose sub-block this position sits in, if it is labelled.

    Looks back over the preceding 200 characters for the LAST party-role word --
    "MARKETER", "CLIENT", "SUPPLIER" -- which is how a signature block separates
    its two halves.

    Scans for the role WORDS rather than for capitalised runs. The first version
    did the latter and worked only because the real Marketing Agreement keeps a
    blank line after each label: when the parser drops those (measured 2026-10-04
    on a generated contract), the block arrives as
    "SUPPLIER Name: Priya Raman ... CLIENT Name: Tom Okafor ...", a greedy run
    matches the phrase "SUPPLIER Name", that is in no vocabulary, and NEITHER
    side gets attributed -- so the buyer's column stayed empty on a document that
    names both signatories plainly.
    """
    window = text[max(0, pos - 200):pos]
    best: Optional[str] = None
    for m in _ROLE_WORD_RE.finditer(window):
        side = side_for_role(m.group(0))
        if side:
            best = side
    return best


def read_signatory(full_text: str) -> Signatory:
    """The supplier's signatory, or nothing. Never a guess."""
    if not full_text:
        return Signatory()
    sec = _SECTION.search(full_text)
    if not sec:
        return Signatory()
    region_start = sec.start()
    region = full_text[region_start:]

    # Every name label, then each one's span: from its own end to the start of the
    # next label (or 300 characters, whichever is sooner). That bound is what makes
    # a one-line block readable, and it is also what stops one signatory's "Title:"
    # being read as the other's.
    labels = list(_NAME_LABEL.finditer(region))
    found: list[tuple[Optional[str], str, str, int, int]] = []
    for i, m in enumerate(labels):
        stop = labels[i + 1].start() if i + 1 < len(labels) else len(region)
        span = region[m.end():min(stop, m.end() + 300)]
        value = _tidy(_cut(span.split("\n")[0]))
        if not _is_person(value):
            continue
        party = _party_label_before(region, m.start())
        found.append((party, value, _squeeze(m.group(0) + value), m.end(), stop))

    if not found:
        return Signatory()

    supplier = next((f for f in found if f[0] == "supplier"), None)
    buyer = next((f for f in found if f[0] == "buyer"), None)

    if supplier is None and buyer is None and len(found) == 1:
        # One signatory, unattributed: that is who signed, and nothing says it is
        # the buyer's, so the buyer's field stays empty rather than guessing.
        _party, name, ev, start, stop = found[0]
        return Signatory(name=name, role=_role_in(region, start, stop), party=None,
                         evidence=ev)

    buyer_name = buyer[1] if buyer else None
    buyer_role = _role_in(region, buyer[3], buyer[4]) if buyer else None

    if supplier is None:
        # Only the other side signed this copy, or nothing could be attributed.
        # Storable now (buyer_signatory_name), and still NOT the supplier's.
        return Signatory(buyer_name=buyer_name, buyer_role=buyer_role,
                         evidence=buyer[2] if buyer else None)

    return Signatory(name=supplier[1], role=_role_in(region, supplier[3], supplier[4]),
                     party="supplier", buyer_name=buyer_name,
                     buyer_role=buyer_role, evidence=supplier[2])


def _role_in(region: str, start: int, stop: int) -> Optional[str]:
    """A job title inside THIS signatory's span, if the block carries one.

    Bounded by the next name label on purpose: searching the whole line read the
    supplier's "Title: Managing Director" as the buyer's title when the parser put
    both signatories on one line.
    """
    span = region[start:min(stop, start + 300)]
    m = _ROLE_LABEL.search(span)
    if not m:
        return None
    value = _tidy(_cut(m.group("value")))
    if not value or len(value) > 60 or re.search(r"\d", value):
        return None
    return value


def _span_for(name: str, full_text: str) -> Span:
    flexible = r"\s+".join(re.escape(tok) for tok in name.split())
    m = re.search(flexible, full_text)
    return Span(page=0, bbox=(0.0, 0.0, 0.0, 0.0),
                text=m.group(0) if m else name)


def signatory_candidates(full_text: str) -> list[Candidate]:
    """`contract_signatory_name` / `contract_signatory_role` from the block."""
    s = read_signatory(full_text)
    out: list[Candidate] = []
    for field, value in ((NAME_FIELD, s.name), (ROLE_FIELD, s.role),
                         (BUYER_NAME_FIELD, s.buyer_name),
                         (BUYER_ROLE_FIELD, s.buyer_role)):
        if not value:
            continue
        out.append(Candidate(
            field=field,
            value=value,
            span=_span_for(value, full_text),
            source="parties",
            pattern_name="contract_signature_block",
            confidence=CONFIDENCE,
        ))
    return out


__all__ = ["Signatory", "read_signatory", "signatory_candidates",
           "NAME_FIELD", "ROLE_FIELD", "BUYER_NAME_FIELD", "BUYER_ROLE_FIELD"]


# ---------------------------------------------------------------------------
# Correcting rows written before the signature block was read. Same four rules as
# contract_parties.decide_correction, and the same reason for each.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SignatoryCorrection:
    name: Optional[str]
    role: Optional[str]
    changed: bool
    reason: str
    buyer_name: Optional[str] = None
    buyer_role: Optional[str] = None


def decide_signatory_correction(
    *,
    full_text: str,
    stored_name: Optional[str],
    stored_role: Optional[str],
    provenance_source: Optional[str],
    stored_buyer_name: Optional[str] = None,
    stored_buyer_role: Optional[str] = None,
) -> SignatoryCorrection:
    """Should this row's signatory change, and to what?

    1. a human-confirmed value is never touched;
    2. a row with no stored text cannot be re-read;
    3. if the document has a signature block, it is the answer;
    4. if it has none, the stored value goes only when it came from the entity
       sweep -- the path that stored "Email Marketing" as a person.
    """
    from .contract_parties import BROKEN_SOURCE, HUMAN_SOURCE

    if (provenance_source or "").lower() == HUMAN_SOURCE:
        return SignatoryCorrection(stored_name, stored_role, False,
                                   "a human confirmed this value; nothing overrules that",
                                   stored_buyer_name, stored_buyer_role)
    if not (full_text or "").strip():
        return SignatoryCorrection(stored_name, stored_role, False,
                                   "no stored text to re-read; left as found",
                                   stored_buyer_name, stored_buyer_role)

    s = read_signatory(full_text)
    if s.name or s.buyer_name:
        # Both sides are decided together, because the block names them together
        # and because buyer_signatory_name was added on 2026-10-04: every row
        # extracted before that holds NULL there however clearly the document
        # states it.
        changed = ((s.name != stored_name) or (s.role != stored_role)
                   or (s.buyer_name != stored_buyer_name)
                   or (s.buyer_role != stored_buyer_role))
        return SignatoryCorrection(
            s.name, s.role, changed,
            "read from the document's own signature block"
            + ("" if changed else " and already stored correctly"),
            s.buyer_name, s.buyer_role,
        )

    if ((provenance_source or "").lower() == BROKEN_SOURCE
            and (stored_name or stored_role or stored_buyer_name or stored_buyer_role)):
        return SignatoryCorrection(
            None, None, True,
            "the document has no signature block naming either signatory and the "
            "stored value came from the entity sweep, which offered a line from "
            "the services list as a person; cleared rather than left as a fact",
            None, None,
        )

    return SignatoryCorrection(stored_name, stored_role, False,
                               "no signature block and the stored value did not "
                               "come from the sweep; left as found",
                               stored_buyer_name, stored_buyer_role)


__all__ += ["SignatoryCorrection", "decide_signatory_correction"]
