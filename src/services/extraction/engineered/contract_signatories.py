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
from .contract_parties import CONFIDENCE, _squeeze, _tidy, side_for_role

NAME_FIELD = "contract_signatory_name"
ROLE_FIELD = "contract_signatory_role"

#: Where the signatures start. A contract's signature block is always announced.
_SECTION = re.compile(
    r"(?i)\b(?:signature(?:s)?(?:\s+and\s+date)?|signed\s+(?:by|for|on\s+behalf)|"
    r"in\s+witness\s+whereof|executed\s+(?:by|as)|signatories|"
    r"agreed\s+and\s+accepted)\b"
)

#: `Name: John Smith`, and the labels extraction_schemas/contract.yaml declares.
_NAME_LABEL = re.compile(
    r"(?:^|[\s\n])(?i:authorised\s+signatory|authorized\s+signatory|signed\s+by|"
    r"authorised\s+by|authorized\s+by|signatory|print\s+name|name)"
    r"\s*[:–]\s*(?P<value>[^\n]*)"
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
    """The supplier's signatory. `buyer_name` is read but not stored anywhere."""

    name: Optional[str] = None
    role: Optional[str] = None
    party: Optional[str] = None
    buyer_name: Optional[str] = None
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


def _party_label_before(text: str, pos: int) -> Optional[str]:
    """The party whose sub-block this position sits in, if it is labelled.

    Looks back over the preceding 200 characters for the LAST party-role word --
    "MARKETER", "CLIENT", "SUPPLIER" -- which is how a signature block separates
    its two halves. The vocabulary is contract_parties', so the two readers cannot
    disagree about what "Marketer" means.
    """
    window = text[max(0, pos - 200):pos]
    best: Optional[str] = None
    for m in re.finditer(r"[A-Za-z][A-Za-z ]{2,24}", window):
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

    found: list[tuple[Optional[str], str, str, int]] = []   # (party, name, evidence, pos)
    for m in _NAME_LABEL.finditer(region):
        value = _tidy(_cut(m.group("value")))
        if not _is_person(value):
            continue
        party = _party_label_before(region, m.start())
        found.append((party, value, _squeeze(m.group(0)), m.start()))

    if not found:
        return Signatory()

    supplier = next((f for f in found if f[0] == "supplier"), None)
    buyer = next((f for f in found if f[0] == "buyer"), None)

    if supplier is None and buyer is None and len(found) == 1:
        # One signatory, unattributed: that is who signed.
        party, name, ev, pos = found[0]
        return Signatory(name=name, role=_role_near(region, pos), party=None, evidence=ev)

    if supplier is None:
        # Only the other side signed this copy, or nothing could be attributed.
        return Signatory(buyer_name=buyer[1] if buyer else None,
                         evidence=buyer[2] if buyer else None)

    return Signatory(name=supplier[1], role=_role_near(region, supplier[3]),
                     party="supplier", buyer_name=buyer[1] if buyer else None,
                     evidence=supplier[2])


def _role_near(region: str, pos: int) -> Optional[str]:
    """A job title on the same signatory's line, if the block carries one."""
    # pos is the START of the name match, which includes the newline before it, so
    # find("\n", pos) would return pos itself and the line would come out empty.
    line_start = region.rfind("\n", 0, pos) + 1
    line_end = region.find("\n", pos + 1)
    line = region[line_start:line_end if line_end > 0 else min(len(region), pos + 300)]
    m = _ROLE_LABEL.search(line)
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
    for field, value in ((NAME_FIELD, s.name), (ROLE_FIELD, s.role)):
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
           "NAME_FIELD", "ROLE_FIELD"]


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


def decide_signatory_correction(
    *,
    full_text: str,
    stored_name: Optional[str],
    stored_role: Optional[str],
    provenance_source: Optional[str],
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
                                   "a human confirmed this value; nothing overrules that")
    if not (full_text or "").strip():
        return SignatoryCorrection(stored_name, stored_role, False,
                                   "no stored text to re-read; left as found")

    s = read_signatory(full_text)
    if s.name:
        changed = (s.name != stored_name) or (s.role != stored_role)
        return SignatoryCorrection(
            s.name, s.role, changed,
            "read from the document's own signature block"
            + ("" if changed else " and already stored correctly"),
        )

    if (provenance_source or "").lower() == BROKEN_SOURCE and (stored_name or stored_role):
        return SignatoryCorrection(
            None, None, True,
            "the document has no signature block naming the supplier's signatory "
            "and the stored value came from the entity sweep, which offered a line "
            "from the services list as a person; cleared rather than left as a fact",
        )

    return SignatoryCorrection(stored_name, stored_role, False,
                               "no signature block and the stored value did not "
                               "come from the sweep; left as found")


__all__ += ["SignatoryCorrection", "decide_signatory_correction"]
