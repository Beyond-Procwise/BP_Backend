"""Data shapes for discrepancy triage.

Every stage after the loader is a pure function over these dataclasses, which is
what lets each one be tested on a hand-built deal and keeps a 5,000-deal run fast.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import Enum, IntEnum
from typing import Optional

ZERO = Decimal("0")


class Outcome(str, Enum):
    """What kind of difference one comparison found (triage spec §5)."""
    MATCH = "MATCH"
    WITHIN_TOL = "WITHIN_TOL"
    EXPLAINED = "EXPLAINED"
    ABSENT_SUBORDINATE = "ABSENT_SUBORDINATE"
    ABSENT_AUTHORITATIVE = "ABSENT_AUTHORITATIVE"
    CONFLICT = "CONFLICT"
    UNVERIFIABLE = "UNVERIFIABLE"


#: Only these go on to scoring; the rest are matches or notes.
SCORED = frozenset({Outcome.CONFLICT, Outcome.ABSENT_AUTHORITATIVE, Outcome.UNVERIFIABLE})
NOTE = frozenset({Outcome.EXPLAINED, Outcome.ABSENT_SUBORDINATE})


class Severity(IntEnum):
    """Higher is worse, so max() picks the most severe and min() caps."""
    S0 = 0
    S3 = 1
    S2 = 2
    S1 = 3


#: The Action Centre's own words. S3 and S0 never reach it.
ACTION_CENTRE_SEVERITY = {Severity.S1: "critical", Severity.S2: "warning"}

CRITICALITY = {"party": 1.0, "currency": 1.0, "money": 0.8, "quantity": 0.8,
               "date": 0.6, "reference": 0.6, "terms": 0.6, "description": 0.2}

CATEGORY = {
    "unit_price": "price", "uniform_uplift": "price", "quantity": "quantity",
    "cumulative_total": "overbilling", "duplicate": "duplicate", "tax_rate": "tax",
    "currency": "currency", "supplier": "supplier", "invoice_date": "date",
    "payment_terms": "terms", "description": "description", "unlinked_line": "linking",
    "rollup": "linking", "bad_po_ref": "linking", "no_po": "linking",
    "line_arithmetic": "arithmetic", "invoice_totals": "arithmetic",
}


def fingerprint(deal_id: str, rule_id: str, cause_key: str) -> str:
    return hashlib.sha1(f"{deal_id}|{rule_id}|{cause_key}".encode()).hexdigest()


def money(amount, currency: Optional[str]) -> str:
    if amount is None:
        return "n/a"
    text = f"{Decimal(amount).quantize(Decimal('0.01')):,}"
    if (currency or "").upper() == "GBP":
        return f"£{text}"
    return f"{text} {currency or ''}".strip()


@dataclass(frozen=True)
class Line:
    line_ref: str
    item_id: Optional[str] = None
    description: Optional[str] = None
    quantity: Optional[Decimal] = None
    uom: Optional[str] = None
    unit_price: Optional[Decimal] = None
    line_amount: Optional[Decimal] = None
    po_id: Optional[str] = None
    delivery_date: Optional[date] = None


@dataclass
class Doc:
    doc_id: str
    doc_type: str                         # quote | purchase_order | invoice
    supplier_id: Optional[str] = None
    currency: Optional[str] = None
    doc_date: Optional[date] = None
    net: Optional[Decimal] = None
    tax: Optional[Decimal] = None
    gross: Optional[Decimal] = None
    payment_terms: Optional[str] = None
    po_id: Optional[str] = None           # invoice -> PO number on its header
    quote_ref: Optional[str] = None       # PO -> quote
    confidence: Optional[float] = None    # 0-1; None = not reported
    fx_to_gbp: Optional[Decimal] = None
    fx_rate_date: Optional[datetime] = None
    lines: list[Line] = field(default_factory=list)

    @property
    def is_credit_note(self) -> bool:
        return self.net is not None and self.net < 0

    @property
    def po_ref(self) -> Optional[str]:
        """The PO this invoice names: the header's, else the first line's."""
        return self.po_id or next((l.po_id for l in self.lines if l.po_id), None)


@dataclass(frozen=True)
class DuplicateFlag:
    invoice_id: str
    earlier_invoice_id: Optional[str]
    amount: Optional[Decimal]


@dataclass
class DocumentSet:
    deal_id: str
    quotes: list[Doc] = field(default_factory=list)
    pos: list[Doc] = field(default_factory=list)
    invoices: list[Doc] = field(default_factory=list)
    duplicates: list[DuplicateFlag] = field(default_factory=list)


@dataclass
class LineLink:
    invoice: Doc
    inv_line: Line
    po: Doc
    po_line: Optional[Line]
    confidence: float
    rollup: bool = False


@dataclass
class Links:
    invoice_po: dict = field(default_factory=dict)   # invoice id -> PO Doc, or None
    bad_refs: set = field(default_factory=set)       # invoice ids naming a PO that is not there
    no_ref: set = field(default_factory=set)         # invoice ids naming no PO at all
    line_links: list = field(default_factory=list)   # list[LineLink]
    po_quote: dict = field(default_factory=dict)     # PO id -> quote Doc, or None


@dataclass
class Result:
    deal_id: str
    rule_id: str
    field_class: str
    outcome: Outcome
    claim_doc: Optional[str]
    field_name: str
    claim_line: Optional[str] = None
    auth_doc: Optional[str] = None
    auth_line: Optional[str] = None
    po_id: Optional[str] = None
    claim_value: Optional[str] = None
    auth_value: Optional[str] = None
    delta: Optional[Decimal] = None        # claim - authoritative, signed
    exposure: Decimal = ZERO               # money at risk, in `currency`
    currency: Optional[str] = None
    fx_to_gbp: Optional[Decimal] = None
    fx_rate_date: Optional[datetime] = None
    basis_total: Optional[Decimal] = None  # claim document total, for the materiality threshold
    # The money behind the comparison, in document currency: claim_amount - auth_amount
    # IS the money at stake. None where the comparison is not about an amount.
    claim_amount: Optional[Decimal] = None
    auth_amount: Optional[Decimal] = None
    confidence: float = 1.0
    tolerance: dict = field(default_factory=dict)
    note: str = ""
    severity: Optional[Severity] = None
    score: Optional[float] = None
    score_inputs: dict = field(default_factory=dict)

    @property
    def cause_key(self) -> str:
        return f"{self.claim_doc or ''}|{self.claim_line or ''}|{self.field_name}"

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.deal_id, self.rule_id, self.cause_key)

    @property
    def exposure_gbp(self) -> Optional[Decimal]:
        if self.fx_to_gbp is None:
            return None
        return (abs(self.exposure) * self.fx_to_gbp).quantize(Decimal("0.01"))


def pct_change(r: Result) -> Optional[Decimal]:
    """The claim's difference as a percentage of the authoritative value."""
    try:
        auth = Decimal(str(r.auth_value))
    except Exception:
        return None
    if auth == 0 or r.delta is None:
        return None
    return r.delta / auth * 100


@dataclass
class Finding:
    """One cause (or one group of same-cause results) and its knock-on effects."""
    deal_id: str
    rule_id: str
    causes: list[Result]
    cause_key: str
    effects: list[Result] = field(default_factory=list)
    headline: str = ""
    text: str = ""

    @property
    def lead(self) -> Result:
        return self.causes[0]

    @property
    def severity(self) -> Severity:
        # Effects count too: an always-S1 effect (over-billing) must never be softened
        # by being attached to a milder cause.
        return max(r.severity for r in (*self.causes, *self.effects) if r.severity is not None)

    @property
    def exposure(self) -> Decimal:
        return sum((abs(r.exposure) for r in self.causes), ZERO)

    @property
    def exposure_gbp(self) -> Optional[Decimal]:
        values = [r.exposure_gbp for r in self.causes]
        return None if any(v is None for v in values) else sum(values, ZERO)

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.deal_id, self.rule_id, self.cause_key)

    @property
    def category(self) -> str:
        return CATEGORY.get(self.rule_id, "other")

    @property
    def confidence(self) -> float:
        return min(r.confidence for r in self.causes)


@dataclass
class Verdict:
    deal_id: str
    verdict: str
    s1: int
    s2: int
    notes: int
    exposure_gbp: Decimal
    incomplete: bool

    @property
    def summary(self) -> str:
        parts = [self.verdict]
        action = self.s1 + self.s2
        if action:
            parts.append(f"{action} finding{'s' if action != 1 else ''} "
                         f"need{'s' if action == 1 else ''} action")
        if self.notes:
            parts.append(f"{self.notes} note{'s' if self.notes != 1 else ''}")
        if self.exposure_gbp:
            parts.append(f"exposure {money(self.exposure_gbp, 'GBP')}")
        return " · ".join(parts)
