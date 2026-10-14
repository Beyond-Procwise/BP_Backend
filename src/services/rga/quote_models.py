"""What a customer quote is made of: the pack it is drawn from, and the tree that draws it.

A quote is a document a customer acts on, so the rule the report types enforce -- nothing
is printed that the pack does not hold -- is tighter here. A report lets a model write
prose around its figures. A quote has no prose block at all:

  * ``QuotePack`` is a ``FactPack`` that also holds the quote's lines, who it is for and
    the terms it is offered on, each under an id and each naming the row it was read from.
  * ``QuoteAST`` is a list of blocks, and every field of every block is either an id that
    must resolve in the pack or one of a fixed set of words. There is no title, no label,
    no text and no number field anywhere in it. A figure, a clause or a name written into
    a quote document by hand is not rejected by a check later -- it has nowhere to go.

The quote blocks are their own union, deliberately not added to ``models.Block``: that
union is the grammar a model composes a report under, and a model writing a report must
never be offered a signature block.

The pack is built from the customer record (``sell_side.quotes.get_quote``), which has no
cost or margin in it. See ``builders/sales_quote``.
"""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.services.analytics.formatting import currency_symbol, format_money_exact
from src.services.rga.models import FACT_ID, FactPack

# L<sales_quote_line_id>: the line's own row id, so a line keeps its id across rebuilds.
LINE_ID = re.compile(r"^L\d+$")
TERM_ID = re.compile(r"^T\d{4}$")
# N<justification_id>.
NOTE_ID = re.compile(r"^N\d+$")


def unit_money(value: Decimal, currency: Optional[str]) -> str:
    """A unit price in full. Catalogue prices run to four places (173.6667), and rounding
    one to pence prints a price that does not multiply up to the line total."""
    exact = value.quantize(Decimal("0.0001")).normalize()
    if exact == exact.quantize(Decimal("0.01")):
        return format_money_exact(exact, currency)
    sign = "-" if exact < 0 else ""
    return f"{sign}{currency_symbol(currency)}{abs(exact):,f}"


def long_date(value: date) -> str:
    """14 September 2026: no reading of it as the ninth of the fourteenth month."""
    return f"{value.day} {value:%B} {value.year}"


def _evidenced(value: str) -> str:
    if not value.strip():
        raise ValueError("a pack entry must name the record it was read from")
    return value


def _distinct(refs: List[str], what: str) -> List[str]:
    repeated = sorted({r for r in refs if refs.count(r) > 1})
    if repeated:
        raise ValueError(f"{what} listed more than once: {', '.join(repeated)}")
    return refs


def _ids(refs: List[str], pattern: re.Pattern, example: str) -> List[str]:
    for ref in refs:
        if not pattern.match(ref):
            raise ValueError(f"{ref!r} is not an id like {example}; a quote document "
                             "names what is in the pack and holds nothing of its own")
    return refs


# --------------------------------------------------------------------------
# The pack.
# --------------------------------------------------------------------------


class QuoteLine(BaseModel):
    """One line as the customer is offered it. No cost, no margin, no tier."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    line_id: str
    line_no: int
    sku: Optional[str] = None
    mpn: Optional[str] = None
    description: str
    unit_of_measure: Optional[str] = None
    quantity: Decimal
    unit_price: Decimal
    line_total: Decimal
    currency: str
    provenance_id: str = Field(min_length=1)

    _id = field_validator("line_id")(lambda v: _ids([v], LINE_ID, "L501")[0])
    _evidence = field_validator("provenance_id")(_evidenced)

    def cell(self, column: str) -> str:
        """The exact text a renderer prints for one column of this line, and the text the
        post-check looks for. One spelling, so a price drawn another way traces to nothing."""
        if column == "line_no":
            return str(self.line_no)
        if column == "quantity":
            return f"{self.quantity.normalize():,f}"
        if column == "unit_price":
            return unit_money(self.unit_price, self.currency)
        if column == "line_total":
            return format_money_exact(self.line_total, self.currency)
        return getattr(self, column) or ""


class QuoteTerm(BaseModel):
    """One term, word for word as the terms library holds it. Never written by a model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    term_id: str
    kind: Literal["general", "payment"]
    text: str = Field(min_length=1)
    provenance_id: str = Field(min_length=1)

    _id = field_validator("term_id")(lambda v: _ids([v], TERM_ID, "T0001")[0])
    _evidence = field_validator("provenance_id")(_evidenced)


class QuoteNote(BaseModel):
    """A sentence already on record as safe for the customer to read (a justification
    marked ``customer_safe``). Printed word for word; never written here."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    note_id: str
    kind: str
    text: str = Field(min_length=1)
    provenance_id: str = Field(min_length=1)

    _id = field_validator("note_id")(lambda v: _ids([v], NOTE_ID, "N801")[0])
    _evidence = field_validator("provenance_id")(_evidenced)


class QuoteIdentity(BaseModel):
    """Who the quote is for and what it is called.

    ``number`` is the customer-facing number, minted when the quote is issued. It is not
    ``quote_ref``, which is an internal draft reference that exists from the first save.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    quote_ref: str
    number: Optional[str] = None
    status: str
    account_name: str
    contact_name: Optional[str] = None
    quote_date: date
    valid_until: date
    currency: str
    provenance_id: str = Field(min_length=1)

    _evidence = field_validator("provenance_id")(_evidenced)


class QuotePack(FactPack):
    """Everything one quote may state. Figures that are totals are facts, as in a report."""

    identity: QuoteIdentity
    lines: List[QuoteLine] = Field(min_length=1)
    terms: List[QuoteTerm] = Field(default_factory=list)
    notes: List[QuoteNote] = Field(default_factory=list)
    total_fact_id: str

    @field_validator("lines")
    @classmethod
    def _line_ids_are_distinct(cls, lines: List[QuoteLine]) -> List[QuoteLine]:
        _distinct([l.line_id for l in lines], "line")
        return lines

    @field_validator("terms")
    @classmethod
    def _term_ids_are_distinct(cls, terms: List[QuoteTerm]) -> List[QuoteTerm]:
        _distinct([t.term_id for t in terms], "term")
        return terms

    @field_validator("notes")
    @classmethod
    def _note_ids_are_distinct(cls, notes: List[QuoteNote]) -> List[QuoteNote]:
        _distinct([n.note_id for n in notes], "note")
        return notes

    def _hashed(self) -> Dict[str, Any]:
        return {
            **super()._hashed(),
            "identity": self.identity.model_dump(mode="json"),
            "lines": [l.model_dump(mode="json") for l in self.lines],
            "terms": [t.model_dump(mode="json") for t in self.terms],
            "notes": [n.model_dump(mode="json") for n in self.notes],
            "total_fact_id": self.total_fact_id,
        }

    def line(self, line_id: str) -> Optional[QuoteLine]:
        return next((l for l in self.lines if l.line_id == line_id), None)

    def term(self, term_id: str) -> Optional[QuoteTerm]:
        return next((t for t in self.terms if t.term_id == term_id), None)

    def note(self, note_id: str) -> Optional[QuoteNote]:
        return next((n for n in self.notes if n.note_id == note_id), None)


# --------------------------------------------------------------------------
# The document tree.
# --------------------------------------------------------------------------

LineColumn = Literal["line_no", "sku", "mpn", "description", "quantity", "unit_of_measure",
                     "unit_price", "line_total"]
_DEFAULT_COLUMNS: List[str] = ["line_no", "sku", "description", "quantity",
                               "unit_of_measure", "unit_price", "line_total"]
# A line drawn without these is not the line the customer is being offered.
REQUIRED_COLUMNS = ("description", "quantity", "unit_price", "line_total")


class _QuoteBlock(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class LineTableBlock(_QuoteBlock):
    """Which lines, and which of their fields. The headings are the renderer's."""

    type: Literal["line_table"] = "line_table"
    line_refs: List[str] = Field(min_length=1)
    columns: List[LineColumn] = Field(default_factory=lambda: list(_DEFAULT_COLUMNS),
                                      min_length=1)

    @field_validator("line_refs")
    @classmethod
    def _lines(cls, refs: List[str]) -> List[str]:
        return _distinct(_ids(refs, LINE_ID, "L501"), "line")

    @field_validator("columns")
    @classmethod
    def _columns(cls, columns: List[str]) -> List[str]:
        missing = [c for c in REQUIRED_COLUMNS if c not in columns]
        if missing:
            raise ValueError(f"a line table must show {', '.join(missing)}")
        return _distinct(columns, "column")


class TotalsBlock(_QuoteBlock):
    type: Literal["totals"] = "totals"
    fact_refs: List[str] = Field(min_length=1)

    @field_validator("fact_refs")
    @classmethod
    def _facts(cls, refs: List[str]) -> List[str]:
        return _distinct(_ids(refs, FACT_ID, "F0001"), "total")


class TermsBlock(_QuoteBlock):
    type: Literal["terms"] = "terms"
    term_refs: List[str] = Field(min_length=1)

    @field_validator("term_refs")
    @classmethod
    def _terms(cls, refs: List[str]) -> List[str]:
        return _distinct(_ids(refs, TERM_ID, "T0001"), "term")


class NotesBlock(_QuoteBlock):
    type: Literal["notes"] = "notes"
    note_refs: List[str] = Field(min_length=1)

    @field_validator("note_refs")
    @classmethod
    def _notes(cls, refs: List[str]) -> List[str]:
        return _distinct(_ids(refs, NOTE_ID, "N801"), "note")


class PaymentBlock(_QuoteBlock):
    type: Literal["payment"] = "payment"
    term_ref: str

    @field_validator("term_ref")
    @classmethod
    def _term(cls, ref: str) -> str:
        return _ids([ref], TERM_ID, "T0001")[0]


class SignatureBlock(_QuoteBlock):
    """A place to sign. Whose name goes beside it comes from the pack, not from here."""

    type: Literal["signature"] = "signature"
    party: Literal["supplier", "customer"]


QuoteBlock = Annotated[
    Union[LineTableBlock, TotalsBlock, TermsBlock, NotesBlock, PaymentBlock,
          SignatureBlock],
    Field(discriminator="type"),
]


class QuoteAST(BaseModel):
    """The quote as a list of blocks, before it is any particular file format."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    blocks: List[QuoteBlock] = Field(default_factory=list)

    def line_refs(self) -> List[str]:
        return [r for b in self.blocks if isinstance(b, LineTableBlock) for r in b.line_refs]

    def fact_refs(self) -> set[str]:
        return {r for b in self.blocks if isinstance(b, TotalsBlock) for r in b.fact_refs}


def unresolved_refs(ast: QuoteAST, pack: QuotePack) -> List[str]:
    """Every id the document names that the pack cannot supply, sorted.

    A term must also be the right kind: the payment block takes a payment term and the
    terms block a general one, so one cannot be passed off as the other.
    """
    missing: set[str] = set()
    for block in ast.blocks:
        if isinstance(block, LineTableBlock):
            missing.update(r for r in block.line_refs if pack.line(r) is None)
        elif isinstance(block, TotalsBlock):
            missing.update(r for r in block.fact_refs if pack.fact(r) is None)
        elif isinstance(block, TermsBlock):
            missing.update(r for r in block.term_refs
                           if getattr(pack.term(r), "kind", None) != "general")
        elif isinstance(block, NotesBlock):
            missing.update(r for r in block.note_refs if pack.note(r) is None)
        elif isinstance(block, PaymentBlock):
            if getattr(pack.term(block.term_ref), "kind", None) != "payment":
                missing.add(block.term_ref)
    return sorted(missing)
