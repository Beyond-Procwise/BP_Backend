"""The typed analytic answer.

An analytic question has an exact answer, and every part of it is decided
before a language model is involved: the figures, the period they cover, the
currency they are stated in, the share one supplier holds, what is missing, and
what the reader should look at next. This module is the shape of that decision.

Two properties do most of the work:

  * **Scope is mandatory and cannot be blank.** A ranking with no period, no
    measure and no stated currency is not an answer; it is a table the reader
    has to interrogate before they can trust it. The type will not build one.
  * **Facts carry their own rendered form.** The insight writer is shown facts,
    never rows, and may use no number that is not already among them. That is
    only checkable if each fact carries the exact text a sentence is allowed to
    quote — so ``display`` is produced here by the shared formatter, and
    ``quotable_numbers()`` is the set the validator holds the model to.

Field names are snake_case, matching every other payload this API returns
(``follow_ups``, ``retrieved_documents``). Conventions otherwise follow
``src/services/facts/models.py``.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional
import re

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

from src.services.analytics.formatting import (
    EMPTY_AMOUNT,
    format_delta,
    format_int,
    format_money,
    format_pct,
)


class Measure(str, Enum):
    """Which spend basis the answer is stated on. Never inferred, always stated."""

    INVOICED = "INVOICED"
    PO = "PO"
    PAID = "PAID"


_MEASURE_LABELS = {
    Measure.INVOICED: "Invoiced spend",
    Measure.PO: "Committed (PO) spend",
    Measure.PAID: "Paid spend",
}


class CurrencyBasis(str, Enum):
    """How the figures came to be in the currency they are stated in.

    ``NATIVE`` means as billed, unconverted — in which case there is no single
    currency and cross-currency ranking is not meaningful.
    """

    CONVERTED = "CONVERTED"
    NATIVE = "NATIVE"
    MANUAL = "MANUAL"


class Confidence(str, Enum):
    ASSERTED = "ASSERTED"
    CORROBORATED = "CORROBORATED"
    UNASSESSED = "UNASSESSED"


class ColumnType(str, Enum):
    MONEY = "money"
    PCT = "pct"
    INT = "int"
    DELTA = "delta"
    TEXT = "text"


class Align(str, Enum):
    LEFT = "left"
    RIGHT = "right"


class Severity(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


_SEVERITY_ORDER = {Severity.HIGH: 0, Severity.MEDIUM: 1, Severity.LOW: 2}


class FactCode(str, Enum):
    """The findings a detector may produce. The headline is written from these."""

    TOP_N_SHARE_OF_TOTAL = "TOP_N_SHARE_OF_TOTAL"
    TOP_1_SHARE = "TOP_1_SHARE"
    TOP_1_TO_TOP_2_RATIO = "TOP_1_TO_TOP_2_RATIO"
    CONCENTRATION_THRESHOLD_BREACHED = "CONCENTRATION_THRESHOLD_BREACHED"
    PERIOD_DELTA = "PERIOD_DELTA"
    CURRENCY_MISMATCH = "CURRENCY_MISMATCH"


class AnomalyCode(str, Enum):
    UNCONVERTED_CURRENCY = "UNCONVERTED_CURRENCY"
    MISSING_PERIOD_DATA = "MISSING_PERIOD_DATA"
    # Ranking across currencies nobody converted is the defect this work exists
    # to end. When the reader has chosen to report as billed, the order of a
    # mixed-currency table says nothing, and the answer must say so rather than
    # present it as a league table.
    RANKING_NOT_COMPARABLE = "RANKING_NOT_COMPARABLE"
    MANUAL_FX_RATE = "MANUAL_FX_RATE"


def _fmt_day(value: str) -> tuple[str, str]:
    """A date as ("1 Apr", "2026"), so a same-year range prints one year."""
    parsed = date.fromisoformat(value)
    return f"{parsed.day} {parsed.strftime('%b')}", str(parsed.year)


class Population(BaseModel):
    """What was counted, and how many of them there were."""

    model_config = ConfigDict(extra="forbid")

    entity: str
    count: int

    def label(self) -> str:
        noun = self.entity.lower()
        if self.count != 1:
            noun = f"{noun}s"
        return f"{format_int(self.count)} {noun}"


class Scope(BaseModel):
    """What this answer covers. Always present, never blank, always shown first."""

    model_config = ConfigDict(extra="forbid")

    measure: Measure
    period_start: str
    period_end: str
    period_label: str
    population: Population
    currency: str
    currency_basis: CurrencyBasis = CurrencyBasis.CONVERTED
    rate_note: Optional[str] = None
    filters_applied: List[str] = Field(default_factory=list)

    @field_validator("period_label")
    @classmethod
    def _label_must_say_something(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("period_label must name the period the answer covers")
        return value.strip()

    @model_validator(mode="after")
    def _converted_figures_must_name_their_currency(self) -> "Scope":
        if self.currency_basis is not CurrencyBasis.NATIVE and not self.currency.strip():
            raise ValueError("a converted figure must state the currency it is stated in")
        return self

    def period_phrase(self) -> str:
        start_day, start_year = _fmt_day(self.period_start)
        end_day, end_year = _fmt_day(self.period_end)
        if start_year == end_year:
            return f"{start_day}–{end_day} {end_year}"
        return f"{start_day} {start_year}–{end_day} {end_year}"

    def line(self) -> str:
        """The one line that opens every analytic answer."""
        currency = self.currency if self.currency.strip() else "as billed"
        parts = [
            _MEASURE_LABELS[self.measure],
            f"{self.period_label} ({self.period_phrase()})",
            self.population.label(),
            currency,
        ]
        if self.rate_note:
            parts.append(self.rate_note)
        parts.extend(self.filters_applied)
        return " · ".join(parts)


class Headline(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    confidence: Confidence = Confidence.ASSERTED


class Column(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    label: str
    type: ColumnType = ColumnType.TEXT
    align: Optional[Align] = None
    decimals: int = 1
    currency: Optional[str] = None
    # The column the inline bar is drawn against — set on the primary measure.
    is_primary: bool = False

    @model_validator(mode="after")
    def _figures_align_right(self) -> "Column":
        if self.align is None:
            object.__setattr__(
                self,
                "align",
                Align.LEFT if self.type is ColumnType.TEXT else Align.RIGHT,
            )
        return self


class Table(BaseModel):
    model_config = ConfigDict(extra="forbid")

    columns: List[Column]
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    totals: Optional[Dict[str, Any]] = None


class Fact(BaseModel):
    """One machine-readable finding, with the exact text a sentence may quote."""

    model_config = ConfigDict(extra="forbid")

    code: FactCode
    value: Optional[Decimal] = None
    type: ColumnType = ColumnType.TEXT
    entity: Optional[str] = None
    entity_ref: Optional[str] = None
    currency: Optional[str] = None
    unit: Optional[str] = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def display(self) -> str:
        if self.value is None:
            return EMPTY_AMOUNT
        if self.type is ColumnType.MONEY:
            return format_money(self.value, self.currency)
        if self.type is ColumnType.PCT:
            return format_pct(self.value)
        if self.type is ColumnType.DELTA:
            return format_delta(self.value)
        if self.type is ColumnType.INT:
            return format_int(self.value)
        # Not normalised: a ratio measured at 1.0 means the leader is level with
        # the next, and printing "1" reads as a rounded-off integer instead.
        return f"{self.value}"


class Anomaly(BaseModel):
    """Something about the answer the reader has to know before trusting it.

    ``subject`` is the short thing the anomaly is about — the currency codes
    with no rate, say — kept apart from the prose so a next-step label can name
    it without parsing a sentence. ``entity_refs`` are the records affected.
    """

    model_config = ConfigDict(extra="forbid")

    code: AnomalyCode
    severity: Severity
    text: str
    subject: Optional[str] = None
    entity_refs: List[str] = Field(default_factory=list)


class NextStep(BaseModel):
    """A step the reader can take, already resolved to an action and its subjects.

    ``action_id`` and ``entity_refs`` are what a click dispatches with. A step
    must not be turned back into a sentence and re-parsed as a question — that
    round trip is where the old follow-up chips lost the entity they were about.
    """

    model_config = ConfigDict(extra="forbid")

    action_id: str
    label: str
    rung: str
    reason: str
    entity_refs: List[str] = Field(default_factory=list)


class Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_counts: Dict[str, int]
    refreshed_at: str
    query_ref: str


_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


class AnalyticAnswer(BaseModel):
    """One analytic answer, whole. Everything true about it is decided here."""

    model_config = ConfigDict(extra="forbid")

    answer_id: str
    scope: Scope
    headline: Headline
    table: Table
    facts: List[Fact] = Field(default_factory=list)
    anomalies: List[Anomaly] = Field(default_factory=list)
    provenance: Provenance
    # Two, hard. A third step means the selection rules did not decide, and
    # trimming silently would hide that behind a plausible answer.
    next_steps: List[NextStep] = Field(default_factory=list, max_length=2)

    @model_validator(mode="after")
    def _worst_anomaly_first(self) -> "AnalyticAnswer":
        ordered = sorted(self.anomalies, key=lambda a: _SEVERITY_ORDER[a.severity])
        object.__setattr__(self, "anomalies", ordered)
        return self

    def quotable_numbers(self) -> set[str]:
        """Every numeric token the insight writer is allowed to put in a sentence.

        Taken off the rendered form of each fact, because the rendered form is
        what a sentence would quote, plus the scope line — the writer is shown
        both, so the period and the population count are legitimately hers. A
        number outside this set did not come from the payload.
        """
        tokens: set[str] = set()
        for fact in self.facts:
            tokens.update(match.group(0) for match in _NUMBER.finditer(fact.display))
            # The unit carries the N in a top-N finding, and a sentence saying
            # "the top 10" is quoting the payload, not inventing a figure.
            if fact.unit:
                tokens.update(match.group(0) for match in _NUMBER.finditer(fact.unit))
        tokens.update(match.group(0) for match in _NUMBER.finditer(self.scope.line()))
        return tokens

    def unquoted_numbers(self, text: str) -> set[str]:
        """The numbers in ``text`` that the payload does not support.

        Entity names are removed before the scan rather than folded into the
        allowed set. A supplier in this corpus is called "Kestrel Supplies 8":
        counting the 8 as quotable would let the same sentence claim "8% of
        spend" and pass. Removing the name instead means naming a supplier
        licences nothing.

        Empty means the sentence is grounded. Anything else is what the model
        made up, and the sentence is discarded for a templated one.
        """
        stripped = text or ""
        names = {fact.entity for fact in self.facts if fact.entity}
        for row in self.table.rows:
            for key in ("supplier", "entity", "name"):
                value = row.get(key)
                if isinstance(value, str) and value:
                    names.add(value)
        # Longest first, so "Kestrel Supplies 8" is removed before a shorter
        # name that happens to be a prefix of it.
        for name in sorted(names, key=len, reverse=True):
            stripped = stripped.replace(name, " ")
        found = {match.group(0) for match in _NUMBER.finditer(stripped)}
        return found - self.quotable_numbers()
