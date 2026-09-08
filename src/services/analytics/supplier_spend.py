"""Top suppliers by spend, decided before any model is involved.

What this replaces
------------------
``corpus_facts._fetch("spend")`` ranks suppliers on raw ``invoice_amount`` with
the currency carried alongside as a label. In the live corpus that puts five
rupee-billing suppliers at the top of the table for no reason other than the
denomination: Harbourline's 102,142,166.94 INR is under £0.8M, behind sterling
suppliers listed below it. The table was not badly worded, it was wrongly
ordered, and no amount of better prose fixes that.

So the ranking is computed here, on values converted into the currency the
reader selected on screen, and every claim the answer could make is derived as
a :class:`Fact` first. The model's turn comes later and it may only rephrase
these.

The builder is pure: rows in, answer out, no database and no clock. Fetching
lives in :mod:`src.services.analytics.repository`, so the arithmetic that
decides what a customer is told can be tested exhaustively without one.

Reporting as billed
-------------------
When the reader has chosen native ("as billed"), there is no common currency
and therefore no single league table. Rather than fall back to the very
ordering this module exists to end, the ranking restarts within each currency,
the currency becomes a column, and the answer says outright that the figures
are not comparable across it.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from src.services.analytics.currency import DisplayCurrency
from src.services.analytics.formatting import format_money, format_pct
from src.services.analytics.models import (
    AnalyticAnswer,
    Anomaly,
    AnomalyCode,
    Column,
    ColumnType,
    Confidence,
    CurrencyBasis,
    Fact,
    FactCode,
    Headline,
    Measure,
    Population,
    Provenance,
    Scope,
    Severity,
    Table,
)
from src.services.analytics.period import Period

class Lens(str, Enum):
    """Which question this answer answers, over one measurement.

    The three are not three builders: they are the same converted spend, seen
    the way each question needs it. Splitting them would be three places for
    the arithmetic to drift apart.
    """

    RANKING = "RANKING"
    CONCENTRATION = "CONCENTRATION"
    TREND = "TREND"


# The rung each lens stands on, read back by the next-step engine so the ladder
# advances instead of offering the reader the screen they are already on.
QUERY_REFS = {
    Lens.RANKING: "supplier_spend_ranking/v1",
    Lens.CONCENTRATION: "supplier_concentration/v1",
    Lens.TREND: "supplier_spend_trend/v1",
}

QUERY_REF = QUERY_REFS[Lens.RANKING]
# What the answer was read from, named as the product names it. The table it
# actually came from is a description of the backend: the output-safety gate
# replaced an entire live answer with "I couldn't retrieve that" the first time
# this line carried one, and it was right to. The internal reference lives in
# repository.py's SQL and in the logs, where an operator reads it.
SOURCE_LABEL = "invoices"
DEFAULT_TOP_N = 10
DEFAULT_CONCENTRATION_THRESHOLD_PCT = Decimal("20")

# Below this, the leader is level with the next supplier rather than ahead of
# it, and the headline says nothing about the gap.
MEANINGFUL_LEAD = Decimal("1.2")

# A supplier with nothing to compare against sorts below every measured move
# rather than being treated as a fall.
_UNMOVED = Decimal("-1e12")

# A change is only a finding if the base it was measured from was real. Live,
# "Windrose Services 14 moved most, +345,261.1%" came off £26.72 of spend the
# year before — arithmetically true, and no use to anyone. The floor is a share
# of the leader's spend rather than a fixed amount, so it travels between
# currencies and between a corpus of thousands and one of ten.
MATERIAL_BASE_SHARE = Decimal("0.01")


@dataclass(frozen=True)
class SupplierSpendRow:
    """One supplier's spend in one currency. Never pre-summed across currencies.

    Splitting by currency is what makes rule 1 of the display-currency contract
    possible — convert each currency once, from its own native amount.
    """

    supplier_id: str
    supplier_name: str
    currency: str
    amount: Decimal
    invoices: int = 0


@dataclass
class _Supplier:
    supplier_id: str
    name: str
    by_currency: Dict[str, Decimal]
    invoices: int
    converted: Optional[Decimal] = None
    excluded_currencies: Tuple[str, ...] = ()

    @property
    def currencies(self) -> List[str]:
        return sorted(self.by_currency)


def _group(rows: Iterable[SupplierSpendRow]) -> List[_Supplier]:
    grouped: Dict[str, _Supplier] = {}
    for row in rows:
        supplier = grouped.get(row.supplier_id)
        if supplier is None:
            supplier = _Supplier(supplier_id=row.supplier_id, name=row.supplier_name,
                                 by_currency={}, invoices=0)
            grouped[row.supplier_id] = supplier
        currency = (row.currency or "").upper()
        supplier.by_currency[currency] = supplier.by_currency.get(currency, Decimal(0)) + row.amount
        supplier.invoices += row.invoices or 0
    return list(grouped.values())


def _pct(part: Decimal, whole: Optional[Decimal]) -> Optional[Decimal]:
    if whole is None or whole == 0:
        return None
    return part / whole * 100


def _ratio(first: Decimal, second: Decimal) -> Optional[Decimal]:
    if second == 0:
        return None
    try:
        return (first / second).quantize(Decimal("0.1"))
    except InvalidOperation:
        return None


def build_supplier_spend_ranking(
    *,
    rows: Sequence[SupplierSpendRow],
    display: DisplayCurrency,
    period: Period,
    population_count: int,
    invoice_count: int,
    answer_id: str,
    refreshed_at: str,
    prior_rows: Sequence[SupplierSpendRow] = (),
    prior_period: Optional[Period] = None,
    top_n: int = DEFAULT_TOP_N,
    concentration_threshold_pct: Decimal = DEFAULT_CONCENTRATION_THRESHOLD_PCT,
    measure: Measure = Measure.INVOICED,
    filters_applied: Optional[List[str]] = None,
    lens: Lens = Lens.RANKING,
) -> AnalyticAnswer:
    """The answer to "top N suppliers by spend", whole and self-describing.

    ``lens`` chooses which of the three questions this answer answers; the
    measurement underneath is identical in all three.
    """

    suppliers = _group(rows)
    anomalies: List[Anomaly] = []

    if display.is_native:
        return _as_billed_answer(
            suppliers=suppliers, display=display, period=period,
            population_count=population_count, invoice_count=invoice_count,
            answer_id=answer_id, refreshed_at=refreshed_at, top_n=top_n,
            measure=measure, filters_applied=filters_applied or [],
        )

    # -- convert, once per currency per supplier ---------------------------
    missing_currencies: List[str] = []
    ranked: List[_Supplier] = []
    unconvertible: List[_Supplier] = []
    for supplier in suppliers:
        result = display.total(
            [(amount, currency) for currency, amount in supplier.by_currency.items()]
        )
        supplier.converted = result.value
        supplier.excluded_currencies = result.excluded_currencies
        for code in result.excluded_currencies:
            if code not in missing_currencies:
                missing_currencies.append(code)
        (ranked if result.value is not None else unconvertible).append(supplier)

    ranked.sort(key=lambda s: (-(s.converted or Decimal(0)), s.name))
    visible = ranked[:top_n]
    population_total = sum((s.converted for s in ranked), Decimal(0)) if ranked else None

    # -- facts -------------------------------------------------------------
    facts: List[Fact] = []
    currency = display.target
    if visible:
        leader = visible[0]
        leader_share = _pct(leader.converted, population_total)
        if leader_share is not None:
            facts.append(Fact(code=FactCode.TOP_1_SHARE, entity=leader.name,
                              entity_ref=leader.supplier_id, value=leader_share,
                              type=ColumnType.PCT))
        top_n_share = _pct(sum((s.converted for s in visible), Decimal(0)), population_total)
        if top_n_share is not None:
            facts.append(Fact(code=FactCode.TOP_N_SHARE_OF_TOTAL, value=top_n_share,
                              type=ColumnType.PCT, unit=str(len(visible))))
            # Concentration is a property of the group, not of its leader. A
            # top-1 rule cannot fire on a book this wide — across 3,510
            # suppliers the largest holds 1.4% — so it would have reported
            # "no concentration risk" whatever the shape of the spend.
            if top_n_share >= concentration_threshold_pct:
                facts.append(Fact(code=FactCode.CONCENTRATION_THRESHOLD_BREACHED,
                                  value=top_n_share, type=ColumnType.PCT,
                                  unit=str(len(visible))))
        if len(visible) > 1:
            ratio = _ratio(visible[0].converted, visible[1].converted)
            if ratio is not None:
                facts.append(Fact(code=FactCode.TOP_1_TO_TOP_2_RATIO, value=ratio,
                                  entity=leader.name, entity_ref=leader.supplier_id,
                                  type=ColumnType.TEXT))

    prior_by_id = {s.supplier_id: s for s in _group(prior_rows)}
    deltas: Dict[str, Decimal] = {}
    prior_converted: Dict[str, Decimal] = {}
    negligible: set = set()
    leader_spend = visible[0].converted if visible else None
    base_floor = (leader_spend or Decimal(0)) * MATERIAL_BASE_SHARE
    if prior_period is not None:
        for supplier in visible:
            previous = prior_by_id.get(supplier.supplier_id)
            if previous is None:
                continue
            was = display.total(
                [(amount, code) for code, amount in previous.by_currency.items()]
            ).value
            if was is not None:
                prior_converted[supplier.supplier_id] = was
            change = _pct(supplier.converted - was, was) if was else None
            if change is None:
                continue
            deltas[supplier.supplier_id] = change
            if was < base_floor:
                negligible.add(supplier.supplier_id)
                facts.append(Fact(code=FactCode.NEGLIGIBLE_BASE, entity=supplier.name,
                                  entity_ref=supplier.supplier_id, value=was,
                                  type=ColumnType.MONEY, currency=currency))
                continue
            facts.append(Fact(code=FactCode.PERIOD_DELTA, entity=supplier.name,
                              entity_ref=supplier.supplier_id, value=change,
                              type=ColumnType.DELTA))

    # How the book as a whole moved, which is the figure a mover is read
    # against: +200% on one supplier means something different when everything
    # else moved with it.
    book_delta: Optional[Decimal] = None
    if prior_period is not None and prior_rows:
        prior_total = display.total(
            [(amount, code) for supplier in _group(prior_rows)
             for code, amount in supplier.by_currency.items()]
        ).value
        if prior_total:
            book_delta = _pct((population_total or Decimal(0)) - prior_total, prior_total)
            if book_delta is not None:
                facts.append(Fact(code=FactCode.PERIOD_DELTA, value=book_delta,
                                  type=ColumnType.DELTA, unit=prior_period.label))

    for supplier in visible:
        if len(supplier.currencies) > 1:
            facts.append(Fact(code=FactCode.CURRENCY_MISMATCH, entity=supplier.name,
                              entity_ref=supplier.supplier_id,
                              value=Decimal(len(supplier.currencies)),
                              type=ColumnType.INT,
                              unit=", ".join(supplier.currencies)))

    # -- anomalies ---------------------------------------------------------
    if missing_currencies:
        names = ", ".join(missing_currencies)
        dropped = len(unconvertible)
        anomalies.append(Anomaly(
            code=AnomalyCode.UNCONVERTED_CURRENCY, severity=Severity.HIGH,
            subject=", ".join(missing_currencies[:2]),
            text=(f"No exchange rate for {names}; "
                  f"{dropped} supplier{'' if dropped == 1 else 's'} left out of the ranking "
                  f"and excluded from the total."),
            entity_refs=[s.supplier_id for s in unconvertible],
        ))
    if display.is_manual(display.target):
        anomalies.append(Anomaly(
            code=AnomalyCode.MANUAL_FX_RATE, severity=Severity.MEDIUM,
            text=f"Figures use a manually entered rate ({display.rate_note()}), not the live one.",
        ))
    if prior_period is not None and not prior_by_id:
        anomalies.append(Anomaly(
            code=AnomalyCode.MISSING_PERIOD_DATA, severity=Severity.LOW,
            text=f"No comparable spend in {prior_period.label}, so no change is shown.",
        ))

    # -- table -------------------------------------------------------------
    scope_filters = list(filters_applied or [])
    listed = list(visible)
    if lens is Lens.TREND:
        # The movers are chosen from the top suppliers by spend, and the answer
        # says so. Ranked across the whole book, the table fills with suppliers
        # that went from £40 to £400 — true, and no use to anybody.
        #
        # Only the ones whose change was actually measured are listed. Live,
        # nine of ten rows had no prior-year spend, so a table asking what moved
        # answered with eight blanks — and a supplier that was never measured
        # has not moved, it is unknown.
        if len(ranked) > len(listed):
            scope_filters.append(f"top {len(listed)} by spend")
        listed = [s for s in listed if s.supplier_id in deltas]
        listed.sort(key=lambda s: (s.supplier_id in negligible,
                                   -(deltas.get(s.supplier_id, _UNMOVED)), s.name))
        if not listed:
            anomalies.append(Anomaly(
                code=AnomalyCode.MISSING_PERIOD_DATA, severity=Severity.HIGH,
                text=(f"No supplier here has comparable spend in "
                      f"{prior_period.label if prior_period else 'the period before'}, "
                      "so no change can be shown.")))

    columns = [
        Column(key="rank", label="#", type=ColumnType.INT),
        Column(key="supplier", label="Supplier", type=ColumnType.TEXT),
        # The bar measures the primary column. In a movement table the order is
        # the movement, so a bar drawn on spend would show the biggest supplier
        # halfway down and read as a mistake.
        Column(key="spend", label=f"{_MEASURE_NOUN[measure]} spend", type=ColumnType.MONEY,
               currency=currency, is_primary=lens is not Lens.TREND),
    ]
    if lens is Lens.TREND:
        columns.append(Column(key="prior", label=f"{prior_period.label if prior_period else 'Prior'}",
                              type=ColumnType.MONEY, currency=currency))
    else:
        columns.append(Column(key="share", label="Share", type=ColumnType.PCT))
    if lens is Lens.CONCENTRATION:
        columns.append(Column(key="cumulative", label="Running share", type=ColumnType.PCT))
    if deltas:
        columns.append(Column(key="delta", label=f"vs {prior_period.label}",
                              type=ColumnType.DELTA))

    table_rows: List[Dict[str, object]] = []
    running = Decimal(0)
    for index, supplier in enumerate(listed, start=1):
        share = _pct(supplier.converted, population_total)
        row: Dict[str, object] = {
            "rank": index,
            "supplier": supplier.name,
            "entity_ref": supplier.supplier_id,
            "spend": supplier.converted,
            "share": share,
            "_flags": (["CURRENCY_MISMATCH"] if len(supplier.currencies) > 1 else [])
                      + (["NEGLIGIBLE_BASE"] if supplier.supplier_id in negligible else []),
        }
        if lens is Lens.CONCENTRATION and share is not None:
            running += share
            row["cumulative"] = running
        if lens is Lens.TREND and supplier.supplier_id in prior_converted:
            row["prior"] = prior_converted[supplier.supplier_id]
        if supplier.supplier_id in deltas:
            row["delta"] = deltas[supplier.supplier_id]
        table_rows.append(row)

    totals = None
    if population_total is not None:
        totals = {"supplier": f"All {population_count:,} suppliers",
                  "spend": population_total, "share": Decimal(100)}
        if book_delta is not None:
            totals["delta"] = book_delta

    manual = display.is_manual(display.target)
    scope = Scope(
        measure=measure,
        period_start=period.start.isoformat(),
        period_end=period.end.isoformat(),
        period_label=period.label,
        population=Population(entity="SUPPLIER", count=population_count),
        currency=currency,
        currency_basis=CurrencyBasis.MANUAL if manual else CurrencyBasis.CONVERTED,
        rate_note=display.rate_note(),
        filters_applied=scope_filters,
    )

    answer = AnalyticAnswer(
        answer_id=answer_id,
        scope=scope,
        headline=Headline(text="", confidence=Confidence.ASSERTED),
        table=Table(columns=columns, rows=table_rows, totals=totals),
        facts=facts,
        anomalies=anomalies,
        provenance=Provenance(source_counts={SOURCE_LABEL: invoice_count},
                              refreshed_at=refreshed_at, query_ref=QUERY_REFS[lens]),
    )
    return _with_templated_headline(answer, manual=manual, lens=lens)


_MEASURE_NOUN = {
    Measure.INVOICED: "Invoiced",
    Measure.PO: "Committed",
    Measure.PAID: "Paid",
}


def _as_billed_answer(
    *, suppliers, display, period, population_count, invoice_count,
    answer_id, refreshed_at, top_n, measure, filters_applied,
) -> AnalyticAnswer:
    """The reader asked for figures as billed, so there is no single league table.

    Each currency is ranked on its own and the answer says the orders are not
    comparable. Presenting one combined order would be the original defect.
    """
    by_currency: Dict[str, List[tuple[str, str, Decimal]]] = {}
    for supplier in suppliers:
        for currency, amount in supplier.by_currency.items():
            by_currency.setdefault(currency, []).append(
                (supplier.supplier_id, supplier.name, amount))

    rows: List[Dict[str, object]] = []
    # Largest currency first by row count, then alphabetically, so the order is
    # stable rather than dictionary-dependent.
    for currency in sorted(by_currency, key=lambda c: (-len(by_currency[c]), c)):
        entries = sorted(by_currency[currency], key=lambda e: (-e[2], e[1]))[:top_n]
        for index, (supplier_id, name, amount) in enumerate(entries, start=1):
            rows.append({"rank": index, "currency": currency, "supplier": name,
                         "entity_ref": supplier_id, "spend": amount, "_flags": []})

    columns = [
        Column(key="rank", label="#", type=ColumnType.INT),
        Column(key="currency", label="Currency", type=ColumnType.TEXT),
        Column(key="supplier", label="Supplier", type=ColumnType.TEXT),
        Column(key="spend", label=f"{_MEASURE_NOUN[measure]} spend (as billed)",
               type=ColumnType.MONEY, is_primary=True),
    ]

    scope = Scope(
        measure=measure,
        period_start=period.start.isoformat(),
        period_end=period.end.isoformat(),
        period_label=period.label,
        population=Population(entity="SUPPLIER", count=population_count),
        currency="",
        currency_basis=CurrencyBasis.NATIVE,
        rate_note=None,
        filters_applied=filters_applied,
    )

    currencies = sorted(by_currency)
    anomaly = Anomaly(
        code=AnomalyCode.RANKING_NOT_COMPARABLE, severity=Severity.HIGH,
        text=("Figures are shown as billed, so suppliers are ranked within each "
              f"currency only ({', '.join(currencies)}). Pick a display currency "
              "to rank them against each other."),
    )

    headline = ("Suppliers are ranked within each currency; as-billed figures cannot be "
                "ranked against each other."
                if rows else f"No invoiced spend in {period.label}.")

    return AnalyticAnswer(
        answer_id=answer_id,
        scope=scope,
        headline=Headline(text=headline, confidence=Confidence.UNASSESSED),
        table=Table(columns=columns, rows=rows, totals=None),
        facts=[],
        anomalies=[anomaly],
        provenance=Provenance(source_counts={SOURCE_LABEL: invoice_count},
                              refreshed_at=refreshed_at, query_ref=QUERY_REF),
    )


def _with_templated_headline(answer: AnalyticAnswer, *, manual: bool,
                             lens: "Lens" = None) -> AnalyticAnswer:
    """The headline the answer ships with until a model earns the right to write it.

    Built from the facts, so it passes the same grounding check the insight
    writer's sentence will face. A fallback held to a weaker standard than the
    thing it replaces is not a fallback.

    One per lens, because a headline that describes the wrong table is worse
    than none: a concentration answer led with its largest supplier says
    nothing about concentration, and a movement answer led the same way names
    the biggest supplier rather than the one that moved.
    """
    facts = {fact.code: fact for fact in answer.facts}
    rows = answer.table.rows
    confidence = Confidence.UNASSESSED if manual else Confidence.ASSERTED

    if not rows:
        # An empty movement table is not an empty period: there is spend, there
        # is simply nothing to compare it with, and saying otherwise would be a
        # false statement about the corpus.
        text = (answer.anomalies[0].text if lens is Lens.TREND and answer.anomalies
                else f"No {answer.scope.measure.value.lower()} spend in "
                     f"{answer.scope.period_label}.")
        return answer.model_copy(update={
            "headline": Headline(text=text, confidence=Confidence.UNASSESSED)})

    if lens is Lens.CONCENTRATION:
        return answer.model_copy(update={
            "headline": Headline(text=_concentration_headline(answer, facts),
                                 confidence=confidence)})
    if lens is Lens.TREND:
        return answer.model_copy(update={
            "headline": Headline(text=_trend_headline(answer, facts),
                                 confidence=confidence)})

    leader = rows[0]["supplier"]
    share = facts.get(FactCode.TOP_1_SHARE)
    ratio = facts.get(FactCode.TOP_1_TO_TOP_2_RATIO)

    sentence = f"{leader} is the largest supplier"
    if share is not None:
        sentence += f" at {share.display} of {answer.scope.measure.value.lower()} spend"
    # A ratio near 1 is not a lead. "1.0 times the next" is a sentence
    # pretending to be a finding, and the live corpus produces exactly that:
    # £288.4K against £274.9K at the top of FY26. The fact still exists for the
    # insight writer; the template simply declines to narrate it.
    if ratio is not None and ratio.value is not None and ratio.value >= MEANINGFUL_LEAD:
        sentence += f", {ratio.display} times the next"
    text = sentence + "."

    breach = facts.get(FactCode.CONCENTRATION_THRESHOLD_BREACHED)
    if breach is not None:
        text += (f" The top {breach.unit} hold {breach.display} of it between them, "
                 "above the concentration threshold.")

    return answer.model_copy(update={"headline": Headline(text=text, confidence=confidence)})


def _concentration_headline(answer: AnalyticAnswer, facts: Dict[FactCode, Fact]) -> str:
    """Concentration is a property of the group, so the group leads the sentence."""
    rows = answer.table.rows
    group = facts.get(FactCode.TOP_N_SHARE_OF_TOTAL)
    breach = facts.get(FactCode.CONCENTRATION_THRESHOLD_BREACHED)
    top_1 = facts.get(FactCode.TOP_1_SHARE)
    measure = answer.scope.measure.value.lower()

    if group is None:
        return f"The top {len(rows)} suppliers by {measure} spend."
    text = (f"The top {group.unit or len(rows)} suppliers hold {group.display} of "
            f"{measure} spend")
    text += ", above the concentration threshold." if breach is not None else "."
    if top_1 is not None and rows:
        text += f" {rows[0]['supplier']} alone holds {top_1.display}."
    return text


def _trend_headline(answer: AnalyticAnswer, facts: Dict[FactCode, Fact]) -> str:
    """The mover, read against how the book as a whole moved."""
    rows = answer.table.rows
    measure = answer.scope.measure.value.lower()
    book = next((f for f in answer.facts
                 if f.code is FactCode.PERIOD_DELTA and f.entity is None), None)
    moved = [row for row in rows
             if row.get("delta") is not None and "NEGLIGIBLE_BASE" not in (row.get("_flags") or [])]
    if not moved:
        return (f"No comparable {measure} spend in {answer.scope.filters_applied[0]}"
                if answer.scope.filters_applied
                else f"No comparable {measure} spend in the year before, so no change is shown.")

    top = moved[0]
    delta = next((f for f in answer.facts
                  if f.code is FactCode.PERIOD_DELTA
                  and f.entity_ref == top.get("entity_ref")), None)
    text = f"{top['supplier']} moved most"
    if delta is not None:
        text += f", {delta.display}"
    text += "."
    if book is not None:
        text += (f" Across all {answer.scope.population.label()}, {measure} spend "
                 f"moved {book.display}.")
    return text
