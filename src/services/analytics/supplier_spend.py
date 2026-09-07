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

QUERY_REF = "supplier_spend_ranking/v1"
SOURCE_TABLE = "proc.bp_invoice_trgt"
DEFAULT_TOP_N = 10
DEFAULT_CONCENTRATION_THRESHOLD_PCT = Decimal("20")


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
) -> AnalyticAnswer:
    """The answer to "top N suppliers by spend", whole and self-describing."""

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
            if leader_share >= concentration_threshold_pct:
                facts.append(Fact(code=FactCode.CONCENTRATION_THRESHOLD_BREACHED,
                                  entity=leader.name, entity_ref=leader.supplier_id,
                                  value=leader_share, type=ColumnType.PCT))
        top_n_share = _pct(sum((s.converted for s in visible), Decimal(0)), population_total)
        if top_n_share is not None:
            facts.append(Fact(code=FactCode.TOP_N_SHARE_OF_TOTAL, value=top_n_share,
                              type=ColumnType.PCT, unit=str(len(visible))))
        if len(visible) > 1:
            ratio = _ratio(visible[0].converted, visible[1].converted)
            if ratio is not None:
                facts.append(Fact(code=FactCode.TOP_1_TO_TOP_2_RATIO, value=ratio,
                                  entity=leader.name, entity_ref=leader.supplier_id,
                                  type=ColumnType.TEXT))

    prior_by_id = {s.supplier_id: s for s in _group(prior_rows)}
    deltas: Dict[str, Decimal] = {}
    if prior_period is not None:
        for supplier in visible:
            previous = prior_by_id.get(supplier.supplier_id)
            if previous is None:
                continue
            was = display.total(
                [(amount, code) for code, amount in previous.by_currency.items()]
            ).value
            change = _pct(supplier.converted - was, was) if was else None
            if change is None:
                continue
            deltas[supplier.supplier_id] = change
            facts.append(Fact(code=FactCode.PERIOD_DELTA, entity=supplier.name,
                              entity_ref=supplier.supplier_id, value=change,
                              type=ColumnType.DELTA))

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
    columns = [
        Column(key="rank", label="#", type=ColumnType.INT),
        Column(key="supplier", label="Supplier", type=ColumnType.TEXT),
        Column(key="spend", label=f"{_MEASURE_NOUN[measure]} spend", type=ColumnType.MONEY,
               currency=currency, is_primary=True),
        Column(key="share", label="Share", type=ColumnType.PCT),
    ]
    if deltas:
        columns.append(Column(key="delta", label=f"vs {prior_period.label}",
                              type=ColumnType.DELTA))

    table_rows: List[Dict[str, object]] = []
    for index, supplier in enumerate(visible, start=1):
        row: Dict[str, object] = {
            "rank": index,
            "supplier": supplier.name,
            "entity_ref": supplier.supplier_id,
            "spend": supplier.converted,
            "share": _pct(supplier.converted, population_total),
            "_flags": ["CURRENCY_MISMATCH"] if len(supplier.currencies) > 1 else [],
        }
        if supplier.supplier_id in deltas:
            row["delta"] = deltas[supplier.supplier_id]
        table_rows.append(row)

    totals = None
    if population_total is not None:
        totals = {"supplier": f"All {population_count:,} suppliers",
                  "spend": population_total, "share": Decimal(100)}

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
        filters_applied=filters_applied or [],
    )

    answer = AnalyticAnswer(
        answer_id=answer_id,
        scope=scope,
        headline=Headline(text="", confidence=Confidence.ASSERTED),
        table=Table(columns=columns, rows=table_rows, totals=totals),
        facts=facts,
        anomalies=anomalies,
        provenance=Provenance(source_counts={SOURCE_TABLE: invoice_count},
                              refreshed_at=refreshed_at, query_ref=QUERY_REF),
    )
    return _with_templated_headline(answer, manual=manual)


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
        provenance=Provenance(source_counts={SOURCE_TABLE: invoice_count},
                              refreshed_at=refreshed_at, query_ref=QUERY_REF),
    )


def _with_templated_headline(answer: AnalyticAnswer, *, manual: bool) -> AnalyticAnswer:
    """The headline the answer ships with until a model earns the right to write it.

    Built from the facts, so it passes the same grounding check the insight
    writer's sentence will face. A fallback held to a weaker standard than the
    thing it replaces is not a fallback.
    """
    facts = {fact.code: fact for fact in answer.facts}
    rows = answer.table.rows

    if not rows:
        text = f"No {answer.scope.measure.value.lower()} spend in {answer.scope.period_label}."
        return answer.model_copy(update={
            "headline": Headline(text=text, confidence=Confidence.UNASSESSED)})

    leader = rows[0]["supplier"]
    share = facts.get(FactCode.TOP_1_SHARE)
    ratio = facts.get(FactCode.TOP_1_TO_TOP_2_RATIO)

    sentence = f"{leader} is the largest supplier"
    if share is not None:
        sentence += f" at {share.display} of {answer.scope.measure.value.lower()} spend"
    if ratio is not None:
        sentence += f", {ratio.display} times the next"
    text = sentence + "."

    if FactCode.CONCENTRATION_THRESHOLD_BREACHED in facts:
        text += " That is above the concentration threshold."

    confidence = Confidence.UNASSESSED if manual else Confidence.ASSERTED
    return answer.model_copy(update={"headline": Headline(text=text, confidence=confidence)})
