"""The executive procurement summary, measured before anything is written.

Every figure below comes from one fixed query with an explicit period predicate.
No user text is interpolated into SQL, and nothing is inferred: a measure that
cannot be taken is recorded as unmeasured rather than defaulted to zero.

THREE THINGS THIS MODULE REFUSES TO DO, EACH BECAUSE OF A VERIFIED FACT ABOUT
THE CORPUS

1. **It does not read ``proc.bp_deal_kpis``.** That view is the obvious source
   for an exec summary and it is unscopeable — it has no date, entity or
   category column, so every figure in it is "all deals, ever"
   (``deal_count = 5042``). A pack that claims a period cannot draw from it
   without misstating its own scope. Aggregates are taken from
   ``proc.bp_deal_overview`` with a period predicate instead.

2. **It does not sum ``invoice_total`` across currencies.** The corpus holds
   five (GBP, EUR, USD, INR, AED) and a native sum is arithmetic on unlike
   units — in 2026-Q1 the INR deals alone total 36.6M against GBP's 2.5M, so a
   bare SUM reports a number that is not money in any currency. Amounts are
   converted one currency at a time through ``analytics.currency``, which
   excludes-and-counts a currency it has no rate for rather than assuming 1:1.

3. **It does not use ``deal_date``.** It is NULL on all 5,042 rows.
   ``first_activity_date`` is populated on all of them, so the period predicate
   is over ``COALESCE(deal_date, first_activity_date)`` — the same COALESCE the
   rest of the product settled on.

All three were verified against the database on 2026-09-12; see
docs/rga/discovery.md §3.8.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, List, Optional, Tuple

from src.services.analytics.models import Confidence
from src.services.rga.factpack import FactBuilder, register
from src.services.rga.models import FindingCode, FormatHint, Origin, Severity

logger = logging.getLogger(__name__)

REPORT_TYPE_ID = "exec_procurement_summary"

SECTION_ORDER = ["exec_summary", "spend", "coverage", "opportunities", "findings"]

# The period predicate, stated once. deal_date is NULL corpus-wide.
_PERIOD = "COALESCE(deal_date, first_activity_date) BETWEEN %s AND %s"

_DEAL_SHAPE = f"""
SELECT COUNT(*)::int                                         AS deals,
       COUNT(DISTINCT supplier_id)::int                      AS suppliers,
       SUM(CASE WHEN three_way_match THEN 1 ELSE 0 END)::int  AS matched,
       COUNT(three_way_match)::int                           AS matchable,
       AVG(cycle_days_quote_to_po)::numeric                  AS avg_cycle,
       COUNT(cycle_days_quote_to_po)::int                    AS cycle_n
  FROM proc.bp_deal_overview
 WHERE {_PERIOD}
"""

# Amounts come out per row, not pre-summed, so each is converted from its own
# native currency. Pre-summing here would be the defect rule 2 above prevents.
_DEAL_AMOUNTS = f"""
SELECT invoice_total, currency
  FROM proc.bp_deal_overview
 WHERE {_PERIOD}
   AND invoice_total IS NOT NULL
"""

# Realised savings are recorded outcomes, not the legacy realised_savings_gbp column: the
# ledger's CURRENT realised_saving rows for opportunities, scoped to the period by the
# outcome's own valid_from (superseded rows excluded). Priced and unpriced rows are
# counted separately -- count(o.amount_gbp) ignores NULLs, count(*) does not -- because an
# unconvertible-currency row (amount_gbp NULL) must not inflate the count that gates
# CORROBORATED vs unmeasured: a period holding only unpriced rows must not report a
# fabricated £0.00 as a measured fact.
_REALISED = """
SELECT coalesce(sum(o.amount_gbp), 0)::numeric           AS realised_gbp,
       count(o.amount_gbp)::int                          AS realised_n,
       count(*) FILTER (WHERE o.amount_gbp IS NULL)::int AS realised_unpriced_n
  FROM proc.bp_value_outcome o
 WHERE o.source_type = 'opportunity' AND o.outcome_type = 'realised_saving'
   AND o.valid_from BETWEEN %s AND %s
   AND NOT EXISTS (SELECT 1 FROM proc.bp_value_outcome s WHERE s.supersedes_id = o.outcome_id)
"""

_OPPORTUNITIES = """
SELECT COUNT(*)::int                        AS n,
       SUM(financial_impact_gbp)::numeric   AS identified_gbp
  FROM proc.bp_opportunity
 WHERE detected_on::date BETWEEN %s AND %s
"""

# Everything counted as "saved" this period: money stopped (avoided), credited back
# (recovered) or booked as a realised opportunity saving. Grouped by outcome_type so the
# caller can both total it and, if needed, see the split. Priced/unpriced counted
# separately for the same reason as _REALISED above.
_SAVED = """
SELECT o.outcome_type,
       coalesce(sum(o.amount_gbp), 0)::numeric           AS gbp,
       count(o.amount_gbp)::int                          AS priced_n,
       count(*) FILTER (WHERE o.amount_gbp IS NULL)::int AS unpriced_n
  FROM proc.bp_value_outcome o
 WHERE o.outcome_type IN ('avoided', 'recovered', 'realised_saving')
   AND o.valid_from BETWEEN %s AND %s
   AND NOT EXISTS (SELECT 1 FROM proc.bp_value_outcome s WHERE s.supersedes_id = o.outcome_id)
 GROUP BY o.outcome_type
"""


def _rates() -> Tuple[dict, Any, bool]:
    """The published USD-quoted batch, or nothing.

    Never an invented rate and never a 1:1 fallback: an unconvertible figure is
    dropped and said to be dropped. Mirrors ``analytics.repository._load_rates``.
    """
    try:
        from src.repositories import fx_rate_repo

        batch = fx_rate_repo.get_latest_batch()
    except Exception:  # noqa: BLE001 - unavailable rates are a finding, not a crash
        logger.exception("rga: exchange rates unavailable")
        return {}, None, True
    if not batch:
        return {}, None, True
    return (batch.get("rates") or {}), batch.get("fetched_at"), False


def _fetch(sql: str, params: tuple) -> List[tuple]:
    from src.services.db import get_conn

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


@register(REPORT_TYPE_ID)
def build(fb: FactBuilder) -> None:
    """Measure the exec summary for ``fb.scope``.

    Scope keys: ``period_start``, ``period_end`` (ISO dates), ``currency``
    (the display currency, default GBP) and ``period_label``.
    """
    start = fb.scope["period_start"]
    end = fb.scope["period_end"]
    target = fb.scope.get("currency", "GBP")

    # ---- shape of the book ------------------------------------------------
    rows = _fetch(_DEAL_SHAPE, (start, end))
    deals, suppliers, matched, matchable, avg_cycle, cycle_n = rows[0]

    fb.add(label="Deals in period", value=Decimal(deals),
           derivation="exec_summary.deals", confidence=Confidence.ASSERTED,
           format_hint=FormatHint.INT, unit="deals")

    fb.add(label="Suppliers transacted with", value=Decimal(suppliers),
           derivation="exec_summary.suppliers", confidence=Confidence.ASSERTED,
           format_hint=FormatHint.INT, unit="suppliers")

    # ---- invoiced spend ---------------------------------------------------
    # A total over five currencies, or no total at all. There is no third option
    # that is honest.
    amounts = _fetch(_DEAL_AMOUNTS, (start, end))
    rates, fetched_at, unavailable = _rates()

    if unavailable or not rates:
        fb.unmeasured(
            label=f"Invoiced spend ({target})",
            derivation="exec_summary.invoiced_total",
            reason="no published exchange rates; a multi-currency total cannot "
                   "be stated and will not be guessed at par",
        )
    else:
        from src.services.analytics.currency import DisplayCurrency

        display = DisplayCurrency(target=target, rates=rates, fetched_at=fetched_at)
        result = display.total(amounts)
        if result.value is None:
            fb.unmeasured(
                label=f"Invoiced spend ({target})",
                derivation="exec_summary.invoiced_total",
                reason=f"no deal amount in the period could be converted "
                       f"to {target}, so no total is stated",
            )
        else:
            # Fully reconciled across currencies against an independent rate
            # source is CORROBORATED. A partial conversion understates the
            # figure, so it drops to UNASSESSED and says which currencies are
            # missing -- an understated total presented as measured is worse
            # than no total.
            complete = result.excluded == 0
            fb.add(
                label=f"Invoiced spend ({target})",
                value=result.value,
                derivation=f"exec_summary.invoiced_total @ {display.rate_note()}",
                confidence=Confidence.CORROBORATED if complete else Confidence.UNASSESSED,
                format_hint=FormatHint.MONEY, currency=target,
            )
            if not complete:
                fb.finding(
                    code=FindingCode.PARTIAL_CURRENCY_CONVERSION,
                    severity=Severity.HIGH,
                    detail=(f"{result.excluded} deal(s) excluded from the "
                            f"{target} total for want of a rate: "
                            f"{', '.join(result.excluded_currencies)}"),
                    blocks_release=False,
                )

    # ---- control coverage -------------------------------------------------
    if matchable:
        fb.add(label="Three-way match rate",
               value=(Decimal(matched) / Decimal(matchable) * 100),
               derivation="exec_summary.three_way_match_rate",
               confidence=Confidence.CORROBORATED, format_hint=FormatHint.PCT)
    else:
        fb.unmeasured(label="Three-way match rate",
                      derivation="exec_summary.three_way_match_rate",
                      reason="no deal in the period carries a match result")

    if cycle_n and avg_cycle is not None:
        fb.add(label="Average quote-to-PO cycle", value=Decimal(avg_cycle),
               derivation="exec_summary.avg_cycle_quote_to_po",
               confidence=Confidence.CORROBORATED, format_hint=FormatHint.INT,
               unit="days")
    else:
        fb.unmeasured(label="Average quote-to-PO cycle",
                      derivation="exec_summary.avg_cycle_quote_to_po",
                      reason="no deal in the period records both a quote and a PO date")

    # ---- opportunities ----------------------------------------------------
    opp_rows = _fetch(_OPPORTUNITIES, (start, end))
    n_opps, identified = opp_rows[0]

    # A COUNT is always measurable, and a measured zero is a finding in itself.
    fb.add(label="Opportunities identified", value=Decimal(n_opps),
           derivation="exec_summary.opportunities_identified",
           confidence=Confidence.ASSERTED, format_hint=FormatHint.INT,
           unit="opportunities")

    if identified is None:
        fb.unmeasured(label="Identified value (GBP)",
                      derivation="exec_summary.opportunity_identified_value",
                      reason="no opportunity was detected in this period, so "
                             "there is no value to state — which is not the same "
                             "as a value of zero")
    else:
        fb.add(label="Identified value (GBP)", value=Decimal(identified),
               derivation="exec_summary.opportunity_identified_value",
               confidence=Confidence.CORROBORATED, format_hint=FormatHint.MONEY,
               currency="GBP")

    realised, realised_n, realised_unpriced_n = _fetch(_REALISED, (start, end))[0]

    if realised_n:
        # A priced figure exists; note (in the derivation, never the unmeasured reason --
        # that may carry no digit) how many further rows this total excludes.
        derivation = "exec_summary.opportunity_realised_value"
        if realised_unpriced_n:
            derivation += (f" (excludes {realised_unpriced_n} realised saving(s) whose "
                           f"currency could not be converted to GBP)")
        fb.add(label="Realised savings (GBP)", value=Decimal(realised or 0),
               derivation=derivation,
               confidence=Confidence.CORROBORATED, format_hint=FormatHint.MONEY,
               currency="GBP", provenance_id="proc.bp_value_outcome")
    elif realised_unpriced_n:
        # Every realised row this period is unpriced: a real saving happened, but no GBP
        # figure can be stated for it -- and £0.00 would be a fabricated one.
        fb.unmeasured(label="Realised savings (GBP)",
                      derivation="exec_summary.opportunity_realised_value",
                      reason="an opportunity saving was realised in this period but its "
                             "currency could not be converted to GBP, so no total is "
                             "stated")
    else:
        fb.unmeasured(label="Realised savings (GBP)",
                      derivation="exec_summary.opportunity_realised_value",
                      reason="no opportunity saving was recorded as realised in "
                             "this period")

    # ---- saved (avoided + recovered + realised) ---------------------------
    saved_rows = _fetch(_SAVED, (start, end))
    saved_total = sum(v for _, v, _, _ in saved_rows)
    saved_priced_n = sum(pn for _, _, pn, _ in saved_rows)
    saved_unpriced_n = sum(un for _, _, _, un in saved_rows)
    if saved_priced_n:
        derivation = "exec_summary.value_saved"
        if saved_unpriced_n:
            derivation += (f" (excludes {saved_unpriced_n} outcome(s) whose currency "
                           f"could not be converted to GBP)")
        fb.add(label="Saved (GBP)", value=Decimal(saved_total),
               derivation=derivation,
               confidence=Confidence.CORROBORATED, format_hint=FormatHint.MONEY,
               currency="GBP", provenance_id="proc.bp_value_outcome")
    elif saved_unpriced_n:
        fb.unmeasured(label="Saved (GBP)", derivation="exec_summary.value_saved",
                      reason="money was stopped, recovered or realised in this period "
                             "but its currency could not be converted to GBP, so no "
                             "total is stated")
    else:
        fb.unmeasured(label="Saved (GBP)", derivation="exec_summary.value_saved",
                      reason="no money was recorded as stopped, recovered or realised "
                             "in this period")
