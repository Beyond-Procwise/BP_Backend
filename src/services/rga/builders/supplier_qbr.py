"""The supplier business review (QBR) fact pack: one supplier, one period, measured (2026-10-20).

A QBR says how a supplier did over a quarter and how that compares with the quarter before. This
builder states only what the platform can trace to its records:

  * what was spent with the supplier, the same for the equal window just before the period, and the
    change between them;
  * how many deals, and how many findings are open against them (and how many of those are
    overbilling against the purchase order);
  * price variance: what was invoiced above or below the PO price on matched lines, by the
    ``ppv.purchase_price_variance`` formula, and how many lines were billed off the PO price.

What it cannot yet measure it says, with the reason, and the composer is told to recommend nothing
on it (the same rule as the supplier criticality review):

  * on-time and in-full delivery: the corpus holds no promised delivery dates and no goods receipts
    (audit rec 7, decision D-25). When receipts exist the pack says the measure is not yet wired
    here, rather than claiming there are none;
  * contract coverage: contracts number their suppliers differently from deals and nothing links
    the two.

Amounts are converted one currency at a time, never summed across currencies and never at par; a
currency with no rate drops the money figures to not assessed together, exactly as the executive
summary and the criticality review do. The supplier's name is not on any figure: a name with digits
in it, copied into prose by the composer, is a typed number the post-check refuses.
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from typing import Any, List, Optional, Tuple

from src.services.analytics.models import Confidence
from src.services.rga.builders.exec_procurement_summary import _PERIOD, _fetch, _rates
from src.services.rga.factpack import FactBuilder, register
from src.services.rga.models import FindingCode, FormatHint, Severity
from src.services.supplier_performance.ppv import purchase_price_variance

REPORT_TYPE_ID = "supplier_qbr"

SECTION_ORDER = ["summary", "spend", "price", "findings", "gaps"]

_SPEND = f"""
SELECT invoice_total, currency
  FROM proc.bp_deal_overview
 WHERE supplier_id = %s
   AND {_PERIOD}
   AND invoice_total IS NOT NULL
"""

_DEALS = f"""
SELECT COUNT(*)::int
  FROM proc.bp_deal_overview
 WHERE supplier_id = %s
   AND {_PERIOD}
"""

_FINDINGS = f"""
SELECT COUNT(*)::int, COUNT(*) FILTER (WHERE f.rule_id = 'cumulative_total')::int
  FROM proc.bp_detection_finding f
  JOIN proc.bp_deal_overview o ON o.deal_id = f.deal_id
 WHERE f.status = 'open'
   AND o.supplier_id = %s
   AND {_PERIOD.replace('deal_date', 'o.deal_date').replace('first_activity_date', 'o.first_activity_date')}
"""

# Invoice lines matched to exactly one PO line on (po_id, item_id), billed in the period. Each row
# carries both header currencies so a pair priced in different currencies can be left out.
_PRICE_LINES = """
WITH pol AS (
    SELECT l.po_id, l.item_id, MIN(l.unit_price) AS up, MIN(h.currency) AS cur, COUNT(*) AS n
      FROM proc.bp_po_line_items_trgt l
      JOIN proc.bp_purchase_order_trgt h USING (po_id)
     WHERE l.item_id IS NOT NULL AND l.unit_price IS NOT NULL
     GROUP BY l.po_id, l.item_id
)
SELECT pol.up, il.unit_price, il.quantity, h.currency, pol.cur
  FROM proc.bp_invoice_line_items_trgt il
  JOIN proc.bp_invoice_trgt h ON h.invoice_id = il.invoice_id
  JOIN pol ON pol.po_id = il.po_id AND pol.item_id = il.item_id
 WHERE pol.n = 1
   AND h.supplier_id = %s
   AND h.invoice_date BETWEEN %s AND %s
   AND il.unit_price IS NOT NULL AND il.quantity IS NOT NULL
"""

_RECEIPTS = "SELECT COUNT(*) FROM proc.bp_goods_receipt_trgt"

COMPOSER_NOTE = (
    "This is one supplier's review. Refer to it as 'the supplier': never write the supplier's "
    "name in a sentence. Compare the period with the period before only through the figures you "
    "are given; do not work out a change, a share or a total yourself, and never number anything "
    "yourself: a table cell is a figure id or words. On-time and in-full delivery and contract "
    "coverage were not measured: say so plainly, and recommend nothing on them. Recommend only "
    "on what is measured: spend, open findings and price variance.")


def _window(start: date, end: date) -> Tuple[date, date]:
    """The equal window ending the day before ``start``."""
    days = (end - start).days + 1
    prior_end = start - timedelta(days=1)
    return prior_end - timedelta(days=days - 1), prior_end


def _total(rows, display) -> Tuple[Optional[Decimal], bool, List[str], int]:
    """A period's spend in the display currency: value, complete?, missing currencies, excluded."""
    if not rows:
        return None, True, [], 0
    result = display.total([(amount, currency) for amount, currency in rows])
    return result.value, result.excluded == 0, list(result.excluded_currencies), result.excluded


@register(REPORT_TYPE_ID, section_order=SECTION_ORDER, composer_note=COMPOSER_NOTE,
          title="Supplier business review")
def build(fb: FactBuilder) -> None:
    """Measure the review for ``fb.scope`` (supplier_id, period_start, period_end, currency)."""
    supplier = str(fb.scope.get("supplier_id") or "").strip()
    if not supplier:
        raise ValueError("a supplier review is written for one supplier: give supplier_id")
    start, end = date.fromisoformat(fb.scope["period_start"]), date.fromisoformat(fb.scope["period_end"])
    if start > end:
        raise ValueError("the period ends before it starts")
    target = fb.scope.get("currency", "GBP")
    prior_start, prior_end = _window(start, end)

    spend_rows = _fetch(_SPEND, (supplier, start, end))
    prior_rows = _fetch(_SPEND, (supplier, prior_start, prior_end))
    rates, fetched_at, unavailable = _rates()

    spend = prior = None
    complete = prior_complete = True
    note = "rates unavailable"
    if rates and not unavailable:
        from src.services.analytics.currency import DisplayCurrency

        display = DisplayCurrency(target=target, rates=rates, fetched_at=fetched_at)
        note = display.rate_note()
        spend, complete, missing, excluded = _total(spend_rows, display)
        prior, prior_complete, prior_missing, prior_excluded = _total(prior_rows, display)
        if not (complete and prior_complete):
            fb.finding(code=FindingCode.PARTIAL_CURRENCY_CONVERSION, severity=Severity.HIGH,
                       detail=(f"{excluded + prior_excluded} deal(s) excluded from the {target} "
                               f"figures for want of a rate: "
                               f"{', '.join(sorted(set(missing + prior_missing)))}"),
                       blocks_release=False)
    # A partial conversion understates a total AND skews the change, so every money figure drops
    # to not assessed together.
    money_confidence = (Confidence.CORROBORATED if complete and prior_complete
                        else Confidence.UNASSESSED)

    # F0001..F0003 -- spend, the prior period's, and the change.
    if spend is None:
        fb.unmeasured(label=f"Invoiced spend ({target})", derivation="supplier_qbr.spend",
                      reason="no spend with the supplier in the period could be stated in the "
                             "display currency")
    else:
        fb.add(label=f"Invoiced spend ({target})", value=spend,
               derivation=f"supplier_qbr.spend @ {note}", confidence=money_confidence,
               format_hint=FormatHint.MONEY, currency=target)
    if prior is None:
        fb.unmeasured(label=f"Invoiced spend in the prior period ({target})",
                      derivation="supplier_qbr.prior_spend",
                      reason="no spend with the supplier in the equal period before could be "
                             "stated in the display currency")
    else:
        fb.add(label=f"Invoiced spend in the prior period ({target})", value=prior,
               derivation=f"supplier_qbr.prior_spend @ {note}", confidence=money_confidence,
               format_hint=FormatHint.MONEY, currency=target)
    if spend is not None and prior:
        fb.add(label="Change in spend on the prior period", value=(spend - prior) / prior * 100,
               derivation="supplier_qbr.spend_change", confidence=money_confidence,
               format_hint=FormatHint.PCT)
    else:
        fb.unmeasured(label="Change in spend on the prior period",
                      derivation="supplier_qbr.spend_change",
                      reason="a change needs spend in both periods, and one of them has none "
                             "that could be stated")

    # F0004..F0006 -- counts, which need no rates.
    (deals,), = _fetch(_DEALS, (supplier, start, end))
    fb.add(label="Deals in the period", value=Decimal(deals), derivation="supplier_qbr.deals",
           confidence=Confidence.ASSERTED, format_hint=FormatHint.INT, unit="deals")
    (open_total, open_over), = _fetch(_FINDINGS, (supplier, start, end))
    fb.add(label="Open findings against the period's deals", value=Decimal(open_total),
           derivation="supplier_qbr.open_findings", confidence=Confidence.ASSERTED,
           format_hint=FormatHint.INT, unit="findings")
    fb.add(label="Open overbilling findings", value=Decimal(open_over),
           derivation="supplier_qbr.open_overbilling_findings", confidence=Confidence.ASSERTED,
           format_hint=FormatHint.INT, unit="findings")

    _price_facts(fb, supplier, start, end, target, rates, unavailable, fetched_at)

    # F0010..F0011 -- what cannot be measured, said plainly (see the module docstring).
    (receipts,), = _fetch(_RECEIPTS, ())
    if receipts:
        reason = ("goods receipts are now on record but the on-time and in-full measure is not "
                  "yet wired into this report, so it is not given")
    else:
        reason = ("no promised delivery dates and no goods receipts are held for this "
                  "supplier's orders, so on-time and in-full delivery cannot be stated")
    fb.unmeasured(label="On-time and in-full delivery", derivation="supplier_qbr.on_time_in_full",
                  reason=reason)
    fb.unmeasured(label="Spend covered by a contract", derivation="supplier_qbr.contract_coverage",
                  reason="contracts on record number their suppliers differently from the "
                         "deals, and nothing links the two, so coverage cannot be measured")


def _price_facts(fb: FactBuilder, supplier: str, start: date, end: date, target: str,
                 rates: Any, unavailable: bool, fetched_at: Any) -> None:
    rows = _fetch(_PRICE_LINES, (supplier, start, end))
    lines, crossed = [], 0
    for po_price, inv_price, qty, inv_ccy, po_ccy in rows:
        if not inv_ccy or not po_ccy or inv_ccy != po_ccy:
            crossed += 1          # a price compared across currencies is not a variance
            continue
        lines.append({"po_unit_price": float(po_price), "invoice_unit_price": float(inv_price),
                      "quantity": float(qty), "currency": inv_ccy})
    if crossed:
        fb.finding(code=FindingCode.PARTIAL_CURRENCY_CONVERSION, severity=Severity.MEDIUM,
                   detail=f"{crossed} invoice line(s) were left out of the price variance: their "
                          "price and the PO's were in different currencies",
                   blocks_release=False)
    result = purchase_price_variance(
        lines=lines, buyer_currency=target,
        fx_rates=None if unavailable else rates,
        fx_table_date=fetched_at.date() if fetched_at is not None else None)
    if result is None or result["total_ppv"] is None:
        for label, derivation in (
                (f"Price variance against purchase orders ({target})", "price_variance"),
                ("Price variance as a share of PO-priced spend", "price_variance_pct"),
                ("Lines billed off the PO price", "lines_billed_off_po_price")):
            fb.unmeasured(label=label, derivation=f"supplier_qbr.{derivation}",
                          reason="no invoice line in the period could be matched to a purchase "
                                 "order line and compared in one currency")
        return
    complete = not result["excluded_currencies"]
    confidence = Confidence.CORROBORATED if complete else Confidence.UNASSESSED
    if not complete:
        fb.finding(code=FindingCode.PARTIAL_CURRENCY_CONVERSION, severity=Severity.MEDIUM,
                   detail=("price variance leaves out lines in a currency with no rate: "
                           + ", ".join(result["excluded_currencies"])), blocks_release=False)
    fb.add(label=f"Price variance against purchase orders ({target})",
           value=Decimal(repr(result["total_ppv"])), derivation="supplier_qbr.price_variance",
           confidence=confidence, format_hint=FormatHint.MONEY, currency=target)
    if result["ppv_pct"] is None:
        fb.unmeasured(label="Price variance as a share of PO-priced spend",
                      derivation="supplier_qbr.price_variance_pct",
                      reason="the PO-priced spend is nothing, so there is no share to state")
    else:
        fb.add(label="Price variance as a share of PO-priced spend",
               value=Decimal(repr(result["ppv_pct"])), derivation="supplier_qbr.price_variance_pct",
               confidence=confidence, format_hint=FormatHint.PCT)
    off = sum(1 for ln in lines
              if ln["po_unit_price"] != ln["invoice_unit_price"]
              and ln["currency"] not in result["excluded_currencies"])
    fb.add(label="Lines billed off the PO price", value=Decimal(off),
           derivation="supplier_qbr.lines_billed_off_po_price", confidence=Confidence.ASSERTED,
           format_hint=FormatHint.INT, unit="lines")
