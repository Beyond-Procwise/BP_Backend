"""Supplier performance measures, registered: purchase price variance and OTIF.

Each is one dimension of a supplier scorecard. The scorecard itself is ``supplier.composite_score``,
which scores a supplier only on the dimensions we hold for them and never charges them for a gap in
our data; these two formulas are inputs to it. Delegates to ``src.services.supplier_performance``.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Mapping, Optional, Sequence

from src.services.supplier_performance import otif as _otif
from src.services.supplier_performance import ppv as _ppv

from ..contract import CURRENCY_CODE, DATE, DAYS, RATIO, RECORD, ROWS, Output, Term
from ..contract import MONEY
from ..registry import GoldenVector, formula
from ..unassessed import UNASSESSED

_OWNER = "supplier_performance"
_FROM = date(2026, 10, 20)


@formula(
    "ppv.purchase_price_variance",
    version="1.0.0",
    owner=_OWNER,
    purpose="What was paid above (+) or below (-) the PO price, on the quantity billed",
    effective_from=_FROM,
    inputs=[
        Term("lines", ROWS, "matched lines: po_unit_price, invoice_unit_price, quantity, currency"),
        Term("buyer_currency", CURRENCY_CODE, "report in this currency; needs a dated table to "
             "convert", required=False),
        Term("fx_rates", RECORD, "dated USD-quoted table: units of currency per 1 USD",
             required=False),
        Term("fx_table_date", DATE, "required whenever a conversion is needed", required=False),
    ],
    output=Output("dict", MONEY,
                  "total_ppv, unfavourable, favourable, standard_spend, ppv_pct, by_currency, "
                  "lines_assessed, lines_unassessed, excluded_currencies; UNASSESSED if no line "
                  "could be assessed"),
    notes=(
        "(invoice unit price - PO unit price) x quantity per matched line; positive = the supplier "
        "billed MORE than the PO price. ppv_pct is of the standard spend (PO price x quantity). A "
        "line with no usable price, quantity or currency is counted as unassessed, never as zero "
        "variance. Mixed currencies are not added without a dated table: by_currency still "
        "reports each, and a currency with no rate is excluded and named, never taken 1:1."
    ),
    golden=[
        GoldenVector(
            inputs={"lines": [
                {"po_unit_price": 10.0, "invoice_unit_price": 11.0, "quantity": 100.0,
                 "currency": "GBP"},
                {"po_unit_price": 5.0, "invoice_unit_price": 4.5, "quantity": 40.0,
                 "currency": "GBP"}]},
            expected={"total_ppv": 80.0, "unfavourable": 100.0, "favourable": -20.0,
                      "standard_spend": 1200.0, "lines_assessed": 2, "currency": "GBP"},
            tolerance=1e-9,
            note="overpaid 100 on one line, underpaid 20 on the other",
        ),
        GoldenVector(
            inputs={"lines": [{"po_unit_price": None, "invoice_unit_price": 1.0, "quantity": 1.0,
                               "currency": "GBP"}]},
            expected=UNASSESSED,
            note="no usable PO price: nothing is assessed, and it is not a zero variance",
        ),
    ],
)
def ppv_purchase_price_variance(
    lines: Sequence[Mapping[str, Any]], buyer_currency: Optional[str] = None,
    fx_rates: Optional[Mapping[str, Any]] = None, fx_table_date: Any = None,
):
    result = _ppv.purchase_price_variance(
        lines=lines, buyer_currency=buyer_currency, fx_rates=fx_rates, fx_table_date=fx_table_date)
    return UNASSESSED if result is None else result


@formula(
    "otif.delivery_performance",
    version="1.0.0",
    owner=_OWNER,
    purpose="The share of deliveries that were on time AND in full, over the deliveries that can "
            "be assessed",
    effective_from=_FROM,
    inputs=[
        Term("deliveries", ROWS, "one row per delivery obligation: promised_date, received_date, "
             "ordered_qty, received_qty"),
        Term("late_tolerance_days", DAYS, "days after the promised date still counted on time",
             minimum=0.0),
        Term("in_full_tolerance", RATIO, "share short of the ordered quantity still counted in "
             "full", minimum=0.0, maximum=1.0),
        Term("early_tolerance_days", DAYS, "days before the promised date still on time; "
             "unstated = early is not penalised", minimum=0.0, required=False),
    ],
    output=Output("dict", RATIO,
                  "on_time_rate, in_full_rate, otif_rate, otif_score (0-100) and the number "
                  "assessed for each; UNASSESSED if nothing could be assessed"),
    notes=(
        "Each rate is over the deliveries assessable for it. A delivery with no promised date, no "
        "receipt, no quantity or an unreadable date is unassessed for that measure, not a failure: "
        "our missing data is not charged to the supplier. The tolerances are inputs from governed "
        "policy, not defaults in code. As of 2026-10-20 the corpus holds no promised dates and no "
        "receipts, so on live data this answers UNASSESSED until they are loaded."
    ),
    golden=[
        GoldenVector(
            inputs={"deliveries": [
                {"promised_date": "2026-03-10", "received_date": "2026-03-10",
                 "ordered_qty": 100.0, "received_qty": 100.0},
                {"promised_date": "2026-03-10", "received_date": "2026-03-12",
                 "ordered_qty": 100.0, "received_qty": 100.0},
                {"promised_date": "2026-03-10", "received_date": "2026-03-10",
                 "ordered_qty": 100.0, "received_qty": 90.0},
                {"promised_date": None, "received_date": None,
                 "ordered_qty": 100.0, "received_qty": 100.0}],
                "late_tolerance_days": 0, "in_full_tolerance": 0.0},
            expected={"on_time_rate": 2 / 3, "in_full_rate": 0.75, "otif_rate": 1 / 3,
                      "otif_assessed": 3, "unassessed": 1},
            tolerance=1e-9,
            note="the fourth has no dates: it counts for in-full only and is not a late delivery",
        ),
        GoldenVector(
            inputs={"deliveries": [{"promised_date": None, "received_date": None,
                                    "ordered_qty": None, "received_qty": None}],
                    "late_tolerance_days": 0, "in_full_tolerance": 0.0},
            expected=UNASSESSED,
            note="nothing assessable: no rate, which is not a rate of zero",
        ),
    ],
)
def otif_delivery_performance(
    deliveries: Sequence[Mapping[str, Any]], late_tolerance_days: float, in_full_tolerance: float,
    early_tolerance_days: Optional[float] = None,
):
    result = _otif.otif(
        deliveries=deliveries, late_tolerance_days=late_tolerance_days,
        in_full_tolerance=in_full_tolerance, early_tolerance_days=early_tolerance_days)
    return UNASSESSED if result is None else result
