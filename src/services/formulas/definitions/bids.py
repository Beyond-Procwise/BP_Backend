"""Bid normalisation, registered. Delegates to ``src.services.bids.normalise``."""
from __future__ import annotations

from datetime import date

from src.services.bids import normalise as _nb

from ..contract import BOOLEAN, CURRENCY_CODE, DAYS, LABEL, RATIO, RECORD, ROWS, TEXT, Output, Term
from ..registry import GoldenVector, formula
from ..unassessed import UNASSESSED, decline

_MASTER = [{"line_id": "L1", "description": "Bracket", "uom": "EA", "qty": 100}]
_BIDS = [
    {"supplier_id": "A", "currency": "GBP", "incoterm": "DAP", "payment_days": 60,
     "lines": [{"line_id": "L1", "unit_price": 10.0, "uom": "EA"}]},
    {"supplier_id": "B", "currency": "GBP", "incoterm": "DAP", "payment_days": 60,
     "lines": [{"line_id": "L1", "unit_price": 9.0, "uom": "EA"}]},
]
_CASE = dict(line_master=_MASTER, bids=_BIDS, base_currency="GBP", fx_rates={"GBP": 1.0},
             fx_type="none needed", fx_date="2026-10-01", target_incoterm="DAP", target_payment_days=60,
             cost_of_capital=0.1)


@formula(
    "bids.normalise",
    version="1.0.0",
    owner="bids",
    purpose="Supplier bids on a like-for-like basis: same lines, units, currency and payment terms, with "
            "every adjustment shown beside the submitted price",
    effective_from=date(2026, 10, 21),
    inputs=[
        Term("line_master", ROWS, "rows like {line_id, description, uom, qty}: the lines and evaluated volumes"),
        Term("bids", ROWS, "one per supplier: {supplier_id, currency, incoterm, payment_days, validity_date, "
             "lines:[{line_id|description, unit_price, uom, price_unit, price_tiers, alternate_flag}], "
             "adjustments:[{type, amount, currency, effect: add|subtract, note}], exceptions:[{text, category, "
             "severity: priced|unpriced|blocking, applies_to, cost_loading}]}"),
        Term("base_currency", CURRENCY_CODE, "the currency every price is shown in"),
        Term("fx_rates", RECORD, "base units per 1 unit of each currency; none is looked up"),
        Term("fx_type", TEXT, "e.g. ECB reference", required=False),
        Term("fx_date", TEXT, "ISO date of the rates", required=False),
        Term("target_incoterm", TEXT, required=False),
        Term("target_payment_days", DAYS, "default 60", required=False, minimum=0.0),
        Term("cost_of_capital", RATIO, "annual; without it payment terms are not adjusted", required=False,
             minimum=0.0, maximum=1.0),
        Term("uom_conversions", ROWS, "{line_id|null, from_uom, to_uom, factor: target units per source unit}",
             required=False),
        Term("missing_line_policy", LABEL, required=False, allowed=_nb.POLICIES),
        Term("last_paid_prices", RECORD, "line_id -> price, for fill_with_last_paid_price", required=False),
        Term("award_date", TEXT, required=False),
        Term("sealed", BOOLEAN, "true for a sealed event", required=False),
        Term("opened_at", TEXT, required=False), Term("as_of", TEXT, required=False),
    ],
    output=Output("dict", RATIO,
                  "matrix (line x supplier cells tracing raw to normalised price), suppliers (totals, completeness, "
                  "missing and unmatched lines, bid adjustments), exceptions, alternates, lowest_per_line, "
                  "cherry_pick_total, best_single_supplier, warnings, assumptions; UNASSESSED if the inputs cannot "
                  "be normalised"),
    notes=(
        "Per cell: tier at the evaluated volume (a break uses the higher tier), price per unit and per master unit "
        "of measure, currency at the supplied rate, then payment terms: price x (1 + r) ^ ((target days - bid "
        "days) / 365). Freight, insurance and exception loadings are shown as bid-level adjustments beside the "
        "price. A rate that is missing, zero or negative makes the line non-comparable, never one-to-one. A bid "
        "on a different Incoterm with no priced adjustment, or on different payment terms with no cost of "
        "capital, is listed as not normalised. Alternates are kept out of the base total. A missing line is "
        "flagged by default; any fill is labelled an assumption. Sealed events refuse a comparison before the "
        "opening time. Submitted values are never changed."
    ),
    golden=[
        GoldenVector(inputs=_CASE, expected={"cherry_pick_total": 900.0,
                                             "best_single_supplier": {"supplier_id": "B", "total": 900.0}},
                     tolerance=1e-9, note="same terms and currency: B at 9.00 x 100 beats A at 10.00 x 100"),
        GoldenVector(inputs={**_CASE, "sealed": True, "opened_at": "2026-10-05T12:00:00Z",
                             "as_of": "2026-10-05T11:00:00Z"},
                     expected=UNASSESSED, note="sealed bids are not compared before the opening time"),
    ],
)
def bids_normalise(
    line_master, bids, base_currency, fx_rates, fx_type=None, fx_date=None, target_incoterm=None,
    target_payment_days=None, cost_of_capital=None, uom_conversions=None, missing_line_policy=None,
    last_paid_prices=None, award_date=None, sealed=None, opened_at=None, as_of=None,
):
    try:
        return _nb.normalise_bids(
            line_master=line_master, bids=bids, base_currency=base_currency, fx_rates=fx_rates or {},
            fx_type=fx_type, fx_date=fx_date, target_incoterm=target_incoterm,
            target_payment_days=60 if target_payment_days is None else target_payment_days,
            cost_of_capital=cost_of_capital, uom_conversions=uom_conversions or (),
            missing_line_policy=missing_line_policy or "flag_only", last_paid_prices=last_paid_prices,
            award_date=award_date, sealed=bool(sealed), opened_at=opened_at, as_of=as_of)
    except _nb.NormaliseError as exc:
        return decline(str(exc))
