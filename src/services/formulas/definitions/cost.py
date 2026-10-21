"""Landed cost and total cost of ownership, registered.

Delegates to ``src.services.cost``. A quote's landed cost is what it costs to get the goods
to the buyer's delivery point; its TCO is the present value of everything it costs to
acquire, run and dispose of them. Neither ever treats an unmentioned cost as zero.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Mapping, Optional, Sequence

from src.services.cost import landed_cost as _landed
from src.services.cost import should_cost as _sc
from src.services.cost import tco as _tco

from ..contract import (
    BOOLEAN, COUNT, CURRENCY_CODE, DATE, FACTOR, LABEL, MONEY, PERCENT, RATIO, RECORD, ROWS, Output, Term,
)
from ..registry import GoldenVector, formula
from ..unassessed import UNASSESSED

_FROM = date(2026, 10, 20)

_LINES = [{"unit_price": 20.0, "quantity": 1000, "currency": "GBP"}]
_SPEC_CASE = dict(
    lines=_LINES, buyer_currency="GBP", cross_border=True, incoterm="FOB Shanghai",
    components=[{"key": "freight", "amount": 1200.0}, {"key": "brokerage", "amount": 150.0}],
    duty_rate=0.04, customs_basis="CIF", default_insurance_rate=0.003, vat_recoverable=True,
)


@formula(
    "landed_cost.total_cost",
    version="1.0.0",
    owner="cost",
    purpose="What it costs to get the ordered goods to the buyer's delivery point, with every "
            "cost that is owed but not stated flagged rather than counted as zero",
    effective_from=_FROM,
    inputs=[
        Term("lines", ROWS, "unit_price, quantity, currency, optional discount_pct and uom"),
        Term("buyer_currency", CURRENCY_CODE, "the currency the answer is in"),
        Term("cross_border", BOOLEAN, "False removes export and import items"),
        Term("incoterm", LABEL, "e.g. 'FOB Shanghai'; none = nothing but goods can be assessed",
             required=False),
        Term("evaluation_date", DATE, required=False),
        Term("fx_rates", RECORD, "dated USD-quoted table: units of currency per 1 USD",
             required=False),
        Term("fx_table_date", DATE, "required whenever a conversion is needed", required=False),
        Term("components", ROWS, "stated costs: key, amount, currency, pay_day, or status "
             "'included' / 'not_applicable'", required=False),
        Term("duty_rate", RATIO, "import duty rate, if known", minimum=0.0, maximum=1.0,
             required=False),
        Term("customs_basis", LABEL, "'CIF' (UK/EU) or 'transaction' (US)", required=False,
             allowed=("CIF", "transaction")),
        Term("default_insurance_rate", RATIO, "policy rate on 110% of goods + freight",
             minimum=0.0, maximum=1.0, required=False),
        Term("vat_recoverable", BOOLEAN, "recoverable import VAT is left out", required=False),
    ],
    output=Output("dict", MONEY,
                  "components (each with a status), landed_total, unit_landed, landed_status, "
                  "missing, estimated, not_stated; UNASSESSED if the goods cannot be priced"),
    notes=(
        "Statuses: stated, included, not_applicable, derived, estimated, missing, and not_stated "
        "(an optional charge the quote does not mention: listed to confirm, not counted as "
        "missing). The Incoterm decides what the buyer owes; with none, every unstated item "
        "except goods is missing. An amount in a currency the dated FX table has no rate for is "
        "missing, never converted 1:1. Insurance defaults to policy rate x 110% x (goods + "
        "freight) and is then 'estimated'; duty is rate x customs value ('CIF' = goods + freight "
        "+ insurance). A total with anything missing is a lower bound (status incomplete)."
    ),
    golden=[
        GoldenVector(
            inputs=_SPEC_CASE,
            expected={
                "landed_total": 22270.76, "unit_landed": 22.27076, "landed_status": "incomplete",
                "missing": ["terminal_handling", "delivery_order_fee", "on_carriage", "unloading"],
                "estimated": ["cargo_insurance"],
            },
            tolerance=0.005,
            note="1,000 x GBP 20, FOB Shanghai, duty 4% on CIF, freight 1,200, brokerage 150",
        ),
        GoldenVector(
            inputs={**_SPEC_CASE, "incoterm": None},
            expected={"landed_status": "incomplete", "landed_total": 21350.0},
            tolerance=0.005,
            note="no Incoterm: only the stated amounts count; the rest is missing",
        ),
        GoldenVector(
            inputs={**_SPEC_CASE, "lines": [{"unit_price": 20.0, "quantity": 1000,
                                             "currency": "NZD"}]},
            expected=UNASSESSED,
            note="NZD with no FX table: the goods cannot be priced, so nothing is assumed 1:1",
        ),
    ],
)
def landed_cost_total_cost(
    lines: Sequence[Mapping[str, Any]], buyer_currency: str, cross_border: bool,
    incoterm: Optional[str] = None, evaluation_date: Optional[date] = None,
    fx_rates: Optional[Mapping[str, Any]] = None, fx_table_date: Optional[date] = None,
    components: Sequence[Mapping[str, Any]] = (), duty_rate: Optional[float] = None,
    customs_basis: Optional[str] = None, default_insurance_rate: Optional[float] = None,
    vat_recoverable: Optional[bool] = None,
):
    result = _landed.landed_cost(
        lines=lines, buyer_currency=buyer_currency, cross_border=cross_border, incoterm=incoterm,
        evaluation_date=evaluation_date, fx_rates=fx_rates, fx_table_date=fx_table_date,
        components=components, duty_rate=duty_rate, customs_basis=customs_basis,
        default_insurance_rate=default_insurance_rate, vat_recoverable=vat_recoverable)
    return UNASSESSED if result is None else result


_LANDED_FOR_TCO = landed_cost_total_cost(**_SPEC_CASE)


@formula(
    "tco.total_cost",
    version="1.0.0",
    owner="cost",
    purpose="The present value, at the evaluation date, of every cash outflow to acquire, use "
            "and dispose of the goods over the horizon, minus inflows",
    effective_from=_FROM,
    inputs=[
        Term("landed", RECORD, "the result of landed_cost.total_cost"),
        Term("cost_of_capital", RATIO, "annual discount rate r, from finance", minimum=0.0,
             maximum=1.0),
        Term("horizon_years", MONEY, "TCO horizon in years", minimum=0.0),
        Term("payment_terms", RECORD, "net_days, discount_pct, discount_days", required=False),
        Term("ap_can_take_discount", BOOLEAN,
             "whether AP reliably pays inside the discount window; unstated = not assumed",
             required=False),
        Term("lifetime_items", ROWS, "key, amount, day or year, direction; landed currency",
             required=False),
    ],
    output=Output("dict", MONEY,
                  "tco, tco_status (inherits landed_status), discount_choice, components with "
                  "pv, excluded_lifetime, warnings"),
    notes=(
        "PV(amount, days) = amount / (1 + r) ** (days / 365), days from the evaluation date. "
        "Unstated timing is day 0. The goods are paid on net terms unless AP takes the early-"
        "payment discount, in which case the cheaper present value is used and recorded. "
        "Lifetime items are included only when supplied; every unsupplied ownership item is "
        "listed in excluded_lifetime. Inventory carrying cost excludes capital cost, because r "
        "already covers it. Contingent charges and supply-disruption risk are not included."
    ),
    golden=[
        GoldenVector(
            inputs=dict(
                landed=_LANDED_FOR_TCO, cost_of_capital=0.08, horizon_years=3,
                payment_terms={"net_days": 60, "discount_pct": 0.02, "discount_days": 10},
                ap_can_take_discount=True,
                lifetime_items=[{"key": "maintenance_support", "amount": 500.0, "year": y}
                                for y in (1, 2, 3)]
                + [{"key": "disposal", "amount": 300.0, "year": 3}]),
            expected={"tco": 23356.17, "discount_choice": "discount",
                      "tco_status": "incomplete"},
            tolerance=0.005,
            note="2/10 net 60 taken; maintenance 500 in years 1-3; disposal 300 in year 3",
        ),
        GoldenVector(
            inputs=dict(landed=_LANDED_FOR_TCO, cost_of_capital=0.08, horizon_years=3,
                        payment_terms={"net_days": 60, "discount_pct": 0.02, "discount_days": 10},
                        ap_can_take_discount=False),
            expected={"discount_choice": "net", "tco": 19748.57 + 2270.76},
            tolerance=0.005,
            note="AP cannot take the discount: net terms, never discount with full-term timing",
        ),
    ],
)
def tco_total_cost(
    landed: Mapping[str, Any], cost_of_capital: float, horizon_years: float,
    payment_terms: Optional[Mapping[str, Any]] = None,
    ap_can_take_discount: Optional[bool] = None,
    lifetime_items: Sequence[Mapping[str, Any]] = (),
):
    if landed is UNASSESSED:
        return UNASSESSED
    return _tco.tco(
        landed=landed, cost_of_capital=cost_of_capital, horizon_years=horizon_years,
        payment_terms=payment_terms, ap_can_take_discount=ap_can_take_discount,
        lifetime_items=lifetime_items)


_SC_BOM = [{"material_id": "PP", "net_weight": 0.5, "price_per_kg": 2.0, "scrap_rate": 0.2,
            "scrap_recovery_price_pct": 0.5}]
_SC_OP = {"process": "injection_moulding", "cycle_time_s": 36, "cavities": 2, "oee": 0.8, "setup_time_h": 2,
          "operators_per_machine": 1, "machine_rate": 80.0, "labour_rate": 20.0}
_SC_CASE = dict(bom=_SC_BOM, routing=[_SC_OP], batch_size=1000, overhead_pct=0.10, sga_pct=0.05,
                margin_pct=0.08, logistics_per_unit=0.05, currency="GBP")


@formula(
    "cost.should_cost",
    version="1.0.0",
    owner="cost",
    purpose="Bottom-up modelled estimate of what a manufactured part should cost: material, conversion, "
            "labour, overhead, SG&A, logistics and margin per unit, with a tornado of drivers",
    effective_from=_FROM,
    inputs=[
        Term("bom", ROWS, "per line: net_weight (kg), price_per_kg, scrap_rate, scrap_recovery_price_pct, "
             "or bought_in_price for a bought-in part"),
        Term("routing", ROWS, "per operation: cycle_time_s, cavities, oee, setup_time_h, operators_per_machine, "
             "labour_rate, and machine_rate or the machine inputs to derive it"),
        Term("batch_size", COUNT, "units per setup", minimum=1.0),
        Term("overhead_pct", RATIO, "of conversion plus labour", minimum=0.0),
        Term("sga_pct", RATIO, "of manufacturing cost", minimum=0.0),
        Term("margin_pct", RATIO, "supplier margin", minimum=0.0),
        Term("logistics_per_unit", MONEY, "packaging, freight and duty per unit", required=False),
        Term("currency", CURRENCY_CODE, "all prices and rates are in this currency", required=False),
        Term("margin_base", LABEL, "what the margin is applied to", required=False,
             allowed=("value_add_only", "total_cost")),
        Term("annual_volume", COUNT, required=False, minimum=0.0),
        Term("sensitivity_pct", PERCENT, "tornado range either side", required=False, minimum=0.0),
        Term("monte_carlo_n", COUNT, "0 = none", required=False, minimum=0.0),
        Term("seed", COUNT, "required when monte_carlo_n is above 0", required=False),
        Term("distributions", RECORD, "driver -> {min, mode, max} multipliers of the base", required=False),
        Term("quote_breakdown", RECORD, "a supplier quote by element, to compare", required=False),
    ],
    output=Output("dict", FACTOR,
                  "unit_cost, per_unit by element, shares, operations, annual, rates_used with source, "
                  "assumptions, sensitivity (tornado, monte_carlo), quote_gap, label 'modelled estimate'; "
                  "UNASSESSED if the inputs cannot be costed"),
    notes=(
        "yield = 1 - scrap; gross = net / yield; material = gross x price less scrap at the recovery share. "
        "Conversion per unit = (cycle / 3600) / (cavities x oee) x machine rate + setup hours x rate / batch. "
        "Labour per unit = the same hours x operators x labour rate. Overhead on conversion plus labour; SG&A on "
        "manufacturing cost; margin on value-add only unless margin_base is total_cost. Every price and rate is "
        "supplied by the caller and listed with its source: a missing labour rate, an oee outside (0, 1], cavities "
        "below 1 or scrap of 1 or more make the whole estimate UNASSESSED. It is a modelled estimate, never a "
        "supplier's actual cost."
    ),
    golden=[
        GoldenVector(
            inputs=_SC_CASE,
            expected={"unit_cost": 2.214959, "label": "modelled estimate"},
            tolerance=1e-6,
            note="material 1.125, conversion 0.66, labour 0.125, overhead 0.0785, SG&A 0.099425, logistics 0.05, margin 0.077034",
        ),
        GoldenVector(
            inputs={**_SC_CASE, "routing": [{**_SC_OP, "cavities": 0}]},
            expected=UNASSESSED,
            note="a tool with no cavities cannot be costed",
        ),
    ],
)
def cost_should_cost(
    bom, routing, batch_size, overhead_pct, sga_pct, margin_pct, logistics_per_unit=None, currency=None,
    margin_base=None, annual_volume=None, sensitivity_pct=None, monte_carlo_n=None, seed=None,
    distributions=None, quote_breakdown=None,
):
    try:
        return _sc.should_cost(
            bom=bom, routing=routing, batch_size=batch_size, overhead_pct=overhead_pct, sga_pct=sga_pct,
            margin_pct=margin_pct, logistics_per_unit=logistics_per_unit or 0.0, currency=currency or "GBP",
            margin_base=margin_base or "value_add_only", annual_volume=annual_volume,
            sensitivity_pct=10.0 if sensitivity_pct is None else sensitivity_pct,
            monte_carlo_n=int(monte_carlo_n or 0), seed=None if seed is None else int(seed),
            distributions=distributions, quote_breakdown=quote_breakdown)
    except _sc.ShouldCostError:
        return UNASSESSED
