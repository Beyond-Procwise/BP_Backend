"""Price harmonisation, registered. Delegates to ``src.services.demand.price_harmonisation``."""
from __future__ import annotations

from datetime import date

from src.services.demand import price_harmonisation as _ph

from ..contract import COUNT, PERCENT, ROWS, Output, Term
from ..registry import GoldenVector, formula
from ..unassessed import UNASSESSED, decline

_LINES = [{"id": "Plant A", "price": 10, "quantity": 1000}, {"id": "Plant B", "price": 12, "quantity": 2000},
          {"id": "Plant C", "price": 13, "quantity": 500}]


@formula(
    "demand.price_harmonisation",
    version="1.0.0",
    owner="demand",
    purpose="The saving from buying one identical item at a reference price (a percentile of the prices paid, "
            "not the lowest) instead of the higher prices actually paid",
    effective_from=date(2026, 10, 21),
    inputs=[
        Term("lines", ROWS, "one per purchase of the same item: {id, price, quantity}, prices already like for like"),
        Term("percentile", PERCENT, "which percentile of the prices is the reference, from the skill settings",
             minimum=0.0, maximum=100.0),
        Term("min_observations", COUNT, "prices needed before a reference is computed; default 3", required=False,
             minimum=1.0),
    ],
    output=Output("dict", PERCENT, "reference_price, lines (each with its saving), total_saving, lowest_price, "
                  "highest_price, method, assumptions; UNASSESSED if the inputs cannot be harmonised"),
    notes=("Reference price is the percentile by linear interpolation between the two nearest prices. Saving per "
           "line is max(0, price - reference) x quantity, so a line at or below the reference saves nothing. "
           "Incoterms, payment terms, currency and unit of measure are not adjusted: normalise first. Fewer "
           "prices than the minimum give no reference."),
    golden=[
        GoldenVector(inputs={"lines": _LINES, "percentile": 25},
                     expected={"reference_price": 11.0, "total_saving": 3000.0}, tolerance=1e-9,
                     note="25th percentile of 10, 12, 13 is 11 (not the lowest, 10): saves 1 x 2000 + 2 x 500"),
        GoldenVector(inputs={"lines": _LINES[:2], "percentile": 25}, expected=UNASSESSED,
                     note="two prices are too few for a reference"),
    ],
)
def demand_price_harmonisation(lines, percentile, min_observations=None):
    try:
        return _ph.harmonise(lines, percentile, 3 if min_observations is None else min_observations)
    except _ph.HarmonisationError as exc:
        return decline(str(exc))
