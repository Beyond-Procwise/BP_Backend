"""The Opportunity Critic's arithmetic, registered.

The critic agent judges; it never calculates. Every number in a verdict comes
from here, with pinned behaviour, so that a verdict can be defended by pointing
at a formula version rather than at a model's mood.

The canonical false positive from the spec is a golden vector, not a comment:
two contracted 4% uplifts against ~3.8% CPI read as inflation. If someone later
widens the band and that vector stops reproducing, this module does not import.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Optional

from ..contract import BOOLEAN, COUNT, DAYS, DATE, MONEY, PERCENT, RATIO, TEXT, Output, Term
from ..registry import GoldenVector, formula
from ..unassessed import UNASSESSED

_OWNER = "opportunity_critic"
_FROM = date(2026, 9, 9)


def _num(value: Any) -> Optional[float]:
    """Coerce to float, or None. Never raises, never guesses."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):  # NaN / inf
        return None
    return out


@formula(
    "critic.annualised_rate",
    version="1.0.0",
    owner=_OWNER,
    purpose="Compound annual rate of change between an anchor price and the current one",
    effective_from=_FROM,
    inputs=[
        Term("anchor_value", MONEY, "the comparator price"),
        Term("current_value", MONEY, "the price today"),
        Term("years", COUNT, "elapsed years between them", minimum=0.0),
    ],
    output=Output("float", PERCENT, "percent per year, or UNASSESSED"),
    notes=(
        "Returns UNASSESSED rather than a number when the anchor is zero or the "
        "span is zero. A zero anchor is not a 0% rise; it is an unusable "
        "comparator, and returning 0.0 would present that as 'no change'."
    ),
    golden=[
        GoldenVector(
            inputs={"anchor_value": 0.041, "current_value": 0.0447, "years": 4.0},
            expected=2.1835, tolerance=0.001,
            note="THE CANONICAL FALSE POSITIVE: managed print, two 4% uplifts "
                 "2022-2026. Must stay inside a 3.8% index +/- 2pp band.",
        ),
        GoldenVector(inputs={"anchor_value": 100.0, "current_value": 100.0, "years": 1.0},
                     expected=0.0),
        GoldenVector(inputs={"anchor_value": 100.0, "current_value": 110.0, "years": 1.0},
                     expected=10.0, tolerance=0.001),
        GoldenVector(inputs={"anchor_value": 0.0, "current_value": 50.0, "years": 4.0},
                     expected=UNASSESSED,
                     note="fail-closed: a zero anchor cannot produce a rate"),
        GoldenVector(inputs={"anchor_value": 100.0, "current_value": 110.0, "years": 0.0},
                     expected=UNASSESSED, note="fail-closed: no elapsed time"),
    ],
)
def annualised_rate(anchor_value=None, current_value=None, years=None):
    anchor, current, span = _num(anchor_value), _num(current_value), _num(years)
    if not anchor or current is None or not span or span <= 0:
        return UNASSESSED
    if anchor <= 0 or current <= 0:
        return UNASSESSED
    return ((current / anchor) ** (1.0 / span) - 1.0) * 100.0


@formula(
    "critic.excess_over_index",
    version="1.0.0",
    owner=_OWNER,
    purpose="The part of a price rise that is above index -- the only part that is an opportunity",
    effective_from=_FROM,
    inputs=[
        Term("annualised_pct", PERCENT, "measured annual rise"),
        Term("index_pct", PERCENT, "the applicable index for category and jurisdiction",
             required=False),
        Term("band_pp", PERCENT, "tolerance in percentage points", minimum=0.0),
    ],
    output=Output("float", PERCENT, "excess percentage points, 0.0 if inside band, "
                                    "or UNASSESSED with no index"),
    notes=(
        "UNASSESSED with no index is the whole point. The prompt forbids guessing "
        "a rate, and there is no index source in this deployment -- so this "
        "formula returns UNASSESSED for every candidate until one is supplied, "
        "and the gap register says so out loud."
    ),
    golden=[
        GoldenVector(inputs={"annualised_pct": 4.0, "index_pct": 3.8, "band_pp": 2.0},
                     expected=0.0,
                     note="PINS THE CANONICAL FALSE POSITIVE: inside band is inflation"),
        GoldenVector(inputs={"annualised_pct": 9.0, "index_pct": 3.8, "band_pp": 2.0},
                     expected=3.2, tolerance=0.001,
                     note="the opportunity is the excess, never the whole delta"),
        GoldenVector(inputs={"annualised_pct": 2.0, "index_pct": 3.8, "band_pp": 2.0},
                     expected=0.0, note="below index is not a negative opportunity"),
        GoldenVector(inputs={"annualised_pct": 9.0, "index_pct": None, "band_pp": 2.0},
                     expected=UNASSESSED,
                     note="fail-closed: no index, no verdict. Never guess a rate."),
    ],
)
def excess_over_index(annualised_pct=None, index_pct=None, band_pp=None):
    measured, index, band = _num(annualised_pct), _num(index_pct), _num(band_pp)
    if measured is None or index is None:
        return UNASSESSED
    ceiling = index + (band or 0.0)
    return max(0.0, measured - ceiling) if measured > ceiling else 0.0


@formula(
    "critic.anchor_age_days",
    version="1.0.0",
    owner=_OWNER,
    purpose="How old the comparator is, so staleness can be tested against a governed rule",
    effective_from=_FROM,
    inputs=[
        Term("anchor_date", DATE, "when the anchor price was observed", required=False),
        Term("current_date", DATE, "when the current price was observed", required=False),
    ],
    output=Output("int", DAYS, "days between, or UNASSESSED if either is undated"),
    notes=(
        "An undated anchor is UNASSESSED, not age zero. The live detector emits "
        "no anchor date at all (spec 6.1), so this returning UNASSESSED is the "
        "normal case until the evidence subagent dates it via invoices."
    ),
    golden=[
        GoldenVector(inputs={"anchor_date": date(2022, 3, 1),
                             "current_date": date(2026, 3, 1)}, expected=1461),
        GoldenVector(inputs={"anchor_date": date(2026, 3, 1),
                             "current_date": date(2026, 3, 1)}, expected=0),
        GoldenVector(inputs={"anchor_date": None, "current_date": date(2026, 3, 1)},
                     expected=UNASSESSED,
                     note="fail-closed: an undated anchor has unknown age, not zero age"),
    ],
)
def anchor_age_days(anchor_date=None, current_date=None):
    if not isinstance(anchor_date, date) or not isinstance(current_date, date):
        return UNASSESSED
    return (current_date - anchor_date).days


@formula(
    "critic.unit_basis_match",
    version="1.0.0",
    owner=_OWNER,
    purpose="Whether anchor and current are priced on the same basis -- mismatch invalidates outright",
    effective_from=_FROM,
    inputs=[
        Term("anchor_basis", TEXT, "per page / per seat / per month ...", required=False),
        Term("current_basis", TEXT, "the same, for the current price", required=False),
    ],
    output=Output("bool", BOOLEAN, "True if comparable, False if not, UNASSESSED if unknown"),
    notes=(
        "Compared after case-folding and separator-stripping only. No synonym "
        "table: deciding that 'per user' means 'per seat' is a judgement, and "
        "judgement belongs to the agent, not to a formula."
    ),
    golden=[
        GoldenVector(inputs={"anchor_basis": "per_page", "current_basis": "per_page"},
                     expected=True),
        GoldenVector(inputs={"anchor_basis": "Per Page", "current_basis": "per_page"},
                     expected=True, note="case and separators are not a difference"),
        GoldenVector(inputs={"anchor_basis": "per_page", "current_basis": "per_month"},
                     expected=False, note="unit mismatch invalidates outright"),
        GoldenVector(inputs={"anchor_basis": None, "current_basis": "per_page"},
                     expected=UNASSESSED,
                     note="fail-closed: an unstated basis is unknown, not matching"),
    ],
)
def unit_basis_match(anchor_basis=None, current_basis=None):
    def _norm(value):
        if value is None:
            return None
        text = str(value).strip().lower()
        for ch in (" ", "-", "_", "/"):
            text = text.replace(ch, "")
        return text or None

    left, right = _norm(anchor_basis), _norm(current_basis)
    if left is None or right is None:
        return UNASSESSED
    return left == right


@formula(
    "critic.fabricated_anchor",
    version="1.0.0",
    owner=_OWNER,
    purpose="Whether an anchor price is an artefact of the miner's missing-data fallbacks",
    effective_from=_FROM,
    inputs=[
        Term("unit_price", MONEY, "the anchor's unit price", required=False),
        Term("line_value", MONEY, "the anchor line's total value", required=False),
        Term("quantity", COUNT, "the anchor line's quantity", required=False),
    ],
    output=Output("bool", BOOLEAN, "True if the anchor is an artefact, not a price"),
    notes=(
        "THIS RULE IS NOT IN THE ORIGINAL PROMPT. It exists because of what the "
        "live detector actually does (spec 6.1): at "
        "opportunity_miner_agent.py:5545 a missing quantity becomes 1.0 and a "
        "missing unit price becomes the whole line value, and the anchor is then "
        "a .min() over that column -- the aggregation that selects for whichever "
        "row is most corrupted downward. A unit price identical to its line "
        "value at quantity exactly 1.0 is the fingerprint. It INVALIDATES rather "
        "than downgrades: a fabricated floor is not a weak comparator, it is not "
        "a price."
    ),
    golden=[
        GoldenVector(inputs={"unit_price": 4540.26, "line_value": 4540.26, "quantity": 1.0},
                     expected=True, note="the fallback fingerprint"),
        GoldenVector(inputs={"unit_price": 100.0, "line_value": 250.0, "quantity": 1.0},
                     expected=False,
                     note="genuine qty-1 lines exist; the coincidence is the signal"),
        GoldenVector(inputs={"unit_price": 50.0, "line_value": 500.0, "quantity": 10.0},
                     expected=False),
        GoldenVector(inputs={"unit_price": None, "line_value": 500.0, "quantity": 10.0},
                     expected=UNASSESSED,
                     note="fail-closed: cannot clear an anchor you cannot see"),
    ],
)
def fabricated_anchor(unit_price=None, line_value=None, quantity=None):
    price, value, qty = _num(unit_price), _num(line_value), _num(quantity)
    if price is None or value is None or qty is None:
        return UNASSESSED
    return abs(qty - 1.0) < 1e-9 and abs(price - value) < 1e-9
