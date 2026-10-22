"""Quality and sustainability arithmetic from the library sheet (rows 102, 107): supplier parts per million
and Scope 3 emissions from activity data and emission factors the user supplies. The platform holds no
inspection register and no emission factors; nothing here guesses either."""
from __future__ import annotations

from datetime import date
from typing import Any, Mapping, Optional, Sequence

from ..contract import COUNT, RATIO, ROWS, Output, Term
from ..registry import GoldenVector, formula

_OWNER = "analytics"
_FROM = date(2026, 10, 22)


def _num(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if out != out or out in (float("inf"), float("-inf")) else out


@formula(
    "quality.ppm",
    version="1.0.0",
    owner=_OWNER,
    purpose="Supplier defect rate in parts per million from defects found and units received or inspected",
    effective_from=_FROM,
    inputs=[
        Term("defective_units", COUNT, "units found defective in the period, from the user's inspection records", required=False, minimum=0.0),
        Term("units_inspected", COUNT, "units received or inspected in the same period", required=False, minimum=0.0),
    ],
    output=Output("float", RATIO, "defective_units / units_inspected * 1,000,000, 1dp; None when nothing was inspected"),
    notes="The denominator is what was inspected, not what was ordered; the caller says which.",
    golden=[
        GoldenVector(inputs={"defective_units": 12, "units_inspected": 48000}, expected=250.0),
        GoldenVector(inputs={"defective_units": 0, "units_inspected": 1000}, expected=0.0),
        GoldenVector(inputs={"defective_units": 5, "units_inspected": 0}, expected=None),
        GoldenVector(inputs={"defective_units": 7, "units_inspected": 5}, expected=None, note="more defects than units inspected is a data error, not a rate"),
    ],
)
def ppm(defective_units=None, units_inspected=None):
    d, n = _num(defective_units), _num(units_inspected)
    if d is None or n is None or n <= 0 or d < 0 or d > n:
        return None
    return round(d / n * 1_000_000.0, 1)


@formula(
    "sustainability.scope3",
    version="1.0.0",
    owner=_OWNER,
    purpose="Scope 3 emissions from activity data and emission factors the user supplies, per line and in total",
    effective_from=_FROM,
    inputs=[
        Term("activities", ROWS, "rows of {activity, quantity, unit, emission_factor (kgCO2e per unit), factor_source}"),
    ],
    output=Output("dict", COUNT, "{total_kgco2e, total_tco2e, lines: [...], lines_without_factor}; None when no line carries both a quantity and a factor"),
    notes=(
        "Factors are never supplied here: a line without a factor is counted in lines_without_factor and left out of the total, "
        "which the caller reports as a lower bound. factor_source is carried through so the answer can cite it."
    ),
    golden=[
        GoldenVector(
            inputs={"activities": [{"activity": "road freight", "quantity": 1000, "unit": "tonne-km", "emission_factor": 0.1, "factor_source": "DEFRA 2025"},
                                   {"activity": "steel", "quantity": 20, "unit": "tonne", "emission_factor": 1900, "factor_source": "supplier EPD"},
                                   {"activity": "air freight", "quantity": 50, "unit": "tonne-km"}]},
            expected={"total_kgco2e": 38100.0, "total_tco2e": 38.1, "lines_without_factor": 1,
                      "lines": [{"activity": "road freight", "kgco2e": 100.0, "factor_source": "DEFRA 2025"},
                                {"activity": "steel", "kgco2e": 38000.0, "factor_source": "supplier EPD"}]},
        ),
        GoldenVector(inputs={"activities": [{"activity": "x", "quantity": 5}]}, expected=None),
    ],
)
def scope3(activities: Sequence[Mapping[str, Any]] = ()):
    lines, missing, total = [], 0, 0.0
    for row in activities or ():
        q, f = _num((row or {}).get("quantity")), _num((row or {}).get("emission_factor"))
        if q is None or f is None or q < 0 or f < 0:
            missing += 1
            continue
        kg = q * f
        total += kg
        lines.append({"activity": str((row or {}).get("activity")), "kgco2e": round(kg, 2), "factor_source": (row or {}).get("factor_source")})
    if not lines:
        return None
    return {"total_kgco2e": round(total, 2), "total_tco2e": round(total / 1000.0, 3), "lines_without_factor": missing, "lines": lines}
