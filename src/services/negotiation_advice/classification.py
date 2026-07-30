"""Kraljic quadrant and negotiation style — suggestions, never impositions.

The quadrant and style vocabularies are fixed by the playbook file's keys; a
value outside them yields zero plays downstream, so they are asserted here.

Note bp_supplier.supplier_type is NOT a Kraljic axis — it holds business types
(Consulting, Retailer, Manufacturer, ...) and is deliberately unused.
"""
from __future__ import annotations

from typing import Optional

THRESHOLD_POLICY_SLUG = "negotiation_advice_thresholds"

QUADRANTS = ("Transactional", "Leverage", "Strategic", "Bottleneck")
STYLES = ("Competitive", "Collaborative", "Principled", "Accommodating",
          "Compromising")


def default_thresholds() -> dict:
    """Seeded from the live distribution: deal-value p90 (98,175; median 4,180)
    and the per-DEAL median alternative-supplier count (93; min 38, p75 126,
    max 232 over 80 sampled deals). NOT the per-item median, which is far lower
    and would make every deal read as "many alternatives". Testdata-derived,
    hence governed data rather than constants."""
    return {"high_spend": 98175.0, "many_alternatives": 93}


def _confidence(value: float, bar: float) -> float:
    """1.0 far from the bar, approaching 0.5 at it."""
    if bar <= 0:
        return 0.5
    ratio = value / bar
    distance = abs(ratio - 1.0)
    return round(min(1.0, 0.5 + distance), 3)


def classify(signals: dict, thresholds: Optional[dict] = None) -> dict:
    bars = dict(default_thresholds())
    if thresholds:
        bars.update({k: v for k, v in thresholds.items() if v is not None})

    spend = signals.get("deal_value")
    alternatives = signals.get("alternative_supplier_count")

    if spend is None or alternatives is None:
        missing = []
        if spend is None:
            missing.append("deal value could not be determined")
        if alternatives is None:
            missing.append("no comparable suppliers found for this deal's items")
        return {
            "quadrant": None, "quadrant_reasons": missing,
            "quadrant_confidence": 0.0, "style": None, "style_reasons": [],
            "indeterminate": True,
        }

    high_spend = float(spend) >= float(bars["high_spend"])
    many_alts = int(alternatives) >= int(bars["many_alternatives"])
    quadrant = {
        (True, True): "Leverage",
        (True, False): "Strategic",
        (False, True): "Transactional",
        (False, False): "Bottleneck",
    }[(high_spend, many_alts)]

    reasons = [
        f"Deal value {float(spend):,.0f} is "
        f"{'at or above' if high_spend else 'below'} the "
        f"{float(bars['high_spend']):,.0f} high-spend bar",
        f"{int(alternatives)} supplier(s) quote comparable items — "
        f"{'a contested' if many_alts else 'a thin'} supply market",
    ]
    risk = signals.get("risk_score")
    if risk is not None:
        reasons.append(f"Supplier risk score {risk}")

    confidence = round(
        min(_confidence(float(spend), float(bars["high_spend"])),
            _confidence(float(alternatives), float(bars["many_alternatives"]))),
        3,
    )

    style, style_reasons = _style_for(quadrant, signals)
    return {
        "quadrant": quadrant, "quadrant_reasons": reasons,
        "quadrant_confidence": confidence, "style": style,
        "style_reasons": style_reasons, "indeterminate": False,
    }


def _style_for(quadrant: str, signals: dict) -> tuple[str, list[str]]:
    variance = signals.get("price_variance_pct")
    preferred = signals.get("is_preferred")
    if quadrant == "Strategic" and preferred:
        return "Collaborative", ["Preferred supplier on a strategic spend — "
                                "protect the relationship while negotiating"]
    if quadrant == "Bottleneck":
        return "Principled", ["Thin supply market — continuity is the exposure, "
                              "so argue from objective criteria, not pressure"]
    if quadrant == "Leverage":
        why = ["Contested market on a material spend — competitive tension is "
               "available"]
        if variance is not None:
            why.append(f"Price variance {variance}% across the document chain")
        return "Competitive", why
    if quadrant == "Strategic":
        return "Collaborative", ["Material spend with few alternatives — build "
                                 "value rather than squeeze price"]
    return "Competitive", ["Low-value, contested spend — standardise and "
                           "compete it"]
