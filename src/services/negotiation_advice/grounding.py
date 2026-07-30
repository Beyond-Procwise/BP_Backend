"""Test each play against the deal's own evidence.

A generic play is only advice when its precondition holds. "Leverage competitor
quotes" is strong when a second quote exists and damaging when it does not — a
buyer who bluffs one loses standing with the supplier. So nothing is marked
`ready` without the evidence behind it.
"""
from __future__ import annotations

from typing import Optional

PLAY_STATES = ("ready", "groundwork", "not_applicable")

# Order matters: the first family whose keyword appears wins. competitive_tension
# must precede price_challenge, or "Leverage competitor quotes to pressure pricing"
# matches on "pric" and gets tested against price variance instead of against the
# existence of a second supplier.
_FAMILY_KEYWORDS = (
    ("competitive_tension", ("competitor", "e-auction", "bidders",
                             "dual-sourc", "competitive")),
    ("overbilling", ("refund", "credit", "non-compliant")),
    ("volume", ("volume", "tiered", "bundle", "rebate")),
    ("price_challenge", ("benchmark", "cost breakdown", "should-cost", "price")),
)


def _family(play_text: str) -> Optional[str]:
    low = (play_text or "").lower()
    for family, keywords in _FAMILY_KEYWORDS:
        if any(k in low for k in keywords):
            return family
    return None


def _ev(label: str, value, source: str) -> dict:
    return {"label": label, "value": value, "source": source}


def assess(play: dict, signals: dict) -> dict:
    family = _family(play.get("play", ""))
    out = dict(play)
    out["family"] = family
    out["evidence"] = []
    out.pop("unlocked_by", None)

    if family is None:
        out["state"] = "ready"
        return out

    if family == "competitive_tension":
        quotes = signals.get("quote_supplier_count") or 0
        alts = signals.get("alternative_supplier_count")
        if quotes >= 2:
            out["state"] = "ready"
            out["evidence"] = [_ev("Quote suppliers on this deal", quotes, "deal")]
        elif alts is not None and alts >= 2:
            out["state"] = "groundwork"
            out["unlocked_by"] = (
                f"Needs a competing quote — {alts} suppliers quote comparable items"
            )
        elif alts is not None:
            # Known to be sole-source: there is no second supplier to go and find,
            # so telling the buyer to run a competitive event is noise, not advice.
            out["state"] = "not_applicable"
        else:
            out["state"] = "groundwork"
            out["unlocked_by"] = ("Needs a competing quote — no comparable "
                                  "suppliers identified yet")
        return out

    if family == "overbilling":
        inv, po = signals.get("invoice_total"), signals.get("po_total")
        if inv is not None and po is not None and inv > po:
            out["state"] = "ready"
            out["evidence"] = [_ev("Invoiced", f"{inv:,.2f}", "deal"),
                               _ev("PO value", f"{po:,.2f}", "deal")]
        else:
            out["state"] = "groundwork"
            out["unlocked_by"] = ("Needs matched invoice and PO totals showing "
                                  "an overcharge")
        return out

    if family == "volume":
        value = signals.get("deal_value")
        if value is not None:
            out["state"] = "ready"
            out["evidence"] = [_ev("Deal value", f"{value:,.2f}", "deal")]
        else:
            out["state"] = "groundwork"
            out["unlocked_by"] = "Needs a known deal value to size a tier"
        return out

    variance = signals.get("price_variance_pct")
    if variance is not None:
        out["state"] = "ready"
        out["evidence"] = [_ev("Price variance", f"{variance}%", "deal")]
    else:
        out["state"] = "groundwork"
        out["unlocked_by"] = "Needs a benchmark or price variance to argue from"
    return out


def apply_states(plays: list[dict], signals: dict) -> list[dict]:
    assessed = [assess(p, signals) for p in plays]
    usable = [p for p in assessed if p["state"] != "not_applicable"]
    usable.sort(key=lambda p: (0 if p["state"] == "ready" else 1,
                               -float(p.get("score") or 0.0)))
    return usable
