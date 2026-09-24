"""Plain-language finding text (spec §8.1, triage spec §10 "Writing findings").

Field and direction first; both documents' values with their ids; exposure in GBP and
the document currency where it differs; knock-on effects on one line.
"""
from __future__ import annotations

from decimal import Decimal

from .model import Finding, Outcome, money, pct_change

_EFFECT = {"cumulative_total": "PO running total", "quantity": "quantity invoiced",
           "unit_price": "unit price"}


def _avg_pct(f: Finding) -> Decimal:
    ps = [p for p in (pct_change(c) for c in f.causes) if p is not None]
    return sum(ps, Decimal("0")) / len(ps) if ps else Decimal("0")


def _headline(f: Finding) -> str:
    r = f.lead
    n = len(f.causes)
    up = r.delta is not None and r.delta > 0
    heads = {
        "unit_price": lambda: f"Unit price {'above' if up else 'below'} PO on line {r.claim_line}",
        "uniform_uplift": lambda: (f"Prices {abs(_avg_pct(f)):.1f}% "
                                   f"{'above' if up else 'below'} PO on {n} lines"),
        "quantity": lambda: f"Quantity above PO {r.po_id} on {n} line{'s' if n != 1 else ''}",
        "cumulative_total": lambda: f"Invoices exceed PO {r.claim_doc} total",
        "duplicate": lambda: f"Possible duplicate of {r.auth_doc or 'an earlier invoice'}",
        "tax_rate": lambda: f"Tax rate {r.claim_value} is not an allowed rate",
        "currency": lambda: f"Invoice currency {r.claim_value} differs from PO ({r.auth_value})",
        "supplier": lambda: "Invoice supplier differs from PO",
        "invoice_date": lambda: "Invoice dated before the PO",
        "payment_terms": lambda: "Payment terms differ from PO",
        "description": lambda: f"Line {r.claim_line} description differs from PO",
        "unlinked_line": lambda: f'Invoice line with no PO line: "{r.claim_value}"',
        "bad_po_ref": lambda: f"Invoice names PO {r.claim_value}, which does not exist",
        "no_po": lambda: "Invoice has no PO",
        "line_arithmetic": lambda: f"Line {r.claim_line} amount is not quantity × price",
        "invoice_totals": lambda: f"Invoice {r.field_name} does not add up",
    }
    head = heads.get(f.rule_id, lambda: f.rule_id.replace("_", " "))()
    if any(c.outcome == Outcome.UNVERIFIABLE for c in f.causes):
        head = f"Could not verify: {head[0].lower()}{head[1:]}"
    return head


def describe(f: Finding) -> Finding:
    r = f.lead
    head = _headline(f)
    if f.exposure_gbp is not None:
        exposure = money(f.exposure_gbp, "GBP")
        if (r.currency or "GBP").upper() != "GBP":
            exposure += f" ({money(f.exposure, r.currency)})"
    else:
        exposure = f"{money(f.exposure, r.currency)} (no FX rate)"
    values = (f"{r.auth_doc or 'expected'} {r.auth_value if r.auth_value is not None else '-'}"
              f" · {r.claim_doc} {r.claim_value if r.claim_value is not None else '-'}")
    effects = ""
    if f.effects:
        effects = "Also changes: " + "; ".join(
            f"{_EFFECT.get(e.rule_id, e.rule_id.replace('_', ' '))} "
            f"({money(abs(e.exposure), e.currency)})" for e in f.effects)
    f.headline = head
    f.text = " · ".join(p for p in (head, exposure, values, r.note, effects) if p)
    return f
