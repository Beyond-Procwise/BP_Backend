"""Purchase price variance (PPV): what was paid above or below the PO price, on the quantity billed.

``(invoice unit price - PO unit price) x quantity``, per matched invoice line. Positive means the
supplier billed MORE than the PO price (unfavourable); negative means less. The percentage is of
the standard spend, the PO price times the same quantity.

Pure arithmetic, no I/O; matching an invoice line to its PO line is the caller's job. Nothing is
guessed: a line with no usable price, quantity or currency, or a currency with no rate in the dated
table, is counted as unassessed and left out, never taken as a zero variance and never taken 1:1.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

from src.services.cost.landed_cost import FxTable


def _num(v: Any) -> Optional[float]:
    try:
        return None if v is None or isinstance(v, bool) else float(v)
    except (TypeError, ValueError):
        return None


def purchase_price_variance(
    *, lines: Sequence[Mapping[str, Any]], buyer_currency: Optional[str] = None,
    fx_rates: Optional[Mapping[str, Any]] = None, fx_table_date: Any = None,
) -> Optional[Dict[str, Any]]:
    """PPV over matched invoice/PO lines, or ``None`` when no line could be assessed."""
    by_ccy: Dict[str, Dict[str, float]] = {}
    unassessed = 0
    for ln in lines:
        po, inv, qty = (_num(ln.get("po_unit_price")), _num(ln.get("invoice_unit_price")),
                        _num(ln.get("quantity")))
        ccy = str(ln.get("currency") or "").strip().upper()
        if po is None or inv is None or qty is None or po < 0 or inv < 0 or qty <= 0 or not ccy:
            unassessed += 1
            continue
        b = by_ccy.setdefault(ccy, {"ppv": 0.0, "unfavourable": 0.0, "favourable": 0.0,
                                    "standard_spend": 0.0, "lines": 0})
        v = (inv - po) * qty
        b["ppv"] += v
        b["unfavourable" if v > 0 else "favourable"] += v if v != 0 else 0.0
        b["standard_spend"] += po * qty
        b["lines"] += 1
    if not by_ccy:
        return None

    target = (str(buyer_currency).strip().upper() if buyer_currency
              else (next(iter(by_ccy)) if len(by_ccy) == 1 else None))
    out: Dict[str, Any] = {
        "by_currency": {c: dict(b, lines=int(b["lines"])) for c, b in by_ccy.items()},
        "currency": None, "total_ppv": None, "unfavourable": None, "favourable": None,
        "standard_spend": None, "ppv_pct": None,
        "lines_assessed": sum(int(b["lines"]) for b in by_ccy.values()),
        "lines_unassessed": unassessed, "excluded_currencies": [],
        "fx_table_date": None,
    }
    if target is None:
        return out    # mixed currencies and no table to put them on one footing

    fx = FxTable(fx_rates, fx_table_date, target)
    totals = {"ppv": 0.0, "unfavourable": 0.0, "favourable": 0.0, "standard_spend": 0.0}
    converted_lines = 0
    for ccy, b in by_ccy.items():
        factor = fx.convert(1.0, ccy)
        if factor is None:
            out["excluded_currencies"].append(ccy)
            out["lines_unassessed"] += int(b["lines"])
            continue
        for k in totals:
            totals[k] += b[k] * factor
        converted_lines += int(b["lines"])
    out["excluded_currencies"].sort()
    out["lines_assessed"] = converted_lines
    if not converted_lines:
        return None
    out.update(currency=target, total_ppv=totals["ppv"], unfavourable=totals["unfavourable"],
               favourable=totals["favourable"], standard_spend=totals["standard_spend"],
               ppv_pct=(totals["ppv"] / totals["standard_spend"] * 100.0
                        if totals["standard_spend"] else None))
    if fx.used and fx_table_date:
        out["fx_table_date"] = (fx_table_date.isoformat() if hasattr(fx_table_date, "isoformat")
                                else str(fx_table_date))
    return out
